import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as k
from Layers import openai_entropy

class AC_Network:
    def __init__(self, input_shape, num_actions, sess, is_training=True, model = None):
        self.input_shape = input_shape
        self.value_s = None
        self.action_s = None
        self.num_actions = num_actions
        self.sess = sess

        # Dicounting hyperparams for loss functions
        self.entropy_coef = 0.01
        self.value_function_coeff = 0.5
        self.max_grad_norm = 40.0
        self.learning_rate = 7e-4
        self.alpha = 0.99
        self.epsilon = 1e-5

        # Model variables
        (_, hight, width, stack) = self.input_shape
        self.input_def = tf.keras.layers.Input(shape=(hight, width*stack, 1), name="input_layer", dtype=tf.float32)
        self.keep_prob = tf.placeholder(tf.float32, (None), "keep_prob")

        # Define Convolution 1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            kernel_initializer=keras.initializers.Orthogonal(gain=.7, seed=None),
            padding="valid",
            activation="relu", 
            name="conv1")
        
        self.maxPool1 = tf.keras.layers.MaxPooling2D(pool_size = (3, 3), name = "maxPool1")
        self.dropout = tf.keras.layers.Dropout(self.keep_prob)

        # define Convolution 2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            kernel_initializer=keras.initializers.Orthogonal(gain=.7, seed=None),
            padding="valid",
            activation="relu",
            name="conv2")

        self.maxPool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name="maxPool2")
        
        # define Convolution 3
        self.conv3 = tf.keras.layers.Conv2D(
            filters=96,
            kernel_size=[2, 2],
            kernel_initializer=keras.initializers.Orthogonal(gain=.7, seed=None),
            padding="valid",
            activation="relu",
            name="conv3")

        self.maxPool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name ="maxPool3")
        self.flattened = tf.keras.layers.Flatten(name="flattening_layer")
        self.hiddenLayer = tf.keras.layers.Dense(
            512,
            activation="relu",
            # bias_regularizer=tf.keras.regularizers.l2(0.01),
            kernel_initializer=keras.initializers.Orthogonal(gain=.7, seed=None),
            name="hidden_layer")

        # Output Layer consisting of an Actor and a Critic
        self._value = tf.keras.layers.Dense(
            1,
            kernel_initializer=keras.initializers.Orthogonal(gain=.7, seed=None),
            activation='linear',
            name="value_layer")
   
        self._policy = tf.keras.layers.Dense(
            7,
            activation='linear',
            kernel_initializer=keras.initializers.Orthogonal(gain=.7, seed=None),
            name="policy_layer")

        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.batch_normalization3 = tf.keras.layers.BatchNormalization()

        # Define the forward pass for the convolutional and hidden layers
        conv1_out = self.conv1(self.input_def)
        conv1_out = self.batch_normalization1(conv1_out) 
        maxPool1_out = self.maxPool1(conv1_out)

        conv2_out = self.conv2(maxPool1_out)
        conv2_out = self.batch_normalization2(conv2_out) 
        maxPool2_out = self.maxPool2(conv2_out)

        conv3_out = self.conv3(maxPool2_out)
        conv3_out = self.batch_normalization3(conv3_out) 
        maxPool3_out = self.maxPool3(conv3_out)
        
        flattened_out = self.flattened(maxPool3_out)
        hidden_out = self.dropout(flattened_out)

        # Actor and the Critic outputs
        self.value = self._value(hidden_out)
        self.logits = self._policy(hidden_out)
        self.action_dist = tf.nn.softmax(self.logits)

        # Final model
        self.model = tf.keras.Model(inputs=[self.input_def, ], outputs=[self.logits, self.value])

        # have to initialize before threading
        self.model._make_predict_function()
    
        # Batch data that will be sent to Model by the coordinator
        self.actions = tf.keras.backend.placeholder(dtype=tf.int32, shape=[None])
        self.actions_hot = tf.one_hot(self.actions, 7, dtype=tf.float32)
        self.advantages = tf.keras.backend.placeholder(dtype=tf.float32, shape=[None])
        self.rewards = tf.keras.backend.placeholder(dtype=tf.float32, shape=[None])
        self.values = tf.keras.backend.placeholder(dtype=tf.float32, shape=[None])
        
        # Responsible Outputs -log π(a_i|s_i)i
        # self.neg_log_prob = tf.reduce_sum(self.logits * self.actions_hot + 1e-10, [1])
        # self.policy_loss = tf.reduce_mean(-1.0 * tf.log(self.neg_log_prob) * self.advantages)

        # Policy Loss:  (1 / n) * ∑ * -log π(a_i|s_i) * A(s_i, a_i) 
        self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.actions_hot)
        self.policy_loss = tf.reduce_mean(self.neg_log_prob * self.advantages)
        
        # Value loss "MSE": (1 / n) * ∑[V(i) - R_i]^2
        self.value_loss = tf.reduce_mean(tf.square(self.rewards - self.values)) 

        # Entropy: - (1 / n) * ∑ P_i * Log (P_i)
        # self.entropy = - tf.reduce_mean(self.logits * tf.log(self.logits + 1e-10))
        # self.entropy = tf.reduce_mean(openai_entropy(self.logits))

        # Total loss: Policy loss - entropy * entropy coefficient + value coefficient * value loss
        #  - (self.entropy * self.entropy_coef)
        self.loss = self.policy_loss + (self.value_loss * self.value_function_coeff)
        params = self.model.trainable_weights
        grads = tf.gradients(self.loss, params)            
        grads, self.global_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

        # Apply Gradients
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=self.epsilon)
        
        # Update network weights
        self.optimize = optimizer.apply_gradients(zip(grads, params))
        
        
