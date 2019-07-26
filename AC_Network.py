import os
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as k
# from Layers import openai_entropy

class AC_Model(tf.keras.Model):
    def __init__(self, input_s, num_actions, is_training=True):
        super(AC_Model, self).__init__()
        self.value_s = None
        self.action_s = None
        self.num_actions = num_actions
        self.is_training = is_training

        # Dicounting hyperparams for loss functions
        self.entropy_coef = 1.0 # 0.01
        self.value_function_coeff = 1.0 # .50
        self.max_grad_norm = 40.0
        self.learning_rate = 7e-4
        self.alpha = 0.99
        self.epsilon = 1e-5

        # Model variables
        (_, hight, width, stack) = input_s
        self.input_def = tf.keras.layers.Input(shape=(hight, width*stack, 1), name="input_layer", dtype=tf.float32)

        # Define Convolution 1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            kernel_initializer=keras.initializers.Orthogonal(gain=2.0, seed=None),
            padding="valid",
            activation="relu", 
            name="conv1",
            trainable=True)
        
        self.maxPool1 = tf.keras.layers.MaxPooling2D(pool_size = (3, 3), name = "maxPool1")

        # define Convolution 2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            kernel_initializer=keras.initializers.Orthogonal(gain=2.0, seed=None),
            padding="valid",
            activation="relu",
            name="conv2")

        self.maxPool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name="maxPool2",
            trainable=True)
        
        # define Convolution 3
        self.conv3 = tf.keras.layers.Conv2D(
            filters=96,
            kernel_size=[2, 2],
            kernel_initializer=keras.initializers.Orthogonal(gain=2.0, seed=None),
            padding="valid",
            activation="relu",
            name="conv3",
            trainable=True)

        self.maxPool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name ="maxPool3",
            trainable=True)
        self.flattened = tf.keras.layers.Flatten(name="flattening_layer",
            trainable=True)
        self.hiddenLayer = tf.keras.layers.Dense(
            512,
            activation="relu",
            # bias_regularizer=tf.keras.regularizers.l2(0.01),
            kernel_initializer=keras.initializers.Orthogonal(gain=2.0, seed=None),
            name="hidden_layer",
            trainable=True)

        # Output Layer consisting of an Actor and a Critic
        self._value = tf.keras.layers.Dense(
            1,
            kernel_initializer=keras.initializers.Orthogonal(gain=2.0, seed=None),
            activation='linear',
            name="value_layer",
            trainable=True)
   
        self._policy = tf.keras.layers.Dense(
            self.num_actions,
            activation='linear',
            kernel_initializer=keras.initializers.Orthogonal(gain=2.0, seed=None),
            name="policy_layer",
            trainable=True)

        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.batch_normalization3 = tf.keras.layers.BatchNormalization()
        self.trainables = [self.conv1,  self.maxPool1, self.conv2, self.maxPool2, self.conv3, self.maxPool3, self.flattened, self._policy, self._value]

    def call(self, input_image, keep_p=1.0):
        conv1_out = self.conv1(input_image)
        conv1_out = self.batch_normalization1(conv1_out) 
        maxPool1_out = self.maxPool1(conv1_out)

        conv2_out = self.conv2(maxPool1_out)
        conv2_out = self.batch_normalization2(conv2_out) 
        maxPool2_out = self.maxPool2(conv2_out)

        conv3_out = self.conv3(maxPool2_out)
        conv3_out = self.batch_normalization3(conv3_out) 
        maxPool3_out = self.maxPool3(conv3_out)
        
        flattened_out = self.flattened(maxPool3_out)

        # hidden_out = None
        hidden_out = flattened_out
        if self.is_training:
            # flattened_out, keep_prob = keep_p)
            hidden_out = tf.nn.dropout(flattened_out, 1.0 - keep_p)
        else:
            hidden_out = flattened_out

        # Actor and the Critic outputs
        self.value = self._value(hidden_out)
        self.logits = self._policy(hidden_out)
        self.action_dist = tf.nn.softmax(self.logits)

        return self.logits, self.action_dist, self.value
    
    # Get watched trainable variables
    def get_variables(self):
        variables = []
        for var in self.trainables:

            variables.append(var.variables)

        return variables

    # pass a tuple of (batch_states, batch_actions, batch_advantages, batch_rewards)
    def train_batch(self, train_data):
        with tf.GradientTape() as tape:
            for var in self.trainables:
                tape.watch(var.variables)
            
            batch_states, batch_actions, batch_advantages, batch_rewards, batch_values, batch_logits = train_data

            actions = tf.Variable(batch_actions, name="Actions", trainable=False)
            rewards = tf.Variable(batch_rewards, name="Rewards")
            actions_hot = tf.one_hot(actions, self.num_actions, dtype=tf.float32)

            logits, action_dist, values = self.call(tf.convert_to_tensor(batch_states, dtype=tf.float32))
            advantages = rewards - values

            # Entropy: - ∑ P_i * Log (P_i)
            entropy = self.softmax_entropy(action_dist)

            # Policy Loss:  (1 / n) * ∑ * -log π(a_i|s_i) * A(s_i, a_i) 
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=actions_hot)
            policy_loss = (neg_log_prob * tf.stop_gradient(advantages)) - (entropy * self.entropy_coef)
            
            # Value loss "MSE": (1 / n) * ∑[V(i) - R_i]^2
            value_loss = tf.square(advantages)

            loss = tf.reduce_mean(policy_loss + (value_loss * self.value_function_coeff)) 

            # Apply Gradients
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=self.epsilon)
            # optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, epsilon=self.epsilon)

            # params = test_4
            params = tape.watched_variables()
            
            grads = tape.gradient(loss, params)

            grads, global_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

            optimizer.apply_gradients(zip(grads, params))
        
            return loss.numpy()
    
    # Makes a step in the environment
    def step(self, observation, keep_per):

        softmax, logits, value = self.call(observation, keep_per)
        return logits.numpy(), softmax.numpy(), value.numpy()

    # Returns the critic estimation of the current state value
    def value(self, observation):
        action_dist, softmax, value = self.call(observation, 1.0)
        return value.numpy()[0]

    def save_model(self): 
        current_dir = os.getcwd()   
        model_save_path = current_dir + '\Model\checkpoint.tf'
        self.save_weights(model_save_path, save_format='tf')

    def load_model(self):
        current_dir = os.getcwd()
        model_save_path = current_dir + '\Model\checkpoint.tf'
        self.load_weights(filepath=model_save_path)

    # Turn logits to softmax and calculate entropy
    def logits_entropy(self, logits):
        
        a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), 1)

    # standard entropy
    def softmax_entropy(self, p0):
        return - tf.reduce_sum(p0 * tf.math.log(p0 + 1e-20), axis=1)