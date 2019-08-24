import os
import numpy as np

import tensorflow as tf

class AC_Model_Small(tf.keras.Model):
    def __init__(self, input_s, num_actions, config, is_training=True):
        super(AC_Model_Small, self).__init__()
        self.value_s = None
        self.action_s = None
        self.num_actions = num_actions
        self.training = is_training

        # Dicounting hyperparams for loss functions
        self.entropy_coef = config['Entropy coeff']
        self.value_function_coeff = config['Value loss coeff']
        self.max_grad_norm = config['Max grad norm']
        self.learning_rate = config['Learning rate']
        self.epsilon = config['Epsilon']
    
          # Define Convolution 1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[8, 8],
            strides=(4, 4),
            kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2.0)),
            padding="valid",
            activation="relu", 
            name="conv1",
            trainable=is_training )
        
        self.maxPool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="maxPool1")

        # define Convolution 2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            strides=[2, 2],
            kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2.0)),
            padding="valid",
            activation="relu",
            name="conv2", 
            trainable=is_training)

        self.maxPool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="maxPool2", trainable=is_training )
        
        self.flattened = tf.keras.layers.Flatten(name="flattening_layer", trainable=is_training )
        
        self.hiddenLayer = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2.0)),
            name="hidden_layer", 
            trainable=is_training )

        self.lstm = tf.keras.layers.SimpleRNN(256, trainable=is_training)

        # Output Layer consisting of an Actor and a Critic
        self._value = tf.keras.layers.Dense(
            1,
            kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2.0)),
            activation='linear',
            name="value_layer",
            trainable=is_training )
   
        self._policy = tf.keras.layers.Dense(
            self.num_actions,
            activation='linear',
            kernel_initializer=keras.initializers.Orthogonal(gain=.3),
            name="policy_layer", trainable=is_training )

        # Batch regularization
        self.batch_reg1 = tf.keras.layers.BatchNormalization()
        self.batch_reg2 = tf.keras.layers.BatchNormalization()

        # Dropout layers to prevent overfitting
        self.spatial_dropout1 = tf.keras.layers.SpatialDropout2D(rate=.5, trainable=is_training)
        self.spatial_dropout2 = tf.keras.layers.SpatialDropout2D(rate=.5, trainable=is_training)
        
        self.linear_dropout = tf.keras.layers.Dropout(rate=.8, trainable=is_training)

    def call(self, input_image, keep_p=1.0):

        # Feature maps one
        conv1_out = self.conv1(input_image)
        batch_reg1_out = self.batch_reg1(conv1_out)
        spatial_dropout1_out = self.spatial_dropout1(batch_reg1_out)
        maxPool1_out = self.maxPool1(spatial_dropout1_out)

        # Feature maps two
        conv2_out = self.conv2(maxPool1_out)
        batch_reg2_out = self.batch_reg2(conv2_out)
        spatial_dropout2_out = self.spatial_dropout2(batch_reg2_out)
        maxPool2_out = self.maxPool2(spatial_dropout2_out)
        
        # Hidden Linear layers
        flattened_out = self.flattened(maxPool2_out)
        hidden_out = self.hiddenLayer(flattened_out)
        hidden_out = self.linear_dropout(hidden_out)
        hidden_out = self.lstm(tf.expand_dims(hidden_out, axis=1))

        # Actor and the Critic outputs
        value = self._value(hidden_out)
        logits = self._policy(hidden_out)
        action_dist = tf.nn.softmax(logits)

        return logits, action_dist, tf.squeeze(value)
    
    # Makes a step in the environment
    def step(self, observation, keep_p=0.0):
        logits, softmax, value = self.call(observation, keep_p=keep_p)
        return logits.numpy(), softmax.numpy(), tf.squeeze(value).numpy()

    # Returns the critic estimation of the current state value
    def value_function(self, observation, keep_p):
        _, _, value = self.call(observation, keep_p)
        return tf.squeeze(value).numpy()

    def save_model_weights(self): 
        try:
            current_dir = os.getcwd()   
            model_save_path = current_dir + '\Model\checkpoint.tf'
            self.save_weights(model_save_path, save_format='tf')
        except:
            print("ERROR: There was an issue saving the model weights.")
            pass

    def load_model_weights(self):
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
        return - tf.reduce_sum(p0 * tf.math.log(p0 + 1e-16), axis=1)

    
  