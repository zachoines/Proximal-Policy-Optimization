import os
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as k

class AC_Model(tf.keras.Model):
    def __init__(self, input_s, num_actions, is_training=True):
        super(AC_Model, self).__init__()
        self.value_s = None
        self.action_s = None
        self.num_actions = num_actions
        self.training = is_training

        # Dicounting hyperparams for loss functions
        self.entropy_coef = 0.01
        self.value_function_coeff = 0.50
        self.max_grad_norm = 50.0
        self.learning_rate = 7e-4
        self.alpha = 0.99
        self.epsilon = 1e-5
        
        # Model variables
        (_, hight, width, stack) = input_s
        self.input_def = tf.keras.layers.Input(shape=(hight, width*stack, 1), name="input_layer", dtype=tf.float32)

        # Define Convolution 1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[8, 8],
            strides=(3, 3),
            kernel_initializer=keras.initializers.Orthogonal(gain=1.0, seed=1),
            padding="valid",
            activation="relu", 
            name="conv1",
            trainable=is_training )
        
        self.maxPool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name="maxPool1")

        # define Convolution 2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            strides=(2, 2),
            kernel_initializer=keras.initializers.Orthogonal(gain=1.0, seed=1),
            padding="valid",
            activation="relu",
            name="conv2", 
            trainable=is_training)

        self.maxPool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name="maxPool2", trainable=is_training )
        
        # define Convolution 3
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[2, 2],
            strides=(1, 1),
            kernel_initializer=keras.initializers.Orthogonal(gain=1.0, seed=1),
            padding="valid",
            activation="relu",
            name="conv3",
            trainable=is_training )

        self.maxPool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name ="maxPool3", trainable=is_training )
        self.flattened = tf.keras.layers.Flatten(name="flattening_layer", trainable=is_training )
        self.hiddenLayer = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_initializer=keras.initializers.Orthogonal(gain=1.0, seed=1),
            name="hidden_layer", 
            trainable=is_training )

        # Output Layer consisting of an Actor and a Critic
        self._value = tf.keras.layers.Dense(
            1,
            kernel_initializer=keras.initializers.Orthogonal(gain=1.0, seed=1),
            activation='linear',
            name="value_layer",
            trainable=is_training )
   
        self._policy = tf.keras.layers.Dense(
            self.num_actions,
            activation='linear',
            kernel_initializer=keras.initializers.Orthogonal(gain=1.0, seed=1),
            name="policy_layer", trainable=is_training )

        # Batch regularization
        self.batch_reg1 = tf.keras.layers.BatchNormalization()
        self.batch_reg2 = tf.keras.layers.BatchNormalization()
        self.batch_reg3 = tf.keras.layers.BatchNormalization()

        # Dropout layers to prevent overfitting
        self.spatial_dropout1 = tf.keras.layers.SpatialDropout2D(rate=.5, seed=1, trainable=is_training)
        self.spatial_dropout2 = tf.keras.layers.SpatialDropout2D(rate=.5, seed=1, trainable=is_training)
        self.spatial_dropout3 = tf.keras.layers.SpatialDropout2D(rate=.5, seed=1, trainable=is_training)
        self.linear_dropout = tf.keras.layers.Dropout(rate=.7, seed=1, trainable=is_training)

    def call(self, input_image, keep_p=1.0):

        # Feature maps one
        conv1_out = self.conv1(input_image)
        # conv1_out = self.batch_reg1(conv1_out)
        # maxPool1_out = self.maxPool1(conv1_out)
        # maxPool1_out = self.spatial_dropout1(maxPool1_out)

        # Feature maps two
        conv2_out = self.conv2(conv1_out)
        # conv2_out = self.batch_reg2(conv2_out)
        # maxPool2_out = self.maxPool2(conv2_out)
        # maxPool2_out = self.spatial_dropout2(maxPool2_out)

        # Feature maps three
        conv3_out = self.conv3(conv2_out)
        # conv3_out = self.batch_reg3(conv3_out)
        # maxPool3_out = self.maxPool3(conv3_out)
        # maxPool3_out = self.spatial_dropout2(maxPool3_out)
        
        # Linear layers
        hidden_out = self.flattened(conv3_out)
        # hidden_out = self.linear_dropout(hidden_out)

        # Actor and the Critic outputs
        value = self._value(hidden_out)
        logits = self._policy(hidden_out)
        action_dist = tf.nn.softmax(logits)

        return logits, action_dist, tf.squeeze(value)
    
    # Makes a step in the environment
    def step(self, observation, keep_per):
        softmax, logits, value = self.call(observation, keep_per)
        return logits.numpy(), softmax.numpy(), value.numpy()

    # Returns the critic estimation of the current state value
    def value_function(self, observation):
        action_dist, softmax, value = self.call(observation, 1.0)
        return value.numpy()[0]

    def save_model_weights(self): 
        current_dir = os.getcwd()   
        model_save_path = current_dir + '\Model\checkpoint.tf'
        self.save_weights(model_save_path, save_format='tf')

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