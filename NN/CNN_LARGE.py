import os
import numpy as np

import tensorflow as tf

class AC_Model_Large(tf.keras.Model):
    def __init__(self, input_s, num_actions, config, is_training=True):
        super(AC_Model_Large, self).__init__()
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
            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)),
            padding="valid",
            activation="relu", 
            name="conv1",
            trainable=is_training, 
            dtype='float64' )
        
        # self.maxPool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="maxPool1")

        # define Convolution 2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[4, 4],
            strides=(2, 2),
            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)),
            padding="valid",
            activation="relu",
            name="conv2", 
            trainable=is_training, 
            dtype='float64')

        # self.maxPool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="maxPool2", trainable=is_training )
        
        # define Convolution 3
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            strides=(1, 1),
            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)),
            padding="valid",
            activation="relu",
            name="conv3",
            trainable=is_training,
            dtype='float64' )

        # self.maxPool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name ="maxPool3", trainable=is_training )
        self.flattened = tf.keras.layers.Flatten(name="flattening_layer", trainable=is_training )
        self.hiddenLayer1 = tf.keras.layers.Dense(
            512,
            activation="relu",
            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)),
            name="hidden_layer1", 
            trainable=is_training, 
            dtype='float64' )

        # self.hiddenLayer2 = tf.keras.layers.Dense(
        #     256,
        #     activation="relu",
        #     kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)),
        #     name="hidden_layer2", 
        #     trainable=is_training, 
        #     dtype='float64' )
        

        # self.dropout1 = tf.keras.layers.Dropout(0.5)
        # self.dropout2 = tf.keras.layers.Dropout(0.5)
   
        # self.spatial_dropout1 = tf.keras.layers.SpatialDropout2D(0.5)
        # self.spatial_dropout2 = tf.keras.layers.SpatialDropout2D(0.5)
        # self.spatial_dropout3 = tf.keras.layers.SpatialDropout2D(0.5)

        # Batch regularization
        # self.batch_reg1 = tf.keras.layers.BatchNormalization()
        # self.batch_reg2 = tf.keras.layers.BatchNormalization()
        # self.batch_reg3 = tf.keras.layers.BatchNormalization()

        # Output Layer consisting of an Actor and a Critic
        self._value = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.orthogonal(np.sqrt(2.0)),
            activation='linear',
            name="value_layer",
            use_bias=True,
            trainable=is_training )
   
        self._policy = tf.keras.layers.Dense(
            self.num_actions,
            activation='linear',
            kernel_initializer=tf.keras.initializers.orthogonal(.01),
            use_bias=True,
            name="policy_layer", trainable=is_training )


    def call(self, input_image, keep_p=1.0):

        # Feature maps
        conv1_out = self.conv1(input_image)
        # conv1_out = self.batch_reg1(conv1_out)
        conv2_out = self.conv2(conv1_out)
        # conv2_out = self.batch_reg2(conv2_out)
        conv3_out = self.conv3(conv2_out)
        # conv3_out = self.batch_reg3(conv3_out)

        # Linear layers
        flattened_out = self.flattened(conv3_out)
        hidden_out1 = self.hiddenLayer1(flattened_out)
        # hidden_out1 = self.dropout1(hidden_out1)
        # hidden_out2 = self.hiddenLayer2(hidden_out1)
        # hidden_out2 = self.dropout2(hidden_out2)
       

        # Actor and the Critic outputs
        value = self._value(hidden_out1)
        logits = self._policy(hidden_out1)
        action_dist = tf.nn.softmax(logits)

        return logits, action_dist, value
    
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
            model_save_path = '.\Model\checkpoint.tf'
            self.save_weights(model_save_path, save_format='tf')
        except:
            print("ERROR: There was an issue saving the model weights.")
            raise

    def load_model_weights(self):
        model_save_path = '.\Model\checkpoint.tf'
        self.load_weights(filepath=model_save_path)
    
    # Open AI entropy
    def logits_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), 1)

    # standard entropy
    def softmax_entropy(self, p0):
        return - tf.reduce_sum(p0 * tf.math.log(p0 + 1e-16), axis=1)

    
  







