import tensorflow as tf
from AC_Network import AC_Network

# Defines the backpropagation step.
class Model:
    def __init__(self, policy_params, sess, network = None):
        self.sess = sess
        self.input_shape, self.batch_size, self.num_actions, self.action_space = policy_params

        if not network:
            self.network = AC_Network(self.input_shape, self.num_actions, sess)
        else:
            self.network = AC_Network(self.input_shape, self.num_actions, sess)
            self.network.model = tf.keras.models.clone_model(network.model)
    
    def get_weights(self):
        return self.network.model.get_weights()

    def set_weights(self, weights):
        self.network.model.set_weights(weights)

    def get_network(self):
        return self.network

    # Makes a step in the environment
    def step(self, observation, keep_per):
        
        softmax, value = self.sess.run([self.network.action_dist, self.network.value], {self.network.input_def: observation, self.network.keep_prob: 1.0})
        return softmax, value[-1]

    # Returns the critic estimation of the current state value
    def value(self, observation):
        value = self.sess.run(self.network.value, {self.network.input_def: observation, self.network.keep_prob: 1.0})
        return value[-1]

    