import tensorflow as tf
from AC_Network import AC_Network

# Defines the backpropagation step.
class Model:
    def __init__(self, policy_params, sess):
        self.sess = sess
        self.input_shape, self.batch_size, self.num_actions, self.action_space = policy_params

        # Local network
        self.network = step_policy = AC_Network(self.input_shape, self.num_actions, sess)

    # Makes a step in the environment
    def step(self, observation, keep_per):
        
        softmax, value = self.sess.run([self.network.action_dist, self.network.value], {self.network.input_def: observation, self.network.keep_prob: 1.0})
        return softmax, value[-1]

    # Returns the critic estimation of the current state value
    def value(self, observation):
        value = self.sess.run(self.network.value, {self.network.input_def: observation, self.network.keep_prob: 1.0})
        return value[-1]

    # Used to copy over global variables to local network 
    def refresh_local_network_params(self):
        pass
        # from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "train")
        # to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "step")
        # op_holder = []
        # for from_var, to_var in zip(from_vars,to_vars):
        #     op_holder.append(to_var.assign(from_var))
        
        # with tf.keras.backend.get_session() as sess:
        #     sess.run(op_holder)