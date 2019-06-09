import tensorflow as tf
from AC_Network import AC_Network

# Defines the backpropagation step.
class Model:
    def __init__(self, policy_params):
        self.sess, self.input_shape, self.batch_size, self.num_actions, self.action_space = policy_params

        # Local network
        self.step_policy = AC_Network(self.sess, self.input_shape, self.num_actions, reuse = False, is_training=False, name = "step")

        # global network
        self.train_policy = AC_Network(self.sess, self.input_shape, self.num_actions, reuse = True, is_training=True, name = "train")

    # Makes a step in the environment
    def step(self, observation):
        return self.step_policy.step(observation)

    # Returns the critic estimation of the current state value
    def value(self, observation):
        return self.step_policy.value(observation)

    # Used to copy over global variables to local network 
    def refresh_local_network_params(self):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "train")
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "step")
        op_holder = []
        for from_var, to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        
        self.sess.run(op_holder)