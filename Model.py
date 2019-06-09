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


    def step(self, observation):
        return self.step_policy.step(observation)

    def value(self, observation):
        return self.step_policy.value(observation)