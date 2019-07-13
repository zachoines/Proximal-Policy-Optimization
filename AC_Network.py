import tensorflow as tf
import numpy as np
from Layers import create_conv, flatten, create_dense, openai_entropy
from Layers import orthogonal_initializer

# defines the foward step.
class AC_Network:
    def __init__(self, sess, input_shape, num_actions, is_training=True, name='train'):
        self.sess = sess
        self.input_shape = input_shape
        self.X_input = None
        self.value_s = None
        self.action_s = None
        self.num_actions = num_actions

        # Dicounting hyperparams for loss functions
        self.entropy_coef = 0.01
        self.value_function_coeff = 0.5
        self.max_grad_norm = 40.0
        self.learning_rate = 7e-4
        self.alpha = 0.99
        self.epsilon = 1e-5

        with tf.variable_scope(name):
            self.keep_per = tf.placeholder(shape=None, dtype=tf.float32)
            self.X_input = tf.placeholder(tf.float32, (None, 96, 384, 1))
            conv1 = create_conv('conv1', self.X_input,
                num_filters=32,
                kernel_size=(7, 7 * 4),
                padding='VALID',
                stride=(1, 1),
                initializer=orthogonal_initializer(np.sqrt(2.0)),
                activation=tf.nn.elu,
                max_pool_enabled=True,
                keep_per=self.keep_per)

            conv2 = create_conv('conv2', conv1,
                num_filters=64,
                kernel_size=(5, 5 * 4),
                padding='VALID',
                stride=(1, 1),
                initializer=orthogonal_initializer(np.sqrt(2.0)),
                activation=tf.nn.elu,
                max_pool_enabled=True,
                keep_per=self.keep_per)

            conv3 = create_conv('conv3', conv2,
                num_filters=64,
                kernel_size=(3, 3 * 4),
                padding='VALID',
                stride=(1, 1),
                initializer=orthogonal_initializer(np.sqrt(2.0)),
                activation=tf.nn.elu,
                max_pool_enabled=True,
                keep_per=self.keep_per)

            conv3_flattened = flatten(conv3)
            fc4 = create_dense('fc4', conv3_flattened,
                output_dim=512,
                initializer=orthogonal_initializer(np.sqrt(2)),
                activation=tf.nn.elu,
                keep_per=self.keep_per)

            self.policy = create_dense('policy_logits', fc4, output_dim=num_actions, initializer=orthogonal_initializer(np.sqrt(1.0)))

            self.value_function = create_dense('value_function', fc4, output_dim=1, initializer=orthogonal_initializer(np.sqrt(1.0)))

            with tf.name_scope('value'):
                self.value_s = self.value_function[:, 0]

            with tf.name_scope('action_distribution'):
                self.action_dist = tf.nn.softmax(self.policy)

            if name == "step":
                # Batch data that will be sent to Model by the coordinator
                self.actions = tf.placeholder(tf.int32, [None])
                self.actions_hot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)
                self.advantages = tf.placeholder(tf.float32, [None])
                self.rewards = tf.placeholder(tf.float32, [None]) 
                self.values = tf.placeholder(tf.float32, [None])
                
                # Responsible Outputs -log π(a_i|s_i)i
                # self.neg_log_prob = tf.reduce_sum(self.policy * self.actions_hot + 1e-10, [1])
                # self.policy_loss = tf.reduce_mean(-1.0 * tf.log(self.log_prob) * self.advantages)
                
                # Policy Loss:  (1 / n) * ∑ * -log π(a_i|s_i) * A(s_i, a_i) 
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.policy, labels=self.actions_hot)
                self.policy_loss = tf.reduce_mean(self.neg_log_prob * self.advantages)
                
                # Value loss "MSE": (1 / n) * ∑[V(i) - R_i]^2
                # self.value_loss = tf.reduce_mean(tf.square(tf.squeeze(self.value_function) - self.rewards)) 
                self.value_loss = tf.reduce_mean(tf.square(self.advantages)) 

                # Entropy: - (1 / n) * ∑ P_i * Log (P_i)
                # self.entropy = - tf.reduce_mean(self.policy * tf.log(self.policy + 1e-10))
                self.entropy = tf.reduce_mean(openai_entropy(self.policy))
                
                # Total loss: Policy loss - entropy * entropy coefficient + value coefficient * value loss
                self.loss = self.policy_loss + (self.value_loss * self.value_function_coeff) - (self.entropy * self.entropy_coef)

                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "step")
                
                grads = tf.gradients(self.loss, params)
                
                grads, self.global_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

                # Apply Gradients 
                grads = list(zip(grads, params))

                # optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay = self.alpha, epsilon = self.epsilon)
                optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, epsilon = self.epsilon)
            
                # Update network weights 
                self.optimize = optimizer.apply_gradients(grads)


    def step(self, observation, keep_per):
        # Take a step using the model and return the predicted policy and value function
        softmax, value = self.sess.run([self.action_dist, self.value_s], {self.X_input: observation, self.keep_per: keep_per})
        return softmax, value

    def value(self, observation, keep_per):
        # Return the predicted value function for a given observation
        return self.sess.run(self.value_s, {self.X_input: observation, self.keep_per: keep_per})


# TODO:: Make this happen. Currently defined in Worker class
def boltzmann_action_select(softmax):
    pass

def noise_and_argmax_action_select(logits):
    # Add noise then take the argmax
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-1.0 * tf.log(noise)), 1)

