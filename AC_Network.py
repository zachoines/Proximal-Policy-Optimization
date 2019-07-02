import tensorflow as tf
import numpy as np
from Layers import create_conv, flatten, create_dense
from Layers import orthogonal_initializer, noise_and_argmax, openai_entropy



# defines the foward step.
class AC_Network:
    def __init__(self, sess, input_shape, num_actions, reuse=False, is_training=True, name='train'):
        self.initial_state = []  # not stateful
        self.sess = sess
        self.input_shape = input_shape
        self.X_input = None
        self.reuse = reuse
        self.value_s = None
        self.action_s = None
        self.initial_state = []
        self.num_actions = num_actions

        # Dicounting hyperparams for loss functions
        self.entropy_coef = 0.01
        self.value_function_coeff = 0.5
        self.max_grad_norm = 0.5
        self.learning_rate =  7e-4
        self.alpha = 0.99
        self.epsilon = 1e-5

        with tf.variable_scope(name):
            self.X_input = tf.placeholder(tf.float32, (None, 96, 384, 1))
            conv1 = create_conv('conv1', self.X_input, num_filters = 32, kernel_size=(7, 7),
                           padding = 'VALID', stride = (1, 1),
                           initializer = orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           max_pool_enabled = True)

            conv2 = create_conv('conv2', conv1, num_filters = 64, kernel_size=(5, 5), padding = 'VALID', stride=(1, 1),
                           initializer = orthogonal_initializer(np.sqrt(2)), activation = tf.nn.relu,
                           max_pool_enabled = True)

            conv3 = create_conv('conv3', conv2, num_filters = 64, kernel_size=(3, 3), padding = 'VALID', stride=(1, 1),
                           initializer = orthogonal_initializer(np.sqrt(2)), activation = tf.nn.relu,
                           max_pool_enabled = True)

            conv3_flattened = flatten(conv3)

            fc4 = create_dense('fc4', conv3_flattened, output_dim = 512, initializer = orthogonal_initializer(np.sqrt(2)),
                        activation = tf.nn.relu, rate = 0.01)

            self.policy_logits = create_dense('policy_logits', fc4, output_dim = num_actions,
                                       initializer = orthogonal_initializer(np.sqrt(1.0)))

            self.value_function = create_dense('value_function', fc4, output_dim=1,
                                        initializer = orthogonal_initializer(np.sqrt(1.0)))

            with tf.name_scope('value'):
                self.value_s = self.value_function[:, 0]

            with tf.name_scope('action'):
                self.action_s = noise_and_argmax(self.policy_logits)


            if name == "step":
                
                # Batch data that will be sent to Model by the coordinator
                self.actions = tf.placeholder(tf.int32, [None]) 
                self.advantages = tf.placeholder(tf.float32, [None])  
                self.rewards = tf.placeholder(tf.float32, [None])  
                self.values = tf.placeholder(tf.float32, [None])
                
                # log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
                # Responsible Outputs -log π(a_i|s_i)
                self.log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy_logits, labels=self.actions)
                
                # Policy Loss:  (1 / n) * ∑ A(s_i, a_i) * -log π(a_i|s_i)
                self.policy_loss = tf.reduce_mean(self.advantages * self.log_prob)
                
                # Value loss: (1 / n) * ∑[V(i) - R_i]^2
                self.value_loss = tf.reduce_mean(tf.square(tf.squeeze(self.value_function) - self.rewards) / 2.0 )
                
                # Entropy: - ∑ P_i * Log (P_i)
                self.entropy = tf.reduce_mean(openai_entropy(self.policy_logits))
                
                # Total loss: Policy loss - entropy * entropy coefficient + value coefficient * value loss
                self.loss = self.policy_loss - self.entropy * self.entropy_coef + self.value_loss * self.value_function_coeff

                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "step")
                
                grads = tf.gradients(self.loss, params)
                
                grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

                # Apply Gradients 
                grads = list(zip(grads, params))

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
                
                # Update network weights 
                self.optimize = optimizer.apply_gradients(grads)


    def step(self, observation, *_args, **_kwargs):
        # Take a step using the model and return the predicted policy and value function
        action, value = self.sess.run([self.action_s, self.value_s], {self.X_input: observation})
        return action, value  # dummy state

    def value(self, observation, *_args, **_kwargs):
        # Return the predicted value function for a given observation
        return self.sess.run(self.value_s, {self.X_input: observation})

