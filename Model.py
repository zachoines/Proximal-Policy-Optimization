import tensorflow as tf
from AC_Network import AN_Network

# Defines the backpropagation step.
class Model:
    def __init__(self, policy, sess):
        self.policy = policy
        self.sess = sess

        # Local network
        self.step_policy = self.policy(self.sess, self.policy.input_shape, self.policy.num_actions, reuse=False,
                                       is_training=False)

        # global network
        self.train_policy = self.policy(self.sess, self.policy.input_shape, self.policy.num_actions, reuse=True,
                                        is_training=True)
        
        # Learning params
        self.entropy_coef = 0.01
        self.value_function_coeff = 0.5
        self.max_gradient_norm = 0.5
        self.learning_rate =  7e-4
        self.alpha = 0.99
        self.epsilon = 1e-5


        
        
    def backpropage(self):    
         
        with tf.variable_scope('train_output'):
            
            # Responsible Output -log(pi)
            negative_log_prob_action = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.train_policy.policy_logits,
                labels=self.actions)

            # 1/n * sum A(si,ai) * -logpi(ai|si)
            self.policy_loss = tf.reduce_mean(self.advantage * negative_log_prob_action)

            # Value loss 1/2 SUM [R - V(s)]^2
            self.value_loss = tf.reduce_mean(tf.square(tf.squeeze(self.train_policy.value_function) - self.reward) / 2.0)

            # Apply Entropy 
            self.entropy = tf.reduce_mean(- tf.reduce_sum(self.train_policy.policy_logits * tf.log(self.train_policy.policy_logits + 1e-6), axis=1))
        
            # Total loss
            self.loss = self.policy_loss - self.entropy * self.entropy_coeff + self.value_loss * self.vf_coeff

            # Gradient Clipping
            with tf.variable_scope("policy"):
                params = tf.trainable_variables()
            
            grads = tf.gradients(self.loss, params)
            
            if self.max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

            # Apply Gradients 
            grads = list(zip(grads, params))
            

            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.alpha,
                                                  epsilon=self.epsilon)
            
            # Update network weights on 
            self.optimize = optimizer.apply_gradients(grads)