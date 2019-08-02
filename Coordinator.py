import os
import copy

import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import cv2
import scipy.signal
import tensorflow.keras as keras
import tensorflow.keras.backend as k

# import local classes
from AC_Network import AC_Model
from Worker import Worker, WorkerThread


class Coordinator:
    def __init__(self, model, local_model, workers, plot, num_envs, num_epocs, num_minibatches, batch_size, gamma, model_save_path):
        self.global_model = model
        self.local_model = local_model
        self.model_save_path = model_save_path
        self.workers = workers
        self.num_envs = num_envs
        self.num_epocs = num_epocs
        self.num_minibatches = num_minibatches
        self.batch_size = batch_size
        self.gamma = gamma
        self.total_loss = 0
        self.plot = plot   
        self.total_steps = 0
        self.pre_train_steps = 0
        self._currentE = 1.0
        self.anneling_steps = num_epocs * num_minibatches 
        
            
    # for Annealing dropout or for Annealing temperature scales from 1.0 to .1
    def _keep_prob(self):
        keep_per = lambda: (1.0 - self._currentE) + 0.1
        startE = 1.0
        endE = 0.1 # Final chance of random action
        pre_train_steps = self.pre_train_steps # Number of steps used before anneling begins
        total_steps = self.total_steps # max steps ones can take
        stepDrop = (startE - endE) / self.anneling_steps

        if self._currentE >= endE and total_steps >= pre_train_steps:
            self._currentE -= stepDrop
            return keep_per()
        else:
            return 1.0      
    
    # Convert numbered actions into one-hot formate [0 , 0, 1, 0, 0]
    def dense_to_one_hot(self, labels_dense, num_classes = 7):
        labels_dense = np.array(labels_dense)
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        labels_one_hot = labels_one_hot.tolist()
        return labels_one_hot

    # Used to copy over global variables to local network 
    def refresh_local_network_params(self):
        self.global_model.save_model_weights()
        global_weights = self.global_model.get_weights()
        self.local_model.set_weights(global_weights)
       
    # pass a tuple of (batch_states, batch_actions,batch_rewards)
    def loss(self, train_data):
        
        batch_states, batch_actions, batch_rewards, batch_advantages = train_data
        actions = tf.Variable(batch_actions, name="Actions", trainable=False)
        rewards = tf.Variable(batch_rewards, name="Rewards", dtype=tf.float32)
        actions_hot = tf.one_hot(actions, self.global_model.num_actions, dtype=tf.float32)

        logits, action_dist, values = self.global_model.call(tf.convert_to_tensor(np.vstack(np.expand_dims(batch_states, axis=1)), dtype=tf.float32))
        
        advantages = rewards - values


        # Entropy: - ∑ P_i * Log (P_i)
        entropy = self.global_model.softmax_entropy(action_dist)
        
        # Policy Loss:  (1 / n) * ∑ * -log π(a_i|s_i) * A(s_i, a_i) 
        # neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=actions_hot)
        neg_log_prob = - tf.math.log(tf.reduce_sum(action_dist * actions_hot, axis=1) + 1e-10)
        print(neg_log_prob.numpy())
        policy_loss = tf.reduce_mean((neg_log_prob * tf.stop_gradient(advantages)) - (entropy * self.global_model.entropy_coef))
        
        # Value loss "MSE": (1 / n) * ∑[V(i) - R_i]^2
        #value_loss = tf.losses.mean_squared_error(rewards, values) * self.global_model.value_function_coeff
        value_loss = tf.reduce_mean(tf.square(rewards - values)) * self.global_model.value_function_coeff
        loss = policy_loss + value_loss

        
        print(policy_loss.numpy())
        print(value_loss.numpy())
        print(entropy.numpy())
        print(loss.numpy())

        return loss
      
    def train(self, train_data):
        with tf.GradientTape() as tape:
            
            for var in self.local_model.layers:
                for w in var.trainable_weights:
                    tape.watch(w)
                for v in var.trainable_variables:
                    tape.watch(v)
            
            loss = self.loss(train_data)

            # Apply Gradients
            params = self.global_model.trainable_variables

            grads = tape.gradient(loss, params)

            grads, global_norm = tf.clip_by_global_norm(grads, self.global_model.max_grad_norm)
            
            # optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.global_model.learning_rate, epsilon=self.global_model.epsilon)
            optimizer = tf.keras.optimizers.Adam(learning_rate=7e-4, epsilon=1e-5)

            optimizer.apply_gradients(zip(grads, params))

            self.refresh_local_network_params()

       
        self.plot.collector.collect("LOSS", loss.numpy())

    # Produces reversed list of discounted rewards
    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    # for debugging processed images
    def displayImage(self, img):
        cv2.imshow('image', np.squeeze(img, axis=0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):
        save_path = ".\Model\\" + "state_vars.txt"

        try: 
            if (os.path.exists(save_path)):
                with open(save_path, 'r') as f:
                    s = f.read()
                    self.total_steps = int("".join(s.split()))
                    f.close()
        except:
            print("No additional saved state variables.")
        
        self.plot.start()
        
        try:
            # Main training loop
            for _ in range(self.num_epocs):

                # ready workers for the next epoc, sync with live plot
                self.plot.stop_request()
                
                for worker in self.workers:
                    worker.reset()

                self.plot.continue_request()
        

                # loop for generating a training session a batch at a time
                for mb in range(self.num_minibatches):
                    
                    self.total_steps += 1   
                    
                    # Copy global network over to local network
                    self.refresh_local_network_params()

                    # Send workers out to threads
                    threads = []
                    prob = self._keep_prob()
                    for worker in self.workers:
                        threads.append(WorkerThread(target=worker.run, args=([prob])))

                    # Start the workers on their tasks
                    for thread in threads:
                        thread.start()
                        
                    batches = []
                    
                    # Wait foreach worker to finish and return their batch
                    for thread in threads:
                        batch = thread.join()
                        batches.append(batch)

                    all_rewards = np.array([])
                    all_states = np.array([])
                    all_actions = np.array([])
                    all_values = np.array([])
                    all_advantages = np.array([])

                    # Calculate discounted rewards for each environment
                    for env in range(self.num_envs):
                        done = False
                        batch_rewards = []
                        batch_observations = []
                        batch_states = []
                        batch_actions = []
                        batch_values = []
                        batch_advantages = []

                        mb = batches[env]
                        
                        # Empty batch
                        if mb == [] or mb == None:
                            continue

                        # For every step made in this env for this particular batch
                        done = False
                        value = None
                        observation = None

                        for step in mb:
                            (state, observation, reward, value, action, done, logits) = step
                            batch_actions.append(action)
                            batch_rewards.append(reward)
                            batch_observations.append(observation)
                            batch_states.append(state)
                            batch_values.append(value)

                        
                        # If we reached the end of an episode or if we filled a batch without reaching termination of episode
                        # we boot-strap the final rewards with the V_s(last_observation)

                        # Discounted bootstraped rewards formula:
                        # G_t == R + γ * V(S_t'):

                        # Advantage
                        # δ_t == R + γ * V(S_t') - V(S_t): 
                        
                        # Or simply... 
                        # δ_t == G_t - V:      

                        if (not done):
                            _, _, boot_strap = self.local_model.step(np.expand_dims(observation, axis=0), 1.0)
                            bootstrapped_rewards = np.asarray(batch_rewards + [boot_strap])
                            discounted_rewards = self.discount(bootstrapped_rewards, self.gamma)[:-1]
                            batch_rewards = discounted_rewards
                            batch_advantages = batch_rewards - batch_values

                            # Same as above...
                            # bootstrapped_values = np.asarray(batch_values + [boot_strap])
                            # advantages = batch_rewards + self.gamma * bootstrapped_values[1:] - bootstrapped_values[:-1]
                            # advantages = self.discount(advantages, self.gamma)

                            all_values = np.concatenate((all_values, batch_values), 0) if all_values.size else np.array(batch_values)
                            all_advantages = np.concatenate((all_advantages, batch_advantages), 0) if all_advantages.size else np.array(batch_advantages)
                            all_rewards = np.concatenate((all_rewards, batch_rewards), 0) if all_rewards.size else np.array(batch_rewards)
                            all_states = np.concatenate((all_states, batch_states), 0) if all_states.size else np.array(batch_states)
                            all_actions = np.concatenate((all_actions, batch_actions), 0) if all_actions.size else np.array(batch_actions)
                           
                        else:
                            
                            boot_strap = 0

                            all_values = np.concatenate((all_values, batch_values), 0) if all_values.size else np.array(batch_values)
                            bootstrapped_rewards = np.asarray(batch_rewards + [boot_strap])
                            discounted_rewards = self.discount(bootstrapped_rewards, self.gamma)[:-1]
                            batch_rewards = discounted_rewards
                            batch_advantages = batch_rewards - batch_values

                            all_advantages = np.concatenate((all_advantages, batch_advantages), 0) if all_advantages.size else np.array(batch_advantages)
                            all_rewards = np.concatenate((all_rewards, batch_rewards), 0) if all_rewards.size else np.array(batch_rewards)
                            all_states = np.concatenate((all_states, batch_states), 0) if all_states.size else np.array(batch_states)
                            all_actions = np.concatenate((all_actions, batch_actions), 0) if all_actions.size else np.array(batch_actions)
                            
                    # We can do this because: d/dx ∑ loss  == ∑ d/dx loss
                    data = (all_states, all_actions, all_rewards, all_advantages)

                    # for state in all_states:
                    #     self.displayImage([state])
                    
                    if data[0].size != 0:
                        self.train(data) 
                    else:
                        break

                try:
                    #Save model and other variables
                    with open(save_path, 'w') as f:
                        try:
                            f.write(str(self.total_steps))
                            f.close()
                        except: 
                            raise
                    self.global_model.save_model_weights()
                    print("Model saved")
                
                except:
                    print("ERROR: There was an issue saving the model!")
                    raise

            print("Training session was sucessfull.")
            return True 

        except:
            print("ERROR: The coordinator ran into an issue during training!")
            raise
      
        self.plot.join()
        


