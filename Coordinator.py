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
    def __init__(self, global_model, local_model, workers, plot, num_envs, num_epocs, batches_per_epoch, batch_size, gamma, model_save_path, anneling_steps):
        self.global_model = global_model
        self.local_model = local_model
        self.model_save_path = model_save_path
        self.workers = workers
        self.num_envs = num_envs
        self.num_epocs = num_epocs
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.gamma = gamma
        self._last_batch_loss = 0
        self.plot = plot   
        self.total_steps = 0
        self.anneling_steps = anneling_steps
        self.pre_train_steps = 0
        
        # Private variables
        self._currentE = 1.0
        self.current_prob = 0.0
        self.anneling_steps = anneling_steps
        self._current_annealed_prob = 1.0
        self._train_data = None
        # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=7e-4, clipnorm=.50)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007, epsilon=1e-5, clipnorm=.50)

        # PPO and training related variables
        self._num_training_sessions_on_sample_data = 8
        self._learning_rate = 0.0007
        self._clip_range = .2
 
    # Annealing entropy to encourage convergence later: 1.0 to 0.01
    def _current_entropy(self):
        startE = .30
        endE = 0.01 

        # Final chance of random action
        stepDrop = (startE - endE) / self.anneling_steps

        if self.total_steps % 256 == 0:
            print("Current entropy is: " + str(self._current_annealed_prob))

        if self._current_annealed_prob >= endE:
            self._current_annealed_prob -= stepDrop
            return self._current_annealed_prob
        else:
            return 0.01 
    
    # STD Mean normalization
    def _normalize(self, x):
        norm_x = (x - x.mean()) / (x.std() + 1e-8)
        # norm_x = (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)
        return norm_x

    def _clip_by_range(self, x, clip_range=[-50.0, 50.0]):
        clipped_x = tf.clip_by_value(x, min(clip_range), max(clip_range))
        return clipped_x
            
    # Annealing temperature scales from .1 to 1.0
    def _keep_prob(self):
        keep_per = lambda: (1.0 - self._currentE) + 0.1
        startE = 1.0
        endE = 0.1 # Final chance of random action
        pre_train_steps = self.pre_train_steps # Number of steps used before anneling begins
        total_steps = self.total_steps # max steps ones can take
        stepDrop = (startE - endE) / self.anneling_steps

        if self._currentE >= endE and total_steps >= pre_train_steps:
            
            self._currentE -= stepDrop
            p = keep_per()

            if self.total_steps % 256 == 0:
                print("Current temp is: " + str(p))
            
            return p

        else:
            print("Anneling Finished")
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
        global_weights = self.global_model.get_weights()
        self.local_model.set_weights(global_weights)
       
    # pass a tuple of (batch_states, batch_actions,batch_rewards)
    def loss(self):
        prob = self.current_prob
        self.sampled_states, self.sampled_actions, self.sampled_rewards, self.sampled_advantages, self.sampled_values, self.sampled_logits = self._train_data
        self.sampled_advantages = self.sampled_advantages
        self.sampled_actions_hot = tf.one_hot(self.sampled_actions, self.global_model.num_actions, dtype=tf.float64)
        self.sampled_rewards = tf.Variable(self.sampled_rewards, name="rewards", dtype=tf.float64)


        logits, _, values = self.global_model.call(tf.convert_to_tensor(np.vstack(np.expand_dims(self.sampled_states, axis=1)), dtype=tf.float64), keep_p=prob)
        values = tf.squeeze(values)
        
        # Policy Loss and Entropy
        self.sampled_neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(labels=self.sampled_actions_hot, logits=self.sampled_logits) 
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(labels=self.sampled_actions_hot, logits=logits) 
        ratio = tf.exp(self.sampled_neg_log_prob - neg_log_prob)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - self._clip_range, 1.0 + self._clip_range)
        self.policy_loss = tf.reduce_mean(tf.minimum(ratio * self.sampled_advantages, clipped_ratio * self.sampled_advantages))
        self.entropy = tf.reduce_mean(self.global_model.logits_entropy(logits))
        
        # Value Loss
        clipped_values = tf.add(self.sampled_values, tf.clip_by_value(values - self.sampled_values, -self._clip_range, self._clip_range))
        self.value_loss = tf.multiply(0.5, tf.reduce_mean(tf.maximum(tf.square(values - self.sampled_rewards), tf.square(clipped_values - self.sampled_rewards))))
        self._last_batch_loss = total_loss = -(self.policy_loss - 0.5 * self.value_loss + 0.01 * self.entropy)

        return total_loss

      
    def train(self, train_data):

        (all_states, all_actions, all_returns, all_advantages, all_values, all_logits) = train_data
        
        
        sample_size = len(all_states)   # Sample size = batch_size * num_workers
        mini_sample_size = sample_size // 4
        indexes = np.arange(sample_size)

        if mini_sample_size == 0:
            self._train_data = train_data

            params = self.global_model.trainable_variables
            self.optimizer.minimize(self.loss, var_list=params)

            self.collect_stats("LOSS", self._last_batch_loss.numpy())
            return
            
        # Here we will take slices of all sample data from the old policy and train the global network
        for _ in range(self._num_training_sessions_on_sample_data): 
            
            np.random.shuffle(indexes)
            
            for start_index in range(0, sample_size, mini_sample_size):

                end_index = start_index + mini_sample_size
                mini_sample_indexes = indexes[start_index: end_index]
                
                mini_sample = ([], [], [], [], [], [])
                for index in mini_sample_indexes:
                    mini_sample[0].append(all_states[index])
                    mini_sample[1].append(all_actions[index])
                    mini_sample[2].append(all_returns[index])
                    mini_sample[3].append(all_advantages[index])
                    mini_sample[4].append(all_values[index])
                    mini_sample[5].append(all_logits[index])
                
                self._train_data = mini_sample

                # Apply Gradients
                params = self.global_model.trainable_variables
                self.optimizer.minimize(self.loss, var_list=params)

                self.collect_stats("LOSS", self._last_batch_loss.numpy())

    # request access to collector and record stats
    def collect_stats(self, key, value):
        while self.plot.busy_notice():
            continue
        self.plot.stop_request()
        self.plot.collector.collect(key, value)
        self.plot.continue_request()
    
    def calculate_advantages(self, dones, rewards, values, bootstrap):
        advantages = np.zeros((len(rewards)))
        last_advantage = 0
        last_value = bootstrap
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = rewards[t] + self.gamma * last_value - values[t]
            last_advantage = delta + self.gamma * 0.95 * last_advantage
            advantages[t] = last_advantage
            last_value = values[t]

        return advantages


    # Produces reversed list of bootstrapped discounted r
    def rewards_discounted(self, rewards, gamma, bootstrap):
        
        discounted_rewards = []
        reward_sum = bootstrap
        
        for reward in rewards[::-1]:  
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        result = discounted_rewards[::-1]
        return result

    # for debugging processed images
    def displayImage(self, img):
        cv2.imshow('image', img)
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
                while self.plot.busy_notice():
                    continue
                
                self.plot.stop_request()
                
                for worker in self.workers:
                    worker.reset()

                self.plot.continue_request()
        

                # loop for generating samples of data to train on the global_network
                for mb in range(self.batches_per_epoch):
                    
                    self.total_steps += 1  

                    # TODO: implement this
                    # progress = update / self.total_steps
                    # self.learning_rate = 2.5e-4 * (1 - progress)
                    # self.clip_range = 0.1 * (1 - progress) 
                    
                    # Copy global network over to local network
                    self.refresh_local_network_params()

                    # Send workers out to threads
                    threads = []
                    prob = self._keep_prob()
                    self.current_prob = prob
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
                    all_dones = np.array([])
                    all_states = np.array([])
                    all_actions = np.array([])
                    all_values = np.array([])
                    all_logits = np.array([])
                    all_dones = np.array([])
                    all_advantages = np.array([])
                    all_returns = np.array([])

                    # Calculate discounted rewards for each environment
                    for env in range(self.num_envs):
                        done = False
                        batch_rewards = []
                        batch_observations = []
                        batch_states = []
                        batch_actions = []
                        batch_values = []
                        batch_logits = []
                        batch_advantages = []
                        batch_dones = []

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
                            batch_logits.append(logits)
                            batch_dones.append(done)

                        
                        # If we reached the end of an episode or if we filled a batch without reaching termination of episode
                        # we boot-strap the final rewards with the V_s(last_observation)

                        # Discounted bootstraped rewards formula:
                        # G_t == R + γ * V(S_t'):

                        # Advantage
                        # δ_t == R + γ * V(S_t') - V(S_t): 
                        
                        # Or simply... 
                        # δ_t == G_t - V: 
                        # 
                        boot_strap = 0.0    

                        # Generalized advantage calculation from OpenAI Baselines
                        # steps = len(batch_rewards)
                        # mb_returns = np.zeros_like(batch_rewards)
                        # advantages = np.zeros_like(batch_rewards)
                        # nextvalues = []
                        # lastgaelam = 0
                        # lam = 0.95
                        boot_strap = 0
                        
                        if (not done):
                            _, _, boot_strap = self.local_model.step(np.expand_dims(observation, axis=0), keep_p=self.current_prob)

                        batch_advantages = self.calculate_advantages(batch_dones, batch_rewards, batch_values, boot_strap)

                        all_dones = np.concatenate((all_dones, batch_dones), 0) if all_dones.size else np.array(batch_dones)
                        all_values = np.concatenate((all_values, batch_values), 0) if all_values.size else np.array(batch_values)
                        all_advantages = np.concatenate((all_advantages, batch_advantages), 0) if all_advantages.size else np.array(batch_advantages)
                        all_logits = np.concatenate((all_logits, batch_logits), 0) if all_logits.size else np.array(batch_logits)
                        all_rewards = np.concatenate((all_rewards, batch_rewards), 0) if all_rewards.size else np.array(batch_rewards)
                        all_states = np.concatenate((all_states, batch_states), 0) if all_states.size else np.array(batch_states)
                        all_actions = np.concatenate((all_actions, batch_actions), 0) if all_actions.size else np.array(batch_actions)

                           
                    # We can do this because: d/dx ∑ loss  == ∑ d/dx loss
                    data = (all_states, all_actions, all_values + all_advantages, all_advantages, all_values, all_logits)

                    if len(data[0]) != 0:
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

            print("Training session was succesfull.")
            return True 

        except:
            print("ERROR: The coordinator ran into an issue during training!")
            raise
      
        self.plot.join()
        


