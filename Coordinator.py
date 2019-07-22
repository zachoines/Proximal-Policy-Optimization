import os

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
    def __init__(self, model, step_models, workers, plot, num_envs, num_epocs, num_minibatches, batch_size, gamma, model_save_path):
        self.global_model = model
        self.local_models = step_models
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
        self._currentE = .90
        self.anneling_steps = num_epocs * num_minibatches 

    # Anneal dropout for optimized exploration in a bayesian network
    def _keep_prob(self):
        keep_per = lambda: (1.0 - self._currentE) + 0.1
        startE = .9
        endE = 0.1 # Final chance of random action
        pre_train_steps = self.pre_train_steps # Number of steps used before anneling begins
        total_steps = self.total_steps # max steps ones can take
        stepDrop = (startE - endE) / self.anneling_steps

        if self._currentE >= endE and total_steps >= pre_train_steps:
            self._currentE -= stepDrop
            return keep_per()
        else:
            return 1.0      
    
    def dense_to_one_hot(self, labels_dense, num_classes = 7):
        labels_dense = np.array(labels_dense)
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    # Used to copy over global variables to local network 
    def refresh_local_network_params(self):
        for model in self.local_models:
            model.set_weights(self.global_model.get_weights())
       
    def train(self, train_data):

        loss, policy_loss, value_loss, entropy = self.global_model.train_batch(train_data)
        
        self.plot.collector.collect("LOSS", loss)
        self.plot.collector.collect("VALUE_LOSS", value_loss)
        self.plot.collector.collect("POLICY_LOSS", policy_loss)
        self.plot.collector.collect("ENTROPY", entropy)

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
                    for worker in self.workers:
                        threads.append(WorkerThread(target=worker.run, args=([self._keep_prob()])))

                    # Start the workers on their tasks
                    for thread in threads:
                        thread.start()
                        
                    batches = []
                    
                    # Wait foreach worker to finish and return their batch
                    for thread in threads:
                        batch = thread.join()
                        batches.append(batch)

                    all_values = np.array([])
                    all_advantages = np.array([])
                    all_rewards = np.array([])
                    all_states = np.array([])
                    all_actions = np.array([])
                    all_logits = np.array([])

                    # Calculate discounted rewards for each environment
                    for env in range(self.num_envs):
                        done = False
                        batch_advantages = []
                        batch_rewards = []
                        batch_values = []
                        batch_observations = []
                        batch_states = []
                        batch_actions = []
                        batch_logits = []

                        mb = batches[env]
                        
                        # Empty batch
                        if mb == [] or mb == None:
                            continue

                        # For every step made in this env for this particular batch
                        done = False
                        state = None
                        value = None
                        observation = None

                        for step in mb:
                            (state, observation, reward, value, action, done, logits) = step
                            batch_rewards.append(reward)
                            batch_values.append(value)
                            batch_observations.append(observation)
                            batch_states.append(state)
                            batch_actions.append(action)
                            batch_logits.append(logits)

                        
                        # If we reached the end of an episode or if we filled a batch without reaching termination of episode
                        # we boot-strap the final rewards with the V_s(last_observation)

                        # Discounted bootstraped rewards formula:
                        # G_t == R + γ * V(S_t'):

                        # Advantage
                        # δ_t == R + γ * V(S_t') - V(S_t): 
                        
                        # Or simply... 
                        # δ_t == G_t - V:      

                        if (not done):
                            _, _, [boot_strap] = self.global_model.step(np.expand_dims(observation, axis=0), 1.0)
                            boot_strap = boot_strap[0]
                            bootstrapped_rewards = np.asarray(batch_rewards + [boot_strap])
                            discounted_rewards = self.discount(bootstrapped_rewards, self.gamma)[:-1]
                            batch_rewards = discounted_rewards
                            batch_advantages = batch_rewards - batch_values

                            # Same as above...
                            # bootstrapped_values = np.asarray(batch_values + [boot_strap])
                            # advantages = batch_rewards + self.gamma * bootstrapped_values[1:] - bootstrapped_values[:-1]
                            # advantages = self.discount(advantages, self.gamma)

                            all_advantages= np.concatenate((all_advantages, batch_advantages), 0) if all_advantages.size else np.array(batch_advantages)
                            all_rewards = np.concatenate((all_rewards, batch_rewards), 0) if all_rewards.size else np.array(batch_rewards)
                            all_states = np.concatenate((all_states, batch_states), 0) if all_states.size else np.array(batch_states)
                            all_actions = np.concatenate((all_actions, batch_actions), 0) if all_actions.size else np.array(batch_actions)
                            all_values = np.concatenate((all_values, batch_values), 0) if all_values.size else np.array(batch_values)
                            all_logits = np.concatenate((all_logits, batch_logits), 0) if all_logits.size else np.array(batch_logits)
                        
                        else:
                            
                            boot_strap = 0

                            bootstrapped_rewards = np.asarray(batch_rewards + [boot_strap])
                            discounted_rewards = self.discount(bootstrapped_rewards, self.gamma)[:-1]
                            batch_rewards = discounted_rewards
                            batch_advantages = batch_rewards - batch_values

                            all_advantages= np.concatenate((all_advantages, batch_advantages), 0) if all_advantages.size else np.array(batch_advantages)
                            all_rewards = np.concatenate((all_rewards, batch_rewards), 0) if all_rewards.size else np.array(batch_rewards)
                            all_states = np.concatenate((all_states, batch_states), 0) if all_states.size else np.array(batch_states)
                            all_actions = np.concatenate((all_actions, batch_actions), 0) if all_actions.size else np.array(batch_actions)
                            all_values = np.concatenate((all_values, batch_values), 0) if all_values.size else np.array(batch_values)
                            all_logits = np.concatenate((all_logits, batch_logits), 0) if all_logits.size else np.array(batch_logits)

                    # We can do this because: d/dx ∑ loss  == ∑ d/dx loss
                    batch_states = np.array(batch_states)
                    data = (all_states, all_actions, all_advantages, all_rewards, all_values, all_logits)
                    
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

                    self.global_model.save_model()
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
        


