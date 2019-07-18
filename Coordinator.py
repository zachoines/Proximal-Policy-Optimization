import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import cv2
import scipy.signal
import tensorflow.keras as keras
import tensorflow.keras.backend as k

# import local classes
from Model import Model
from AC_Network import AC_Network
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

        (batch_states,
        batch_actions,
        batch_advantages, batch_rewards) = train_data
        
        # Now perform tensorflow session to determine policy and value loss for this batch
        # np.ndarray.tolist(self.dense_to_one_hot(batch_actions))
        feed_dict = { 
            self.global_model.step_policy.keep_per: 1.0,
            self.global_model.step_policy.X_input: batch_states.tolist(),
            self.global_model.step_policy.actions: batch_actions.tolist(),
            self.global_model.step_policy.rewards: batch_rewards.tolist(),
            self.global_model.step_policy.advantages: batch_advantages.tolist(),
        }
        
        # Run tensorflow graph, return loss without updateing gradients 
        optimizer, loss, entropy, policy_loss, value_loss, neg_log_prob, global_norm = self.sess.run(
            [self.global_model.step_policy.optimize,
             self.global_model.step_policy.loss, 
             self.global_model.step_policy.entropy, 
             self.global_model.step_policy.policy_loss, 
             self.global_model.step_policy.value_loss, 
             self.global_model.step_policy.neg_log_prob, 
             self.global_model.step_policy.global_norm], feed_dict)
        
        self.plot.collector.collect("LOSS", loss)

    # Produces reversed list of discounted rewards
    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    # for debugging processed images
    def displayImage(self, img):
        cv2.imshow('image', np.squeeze(img, axis=0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):
        
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

                    # Copy global network over to local network
                    self.refresh_local_network_params()

                    # Send workers out to threads
                    threads = []
                    for worker in self.workers:
                        threads.append(WorkerThread(target=worker.run, args=()))

                    # Start the workers on their tasks
                    for thread in threads:
                        thread.start()
                        
                    batches = []
                    
                    # Wait foreach worker to finish and return their batch
                    for thread in threads:
                        batch = thread.join()
                        batches.append(batch)

                    all_advantages = np.array([])
                    all_rewards = np.array([])
                    all_states = np.array([])
                    all_actions = np.array([])

                    # Calculate discounted rewards for each environment
                    for env in range(self.num_envs):
                        done = False
                        batch_advantages = []
                        batch_rewards = []
                        batch_values = []
                        batch_observations = []
                        batch_states = []
                        batch_actions = []

                        mb = batches[env]
                        
                        # Empty batch
                        if mb == []:
                            continue

                        # For every step made in this env for this particular batch
                        done = False
                        state = None
                        value = None
                        observation = None

                        for step in mb:
                            (state, observation, reward, value, action, done) = step
                            batch_rewards.append(reward)
                            batch_values.append(value)
                            batch_observations.append(observation)
                            batch_states.append(state)
                            batch_actions.append(action)

                        
                        # If we reached the end of an episode or if we filled a batch without reaching termination of episode
                        # we boot-strap the final rewards with the V_s(last_observation)

                        # Discounted bootstraped rewards formula:
                        # G_t == R + γ * V(S_t'):

                        # Advantage
                        # δ_t == R + γ * V(S_t') - V(S_t): 
                        
                        # Or simply... 
                        # δ_t == G_t - V:      

                        if (not done):
                            [boot_strap] = self.global_model.value([observation])
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

                    # We can do this because: d/dx ∑ loss  == ∑ d/dx loss
                    batch_states = np.array(batch_states)
                    data = (all_states, all_actions, all_advantages, all_rewards)
                    
                    if data[0].size != 0:
                        self.train(data) 
                    else:
                        break

                try:
                    # saver = tf.train.Saver()
                    # save_path = saver.save(self.sess, self.model_save_path + "\model.ckpt")
                    print("Model saved at " + save_path + ".")
                
                except:
                    print("ERROR: There was an issue saving the model!")
                    raise

            print("Training session was sucessfull.")
            return True 

        except:
            print("ERROR: The coordinator ran into an issue during training!")
            raise
      
        self.plot.join()
        


