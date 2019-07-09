import tensorflow as tf
import numpy as np
import cv2
import scipy.signal

# import local classes
from Model import Model
from AC_Network import AC_Network
from Worker import Worker, WorkerThread


class Coordinator:
    def __init__(self, session, model, workers, plot, num_envs, num_epocs, num_minibatches, batch_size, gamma, model_save_path):
        self.sess = session
        self.model = model
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
    
    def train(self, train_data):

        (batch_states,
        batch_actions,
        batch_advantages, batch_rewards) = train_data
        
        # Now perform tensorflow session to determine policy and value loss for this batch
        # np.ndarray.tolist(self.dense_to_one_hot(batch_actions))
        feed_dict = { 
            self.model.step_policy.X_input: batch_states,
            self.model.step_policy.actions: batch_actions,
            self.model.step_policy.rewards: batch_rewards,
            self.model.step_policy.advantages: batch_advantages,
        }
        
        # Run tensorflow graph, return loss without updateing gradients 
        optimizer, loss, entropy, policy_loss, value_loss, log_prob, self.global_norm = self.sess.run([self.model.step_policy.optimize, self.model.step_policy.loss, self.model.step_policy.entropy, self.model.step_policy.policy_loss, self.model.step_policy.value_loss, self.model.step_policy.log_prob, self.model.step_policy.global_norm],  feed_dict)
        
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
                    self.model.refresh_local_network_params()

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

                    
                    # Calculate discounted rewards for each environment
                    for env in range(self.num_envs):
                        done = False
                        batch_advantages = []
                        batch_rewards = []
                        batch_values = []
                        batch_dones = []
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

                        for step in mb:
                            (state, observation, reward, [value], action, done) = step
                            batch_rewards.append(reward)
                            batch_values.append(value)
                            batch_dones.append(done)
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
                            
                            [boot_strap] = self.model.value([state])

                            bootstrapped_rewards = np.asarray(batch_rewards + [boot_strap])
                            discounted_rewards = self.discount(bootstrapped_rewards, self.gamma)[:-1]
                            advantages = discounted_rewards - batch_values

                            data = (batch_states, batch_actions, advantages, discounted_rewards)
                            self.train(data)

                            # Same as above...
                            # bootstrapped_values = np.asarray(batch_values + [boot_strap])
                            # advantages = batch_rewards + self.gamma * bootstrapped_values[1:] - bootstrapped_values[:-1]
                            # advantages = self.discount(advantages, self.gamma)
                        
                        else:
                            
                            boot_strap = 0

                            bootstrapped_rewards = np.asarray(batch_rewards + [boot_strap])
                            discounted_rewards = self.discount(bootstrapped_rewards, self.gamma)[:-1]
                            advantages = discounted_rewards - batch_values

                            data = (batch_states, batch_actions, advantages, discounted_rewards)
                            self.train(data)

                            # Same as above...
                            # bootstrapped_values = np.asarray(batch_values + [boot_strap])
                            # advantages = batch_rewards + self.gamma * bootstrapped_values[1:] - bootstrapped_values[:-1]
                            # advantages = self.discount(advantages, self.gamma)

                try:
                    saver = tf.train.Saver()
                    save_path = saver.save(self.sess, self.model_save_path + "\model.ckpt")
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
        


