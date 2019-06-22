import tensorflow as tf
import numpy as np
import cv2

# import local classes
from Model import Model
from AC_Network import AC_Network
from Worker import Worker, WorkerThread


class Coordinator:
    def __init__(self, session, model, workers, num_envs, num_epocs, num_minibatches, batch_size, gamma):
        self.sess = session
        self.model = model
        self.workers = workers
        self.num_envs = num_envs
        self.num_epocs = num_epocs
        self.num_minibatches = num_minibatches
        self.batch_size = batch_size
        self.gamma = gamma

    # Produces reversed list of discounted rewards
    def discounted_rewards(self, rewards, dones, gamma):
        discounted = []
        r = 0
        # Start from downwards to upwards like Bellman backup operation.
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)  # fixed off by one bug
            discounted.append(r)
        return np.array(discounted[::-1])

    # for debugging processed images
    def displayImage(self, img):
        cv2.imshow('image', np.squeeze(img, axis=0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):
        
        try:
            # Main training loop
            for epoch in range(self.num_epocs):

                # ready workers for the next epoc
                for worker in self.workers:
                    worker.reset()

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
                        batches.append(thread.join())

                    all_batches_discounted_rewards = np.array([])
                    all_batches_advantages = np.array([])
                    all_batches_values = np.array([])
                    all_batches_actions = np.array([])
                    all_batches_loss = np.array([])
                    all_batches_observations = np.array([])
                    all_batches_states = []
                    all_batches_rewards = np.array([])
                    
                    # Calculate discounted rewards for each environment
                    for env in range(self.num_envs):
                        done = False
                        bootstrap_value = 0
                        total_discounted_rewards = 0
                        steps = 0
                        batch_advantages = np.array([])
                        batch_rewards = []
                        batch_values = np.array([])
                        batch_dones = []
                        batch_observations = []
                        batch_states = []
                        batch_actions = []

                        mb = batches[env]

                        # For every step made in this env for this particular batch
                        for step in mb:
                            steps += 1
                            (state, observation, reward, value, action, done) = step
                            batch_rewards.append(reward)
                            batch_values = np.append(batch_values, value)
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
                            if (steps % self.batch_size == 0):
                                
                                # Bootstrap terminal state value onto list of discounted retur                    batch_rewards = adjusted_discounterd_rewards(batch_rewards)
                                batch_rewards = self.discounted_rewards(batch_rewards + [0.0], batch_dones, self.gamma)
                                V_S = np.append(batch_values[:-1], 0.0)
                                V_next = np.append(np.array(0.0), batch_values[1:])
                                batch_advantages = batch_rewards + self.gamma * ((V_next) - (V_S))
                                break
                            elif done:
                                
                                # Generate a reversed dicounted list of returns without boostrating (adding V(s_terminal)) on non-terminal state
                                batch_rewards = self.discounted_rewards(batch_rewards + [0.0], batch_dones, self.gamma)
                                V_S = np.append(batch_values[:-1], 0.0)
                                V_next = np.append(np.array(0.0), batch_values[1:])
                                batch_advantages = batch_rewards + self.gamma * ((V_next) - (V_S))
                                break
                            else:
                                
                                # Continue accumulating batch data
                                continue

                        


                        # Collect all individual batch data from each env
                        all_batches_values = np.concatenate((all_batches_values, batch_values), axis=0)
                        all_batches_discounted_rewards = np.concatenate((all_batches_discounted_rewards, batch_rewards), axis = 0)
                        all_batches_advantages = np.concatenate((all_batches_advantages, batch_advantages), axis = 0)
                        all_batches_actions = np.concatenate((all_batches_actions, batch_actions), axis = 0)
                        
                        # all_batches_observations = np.concatenate((all_batches_observations, batch_observations), axis = 0)
                        all_batches_states += batch_states
                        all_batches_rewards = np.concatenate((all_batches_rewards, batch_rewards), axis = 0)

                    # Now perform tensorflow session to determine policy and value loss for this batch
                    feed_dict = { 
                        self.model.step_policy.X_input: all_batches_states,
                        self.model.step_policy.actions: all_batches_actions,
                        self.model.step_policy.advantages: all_batches_advantages,
                        self.model.step_policy.values: all_batches_values,
                        self.model.step_policy.rewards: all_batches_rewards, 
                    }
                    
                    # Run tensorflow graph, return loss without updateing gradients 
                    loss, optimizer = self.sess.run([self.model.step_policy.loss, self.model.step_policy.optimize], feed_dict)
            print("Training session was sucessfull.")
            return True
        except:
            print("ERROR: The coordinator ran into an issue during training!")
            raise  
            return False
        

