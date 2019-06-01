# Utility python classes
import threading
import numpy as np
import tensorflow as tf
import cv2

# Importing the packages for OpenAI and MARIO
import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Locally defined classes
from Wrappers import preprocess
from Worker import Worker, WorkerThread
from AC_Network import AC_Network

# for debugging processed images
def displayImage(img):
    cv2.imshow('image', np.squeeze(img, axis=0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Environments to run
env_1 = 'SuperMarioBros-v0'
env_2 = 'SuperMarioBros-v0'
env_3 = 'SuperMarioBros2-v0'
env_4 = 'SuperMarioBros2-v0'

env_names = [env_1, env_2, env_3, env_4]
#env_names = [env_1]

# Enviromental vars
num_envs = len(env_names)
batch_size = 32
num_minibatches = 8
num_epocs = 16
gamma = .99

tf.reset_default_graph()
# Create a new tf session
config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=num_envs,
                            inter_op_parallelism_threads=num_envs)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Make the super mario gym environment and apply wrappers
envs = []
for env in env_names:
    env = gym.make(env)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    env = preprocess.GrayScaleImage(env, sess, height = 96, width = 96, grayscale = True)
    envs.append(env)

(HEIGHT, WIDTH, CHANNELS) = envs[0].observation_space.shape
NUM_ACTIONS = envs[0].env.action_space.n
NUM_STATE = (1, HEIGHT, WIDTH, CHANNELS)


# Init the Network and Workers, starting then onto their own thread
network = AC_Network(sess, NUM_STATE, NUM_ACTIONS)
sess.run(tf.global_variables_initializer())

workers = [Worker(network, env, batch_size = 32, render = False) for env in envs]

# Main training loop
for epoch in range(num_epocs):

    # ready workers for the next epoc
    for worker in workers:
        worker.reset()

    # loop for generating a training session a batch at a time
    for mb in range(num_minibatches):

        # Send workers out to threads
        threads = []
        for worker in workers:
            threads.append(WorkerThread(target=worker.run, args=()))

        # Start the workers on their tasks
        for thread in threads:
            thread.start()
            
        batches = []
        # Wait foreach worker to finish and return their batch
        for thread in threads:
            batches.append(thread.join())

        # Calculate discounted rewards for each environment
        for env in range(num_envs):
            done = False
            bootstrap_value = 0
            total_discounted_rewards = 0
            steps = 0
            batch_advantages = []
            batch_values_and_rewards = []

            mb = batches[env]
            for step in mb:
                steps += 1
                (sate, observation, reward, value, done) = step
                

                # displayImage(observation)

                # If we reached the end of an episode or if we filled a batch without reaching termination of episode
                # we boot strap the final rewards with the v_s(last_observation)
                if (steps % batch_size == 0):
                    total_discounted_rewards += value
                    break
                elif done:
                    total_discounted_rewards = 0
                    break
                else:
                    # collect step rewards and values
                    batch_values_and_rewards.append((reward, value))
                    continue
        
            
            # from t - 1 to t_start, find discounted rewards and advantages
            for (reward, value) in reversed(batch_values_and_rewards):
                total_discounted_rewards += reward + gamma * total_discounted_rewards
                batch_advantages.append(reward - value)

            lastgaelam = 0

            # From last step to first step
            for t in reversed(range(self.steps)):
                # If t == before last step
                if t == steps - 1:
                    # If a state is done, nextnonterminal = 0
                    # In fact nextnonterminal allows us to do that logic

                    #if done (so nextnonterminal = 0):
                    #    delta = R - V(s) (because self.gamma * nextvalues * nextnonterminal = 0) 
                    # else (not done)
                        #delta = R + gamma * V(st+1)
                    nextnonterminal = 1.0 - self.dones
                    
                    # V(t+1)
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[t+1]
                    
                    nextvalues = mb_values[t+1]

                # Delta = R(st) + gamma * V(t+1) * nextnonterminal  - V(st)
                delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]

                # Advantage = delta + gamma *  λ (lambda) * nextnonterminal  * lastgaelam
                mb_advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

            # Returns
            mb_returns = mb_advantages + mb_values
                


