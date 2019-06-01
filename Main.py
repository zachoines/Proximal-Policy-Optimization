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

# Produces reversed list of discounted rewards
def discounted_rewards(rewards, dones, gamma):
        discounted = []
        r = 0
        # Start from downwards to upwards like Bellman backup operation.
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)  # fixed off by one bug
            discounted.append(r)
        return discounted[::-1]

# Adjusted discounterd rewards should be used when rewards received from the env at each step are the sum of all previous
# rewards plus the reward for the current step. Ex.) [1, 2, 3] -> [1, 1, 1] 
def adjusted_discounterd_rewards(rewards):

    adjusted_rewards = []
    previous_reward = 0
    for reward in rewards:
        adjusted_rewards.append(rewards - previous_reward)
        previous_reward = reward
    
    return adjusted_rewards


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

# Enviromental vars
num_envs = len(env_names)
batch_size = 32
num_minibatches = 8
num_epocs = 16
gamma = .99

# Create a new tf session
tf.reset_default_graph()
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

        all_batches_discounted_rewards = []
        all_batches_advantages = []
        # Calculate discounted rewards for each environment
        for env in range(num_envs):
            done = False
            bootstrap_value = 0
            total_discounted_rewards = 0
            steps = 0
            batch_advantages = []
            batch_rewards = []
            batch_values = []
            batch_dones = []

            mb = batches[env]

            # For every step made in this env for this particular batch
            for step in mb:
                steps += 1
                (state, observation, reward, value, done) = step
                batch_rewards.append(reward)
                batch_values.append(reward)
                batch_dones.append(done)
                

                # displayImage(observation)

                # If we reached the end of an episode or if we filled a batch without reaching termination of episode
                # we boot strap the final rewards with the v_s(last_observation)
                if (steps % batch_size == 0):
                    
                    # Bootstrap terminal state value onto list of discounted retur                    batch_rewards = adjusted_discounterd_rewards(batch_rewards)
                    batch_rewards = discounted_rewards(batch_rewards, batch_dones, gamma)
                    break
                elif done:
                    # Generate a reversed dicounted list of returns without boostrating (adding V(s_terminal)) on non-terminal state
                    batch_rewards = discounted_rewards(batch_rewards + [value], gamma)[:-1]
                    break
                else:
                    # Continue accumulating batch data
                    continue

            # Collect individual batch data
            all_batches_discounted_rewards.append(batch_rewards)
        

            
