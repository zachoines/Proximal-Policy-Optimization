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
from Model import Model

# Produces reversed list of discounted rewards
def discounted_rewards(rewards, dones, gamma):
    discounted = []
    r = 0
    # Start from downwards to upwards like Bellman backup operation.
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)  # fixed off by one bug
        discounted.append(r)
    return np.array(discounted[::-1])

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
learning_rate =  7e-4

# Create a new tf session
tf.reset_default_graph()
config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=num_envs,
                            inter_op_parallelism_threads=num_envs)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Make the super mario gym environments and apply wrappers
envs = []
for env in env_names:
    env = gym.make(env)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    env = preprocess.GrayScaleImage(env, sess, height = 96, width = 96, grayscale = True)
    envs.append(env)

(HEIGHT, WIDTH, CHANNELS) = envs[0].observation_space.shape
NUM_ACTIONS = envs[0].env.action_space.n
ACTION_SPACE = envs[0].env.action_space
NUM_STATE = (1, HEIGHT, WIDTH, CHANNELS)



# Init the Network, model, and Workers, starting them onto their own thread
network_params = (sess, NUM_STATE, batch_size, NUM_ACTIONS, ACTION_SPACE)
model = Model(network_params)
sess.run(tf.global_variables_initializer())

workers = [Worker(model, env, batch_size = 32, render = False) for env in envs]

# Main training loop
for epoch in range(num_epocs):

    # ready workers for the next epoc
    for worker in workers:
        worker.reset()

    # loop for generating a training session a batch at a time
    for mb in range(num_minibatches):

        # Copy global network over to local network
        model.refresh_local_network_params

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

        all_batches_discounted_rewards = np.array([])
        all_batches_advantages = np.array([])
        all_batches_values = np.array([])
        all_batches_actions = np.array([])
        all_batches_loss = np.array([])
        all_batches_observations = np.array([])
        all_batches_states = []
        all_batches_rewards = np.array([])
        
        # Calculate discounted rewards for each environment
        for env in range(num_envs):
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

                # displayImage(observation)

                # If we reached the end of an episode or if we filled a batch without reaching termination of episode
                # we boot-strap the final rewards with the V_s(last_observation)

                # Discounted bootstraped rewards formula:
                # G_t == R + γ * V(S_t'):

                # Advantage
                # δ_t == R + γ * V(S_t') - V(S_t): 
                if (steps % batch_size == 0):
                    
                    # Bootstrap terminal state value onto list of discounted retur                    batch_rewards = adjusted_discounterd_rewards(batch_rewards)
                    batch_rewards = discounted_rewards(batch_rewards + [0.0], batch_dones, gamma)
                    V_S = np.append(batch_values[:-1], 0.0)
                    V_next = np.append(np.array(0.0), batch_values[1:])
                    batch_advantages = batch_rewards + gamma * ((V_next) - (V_S))
                    break
                elif done:
                    
                    # Generate a reversed dicounted list of returns without boostrating (adding V(s_terminal)) on non-terminal state
                    batch_rewards = discounted_rewards(batch_rewards + [0.0], batch_dones, gamma)
                    V_S = np.append(batch_values[:-1], 0.0)
                    V_next = np.append(np.array(0.0), batch_values[1:])
                    batch_advantages = batch_rewards + gamma * ((V_next) - (V_S))
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
            model.step_policy.X_input: all_batches_states,
            model.step_policy.actions: all_batches_actions,
            model.step_policy.advantages: all_batches_advantages,
            model.step_policy.values: all_batches_values,
            model.step_policy.rewards: all_batches_rewards, 
        }
        
        # Run tensorflow graph, return loss without updateing gradients 
        loss, optimizer = sess.run([model.step_policy.loss, model.step_policy.optimize], feed_dict)



        
        

            
