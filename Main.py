# Utility python classes
import threading
import numpy as np
import tensorflow as tf

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
from Worker import Worker
from AC_Network import AC_Network



### SANITY CHECK SECTION ###
env_1 = 'SuperMarioBros-v0'
env_2 = 'SuperMarioBros-v0'
env_3 = 'SuperMarioBros2-v0'
env_4 = 'SuperMarioBros2-v0'

env_names = [env_1, env_2, env_3, env_4]
env_names = [env_1]

# Enviromental vars
num_envs = len(env_names)
num_minibatches = 8

# Create a new tf session
# config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=num_envs, inter_op_parallelism_threads=num_envs)
# config.gpu_options.allow_growth = True
# config=config
# sess = tf.Session(config)
sess = tf.Session()

# Make the super mario gym environment and apply wrappers
envs = []
for env in env_names:
    env = gym.make(env)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    env = preprocess.GrayScaleImage(env, sess, height = 64, width = 64, grayscale = True)
    envs.append(env)

(HEIGHT, WIDTH, CHANNELS) = envs[0].observation_space.shape
NUM_ACTIONS = envs[0].env.action_space.n
NUM_STATE = (None, HEIGHT, WIDTH, CHANNELS)
# NONE_STATE = np.zeros(NUM_STATE)



# Init the Network and Workers, starting then onto their own thread
network = AC_Network(sess, NUM_STATE, NUM_ACTIONS)
sess.run(tf.global_variables_initializer())

workers = [Worker(network, env, batch_size = 32, num_batches = 8, render = False) for env in envs]

threads = []

for worker in workers:
    t = threading.Thread(target = worker.run(), args=())
    threads.append(t)
    t.start()

for thread in threads:
    thread.join()

total_batches = []
for worker in workers:
    total_batches.append(worker.get_batches())
