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
from Wrappers.Monitor import Monitor
from Worker import Worker, WorkerThread
from AC_Network import AC_Network
from Model import Model
from Coordinator import Coordinator

# Environments to run
env_1 = 'SuperMarioBros-v0'
env_2 = 'SuperMarioBros-v0'
env_3 = 'SuperMarioBros2-v0'
env_4 = 'SuperMarioBros2-v0'

# env_names = [env_1, env_2, env_3, env_4]
env_names = [env_1]

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
counter = 1
record = False
for env in env_names:
    if (counter == num_envs):
        counter += 1
        record = True

    env = gym.make(env)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    env = Monitor(env, record = record)
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

coordinator = Coordinator(sess, model, workers, num_envs, num_epocs, num_minibatches, batch_size, gamma)
coordinator.run()
