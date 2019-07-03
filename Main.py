# install requirements with:
# pip install -r requirements.txt

# Utility python classes
import threading
from multiprocessing import Process
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Importing the packages for OpenAI and MARIO
import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
# from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

# Locally defined classes
from Wrappers import preprocess
from Wrappers.Monitor import Monitor
from Wrappers.Stats import Stats, Collector, AsynchronousPlot
from Worker import Worker, WorkerThread
from AC_Network import AC_Network
from Model import Model
from Coordinator import Coordinator

# Define a movement set
CUSTOM_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A', 'right'],
    ['A'],
    ['left'],
]

# Environments to run
env_1 = 'SuperMarioBros-v0'
env_2 = 'SuperMarioBros-v0'
env_3 = 'SuperMarioBros2-v0'
env_4 = 'SuperMarioBros2-v0'

env_names = [env_1, env_2, env_3, env_4]
# env_names = [env_1]

# Configuration
current_dir = os.getcwd()
model_save_path = current_dir + '.\Model'
video_save_path = current_dir + '.\Videos'
record = True

# Enviromental vars
num_envs = len(env_names)
batch_size = 16
num_minibatches = 256
num_epocs = 64
gamma = .99
learning_rate =  7e-4

# Create a new tf session with graphics enabled
tf.reset_default_graph()
config = tf.ConfigProto()

# GPU related configuration here:
config.allow_soft_placement = True 
config.gpu_options.allow_growth = True

# CPU related configuration here:
# config.intra_op_parallelism_threads = num_envs
# config.inter_op_parallelism_threads = num_envs

sess = tf.Session(config=config)


# Make the super mario gym environments and apply wrappers
envs = []
collector = Collector()
plot = AsynchronousPlot(collector)

# Apply env wrappers
for env in env_names:
    env = gym.make(env)
    env = preprocess.FrameSkip(env, 4)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = Monitor(env, env.observation_space.shape, savePath = video_save_path,  record = record)
    env = preprocess.GrayScaleImage(env, sess, height = 96, width = 96, grayscale = True)
    env = preprocess.FrameStack(env, 4)
    env = Stats(env, collector)
    envs.append(env)

(HEIGHT, WIDTH, CHANNELS) = envs[0].observation_space.shape
NUM_ACTIONS = envs[0].env.action_space.n
ACTION_SPACE = envs[0].env.action_space
NUM_STATE = (1, HEIGHT, WIDTH, CHANNELS)


# Init the Network, model, and Workers, starting them onto their own thread
network_params = (sess, NUM_STATE, batch_size, NUM_ACTIONS, ACTION_SPACE)
model = Model(network_params)
sess.run(tf.global_variables_initializer())


# Load model if exists
saver = tf.train.Saver()
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
else:
    try:
        if (os.path.exists(model_save_path + "\checkpoint")):
            saver.restore(sess, model_save_path + "\model.ckpt")
            print("Model restored.")
        else:
            print("Creating new model.")
    except:
        print("ERROR: There was an issue loading the model!")
        raise

if not os.path.exists(video_save_path):
    os.makedirs(video_save_path)

# Init coordinator and send out the workers
workers = [Worker(model, env, batch_size = batch_size, render = False) for env in envs]
coordinator = Coordinator(sess, model, workers, plot, num_envs, num_epocs, num_minibatches, batch_size, gamma, model_save_path)

# Train and save
if coordinator.run():
    try:
        save_path = saver.save(sess, model_save_path + "\model.ckpt")
        print("Model saved.")
    except:
        print("ERROR: There was an issue saving the model!")
