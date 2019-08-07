# install requirements with:
# pip install -r requirements.txt

# Utility python classes
import threading
from multiprocessing import Process
import os
import sys
import numpy as np
import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib


print("GPU Available: ", tf.test.is_gpu_available())


# Importing the packages for OpenAI and MARIO
import gym
# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

# Locally defined classes
from Wrappers import preprocess
from Wrappers.Monitor import Monitor
from Wrappers.Stats import Stats, Collector, AsynchronousPlot
from Worker import Worker, WorkerThread
from AC_Network import AC_Model
from Coordinator import Coordinator

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
if gpus:
  try:
    
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# Environments to run
env_1 = 'BreakoutDeterministic-v4'
env_2 = 'Breakout-v4'
env_3 = 'BreakoutDeterministic-v0'
env_4 = 'MsPacmanDeterministic-v4'
env_5 = 'MsPacman-v0'


# env_names = [env_1, env_2, env_3, env_4]
env_names = ['Breakout-v0']

# Configuration
current_dir = os.getcwd()
model_save_path = current_dir + '.\Model'
video_save_path = current_dir + '.\Videos'
record = True

# Enviromental vars
num_envs = len(env_names)
batch_size = 8
batches_per_epoch = sys.maxsize
num_epocs = 512 * 4
gamma = .99
learning_rate = 7e-4
anneling_steps = num_epocs**2 

# Make the super mario gym environments and apply wrappers
envs = []
collector = Collector()
collector.set_dimensions(["CMA", "LOSS", "LENGTH"])
plot = AsynchronousPlot(collector, live=False)

# Apply env wrappers
for env in env_names:
    env = gym.make(env) 
    # env = preprocess.FrameSkip(env, 4)
    # env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = Monitor(env, env.observation_space.shape, savePath=video_save_path, record=record)
    env = preprocess.GrayScaleImage(env, height=96, width=96, grayscale=True)
    # env = preprocess.FrameStack(env, 4)
    env = Stats(env, collector)
    envs.append(env)

(HEIGHT, WIDTH, CHANNELS) = envs[0].observation_space.shape
NUM_ACTIONS = envs[0].env.action_space.n
ACTION_SPACE = envs[0].env.action_space
NUM_STATE = (1, HEIGHT, WIDTH, CHANNELS)  

if not os.path.exists(video_save_path):
    os.makedirs(video_save_path)

if not os.path.exists('.\stats'):
    os.makedirs('.\stats')


workers = []
network_params = (NUM_STATE, batch_size, NUM_ACTIONS, ACTION_SPACE)

# Init Global and Local networks. Generate Weights for them as well.
Global_Model = AC_Model(NUM_STATE, NUM_ACTIONS, is_training=True)
(_, hight, width, stack) = NUM_STATE
Global_Model(tf.convert_to_tensor(np.random.random((1, hight, width*stack, 1)), dtype=tf.float32))
step_model = AC_Model(NUM_STATE, NUM_ACTIONS, is_training=True)
step_model(tf.convert_to_tensor(np.random.random((1, hight, width*stack, 1)), dtype=tf.float32))


# Load model if exists
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
else:
    try:
        if (os.path.exists(model_save_path + "\checkpoint")):
            
            Global_Model.load_model_weights()
            step_model.load_model_weights()
            for env in envs:
                workers.append(Worker(step_model, env, batch_size=batch_size, render=False))
            
            print("Model restored.")
        
        else:
            
            for env in envs:
                workers.append(Worker(step_model, env, batch_size=batch_size, render=False))
            
            print("Creating new model.")
    except:
        print("ERROR: There was an issue loading the model!")
        raise

coordinator = Coordinator(Global_Model, step_model, workers, plot, num_envs, num_epocs, batches_per_epoch, batch_size, gamma, model_save_path, anneling_steps)

# Train and save
if coordinator.run():
    try:
        Global_Model.save_model_weights()
        print("Model saved.")
        print("Now testing results....")
    except:
        print("ERROR: There was an issue saving the model!")
