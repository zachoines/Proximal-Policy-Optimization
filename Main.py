# install requirements with:
# pip install -r requirements.txt

# Utility python classes
import threading
from multiprocessing import Process
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib


# Importing the packages for OpenAI and MARIO
import gym
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



def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


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
env_5 = 'SuperMarioBros-v0'

env_names = [env_1]

# Configuration
current_dir = os.getcwd()
model_save_path = current_dir + '.\Model'
video_save_path = current_dir + '.\Videos'
record = True

# Enviromental vars
num_envs = len(env_names)
batch_size = 16
num_minibatches = 512
num_epocs = 32
gamma = .99
learning_rate = 7e-4

# Make the super mario gym environments and apply wrappers
envs = []
collector = Collector()
collector.set_dimensions(["CMA", "LOSS"])
plot = AsynchronousPlot(collector, live = False)

# Apply env wrappers
for env in env_names:
    env = gym_super_mario_bros.make(env) # gym.make(name), for other env's in the future.
    env = preprocess.FrameSkip(env, 4)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = Monitor(env, env.observation_space.shape, savePath = video_save_path,  record = record)
    env = preprocess.GrayScaleImage(env, height=96, width=96, grayscale=True)
    env = preprocess.FrameStack(env, 4)
    env = Stats(env, collector)
    envs.append(env)

(HEIGHT, WIDTH, CHANNELS) = envs[0].observation_space.shape
NUM_ACTIONS = envs[0].env.action_space.n
ACTION_SPACE = envs[0].env.action_space
NUM_STATE = (1, HEIGHT, WIDTH, CHANNELS)


# Load model if exists
# saver = tf.train.Saver()
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
else:
    try:
        if (os.path.exists(model_save_path + "\checkpoint")):
            # saver.restore(sess, model_save_path + "\model.ckpt")
            print("Model restored.")
        else:
            print("Creating new model.")
    except:
        print("ERROR: There was an issue loading the model!")
        raise

if not os.path.exists(video_save_path):
    os.makedirs(video_save_path)

if not os.path.exists('.\stats'):
    os.makedirs('.\stats')

# Init coordinator and send out the workers
anneling_steps = num_epocs * num_minibatches * batch_size

# K.manual_variable_initialization(True)
workers = []
network_params = (NUM_STATE, batch_size, NUM_ACTIONS, ACTION_SPACE)
tf.reset_default_graph()
config=tf.ConfigProto(allow_soft_placement=True)
main_sess = tf.Session(config=config)

# K.set_session(main_sess)
gpus = get_available_gpus()
device = gpus[0] if gpus else "cpu"
with tf.device(device):
    Train_Model = Model(network_params, main_sess)

step_models = []
for env in envs:
    
    step_Model = Model(network_params, main_sess, Train_Model.get_network()) 
    step_models.append(step_Model)
    gpus = get_available_gpus()
    device = gpus[0] if gpus else "cpu"
    with tf.device(device):
        workers.append(Worker(step_Model, env, anneling_steps, batch_size=batch_size, render=False))

main_sess.run(tf.global_variables_initializer())

coordinator = Coordinator(Train_Model, step_models, workers, plot, num_envs, num_epocs, num_minibatches, batch_size, gamma, model_save_path)


# Train and save
if coordinator.run():
    try:
        # save_path = saver.save(sess, model_save_path + "\model.ckpt")
        print("Model saved.")
        print("Now testing results....")
    except:
        print("ERROR: There was an issue saving the model!")
