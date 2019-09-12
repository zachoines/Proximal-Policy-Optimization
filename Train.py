# Utility python classes
import threading
from multiprocessing import Process
import os
import sys
import imp
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

# Importing the packages for OpenAI
import gym
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

# Locally defined classes
from Wrappers import preprocess
from Wrappers.Normalize import Normalize, MsPacmanWrapper
from Wrappers.Monitor import Monitor
from Wrappers.Stats import Stats, Collector, AsynchronousPlot
from Worker import Worker, WorkerThread
from NN.CNN_LARGE import AC_Model_Large
from NN.CNN_SMALL import AC_Model_Small
from Coordinator import Coordinator
from Wrappers.preprocess import FrameSkip

# Dynamically import and load wrapper classes for env's. Configure in Config.py.
def env_wrapper_import(class_name, env):
    module = __import__('Wrappers')
    my_class = getattr(module.Normalize, class_name)
    return my_class(env)

# Dynamically import and load CNN classes for env's. Configure in Config.py
def CNN_class_import(class_name, params):
    module = __import__('NN')
    my_class = getattr(module, class_name)
    return my_class(*params)

class Train():

    def __init__(self, config):

        self._config = config

        def get_available_gpus():
            local_device_protos = device_lib.list_local_devices()
            return [x.name for x in local_device_protos if x.device_type == 'GPU']

        print("GPU Available: ", tf.test.is_gpu_available())

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

        env_names = []
        for _ in range(self._config['Number of worker threads']):
            env_names.append(self._config['Environment Name'  ])

        # Configuration
        # current_dir = os.getcwd()
        self._model_save_path = '.\Model'
        self._video_save_path = '.\Videos'
        self.record = True

        # Make the super mario gym environments and apply wrappers
        self._envs = []
        collector = Collector()
        collector.set_dimensions(["CMA", "EMA", "SMA", "LENGTH", "LOSS", 'TOTAL_EPISODE_REWARDS'])
        self._plot = AsynchronousPlot(collector, live=False)

        # Apply env wrappers
        counter = 0
        for env_name in env_names:
            env = gym.make(env_name) 
     
            if env_name == 'SuperMarioBros-v0':
                env = JoypadSpace(env, COMPLEX_MOVEMENT)

            # Load wrapper class
            if self._config['Wrapper class'] != '':
                env = env_wrapper_import(self._config['Wrapper class'], env)
            env = Stats(env, collector)
            
            env = Monitor(env, env.observation_space.shape, savePath=self._video_save_path, record=self.record)

            env = preprocess.GrayScaleImage(env, height=84, width=84, grayscale=self._config['Grayscale'])
            env = preprocess.FrameStack(env, 4)
        
            self._envs.append(env)

        self.NUM_STATE = self._envs[0].observation_space.shape
        self.NUM_ACTIONS = self._envs[0].env.action_space.n
        self.ACTION_SPACE = self._envs[0].env.action_space

        if not os.path.exists(self._video_save_path):
            os.makedirs(self._video_save_path)

        if not os.path.exists(self._model_save_path):
            os.makedirs(self._model_save_path)

        if not os.path.exists('.\stats'):
            os.makedirs('.\stats')

    def start(self):
        workers = []
        network_params = (self.NUM_STATE, self._config['Max steps taken per batch'], self.NUM_ACTIONS, self.ACTION_SPACE)

        # Init Global and Local networks. Generate Weights for them as well.
        if self._config['CNN type'] == '':
            self._global_model = AC_Model_Large(self.NUM_STATE, self.NUM_ACTIONS, self._config, is_training=True)
            self._global_model(tf.convert_to_tensor(np.random.random((1, *self.NUM_STATE)), dtype='float64'))
            self._step_model = AC_Model_Large(self.NUM_STATE, self.NUM_ACTIONS, self._config, is_training=True)
            self._step_model(tf.convert_to_tensor(np.random.random((1, *self.NUM_STATE)), dtype='float64'))
        else:     
            self._global_model = CNN_class_import(self._config['CNN type'], (self.NUM_STATE, self.NUM_ACTIONS, self._config, True))
            self._global_model(tf.convert_to_tensor(np.random.random((1, *self.NUM_STATE)), dtype='float64'))
            self._step_model = CNN_class_import(self._config['CNN type'], (self.NUM_STATE, self.NUM_ACTIONS, self._config, True))
            self._step_model(tf.convert_to_tensor(np.random.random((1, *self.NUM_STATE)), dtype='float64'))

        # Load model if exists
        if not os.path.exists(self._model_save_path):
            os.makedirs(self._model_save_path)
        else:
            try:
                if (os.path.exists(self._model_save_path + "\checkpoint")):
                    
                    self._global_model.load_model_weights()
                    self._step_model.load_model_weights()
                    for env in self._envs:
                        workers.append(Worker(self._step_model, env, batch_size=self._config['Max steps taken per batch'], render=False))
                    
                    print("Model restored.")
                
                else:
                    
                    for env in self._envs:
                        workers.append(Worker(self._step_model, env, batch_size=self._config['Max steps taken per batch'], render=False))
                    
                    print("Creating new model.")
            except:
                print("ERROR: There was an issue loading the model!")
                raise

        coordinator = Coordinator(  
                                    self._global_model, 
                                    self._step_model, 
                                    workers, 
                                    self._plot, 
                                    self._model_save_path, 
                                    self._config
                                )

        # Train and save
        try:
            if coordinator.run():
                try:
                    self._global_model.save_model_weights()
                    print("Model saved.")
                    return True
                except:
                    print("ERROR: There was an issue saving the model!")
                    raise 
                
        except:
            print("ERROR: There was an issues during training!")
            raise 


        

    


    
