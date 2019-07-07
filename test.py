# Utility python classes
import os
import numpy as np
import tensorflow as tf

# Importing the packages for OpenAI and MARIO
import gym
from gym import wrappers
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

# Locally defined classes
from Wrappers import preprocess
from Wrappers.Stats import Stats, Collector, AsynchronousPlot
from Model import Model

def action_select(softmax):
        
    temperature = 1.0
    exp_preds = np.exp(softmax / temperature)
    preds = exp_preds / np.sum(exp_preds)
    
    [probas] = np.random.multinomial(1, preds, 1)
    action = np.argmax(probas)
    
    return action


# Environments to run
env_1 = 'SuperMarioBros-v0'
env_2 = 'SuperMarioBros2-v0'
env_names = [env_1]

# Configuration
current_dir = os.getcwd()
model_save_path = current_dir + '.\Model'      


# Create a new tf session with graphics enabled
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.reset_default_graph()
config = tf.ConfigProto()
sess = tf.Session(config=config)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = preprocess.FrameSkip(env, 4)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = preprocess.GrayScaleImage(env, sess, height = 96, width = 96, grayscale = True)
env = preprocess.FrameStack(env, 4)


(HEIGHT, WIDTH, CHANNELS) = env.observation_space.shape
NUM_ACTIONS = env.env.action_space.n
ACTION_SPACE = env.env.action_space
NUM_STATE = (1, HEIGHT, WIDTH, CHANNELS)


network_params = (sess, NUM_STATE, 16, NUM_ACTIONS, ACTION_SPACE)
model = Model(network_params)
sess.run(tf.global_variables_initializer())


def run():

    # Load model if exists
    saver = tf.train.Saver()
    try:
        if (os.path.exists(model_save_path + "\checkpoint")):
            saver.restore(sess, model_save_path + "\model.ckpt")
            print("Model restored.")
        else:
            print("No model.")
            exit(-1)
    except:
        print("ERROR: There was an issue loading the model!")
        raise



    try:
        done = True
        state = None
        
        while True:
            if done:
                state = env.reset()
            [action], _ = model.step(state)
            action = action_select(action)
            state, _, done, _ = env.step(action)
            env.render()
    
    except (KeyboardInterrupt, SystemExit):
        print('KeyboardInterrupt caught')
        raise 
    finally:
        env.close()
 

    



