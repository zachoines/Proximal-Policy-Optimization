# Utility python classes
import os
import sys
import numpy as np
import time
import tensorflow as tf

# Importing the packages for OpenAI and MARIO
import gym

# Locally defined classes
from NN.CNN_LARGE import AC_Model_Large
from NN.CNN_SMALL import AC_Model_Small

class Test():
    def __init__(self, config):
        self._config = config
        # Configuration
        current_dir = os.getcwd()
        model_save_path = current_dir + '.\Model'

        # Apply env wrappers
        env = gym.make(config['Environment Name'])
        NUM_STATE = env.observation_space.shape
        NUM_ACTIONS = env.env.action_space.n
        ACTION_SPACE = env.env.action_space

        network_params = (NUM_STATE, 1.0, NUM_ACTIONS, ACTION_SPACE)

        if self._config['CNN type'] == 'large':
            self.Test_Model = AC_Model_Large(NUM_STATE, NUM_ACTIONS, self._config, is_training=False)
        else:
            self.Test_Model = AC_Model_Small(NUM_STATE, NUM_ACTIONS, is_training=False)

        # Load model if exists
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        else:
            try:
                if (os.path.exists(model_save_path + "\checkpoint")):
                    
                    Test_Model.load_model_weights()
                    print("Model restored.")
                    print("Now running environment...")
                    self._run(10000, env, Test_Model, True)
                
                else:
                    raise("There is no model available")
            except:
                print("ERROR: There was an issue running the model!")
                raise

    # Generate an batch worth of observations. Return nothing.
    def _run(self, num_steps, env, network, render):
        s = env.reset()
        done = False
        for step in range(num_steps):
            time.sleep(.00001)
            
            # Make a prediction and take a step if the epoc is not done
            if not done:
                [logits], d, _ = network.step(np.expand_dims(s, axis=0), 1.0)
                action = self._action_select(logits)
                s_t, reward, d, stuff = env.step(action)
                done = d
                s = s_t

                # render the env
                if (render and not done):
                    env.render()
            
            else:
                done = False
                if (render):
                    s = env.reset()
                    env.render()
        print("Test run has finished.")
        return

    # Action selection
    def _action_select(self, dist):
        dist = tf.nn.softmax(dist).numpy() 
        a = np.argmax(dist)
        return a

    

