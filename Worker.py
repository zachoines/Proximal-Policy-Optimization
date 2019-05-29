import time, random, threading
import numpy as np
import tensorflow as tf


# Class to represent a worker in an environment. Call run() to generate a batch. 
class Worker:

    def __init__(self, network, env, batch_size = 32, num_batches = 8, render = True):

        # Epsisode collected variables
        self._batch_buffer = []
        self._done = False
        self._render = render

        self.network = network
        self.env = env
        self.batch_size = batch_size 
        self.num_batches = num_batches
        self.s = env.reset()

    # Reset worker and evironment variables in preperation for a new epoc
    def reset():
        self._batch_buffer = []
        self._done = False
        self.s = env.reset()


    # Generate an epocs worth of observations. Return nothing.
    def run(self):
        
        # Generate a minibatches 
        for mb in range(self.num_batches):
        
            for step in range(self.batch_size):

                # Make a prediction and take a step if the epoc is not done
                if not self._done:
                    [action], value = self.network.step(self.s)
                    # action = self.action_select(actions)
                    s_t, reward, d, _ = self.env.step(action)
                    self._done = d

                    self._batch_buffer.append((self.s, s_t, reward, value, d))
                    s = s_t

                    # render the env
                    if (self._render):
                        self.env.render()
                # if the episode is already _done, generate a null entry
                else:
                    self._batch_buffer.append((None, None, 0, 0, True))

        
                
      

        
            

            

    # Get all the batches in this epoc
    def get_batches(self):
        return self._batch_buffer
            

    # TODO::Delegate action selction to netork by making a new tensor session to find Boltzmann probabilities action selection
    def action_select(self, softmax):
        temperature = .7
        EPSILON = 10e-16 # to avoid taking the log of zero
        
        (np.array(softmax) + EPSILON).astype('float64')
        preds = np.log(softmax) / temperature
        
        exp_preds = np.exp(preds)
        
        preds = exp_preds / np.sum(exp_preds)
        
        probas = np.random.multinomial(1, preds, 1)
        return probas[0]

