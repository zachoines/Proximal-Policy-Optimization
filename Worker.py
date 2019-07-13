import time, random
from threading import Thread
import numpy as np
import tensorflow as tf

class WorkerThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
        self._target = target
        self._args = args
        self._kwargs = kwargs

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

# Class to represent a worker in an environment. Call run() to generate a batch. 
class Worker():

    def __init__(self, network, env, anneling_steps, batch_size = 32, render = True):

        # Epsisode collected variables
        self._batch_buffer = []
        self._done = False
        self._render = render
        self.total_steps = 0
        self.network = network
        self.env = env
        self.batch_size = batch_size 
        self.s = None
        self.NUM_ACTIONS = env.action_space.n
        self.NONE_STATE = np.zeros(self.env.observation_space.shape)
        self._currentE = 1.0
        self.anneling_steps = anneling_steps # How many steps of training to reduce startE to endE.

    # Anneal dropout for optimized exploration in a bayesian network
    def _keep_prob(self):
        keep_per = lambda: (1.0 - self._currentE) + 0.1

        startE = 1.0 # Random action chance
        endE = 0.1 # Final chance of random action
        pre_train_steps = 0 # Number of steps used before anneling begins
        stepDrop = (startE - endE) / self.anneling_steps

        if self._currentE > endE and self.total_steps > pre_train_steps:
            self._currentE -= stepDrop
            return keep_per()
        else:
            return 1.0 
        

    # Reset worker and evironment variables in preperation for a new epoc
    def reset(self):
        self._batch_buffer = []
        self._done = False
        [self.s] = self.env.reset()


    # Generate an batch worth of observations. Return nothing.
    def run(self):
        batch = []
        for step in range(self.batch_size):

            # Make a prediction and take a step if the epoc is not done
            if not self._done:
                self.total_steps += 1
                [actions_dist], value = self.network.step([self.s], self._keep_prob())
                action = self.action_select(actions_dist)
                [s_t], reward, d, _ = self.env.step(action)
                self._done = d

                batch.append((self.s, s_t , reward, value, action, d))
                s = [s_t]

                # render the env
                if (self._render):
                    self.env.render()

            else:
                self._batch_buffer.append(batch)
                return batch

        self._batch_buffer.append(batch)
        if batch == None:
            print('There is an issue!')
        return batch

    

    # Get all the batches in this epoc
    def get_batches(self):
        return self._batch_buffer
            

    # TODO::Delegate action selction to AC_Netork class by making a Boltzmann probabilities action selection
    # Boltzmann Softmax style action selection
    def action_select(self, softmax):
        
        temperature = .8
        exp_preds = np.exp(softmax / temperature)
        preds = exp_preds / np.sum(exp_preds)
        
        [probas] = np.random.multinomial(1, preds, 1)
        action = np.argmax(probas)
        
        return action

