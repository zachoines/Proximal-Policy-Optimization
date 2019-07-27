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

    def __init__(self, network, env, batch_size = 32, render = True, exploration="bayesian"):

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
        self.exploration = exploration
        
        
    # Reset worker and evironment variables in preperation for a new epoc
    def reset(self):
        self._batch_buffer = []
        self._done = False
        [self.s] = self.env.reset()


    # Generate an batch worth of observations. Return nothing.
    def run(self, keep_prob):
        batch = []
        for step in range(self.batch_size):

            # Make a prediction and take a step if the epoc is not done
            if not self._done:
                self.total_steps += 1
                [logits], _ , value = self.network.step(np.expand_dims(self.s, axis=0), 0.5)
                action = self.action_select(logits, exploration="boltzmann", temperature=keep_prob)
                [s_t], reward, d, _ = self.env.step(action)
                self._done = d

                batch.append((self.s, s_t , reward / 15.0, value, action, d, logits.tolist()))
                self.s = s_t

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
            

    # Boltzmann Softmax style action selection
    def action_select(self, dist, exploration="boltzmann", temperature=.8):
        

        if exploration == "boltzmann":        

            # dist = tf.nn.softmax(dist/temperature).numpy()
            # a = np.random.choice(dist,p=dist)
            # a = np.argmax(dist == a)

            exp_preds = np.exp((dist + 1e-16) / temperature ).astype("float64")
            preds = exp_preds / np.sum(exp_preds)
            
            [probas] = np.random.multinomial(1, preds, 1)
            a = np.argmax(probas)
            return a

        elif exploration == "bayesian":
            return np.argmax(dist)

