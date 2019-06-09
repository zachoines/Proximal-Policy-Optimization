import tensorflow as tf

# import local classes
from Model import Model
from AC_Network import AC_Network
from Worker import Worker

class Coordinator:
    def __init__(model, sess, environments, num_minibatches = 8, render = 4, threads = 4):

        self.model = model
        self.sess = sess
        self.envs = environments
        self.mb = num_minibatches

    def run():
        pass

    def collect_batches_and_run_updates():
        pass