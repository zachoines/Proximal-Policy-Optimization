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
        # Init the Network and Workers, starting then onto their own thread
        sess.run(tf.global_variables_initializer())
        workers = [Worker(self.model, env, render = False) for env in self.envs]

        threads = []

        for _ in range(num_minibatches):
            for worker in workers:
                t = threading.Thread(target = worker.run(), args=())
                threads.append(t)
                t.start()

            for thread in threads:
                thread.join()

        total_batches = []
        for worker in workers:
            total_batches.append(worker.get_batches())


    def collect_batches_and_run_updates():
        pass