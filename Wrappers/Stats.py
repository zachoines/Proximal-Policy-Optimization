# Python core libs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from collections import OrderedDict
from multiprocessing import Process, Queue, Lock
import threading
import time

# OpenAI Gym classes
import gym
from gym.core import RewardWrapper

# An action Wrapper class for environments
class Stats(RewardWrapper):
    def __init__(self, env, collector):
        super(Stats, self).__init__(env)
        self.numSteps = 0
        self.collector = collector
        self.CMA = 0

    def reset(self, **kwargs):
        if self.numSteps > 0:
            self.collector.collect('CMA', self.CMA)
        self.numSteps = 0
        self.CMA = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.numSteps += 1
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        # Cumulative moving average
        self.CMA = (reward + (( self.numSteps - 1 ) * self.CMA )) / self.numSteps
        return reward

class AsynchronousPlot(threading.Thread):
    def __init__(self, collector):
        super(AsynchronousPlot, self).__init__()
        self.collector = collector
        self.stoprequest = threading.Event()
         
    def run(self):
        # style.use('fivethirtyeight')
        self._fig = plt.figure()
        self._ani = animation.FuncAnimation(self._fig, self._update_graph, interval = 100, save_count=200)
        plt.show()
    
    def continue_request(self):
        self.stoprequest.clear()
       
    def stop_request(self):
        self.stoprequest.set() 
        
    def join(self, timeout=None):
        self._ani.save('Results.mp4')
        self.stoprequest.set()
        super(AsynchronousPlot, self).join(timeout)
    
    def _update_graph(self, i):
        
        if not self.stoprequest.isSet():
            if (len(self.collector.dimensions) > 0):
                name = self.collector.dimensions[0]
            else:
                # Not ready to show anything
                return
            data = self.collector.get_dimension(name)
            axis = self._fig.add_subplot(1,1,1)
            xar = []
            yar = []

            for x, y in data.items():
                xar.append(int(x))
                yar.append(int(y))

            axis.clear()
            axis.plot(xar,yar)
            
            if (len(self.collector.get_dimensions()) > 1):
                for name in self.collector.dimensions():
                    data = self.collector.get_dimension(name)
                    axis = self._fig.add_subplot(1,1,1)
                    xar = []
                    yar = []

                    for x, y in data.items():
                        xar.append(int(x))
                        yar.append(int(y))

                    axis.plot(xar,yar)


# Data structor for collecting multiple 2d dimentions
class Collector:
    def __init__(self):
        # datastructure is this: [( Data_name : { x_0 : y_0, x_1 : y_1, ....}), ....]
        self.current_milli_time = lambda: int(round(time.time()))
        self.tstart = self.current_milli_time()
        self.dimensions = []
        self._data = []

    def collect(self, name, data):
        
        current_time = self._current_run_time()

        if (name not in self.dimensions):
            self.dimensions.append(name)
            entry = (name, OrderedDict([(current_time, data)]))
            self._data.append(entry)
        else:
            
            dim_values = None
            
            try:
                ( _ , dim_values) = next((value for key, value in enumerate(self._data) if value[0] == name), None)
                dim_values[current_time] = data 
            except GeneratorExit:
                pass
                
                
    def get_dimensions(self):
        return self.dimensions

    def get_dimension(self, name):
        dim_values = None
        try:
            ( _ , dim_values) = next((value for key, value in enumerate(self._data) if value[0] == name), None)
        except GeneratorExit:
            pass
            

        return dim_values
            
    def remove(self, name):
        if name in self.dimensions:
            try:
                index = next((key for key, value in enumerate(self._data) if value[0] == name), None)
                return self._data.pop(index)
            except GeneratorExit:
                pass
            
            return None

    def _current_run_time(self):
        return (self.current_milli_time() - self.tstart)
  
  

    
        


        