# Python core libs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from collections import OrderedDict
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
        self.numSteps = 0
        self.collector.collect('CMA', self.CMA)
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

# Class designed to collect along multiple 2d dimentions and graph live results
class Collector:
    def __init__(self):
        # datastructure is this: [( Data_name : { x_0 : y_0, x_1 : y_1, ....}), ....]
        self.current_milli_time = lambda: int(round(time.time() * 1000))
        self.dimensions = []
        self._data = []
        
        # Setup matplotlib stuff here
        style.use('fivethirtyeight')
        self._fig = plt.figure()
        self._ani = animation.FuncAnimation(self._fig, self._update_graph, interval = 1000)
        plt.show()

    def collect(self, name, data):
        current_time = self.current_milli_time()

        if (name not in self.dimensions):
            self.dimensions.append(name)
            entry = (name, OrderedDict([(current_time, data)]))
            self._data.append(entry)
        else:
            index = next((key for key, value in enumerate(self._data) if value[0] == name), None)
            (dim_name, dim_values) = self._data[index]
            dim_values.update( ( current_time , data ) )
            self._data[index] = (dim_name, dim_values)

    def get_dimension(self, name):
        index = next((key for key, value in enumerate(self._data) if value[0] == name), None)
        (dim_name, dim_values) = self._data[index]
        return dim_values
            
    def remove(self, name):
        if name in self.dimensions:
            index = next((key for key, value in enumerate(self._data) if value[0] == name), None)
            return self._data.pop(index)

    def _update_graph(self, i):

        for name in self.dimensions:
            data = self.get_dimension(name)
            axis = self._fig.add_subplot(1,1,1)
            xar = []
            yar = []

            for x, y in data:
                xar.append(int(x))
                yar.append(int(y))

            axis.clear()
            axis.plot(xar,yar)
        

        