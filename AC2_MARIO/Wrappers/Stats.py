# Python core libs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

# OpenAI Gym classes
import gym
from gym.core import RewardWrapper

# An action Wrapper class for environments
class Stats(RewardWrapper):
    def __init__(self, env, collector):
        super(Stats, self).__init__(env)
        self.numSteps = 0
        self.averageRewards = 0

    def reset(self, **kwargs):
        self.numSteps = 0
        self.averageRewards = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.numSteps += 1
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        self.averageRewards += reward / self.numSteps
        return reward

# Class designed to collect information ('stat name', timestamp, value) and graph live results
class Collector:
    def __init__(self):
        style.use('fivethirtyeight')
        fig = plt.figure()
        self.ax1 = fig.add_subplot(1,1,1)
        self.ani = animation.FuncAnimation(fig, animate, interval=1000)

    def animate(self, i):
        xar = []
        yar = []
        for eachLine in dataArray:
            if len(eachLine)>1:
                x,y = eachLine.split(',')
                xar.append(int(x))
                yar.append(int(y))
        ax1.clear()
        ax1.plot(xar,yar)

    def run():
        plt.show()