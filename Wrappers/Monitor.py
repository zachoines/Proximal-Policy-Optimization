# Python core libs
import numpy as np
from os.path import isfile, join
import cv2
import time
import datetime
import glob

# OpenAI Gym classes
import gym
from gym.core import ObservationWrapper
from gym.spaces.box import Box
 


class Monitor(ObservationWrapper):
    def __init__(self, env, env_shape, record = False, savePath = "\Videos", random_samples = False, save_images_to_disk = False):
        super(Monitor, self).__init__(env)
        (HEIGHT, WIDTH, CHANNELS) = env_shape
        self._height = HEIGHT
        self._width = WIDTH
        self._channels = CHANNELS
        self._record = record
        self._savePath = savePath
        self._is_running = False
        self._session_video = None
        self._timestamp = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if self._record:
            return self.observation(observation), reward, done, info
        else:
            return observation, reward, done, info

    def reset(self, **kwargs):
        if self._session_video:
            self._session_video.release()
        
        # self._timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
        current_milli_time = lambda: int(round(time.time() * 1000))

        self._session_video = cv2.VideoWriter(self._savePath + "\\" + str(current_milli_time()) + '.avi' , apiPreference = 0, fourcc = cv2.VideoWriter_fourcc(*'DIVX'), fps = 30, frameSize = (self._width, self._height))
    
        observation = self.env.reset(**kwargs)
        self._is_running = False

        return self.observation(observation)

    def observation(self, observation):
        if not self._is_running:
            self._is_running = True

        try:
            self._session_video.write(observation)
        except:
            print("There was an issue generating episode video.")



        return observation 

    def _displayImage(self, img):
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
