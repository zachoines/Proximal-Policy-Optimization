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
    def __init__(self, env, env_shape, record = False, savePath = "/Videos", random_samples = False, save_images_to_disk = False):
        super(Monitor, self).__init__(env)
        (HEIGHT, WIDTH, CHANNELS) = env_shape
        self._height = HEIGHT
        self._width = WIDTH
        self._channels = CHANNELS
        self._frame_array = []
        self._history = []
        self._fps = None
        self._record = record
        self._savePath = savePath
        self._start_time = 0.0
        self._is_running = False

    def step(self, action):
        if not self._is_running:
            self._is_running = True
        
        observation, reward, done, info = self.env.step(action)
        # if (not done):
        #     self._fps = 1.0 / (time.time() - self._start_time)
        if self._record:
            # for image in self._frame_array:
            #     self._displayImage(image)
            self._displayImage(observation)
            self._frame_array.append(observation)
            for image in self._frame_array:
                self._displayImage(image)

            return observation, reward, done, info
        else:
            return observation, reward, done, info

    def reset(self, **kwargs):
       
        # Save previous episode buffer
        if (len(self._frame_array) > 1):
            self._convert_frames_to_video()
        
        observation = self.env.reset(**kwargs)
        self._frame_array = []
        self._fps = None
        self.start_time = 0.0
        self._is_running = False

        return self.observation(observation)

    def observation(self, observation):
        return observation

    def _reset_monitor(self):
        self._frame_array = []
        self._fps = None
        self.start_time = 0.0

    def _episodeFPS(self):
        return self._fps
    
    def _displayImage(self, img):
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
    def _convert_frames_to_video(self):
        
        img_array = self._frame_array
        size = None

        # TODO:: Set option to read images from files at end of episode to
        # then create a video image
        if (False):
            for filename in glob.glob('./Images/*.jpg'):
                img = cv2.imread(filename)
                size = (self._width, self._height)
                img_array.append(img)
        else:   
            if (len(img_array)):   
                size = (self._width, self._height)
        
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
        out = cv2.VideoWriter(timestamp + '.avi' , cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
        try:
            for image in img_array:
                out.write(image)
        except:
            print("There was an issue generating episode video.")
        out.release()
