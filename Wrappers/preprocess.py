# Image Preprocessing

# Importing the libraries
import numpy as np
import os
import cv2
from collections import deque

import gym
from gym.core import ObservationWrapper
from gym.spaces.box import Box
import tensorflow as tf

# Preprocessing the Images

class GrayScaleImage(ObservationWrapper):
    
    def __init__(self, env, sess, height = 96, width = 96, grayscale = True):
        super(GrayScaleImage, self).__init__(env)

        with tf.name_scope("cnn_input"):
            self.cnn_input = tf.placeholder(tf.uint8, (1, height, width, 1))
        self.img_size = (height, width)
        self.grayscale = grayscale
        n_channels = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [height, width, n_channels])
        self.sess = sess
    
    # for debugging processed images
    def _displayImage(self, img):
        # img = self.sess.run(img)
        cv2.imshow('image',np.squeeze(img, 0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def observation(self, img):
        # Convert to grayscale if enabled using mean method
        # if self.grayscale:
            # img = np.mean(img, axis=-1,keepdims=1)
            # img = np.max(img, axis = -1, keepdims = 1) / 2 +  np.min(img, axis = -1, keepdims = 1) / 2
        
        # if self.grayscale:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        
        img = tf.image.resize_images(img, self.img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if self.grayscale:
            img = tf.image.rgb_to_grayscale(img)
      
        img = tf.image.per_image_standardization(img)
        self.cnn_input = tf.expand_dims(img, 0)
        
        with tf.Session() as sess:
            img = sess.run(self.cnn_input)

        return img

class FrameSkip(gym.Wrapper):
    
    def __init__(self, env, skipped_frames):
        super(FrameSkip, self).__init__(env)
        self.n = skipped_frames

    def step(self, action):
        done = False
        total_reward = 0
        for _ in range(self.n):
            o, r, done, info = self.env.step(action)
            total_reward += r
            if done: break
        return o, total_reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        # self.observation_space = Box(low=0, high=255, shape=(shp[0], shp[1], k))
        self.observation_space = Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k): self.frames.append(ob)
        return self.observation()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self.observation(), reward, done, info

    def observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)