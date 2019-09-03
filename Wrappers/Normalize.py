# OpenAI Gym classes
import gym
from gym.core import Wrapper

# A Class to normalize state and rewards at will
class Normalize(Wrapper):
    def __init__(self, env):
        super(Normalize, self).__init__(env)
        self.num_lives = 3

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation) 

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        # Punish lost lives
        if info['ale.lives'] < self.num_lives:
            self.num_lives = info['ale.lives'] 
            reward -= 10

        # Punish terminal states
        if done: 
            reward -= 20

        # Add a living cost
        return self.observation(observation), self.reward(reward - .1), done, info

    def reward(self, reward):
        return (reward / 50) 

    def observation(self, observation):
        return observation

 