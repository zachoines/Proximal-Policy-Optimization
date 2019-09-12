# OpenAI Gym classes
import gym
from gym.core import Wrapper
import numpy as np

# A Class to normalize state, actions, and rewards at will
class Normalize(Wrapper):
    def __init__(self, env):
        super(Normalize, self).__init__(env)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation) 

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        return self.observation(observation), self.reward(reward), done, info
    
    # Bring rewards to reasonable scale. Rewards can get very large, leading to destructive loss.
    def reward(self, reward):
        return reward

    def observation(self, observation):
        return observation
        
# Reduces original discrete action and normalizes rewards.
class MsPacmanWrapper(Wrapper):

    def __init__(self, env):
        super(MsPacmanWrapper, self).__init__(env)
        self.num_lives = 3
        self._reversed = False
        self._action_desc = {0:'NOOP', 1:'UP', 2:'RIGHT', 3:'LEFT', 4:'DOWN' }
        self._actions = [0, 1, 2, 3, 4]
        self._last_action = None
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def reversed(self):
        return self._reversed
       
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation) 

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        #### Original Rewards detail ####
        # Total Rewards: 220(dots) × 10 + 4(pills) × 40 + 4(ghosts) × (200 + 400 + 800 + 1600) = 14360 points
        # Ghost: 200 * N^2 (Where 'N' represents number of ghosts eaten within this episode)
        # Fruit: 100
        # Pill: 10
        # Power Pill: 40

        ### New cost/rewards function ###
        step = -0.01
        reverse = 0.0
        life_loss = -2
        death = -2

        # Check for going backwards
        if self._reversed != None:
            a = action
            if self._last_action == 1:
                if a == 4:
                    self._reversed == True
                else:
                    self._reversed = False
            elif self._last_action == 2:
                if a == 3:
                    self._reversed == True
                else:
                    self._reversed = False
            elif self._last_action == 3:
                if a == 2:
                    self._reversed == True
                else:
                     self._reversed = False
            elif self._last_action == 4:
                if a == 1:
                    self._reversed == True
                else:
                    self._reversed = False
        else:
            self._reversed = False
            self._last_action = a

        # Add a living cost to prevent running into walls endlessly
        rewards = 0
        rewards += step

        # +1 for ghost was caught or high level fruit
        rewards += (reward // 200) * 3
        reward = (reward % 200)

        # +1 for other fruit with odd multiples of 100
        rewards += (reward // 100) * 2
        reward = (reward % 100) 

        # +1 for power pill
        rewards += (reward // 40) * 2
        reward = (reward % 40)

        # +1 for dot
        rewards += (reward // 10)
        reward = reward % 10

        # Punish going backwards
        if self._reversed:
             rewards += reverse

        # Punish lost lives
        if info['ale.lives'] < self.num_lives:
            self.num_lives = info['ale.lives'] 
            rewards += life_loss
        
        # Punish death
        if (done):
            rewards += death
            
        # clip everything 
        rewards = np.clip(rewards, -4, 4)
        
        return self.observation(observation), self.reward(rewards), done, info
    
    # Bring rewards to reasonable scale. Rewards can get very large, leading to destructive loss.
    def reward(self, reward):

        return reward

    def observation(self, observation):
        return observation