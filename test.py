from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY, TEST
from Wrappers.preprocess import FrameStack, GrayScaleImage
import tensorflow as tf

sess = tf.Session()
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = GrayScaleImage(env, sess)
env = FrameStack(env, 4)
env = JoypadSpace(env, TEST)


done = True
for step in range(5000):
    if done:
        state = env.reset()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    env.render()

env.close()