import sys
import tensorflow as tf
import numpy as np

from Train import Train
from Test import Test

# install requirements with:
# pip install -r requirements.txt

config = {

    # Environmental variables
    'Environment Name' : 'Breakout-v0',                                 # Env given to each worker. Play with 'MsPacman-v0' as well.
    'Number of worker threads' : 8,                                     # NUmber of parallel envs on their own thread.

    # Sample loop variables
    'Number of environment episodes' : 512,                             # How many times we reboot test envs.
    'Max Number of sample batches per environment episode' : 512,       # May end earlier.
    'Max steps taken per batch' : 128,                                  # Number steps agent takes in env.
    'Max timsteps' : 512 * 512,                                         # Episodes * batches. likely will end far before as envs may terminate early.
    
    # Training loop variables
    'Training epochs' : 8,                                              # Number of times we train on a sample gathered from env.
    'Mini batches per training epoch' : 8,                              # How many updates per epoch per batch.
    
    # Learning variables
    'Epsilon' : 1e-5,                                                   # Noise factor for adam optimizer.
    'Gamma' : 0.99,                                                     # discount factor for rewards.
    'Learning rate' : 0.0007,                                           # Learning rate for adam optimizer.
    'PPO clip range' : 0.2,                                             # Max ratio for PPO loss function .10 ~ .20.
    'Max grad norm' : 0.5,                                              # Clip norm feed to adam optimizer.
    'Normalize advantages' : False,                                     # Normalize advantages in mini-batch sent to loss function.

    # Loss function coefficient     
    'Value loss coeff' : 0.5,                                           # Discount factor dor value loss in PPO loss function.
    'Entropy coeff' : 0.01,                                             # Discount factor applied to entropy bonus in PPO loss function. HIgher means more agent exploration.

    # CNN options
    'CNN type' : 'large',   # or 'small'                                # 'Large' vs. 'Small.' Means one has more convolution layers.

    # Decay options
    'Pre training steps' : 0,                                           # Steps taken before annealing starts.
    'Anneling_steps' : 128 * 256 * 8 ,                                  # Env restarts * batches * training epochs.
    'Decay clip and learning rate' : True                               # Decay the PPO clip rate.

}

if __name__ == "__main__":
    
    # 42 is the answer to everything
    np.random.seed(42)
    tf.random.set_seed(42)

    train_session = Train(config)

    if (train_session.start()):
        test_session = Test(config)
        test_session.start()
    