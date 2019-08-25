import sys
import tensorflow as tf
import numpy as np

from Train import Train
from Test import Test

# install requirements with:
# pip install -r requirements.txt

config = {

    # Environmental variables
    'Environment Name' : 'Breakout-v0',    # 'MsPacman-v0'
    'Number of worker threads' : 8,

    # Sample loop variables
    'Number of global sessions' : 256,
    'Max Number of sample batches per environment episode' : 512,
    'Max steps taken per batch' : 128, 
    
    # Training loop variables
    'Training epochs' : 4,
    'Mini batches per training epoch' : 4,
    
    # Learning variables
    'Epsilon' : 1e-5,
    'Gamma' : 0.99,
    'Learning rate' : 0.0007,
    'PPO clip range' : 0.2,
    'Max grad norm' : 0.5,

    # Loss function coefficient 
    'Value loss coeff' : 0.5,
    'Entropy coeff' : 0.01,

    # CNN options
    'CNN type' : 'large',   # or 'small'

    # Decay options
    'Pre training steps' : 0,
    'Anneling_steps' : 512 * 256,
    'Decay clip and learning rate' : True

}

if __name__ == "__main__":
    # Because 42 is the answer to everything
    np.random.seed(42)
    tf.random.set_seed(42)

    train_session = Train(config)

    if (train_session.start()):
        test_session = Test(config)
        test_session.start()
    