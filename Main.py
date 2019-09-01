import tensorflow as tf
import numpy as np

from Train import Train
from Test import Test
from Config import config_pacman as config

# install requirements with:
# pip install -r requirements.txt

if __name__ == "__main__":
    
    # Basic numberic global config settings
    # 42 is the answer to everything
    tf.keras.backend.set_floatx('float64')
    np.random.seed(42)
    tf.random.set_seed(42)

    train_session = Train(config)

    if (train_session.start()):
        Test(config)
    

