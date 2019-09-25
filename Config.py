# Trained on eight threads and GPU
config_pacman = {

    # Environmental variables
    'Environment Name' : 'MsPacman-v0',                                 # Env given to each worker. Play with 'MsPacman-v0' or 'SuperMarioBros-v0' as well.
    'Number of worker threads' : 8,                                     # Number of parallel envs on their own thread.
    'Wrapper class' : 'MsPacmanWrapper',                                # Optionally defined OpenAI env wrapper sub-class to modify anything relating to this env.

    # Sample loop variables
    'Number of environment episodes' : 8192*2,                          # How many times we reboot test envs.
    'Max Number of sample batches per environment episode' : 1,         # 
    'Max steps taken per batch' : 128,                                   # Number steps agent takes in env.
    'Max timsteps' : 8192*2,                                            # Episodes * batches. 
    
    # Training loop variables
    'Training epochs' : 8,                                              # Number of times we train on a sample gathered from env.
    'Mini batches per training epoch' : 8,                              # How many updates per epoch per batch.
    
    # Learning variables
    'Epsilon' : 1e-5,                                                   # Noise factor for adam optimizer.
    'Gamma' : 0.99,                                                     # discount factor for rewards.
    'Learning rate' : 1e-3,                                             # Learning rate for adam optimizer.
    'PPO clip range' : .20,                                              # Max ratio for PPO loss function .10 ~ .20.
    'Max grad norm' : 0.5,                                              # Clip norm feed to adam optimizer.
    'Normalize advantages' : False,                                     # Normalize advantages in mini-batch sent to loss function.

    # Loss function coefficient     
    'Value loss coeff' : 0.5,                                           # Discount factor for value loss in PPO loss function.
    'Entropy coeff' : 0.01,                                             # Discount factor applied to entropy bonus in PPO loss function. HIgher means more agent exploration.

    # CNN options
    'CNN type' : '',                                                    # " '' " means 'CNN_LARGE' class will be used. 'Large' vs 'small' means more/less convolutional layers
    'Grayscale' : True,

    # Decay options
    'Pre training steps' : 0,                                           # Steps taken before annealing starts.
    'Anneling_steps' : 128 * 256 * 8 ,                                  # Env restarts * batches * training epochs.
    'Decay clip and learning rate' : True,                              # Decay the PPO clip rate.
    'Learning rate decay' : 3e-6,                                       # LR * (1 / (1 + decay * (iterations)))
    'Min clip' : 0.0001

}

# Trained on eight threads and GPU
config_breakout = {

    # Environmental variables
    'Environment Name' : 'Breakout-v0',                                 # Env given to each worker. Play with 'MsPacman-v0' or 'SuperMarioBros-v0' as well.
    'Number of worker threads' : 8,                                     # Number of parallel envs on their own thread.
    'Wrapper class' : '',                                               # Optionally defined OpenAI env wrapper sub-class to modify anything relating to this env.            

    # Sample loop variables
    'Number of environment episodes' : 8192,                            # How many times we reboot test envs.
    'Max Number of sample batches per environment episode' : 1,         # 
    'Max steps taken per batch' : 128,                                  # Number steps agent takes in env.
    'Max timsteps' : 8192,                                              # Episodes * batches. 
    
    # Training loop variables
    'Training epochs' : 4,                                              # Number of times we train on a sample gathered from env.
    'Mini batches per training epoch' : 8,                              # How many updates per epoch per batch.

    # Learning variables
    'Epsilon' : 1e-5,                                                   # Noise factor for adam optimizer.
    'Gamma' : 0.99,                                                     # discount factor for rewards.
    'Learning rate' : 7e-4,                                             # Learning rate for adam optimizer.
    'PPO clip range' : 0.2,                                             # Max ratio for PPO loss function .10 ~ .20.
    'Max grad norm' : 0.5,                                              # Clip norm feed to adam optimizer.
    'Normalize advantages' : False,                                     # Normalize advantages in mini-batch sent to loss function.

    # Loss function coefficient     
    'Value loss coeff' : 0.5,                                           # Discount factor for value loss in PPO loss function.
    'Entropy coeff' : 0.01,                                             # Discount factor applied to entropy bonus in PPO loss function. HIgher means more agent exploration.

    # CNN options
    'CNN type' : 'CNN_SMALL',                                           # Enter CNN Class names here.
    'Grayscale' : True,

    # Decay options
    'Pre training steps' : 0,                                           # Steps taken before annealing starts.
    'Anneling_steps' : 128 * 256 * 8 ,                                  # Env restarts * batches * training epochs.
    'Decay clip and learning rate' : True,                              # Decay the PPO clip rate.
    'Learning rate decay' : 8e-4,                                       # LR * (1 / (1 + decay * (iterations)))
    'Min clip' : 0.001

}

# Trained on two threads (laptop resources)
config_pong = {

    # Environmental variables
    'Environment Name' : 'Pong-v0',                                     # Env given to each worker. Play with 'MsPacman-v0' or 'SuperMarioBros-v0' as well.
    'Number of worker threads' : 2,                                     # Number of parallel envs on their own thread.
    'Wrapper class' : '',                                               # Optionally defined OpenAI env wrapper sub-class to modify anything relating to this env.

    # Sample loop variables
    'Number of environment episodes' : 8192,                            # How many times we reboot test envs.
    'Max Number of sample batches per environment episode' : 1,         # 
    'Max steps taken per batch' : 512,                                  # Number steps agent takes in env.
    'Max timsteps' : 8192,                                              # Episodes * batches. 
    
    # Training loop variables
    'Training epochs' : 4,                                              # Number of times we train on a sample gathered from env.
    'Mini batches per training epoch' : 4,                              # How many updates per epoch per batch.

    # Learning variables
    'Epsilon' : 1e-5,                                                   # Noise factor for adam optimizer.
    'Gamma' : 0.99,                                                     # discount factor for rewards.
    'Learning rate' : 7e-4,                                             # Learning rate for adam optimizer.
    'PPO clip range' : 0.2,                                             # Max ratio for PPO loss function .10 ~ .20.
    'Max grad norm' : 0.5,                                              # Clip norm feed to adam optimizer.
    'Normalize advantages' : False,                                     # Normalize advantages in mini-batch sent to loss function.

    # Loss function coefficient     
    'Value loss coeff' : 0.5,                                           # Discount factor for value loss in PPO loss function.
    'Entropy coeff' : 0.01,                                             # Discount factor applied to entropy bonus in PPO loss function. HIgher means more agent exploration.

    # CNN options
    'CNN type' : 'CNN_SMALL',                                           # Enter CNN class name here. 
    'Grayscale' : False,

    # Decay options
    'Pre training steps' : 0,                                           # Steps taken before annealing starts.
    'Anneling_steps' : 128 * 256 * 8 ,                                  # Env restarts * batches * training epochs.
    'Decay clip and learning rate' : True,                              # Decay the PPO clip rate.
    'Learning rate decay' : 3e-6,                                       # LR * (1 / (1 + decay * (iterations)))
    'Min clip' : 0.0001

}