# Utility python classes
import os
import numpy as np
import tensorflow as tf
from Layers import openai_entropy


# Importing the packages for OpenAI and MARIO
import gym
from gym import wrappers
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from tensorflow.keras import backend as K
import scipy
import scipy.signal

# Locally defined classes
from Wrappers import preprocess



# Locally defined classes
from Worker import Worker, WorkerThread
from AC_Network import AC_Network


class Coordinator:
    def __init__(self, model, workers,  num_envs, num_epocs, num_minibatches, batch_size, gamma):
        self.model = model
        self.workers = workers
        self.num_envs = num_envs
        self.num_epocs = num_epocs
        self.num_minibatches = num_minibatches
        self.batch_size = batch_size
        self.gamma = gamma
        self.total_loss = 0   
    
    def dense_to_one_hot(self, labels_dense, num_classes = 7):
        labels_dense = np.array(labels_dense)
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
 

    def train(self, train_data):

        (batch_states,
        batch_actions,
        batch_advantages, batch_rewards) = train_data
        
        # Now perform tensorflow session to determine policy and value loss for this batch
        # np.ndarray.tolist(self.dense_to_one_hot(batch_actions))
        feed_dict = { 
            self.model.step_policy.keep_per: 1.0,
            self.model.step_policy.X_input: batch_states.tolist(),
            self.model.step_policy.actions: batch_actions.tolist(),
            self.model.step_policy.rewards: batch_rewards.tolist(),
            self.model.step_policy.advantages: batch_advantages.tolist(),
        }
        
        # Run tensorflow graph, return loss without updateing gradients 
        optimizer, loss, entropy, policy_loss, value_loss, neg_log_prob, global_norm = self.sess.run(
            [self.model.step_policy.optimize,
             self.model.step_policy.loss, 
             self.model.step_policy.entropy, 
             self.model.step_policy.policy_loss, 
             self.model.step_policy.value_loss, 
             self.model.step_policy.neg_log_prob, 
             self.model.step_policy.global_norm], feed_dict)
        

    # Produces reversed list of discounted rewards
    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    # for debugging processed images
    def displayImage(self, img):
        cv2.imshow('image', np.squeeze(img, axis=0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):

        try:
            # Main training loop
            for _ in range(self.num_epocs):

                for worker in self.workers:
                    worker.reset()

                # loop for generating a training session a batch at a time
                for mb in range(self.num_minibatches):

                    # Copy global network over to local network
                    # self.model.refresh_local_network_params()
                    #for env in envs:
                    #   model_copy = keras.models.clone_model(model)
                    #   model_copy.set_weights(model.get_weights())

                    # Send workers out to threads
                    threads = []
                    for worker in self.workers:
                        threads.append(WorkerThread(target=worker.run, args=()))

                    # Start the workers on their tasks
                    for thread in threads:
                        thread.start()
                        
                    batches = []
                    
                    # Wait foreach worker to finish and return their batch
                    for thread in threads:
                        batch = thread.join()
                        batches.append(batch)

                    all_advantages = np.array([])
                    all_rewards = np.array([])
                    all_states = np.array([])
                    all_actions = np.array([])

                    # Calculate discounted rewards for each environment
                    for env in range(self.num_envs):
                        done = False
                        batch_advantages = []
                        batch_rewards = []
                        batch_values = []
                        batch_observations = []
                        batch_states = []
                        batch_actions = []

                        mb = batches[env]
                        
                        # Empty batch
                        if mb == []:
                            continue

                        # For every step made in this env for this particular batch
                        done = False
                        state = None
                        value = None
                        observation = None

                        for step in mb:
                            (state, observation, reward, value, action, done) = step
                            batch_rewards.append(reward)
                            batch_values.append(value)
                            batch_observations.append(observation)
                            batch_states.append(state)
                            batch_actions.append(action)

                        
                        # If we reached the end of an episode or if we filled a batch without reaching termination of episode
                        # we boot-strap the final rewards with the V_s(last_observation)

                        # Discounted bootstraped rewards formula:
                        # G_t == R + γ * V(S_t'):

                        # Advantage
                        # δ_t == R + γ * V(S_t') - V(S_t): 
                        
                        # Or simply... 
                        # δ_t == G_t - V:      

                        if (not done):
                            [boot_strap] = sess.run(self.model.value, {self.model.input_def: [observation], self.model.keep_prob: 1.0})
                            # [boot_strap] = self.model.value_s([observation], 1.0)
                            print(boot_strap)
                            bootstrapped_rewards = np.asarray(batch_rewards + boot_strap)
                            discounted_rewards = self.discount(bootstrapped_rewards, self.gamma)[:-1]
                            batch_rewards = discounted_rewards
                            batch_advantages = batch_rewards - batch_values

                            # Same as above...
                            # bootstrapped_values = np.asarray(batch_values + [boot_strap])
                            # advantages = batch_rewards + self.gamma * bootstrapped_values[1:] - bootstrapped_values[:-1]
                            # advantages = self.discount(advantages, self.gamma)

                            all_advantages = np.concatenate((all_advantages, batch_advantages), 0) if all_advantages.size else np.array(batch_advantages)
                            all_rewards = np.concatenate((all_rewards, batch_rewards), 0) if all_rewards.size else np.array(batch_rewards)
                            all_states = np.concatenate((all_states, batch_states), 0) if all_states.size else np.array(batch_states)
                            all_actions = np.concatenate((all_actions, batch_actions), 0) if all_actions.size else np.array(batch_actions)
                        
                        else:
                            
                            boot_strap = 0

                            bootstrapped_rewards = np.asarray(batch_rewards + [boot_strap])
                            discounted_rewards = self.discount(bootstrapped_rewards, self.gamma)[:-1]
                            batch_rewards = discounted_rewards
                            batch_advantages = batch_rewards - batch_values

                            all_advantages= np.concatenate((all_advantages, batch_advantages), 0) if all_advantages.size else np.array(batch_advantages)
                            all_rewards = np.concatenate((all_rewards, batch_rewards), 0) if all_rewards.size else np.array(batch_rewards)
                            all_states = np.concatenate((all_states, batch_states), 0) if all_states.size else np.array(batch_states)
                            all_actions = np.concatenate((all_actions, batch_actions), 0) if all_actions.size else np.array(batch_actions)

                    # We can do this because: d/dx ∑ loss  == ∑ d/dx loss
                    batch_states = np.array(batch_states)
                    data = (all_states, all_actions, all_advantages, all_rewards)
                    
                    if data[0].size != 0:
                        self.train(data) 
                    else:
                        break

                try:
                    # saver = tf.train.Saver()
                    # save_path = saver.save(self.sess, self.model_save_path + "\model.ckpt")
                    print("Model saved at " + save_path + ".")
                
                except:
                    print("ERROR: There was an issue saving the model!")
                    raise

            print("Training session was sucessfull.")
            return True 

        except:
            print("ERROR: The coordinator ran into an issue during training!")
            raise
      




# defines the foward step.
class AC_Network:
    def __init__(self, input_shape, num_actions, is_training=True, name='train'):
        self.input_shape = input_shape
        self.value_s = None
        self.action_s = None
        self.num_actions = num_actions

        # Dicounting hyperparams for loss functions
        self.entropy_coef = 0.01
        self.value_function_coeff = 0.5
        self.max_grad_norm = 40.0
        self.learning_rate = 7e-4
        self.alpha = 0.99
        self.epsilon = 1e-5

        # Model variables
        (_, hight, width, stack) = self.input_shape
        self.input_def = tf.keras.layers.Input(shape=(hight, width*stack, 1), name="input_layer", dtype=tf.float32)
        # self.keep_prob = tf.keras.layers.Input(dtype=tf.float32, shape = (None), name="keep_prob", tensor=tf.placeholder_with_default(1.0, shape=(None)))
        self.keep_prob = tf.placeholder(tf.float32, (None), "keep_prob")
        # self.keep_prob = K.placeholder(dtype=tf.float32, name="keep_prob")


        # Define Convolution 1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            kernel_initializer='orthogonal',
            padding="valid",
            activation="relu", 
            name="conv1")
        
        self.maxPool1 = tf.keras.layers.MaxPooling2D(pool_size = (3, 3), name = "maxPool1")
        self.dropout1 = tf.keras.layers.Dropout(self.keep_prob)

        # define Convolution 2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            kernel_initializer='orthogonal',
            padding="valid",
            activation="relu",
            name="conv2")

        self.maxPool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name="maxPool2")
        self.dropout2 = tf.keras.layers.Dropout(self.keep_prob)
        
        # define Convolution 3
        self.conv3 = tf.keras.layers.Conv2D(
            filters=96,
            kernel_size=[2, 2],
            kernel_initializer='orthogonal',
            padding="valid",
            activation="relu",
            name="conv3")

        self.maxPool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name ="maxPool3")
        self.dropout3 = tf.keras.layers.Dropout(self.keep_prob)
        self.flattened = tf.keras.layers.Flatten(name="flattening_layer")
        self.hiddenLayer = tf.keras.layers.Dense(
            512,
            activation="relu",
            # bias_regularizer=tf.keras.regularizers.l2(0.01),
            kernel_initializer='orthogonal',
            name="hidden_layer")

        self.dropout4 = tf.keras.layers.Dropout(self.keep_prob)

        # Output Layer consisting of an Actor and a Critic
        self._value = tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer='orthogonal',
            name="value_layer")
   
        self._policy = tf.keras.layers.Dense(
            7,
            kernel_initializer='orthogonal',
            name="policy_layer")

        # Define the forward pass for the convolutional and hidden layers
        conv1_out = self.conv1(self.input_def)
        maxPool1_out = self.maxPool1(conv1_out)
        dropout1 = self.dropout1(maxPool1_out)

        conv2_out = self.conv2(dropout1)
        maxPool2_out = self.maxPool2(conv2_out)
        dropout2 = self.dropout2(maxPool2_out)

        conv3_out = self.conv3(dropout2)
        maxPool3_out = self.maxPool3(conv3_out)
        dropout3 = self.dropout3(maxPool3_out)
        
        flattened_out = self.flattened(dropout3)
        hidden_out = self.dropout4(flattened_out)

        # Actor and the Critic outputs
        self.value = self._value(hidden_out)
        self.logits = self._policy(hidden_out)
        self.action_dist = tf.nn.softmax(self.logits)

        # Final model
        self.model = tf.keras.Model(inputs=[self.input_def, ], outputs=[self.logits, self.value])

        # have to initialize before threading
        self.model._make_predict_function()
    
        # Batch data that will be sent to Model by the coordinator
        self.actions = tf.placeholder(tf.int32, [None])
        self.actions_hot = tf.one_hot(self.actions, 7, dtype=tf.float32)
        self.advantages = tf.placeholder(tf.float32, [None])
        self.rewards = tf.placeholder(tf.float32, [None]) 
        self.values = tf.placeholder(tf.float32, [None])
        
        # Responsible Outputs -log π(a_i|s_i)i
        # self.neg_log_prob = tf.reduce_sum(self.policy * self.actions_hot + 1e-10, [1])
        # self.policy_loss = tf.reduce_mean(-1.0 * tf.log(self.log_prob) * self.advantages)

        # Policy Loss:  (1 / n) * ∑ * -log π(a_i|s_i) * A(s_i, a_i) 
        self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.actions_hot)
        self.policy_loss = tf.reduce_mean(self.neg_log_prob * self.advantages)
        
        # Value loss "MSE": (1 / n) * ∑[V(i) - R_i]^2
        # self.value_loss = tf.reduce_mean(tf.square(tf.squeeze(self.value_function) - self.rewards)) 
        self.value_loss = tf.reduce_mean(tf.square(self.advantages)) 

        # Entropy: - (1 / n) * ∑ P_i * Log (P_i)
        # self.entropy = - tf.reduce_mean(self.policy * tf.log(self.policy + 1e-10))
        self.entropy = tf.reduce_mean(openai_entropy(self.logits))

        # Total loss: Policy loss - entropy * entropy coefficient + value coefficient * value loss
        self.loss = self.policy_loss + (self.value_loss * self.value_function_coeff) - (self.entropy * self.entropy_coef)

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads = tf.gradients(self.loss, params)            
        grads, self.global_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

        # Apply Gradients
        grads = list(zip(grads, params))

        # optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay = self.alpha, epsilon = self.epsilon)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, epsilon = self.epsilon)
    
        # Update network weights
        self.optimize = optimizer.apply_gradients(grads)
        
        


    def step(self, observation, keep_prob):
        # Take a step using the model and return the predicted policy and value function
              
        softmax, value = sess.run([self.action_dist, self.value], {self.input_def: observation, self.keep_prob: keep_prob})
        return softmax, value[-1]

    def value_s(self, observation, keep_prob):
        softmax, value = sess.run([self.action_dist, self.value], {self.input_def: observation, self.keep_prob: keep_prob})
        return value[-1]
        # self.model.predict([observation,  keep_prob])
        # Return the predicted value function for a given observation
        #with tf.keras.backend.get_session() as sess:
        # value = sess.run([self.value], {self.input_def: observation, self.keep_prob: keep_prob})
        # return value[-1]



def action_select(softmax):
        
    temperature = 1.0
    exp_preds = np.exp(softmax / temperature)
    preds = exp_preds / np.sum(exp_preds)
    
    [probas] = np.random.multinomial(1, preds, 1)
    action = np.argmax(probas)
    
    return action


# Environments to run
env_1 = 'SuperMarioBros-v0'
env_2 = 'SuperMarioBros2-v0'
env_names = [env_1]



# Tf session
tf.reset_default_graph()
config = tf.ConfigProto()
sess = tf.Session(config=config)


envs = []
# Apply wrappers
for env in env_names:
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = preprocess.FrameSkip(env, 4)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = preprocess.GrayScaleImage(env, height = 96, width = 96, grayscale = True)
    env = preprocess.FrameStack(env, 4)
    envs.append(env)

# Env variables
(HEIGHT, WIDTH, CHANNELS) = env.observation_space.shape
NUM_ACTIONS = env.env.action_space.n
ACTION_SPACE = env.env.action_space
NUM_STATE = (1, HEIGHT, WIDTH, CHANNELS)
num_epocs = 5
num_minibatches = 2
batch_size = 16
anneling_steps = num_epocs * num_minibatches * batch_size


model = AC_Network((1, HEIGHT, WIDTH, 4), ACTION_SPACE)



# Init coordinator and send out the workers


# 

workers = [Worker(AC_Network((1, HEIGHT, WIDTH, 4), ACTION_SPACE), env, anneling_steps, batch_size=batch_size, render=False) for env in envs]
# model, workers,  num_envs, num_epocs, num_minibatches, batch_size, gamma):
coordinator = Coordinator(model, workers, len(env_names), num_epocs, num_minibatches, batch_size, .99)
sess.run(tf.global_variables_initializer())
# Train and save
if coordinator.run():
    try:
        # save_path = saver.save(sess, model_save_path + "\model.ckpt")
        print("Model saved.")
        print("Now testing results....")
    except:
        print("ERROR: There was an issue saving the model!")


