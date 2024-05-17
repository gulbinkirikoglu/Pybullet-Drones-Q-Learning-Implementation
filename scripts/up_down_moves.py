################################################# LIBRARIES #############################################################

from collections import deque, namedtuple
import time
import  gymnasium as gym
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense

# from gym_pybullet_drones.examples import utils
from scripts import utils

######################################## PARAMETERS AND ENVIROMENT ######################################################

MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.99              # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
MINIBATCH_SIZE = 32       # Mini-batch size.
TAU = 1e-3                # Soft update parameter.
E_DECAY = 0.995           # ε-decay rate for the ε-greedy policy.
E_MIN = 0.01              # Minimum ε value for the ε-greedy policy.


# Load the enviroment
env = gym.make('hover-aviary-v0')
env.reset()

# Define observation space size as "state size"
# state_size = env.observation_space.shape[1]
# print("aaaa:", state_size) # 72

state_size = 2
state_size = tuple([state_size])

# Define number of actions
action_size = 2

################################################## NN MODELS ############################################################


# Create the Q-Network
q_network = Sequential([
    Input(shape=state_size), 
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(action_size, activation='linear'),  # (None, 1, 3)
])
# q_network.summary()  

# Create the target Q^-Network
target_q_network = Sequential([
    Input(shape=state_size), 
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(action_size, activation='linear'),  # (None, 1, 3)
])
# q_network.summary() 

############################################### FUNCTIONS ###########################################################

optimizer  = Adam(learning_rate=ALPHA)
# print(optimizer)

# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


# Calculate_loss
def compute_loss(experiences, gamma, q_network, target_q_network):

    states, actions, rewards, next_states, done_vals = experiences
    # print("exp-act:", actions.shape)
    # actions = tf.reshape(actions,(-1,4))
    # print("resize-act:", actions.shape)

    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))

    # Get the q_values and reshape to match y_targets
    q_values = q_network(states)
    # print("shapeeeeee:", q_values.shape)

    # print("shape2:", (tf.range(q_values.shape[0])).shape, "shape3:", (tf.cast(actions, tf.int32)).shape)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),tf.cast(actions, tf.int32)], axis=1))
    loss = MSE(y_targets, q_values)
    
    return loss


# eğitim
@tf.function
def agent_learn(experiences, gamma):
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)
    
    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    
    # update the weights of target q_network
    utils.update_target_network(q_network, target_q_network, TAU)
    

################################################ TRAINING ########################################################

start = time.time()
num_episodes = 1000
max_num_timesteps = 500                        
total_point_history = []
height_list = []
height_list_all = []

num_p_av = 100   
epsilon = 1   
memory_buffer = deque(maxlen=MEMORY_SIZE)  
target_q_network.set_weights(q_network.get_weights())

for i in range(num_episodes):
    state = env.reset()
    state = state[0][0]
    indices = [2, 9]
    state = state[indices] 
    total_points = 0
    height = [] 

    for t in range(max_num_timesteps):
        # print("state:", state)
        state_qn = np.expand_dims(state, axis=0)  
        q_values = q_network(state_qn)

        action = utils.get_action(q_values, epsilon)   
        # print("discrete action:", action)
        
        a = np.array([[-1, -1, -1, -1]])
        array_action = 2*action + a
        # print("array action1:", array_action)
        # action = array_action.reshape(1, -1)

        next_state, reward, done, info, _ = env.step(array_action)
        next_state = next_state[0]
        indices = [2, 9]  
        next_state = next_state[indices]
        # print("next-state:", next_state)
        h = next_state[0]

        memory_buffer.append(experience(state, action, reward, next_state, done))
        update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer, MINIBATCH_SIZE)

        if update:
            experiences = utils.get_experiences(memory_buffer, MINIBATCH_SIZE)
            agent_learn(experiences, GAMMA)
        
        height.append(h)     
        state = next_state.copy()
        total_points += reward
        
        if done:
            break

    height_list_all.extend(height)
    height_list.append(h)

    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    epsilon = utils.get_new_eps(epsilon, E_DECAY, E_MIN)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    if av_latest_points >= 500.0:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        print('Q-NETWORK:', q_network)
        q_network.save('./model.h5')
        break

tot_time = time.time() - start
print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

utils.plot_history(total_point_history, filename="./drone_points.png")
utils.plot_heights(height_list,  filename="./drone_heights.png")
# utils.plot_heights(height_list_all, filename="/home/gulbin/Desktop/drone_height_all.png")

