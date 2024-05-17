"""
This script is used to observe the movements of the drone according to 
the desired engine powers in a simulation environment using the Gymnasium library.
"""
import random
import time
import gymnasium as gym
import numpy as np

env = gym.make('hover-aviary')

env.reset()

state_size = env.observation_space.shape
print('State Shape:', state_size)

num_actions = env.action_space.shape
print('Number of actions:', num_actions)

action_list = [np.array([1.0, 1.0, 1.0, 1.0])]

"""
action_list = [np.array([0.0, 0.0, 0.0, 0.0]),
               np.array([0.0, 0.0, 0.0, 1.0]),
               np.array([0.0, 0.0, 1.0, 0.0]),
               np.array([0.0, 0.0, 1.0, 1.0]),
               np.array([0.0, 1.0, 0.0, 0.0]), 
               np.array([0.0, 1.0, 0.0, 1.0]),
               np.array([0.0, 1.0, 1.0, 0.0]),
               np.array([0.0, 1.0, 1.0, 1.0]),
               np.array([1.0, 0.0, 0.0, 0.0]),
               np.array([1.0, 0.0, 0.0, 1.0]),
               np.array([1.0, 0.0, 1.0, 0.0]),
               np.array([1.0, 0.0, 1.0, 1.0]),
               np.array([1.0, 1.0, 0.0, 0.0]),
               np.array([1.0, 1.0, 0.0, 1.0]),
               np.array([1.0, 1.0, 1.0, 0.0])]
"""
               
for _ in range(1000):
    # use this loop to move drone randomly in simulation
    action = random.choice(action_list) 
    action = np.reshape(action, (1, 4))
    #print(action)

    # Take a step in the environment using the action
    obs, reward, done, info, _ = env.step(action)

    # Render the environment
    env.render()
    
    time.sleep(0.1)
