from tensorflow.keras.models import load_model
from scripts import utils
import  gymnasium as gym
import numpy as np


# Load the saved model
loaded_model = load_model("model.h5")

# Load the enviroment
env = gym.make('hover-aviary-v0')
state = env.reset()    

state = state[0][0]
# print("---------", state)

max_num_timesteps = 10000
epsilon = 1.0

heights = []

for t in range(max_num_timesteps):
        # print("state:", state)
        indices = [2, 9]
        state = state[indices]
        # print("state-aaaa:", state)
        state_qn = np.expand_dims(state, axis=0)  
        # print("state2:", state_qn, state_qn.shape)
        q_values = loaded_model(state_qn)
        #print(q_values.shape, q_values)

        action = np.argmax(q_values.numpy())
        print("aa:", action)

        a = np.array([[-1, -1, -1, -1]])
        array_action = 2*action + a
        # print("array:", array_action)

        next_state, reward, done, info, _ = env.step(array_action)

        state = next_state.copy()
        # print("copy:", state)
        state = state[0]
        # print("copy2:", state)
        h = state[2]
        heights.append(h)
        
        if done:
            break

utils.plot_heights(heights,  filename="./model_drone_heights.png")