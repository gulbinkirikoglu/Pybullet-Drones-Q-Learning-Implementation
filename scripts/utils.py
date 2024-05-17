import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

def get_experiences(memory_buffer, minibatch_size):
    experiences = random.sample(memory_buffer, k=minibatch_size)
    states = tf.convert_to_tensor(
        np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
    )
    # print('\nexp-state-shape',states.shape)
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
    )
    # print('\nexp-actions-shape', actions)
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
    )
    # print('\nexp-rewards-shape',rewards.shape)
    next_states = tf.convert_to_tensor(
        np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
    )
    # print('\nexp-n-state-shape',next_states.shape)
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
        dtype=tf.float32,
    )
    # print('\nexp-done-shape',done_vals.shape)
    return (states, actions, rewards, next_states, done_vals)


def check_update_conditions(t, num_steps_upd, memory_buffer, minibatch_size):
    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > minibatch_size:
        return True
    else:
        return False


def get_new_eps(epsilon, e_decay, e_min):
    return max(e_min, e_decay * epsilon)


def get_action(q_values, epsilon):        
    if random.random() > epsilon:
        act = np.argmax(q_values.numpy()[0])
        # print("\nACT ARGMAX:", act)
        return act
    else:
        act = random.choice(np.arange(2))
        # print("\nACT RANDOM:", act)
        return act


def update_target_network(q_network, target_q_network, TAU):
    for target_weights, q_net_weights in zip(
        target_q_network.weights, q_network.weights
    ):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)


def plot_heights(heights, filename, ref_height=1):

    lower_limit = 0
    upper_limit = len(heights)

    heights = heights[lower_limit:upper_limit]
    episode_num = [x for x in range(lower_limit, upper_limit)]
    
    plt.figure(figsize=(10, 7), facecolor="white")
    plt.plot(episode_num, heights, linewidth=1, color="blue")
    plt.plot(episode_num, [ref_height]*len(episode_num), linewidth=1, color="red")

    text_color = "black"

    ax = plt.gca()
    ax.set_facecolor("white")
    plt.grid()

    plt.xlabel("Episodes", color=text_color, fontsize=30)
    plt.ylabel("Heights", color=text_color, fontsize=30)
    plt.title("Height Over Episodes")

    filename = filename
    plt.savefig(filename)

def plot_history(point_history, filename, ref_points=467):

    lower_limit = 0
    upper_limit = len(point_history)

    points = point_history[lower_limit:upper_limit]
    episode_num = [x for x in range(lower_limit, upper_limit)]

    plt.figure(figsize=(10, 7), facecolor="white")
    plt.plot(episode_num, points, linewidth=1, color="blue")
    plt.plot(episode_num, [ref_points]*len(episode_num), linewidth=1, color="red")

    text_color = "black"

    ax = plt.gca()
    ax.set_facecolor("white")
    plt.grid()

    plt.xlabel("Episodes", color=text_color, fontsize=30)
    plt.ylabel("Total Points", color=text_color, fontsize=30)
    plt.title("Total Points Over Episodes")
    
    filename = filename
    plt.savefig(filename)

def str_to_numpy(s):
    if len(s) > 4:
        raise ValueError("Input string must be at most 4 digits long")
    s = s.zfill(4)  # String uzunluğunu 4 haneli yapmak için başına sıfır ekler
    return np.array(list(map(int, s)), dtype=np.int32)


def image_to_np_array(image):
    return np.array(image)
