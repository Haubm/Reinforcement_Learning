from dqn_tf import Agent
import numpy as np
import gymnasium as gym
from utils_2 import plot_learning_curve
import tensorflow as tf

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2',render_mode = 'rgb_array')
    mem_size = 1000000
    input_dims = env.observation_space.shape
    print(input_dims)
    print(env.observation_space)
    state_memory = np.zeros((mem_size, *input_dims),
                                        dtype=np.float32)
    print(state_memory[0])