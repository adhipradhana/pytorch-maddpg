import torch
import visdom

from utils.MADDPG import MADDPG

random_seed = 1234
n_states = 41
n_actions = 5
n_agents = 2
n_episode = 10
max_steps = 1000
buffer_capacity = 1000000
batch_size = 1
episodes_before_train = 0

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, buffer_capacity, episodes_before_train)
maddpg.episode_done = 1
maddpg.update_policy()