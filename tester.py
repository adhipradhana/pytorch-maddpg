#!/usr/bin/env python
# coding: utf-8

# # Battle Royale Environment Trainer
# This notebook is for training Battle Royale agents. MADDPG is used for training the agents.

# ## Setup Environment Dependencies

# In[ ]:

import torch
import random
import sys
import numpy as np
from gym_unity.envs import UnityEnv
from utils.MADDPG import MADDPG
from datetime import datetime

print("Python version:")
print(sys.version)
print(sys.executable)

# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")


# ## Start Environment

# In[ ]:


# Environment name
# Remember to put battle royale environment configuration within the config folder
env_name = "environment/new/battle-royale-static"

env = UnityEnv(env_name, worker_id=1, use_visual=False, multiagent=True)

print(str(env))


# ## Model Variables

# In[ ]:


random_seed = random.randint(0,1000000)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
n_agents = env.number_agents
n_episode = 15
max_steps = 10
buffer_capacity = 1000
batch_size = 7
episodes_before_train = 5


# In[ ]:


# setup seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, buffer_capacity, episodes_before_train, use_approx=True)

FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor


# ## Run Model

# In[ ]:

current_time = str(datetime.now())
print("Testing model...")
for i_episode in range(n_episode):
    # reset environment
    obs = env.reset()
    obs = np.stack(obs)
    
    # convert observation to tensor
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float()
    
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    for i_step in range(max_steps):
        obs = obs.type(FloatTensor)
        actions = maddpg.select_action(obs).data.cpu()
        actions_list = actions.tolist()
        
        obs_, reward, done, _ = env.step(actions_list)
        
        reward = torch.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = torch.from_numpy(obs_).float()
        if i_step != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, actions, next_obs, reward)
        
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()

        
        if True in done:
            break

    maddpg.episode_done += 1
    print("Episode: {}, reward = {}".format(i_episode, total_reward))

    # save model
    if (maddpg.episode_done + 1 == n_episode):
        maddpg.save(current_time, maddpg.episode_done)


# ## Close Environment

# In[ ]:


env.close()

