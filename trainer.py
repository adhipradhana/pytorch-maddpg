#!/usr/bin/env python
# coding: utf-8

# # Battle Royale Environment Trainer
# This notebook is for training Battle Royale agents. MADDPG is used for training the agents.

# ## Setup Environment Dependencies

# In[1]:


import sys
from gym_unity.envs import UnityEnv

print("Python version:")
print(sys.version)
print(sys.executable)

# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")


# ## Start Environment

# In[2]:


# Environment name
# Remember to put battle royale environment configuration within the config folder
env_name = "environment/battle-royale-static"

env = UnityEnv(env_name, worker_id=4, use_visual=False, multiagent=True)

print(str(env))


# ## Examine Observation Space

# In[3]:


# Examine observation space
observation = env.observation_space
env.reset()
print("Agent observation space type: {}".format(observation))


# ## Examine Action Space

# In[4]:


# Examine action space
action = env.action_space
print("Agent action space type: {}".format(action))


# ## Agents Training
# This part shows agent training using MADDPG algoritm

# ### Setup Algorithm Dependencies

# In[5]:


from datetime import datetime
import torch
import visdom
import numpy as np
import random

from utils.MADDPG import MADDPG


# ### Setup Algoritm Parameters

# In[6]:


random_seed = random.randint(0,1000000)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
n_agents = env.number_agents
n_episode = 5000
max_steps = 100
buffer_capacity = 1000000
batch_size = 1000
episodes_before_train = 100
checkpoint_episode = 1000


# ### Setup MADDPG

# In[8]:


# setup seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, buffer_capacity, episodes_before_train)

FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor

vis = visdom.Visdom(port=8097, log_to_filename='log/maddpg.log')


# ### MADDPG Training

# In[ ]:


win_avg = None
win_curr = None
current_time = str(datetime.now())
reward_record = [[] for i in range(n_agents+1)]

print("Exploration begins...")
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
    
    reward_record[0].append(total_reward.cpu())  
    for i in range(n_agents):
        reward_record[i+1].append(rr[i])
    
    y_axis = np.asarray([np.mean(reward_record[i]) for i in range(n_agents+1)]).reshape((1,n_agents+1))
  
    if maddpg.episode_done == maddpg.episodes_before_train:
        print("Training begins...")
        print("MADDPG on Battle Royale")
              
    if win_avg is None:
        win_avg = vis.line(X=np.arange(i_episode, i_episode+1),
                       Y=np.array(y_axis),
                       opts=dict(
                           ylabel="Average Reward",
                           xlabel="Episode",
                           title="Average Reward | MADDPG on Battle Royale | " + \
                               "Agent: {} | ".format(n_agents) + \
                               "Time: {}\n".format(current_time),
                           legend=["Total"] +
                           ["Agent-{}".format(i) for i in range(n_agents)]))
    else:
        vis.line(X=np.array(
            [np.array(i_episode).repeat(n_agents+1)]),
                 Y=np.array(y_axis),
                 win=win_avg,
                 update="append")
        
    if win_curr is None:
        win_curr = vis.line(X=np.arange(i_episode, i_episode+1),
                       Y=np.array([
                           np.append(total_reward, rr)]),
                       opts=dict(
                           ylabel="Reward",
                           xlabel="Episode",
                           title="Current Reward | MADDPG on Battle Royale | " + \
                               "Agent: {} | ".format(n_agents) + \
                               "Time: {}\n".format(current_time),
                           legend=["Total"] +
                           ["Agent-{}".format(i) for i in range(n_agents)]))
    else:
        vis.line(X=np.array(
            [np.array(i_episode).repeat(n_agents+1)]),
                 Y=np.array([np.append(total_reward,rr)]),
                 win=win_curr,
                 update="append")
        
    # save model
    if (maddpg.episode_done % checkpoint_episode == 0):
        maddpg.save(current_time, maddpg.episode_done)


# ## Close Environment

# In[9]:


env.close()


# In[ ]:




