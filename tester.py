#!/usr/bin/env python
# coding: utf-8

# # Battle Royale Environment Trainer
# This notebook is for training Battle Royale agents. MADDPG is used for training the agents.

# ## Setup Environment Dependencies

# In[ ]:


import sys
from gym_unity.envs import UnityEnv

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
env_name = "environment/battle-royale-static"

env = UnityEnv(env_name, worker_id=3, use_visual=False, multiagent=True)

print(str(env))


# ## Testing Model

# ## Model Variables

# In[ ]:


random_seed = 6272727
n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
n_agents = env.number_agents
n_episode = 50
max_steps = 2000
buffer_capacity = 1000000
batch_size = 1000
episodes_before_train = 50


# In[ ]:


# setup seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, buffer_capacity, episodes_before_train)

FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor


# ## Loading Model

# In[ ]:


import os

path = os.path.join(os.getcwd(), 'checkpoint', 'Time_2020-03-25_08-42-13.632845_NAgent_2', 'Time_2020-03-25_08-42-13.632845_NAgent_2_Episode_180.pth')
maddpg.load(path, map_location='cpu')


# ## Run Model

# In[ ]:


print("Testing model...")
for i_episode in range(n_episode):
    # reset environment
    obs = env.reset()
    obs = np.stack(obs)
    
    # convert observation to tensor
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float()
    
    total_reward = 0.0
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
        obs = next_obs

        # check if done
        if True in done:
            break

    maddpg.episode_done += 1
    print("Episode: {}, reward = {}".format(i_episode, total_reward))


# ## Close Environment

# In[ ]:


env.close()

