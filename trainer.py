
# coding: utf-8

# # Battle Royale Environment Trainer
# This notebook is for training Battle Royale agents. MADDPG is used for training the agents.

# ## Setup Environment Dependencies

# In[1]:


import sys
from gym_unity.envs import UnityEnv

get_ipython().run_line_magic('matplotlib', 'inline')

print("Python version:")
print(sys.version)
print(sys.executable)

# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")


# ## Start Environment

# In[11]:


# Environment name
# Remember to put battle royale environment configuration within the config folder
env_name = "environment_linux/battle-royale-static.x86_64"

env = UnityEnv(env_name, worker_id=4, use_visual=False, multiagent=True)

print(str(env))


# ## Examine Observation Space

# In[12]:


# Examine observation space
observation = env.observation_space
print("Agent observation space type: {}".format(observation))


# ## Examine Action Space

# In[13]:


# Examine action space
action = env.action_space
print("Agent action space type: {}".format(action))


# ## Agents Training
# This part shows agent training using MADDPG algoritm

# ### Setup Algorithm Dependencies

# In[14]:


from datetime import datetime
import torch
import visdom
import numpy as np

from utils.MADDPG import MADDPG
from utils.RandomProcess import OUNoise


# ### Setup Algoritm Parameters

# In[15]:


random_seed = 4966
n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
n_agents = env.number_agents
n_episode = 200
max_steps = 10000
buffer_capacity = 1000000
batch_size = 1000
episodes_before_train = 10
checkpoint_episode = 20


# ### Setup MADDPG

# In[16]:


# setup seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, buffer_capacity, episodes_before_train)
noise = OUNoise(env.action_space)

FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor

vis = visdom.Visdom(port=8097)


# ### MADDPG Training

# In[ ]:


win = None
current_time = str(datetime.now())
reward_record = []

print("Exploration begins...")
for i_episode in range(n_episode):
    # reset environment
    obs = env.reset()
    obs = np.stack(obs)
    noise.reset()
    
    # convert observation to tensor
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float()
    
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    for i_step in range(max_steps):
        obs = obs.type(FloatTensor)
        actions = maddpg.select_action(obs).data.cpu()
        actions_list = [noise.get_action(action) for action in actions.tolist()]
        
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
        
        # check if done
        if True in done:
            break

    maddpg.episode_done += 1
    print("Episode: {}, reward = {}".format(i_episode, total_reward))
    reward_record.append(total_reward)
    
    if maddpg.episode_done == maddpg.episodes_before_train:
        print("Training begins...")
        print("MADDPG on Battle Royale")
              
    if win is None:
        win = vis.line(X=np.arange(i_episode, i_episode+1),
                       Y=np.array([
                           np.append(total_reward.cpu(), rr)]),
                       opts=dict(
                           ylabel="Reward",
                           xlabel="Episode",
                           title="MADDPG on Battle Royale | " + \
                               "Agent: {} | ".format(n_agents) + \
                               "Time: {}\n".format(current_time),
                           legend=["Total"] +
                           ["Agent-".format(i) for i in range(n_agents)]))
    else:
        vis.line(X=np.array(
            [np.array(i_episode).repeat(n_agents+1)]),
                 Y=np.array([np.append(total_reward.cpu(),rr)]),
                 win=win,
                 update="append")
        
    # save model
    if (maddpg.episode_done % checkpoint_episode == 0):
        maddpg.save(current_time, maddpg.episode_done)


# ## Close Environment

# In[9]:


env.close()

