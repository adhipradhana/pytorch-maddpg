import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = dim_action * n_agent

        fc1 = 256
        fc2 = 256
        fc3 = 256

        # input layer
        self.fc1 = nn.Linear(obs_dim, fc1)
        self.fc2 = nn.Linear(fc1+act_dim, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.output = nn.Linear(fc3, 1)

        #last layer weight and bias initialization 
        torch.nn.init.uniform_(self.output.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.output.bias, a=-3e-3, b=3e-3)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.fc1(obs))
        combined = torch.cat([result, acts], 1)
        result = F.relu(self.fc2(combined))
        result = F.relu(self.fc3(result))
        result = self.output(F.relu(result))

        return result


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()

        fc1 = 64
        fc2 = 64
        
        # network mapping state to action 
        self.fc1 = nn.Linear(dim_observation, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.output = nn.Linear(fc2, dim_action)

        #last layer weight and bias initialization 
        torch.nn.init.uniform_(self.output.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.output.bias, a=-3e-3, b=3e-3)

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.fc1(obs))
        result = F.relu(self.fc2(result))
        result = F.tanh(self.output(result))

        return result

