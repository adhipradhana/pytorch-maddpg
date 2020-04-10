from utils.model import Critic, Actor
import torch
from copy import deepcopy
from utils.memory import ReplayMemory, Experience
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import os
from datetime import datetime

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        # for gaussian noise
        self.var = [1.0 for i in range(n_agents)]

        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = BoolTensor(list(map(lambda s: s is not None, batch.next_states)))
            
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).type(FloatTensor)
            action_batch = torch.stack(batch.actions).type(FloatTensor)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)

            # non_final_next_states: (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)

            # calculating current_Q : Q(x, a1, ..., an)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_action)

            # calculating for target_Q : y = r + Q(x', a'1, ... , a'n) --> for all states non final
            # TODO: Implementing inferring of other's agent policy
            non_final_next_actions = [self.actors_target[i](non_final_next_states[:,i,:]) for i in range(self.n_agents)]
            non_final_next_actions = torch.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0,1).contiguous())

            target_Q = torch.zeros(self.batch_size).type(FloatTensor)

            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1,self.n_agents * self.n_actions)
            ).squeeze()
            
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1))

            # calculating critic loss from current_Q and target_Q
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            # calculating actor loss
            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch):
        # state_batch dimention: n_agents x state_dim

        # Define type of tensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        # create actions tensor
        actions = torch.zeros(
            self.n_agents,
            self.n_actions)

        # iterating for all agents
        for i in range(self.n_agents):
            # get all observation
            sb = state_batch[i, :].detach()

            # calculate forward
            act = self.actors[i](sb.unsqueeze(0)).squeeze()
            
            ## add gaussian noise
            act += torch.from_numpy(
                np.random.randn(self.n_actions) * self.var[i]).type(FloatTensor)

            # update gaussian noise
            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.999998
            act = torch.clamp(act, -1.0, 1.0)

            actions[i, :] = act
        self.steps_done += 1

        return actions

    def save(self, time, episode):
        # check path exists
        cwd = os.getcwd()
        save_dir = os.path.join(cwd, 'checkpoint')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # create filename
        time = time.replace(' ','_')
        filename = 'Time_{}_NAgent_{}_Episode_{}.pth'.format(time, self.n_agents, episode)
        save_dir = os.path.join(save_dir, filename)

        # create saving dictionary
        checkpoint = dict()

        # saving model
        for i in range(self.n_agents):
            checkpoint['actor_{}'.format(i)] = self.actors[i].state_dict()
            checkpoint['critic_{}'.format(i)] = self.critics[i].state_dict()
            checkpoint['actor_target_{}'.format(i)] = self.actors_target[i].state_dict()
            checkpoint['critic_target_{}'.format(i)] = self.critics_target[i].state_dict()
            checkpoint['actor_optimizer_{}'.format(i)] = self.actor_optimizer[i].state_dict()
            checkpoint['critic_optimizer_{}'.format(i)] = self.critic_optimizer[i].state_dict()
            checkpoint['var_{}'.format(i)] = self.var[i]
        
        # saving model info
        checkpoint['n_agents'] = self.n_agents
        checkpoint['episode'] = episode
        checkpoint['time'] = str(datetime.now())

        # save
        torch.save(checkpoint, save_dir)

    def load(self, path, map_location):
        checkpoint = torch.load(path, map_location=map_location)

        # loading model
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint['actor_{}'.format(i)])
            self.critics[i].load_state_dict(checkpoint['critic_{}'.format(i)])
            self.actors_target[i].load_state_dict(checkpoint['actor_target_{}'.format(i)])
            self.critics_target[i].load_state_dict(checkpoint['critic_target_{}'.format(i)])
            self.actor_optimizer[i].load_state_dict(checkpoint['actor_optimizer_{}'.format(i)])
            self.critic_optimizer[i].load_state_dict(checkpoint['critic_optimizer_{}'.format(i)])
            self.var[i] = checkpoint['var_{}'.format(i)]

    def load_agent(self, path, agent_number, map_location):
        '''strictly for testing, do not use for resume training due to critic's network's size differents'''

        checkpoint = torch.load(path, map_location=map_location)

        #loading from 1 agent
        if agent_number >= self.n_agents or agent_number < 0:
            agent_number = 0

        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint['actor_{}'.format(agent_number)])
            self.actors_target[i].load_state_dict(checkpoint['actor_target_{}'.format(agent_number)])
            self.actor_optimizer[i].load_state_dict(checkpoint['actor_optimizer_{}'.format(agent_number)])
            self.var[i] = checkpoint['var_{}'.format(agent_number)]


