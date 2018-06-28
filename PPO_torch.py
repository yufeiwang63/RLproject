#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:58:01 2018

@author: yufei
"""

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
from collections import deque
from torch_networks import AC_v_fc_network, CAC_a_fc_network, CAC_a_sigma_fc_network
from helper_functions import SlidingMemory, PERMemory
import warnings

warnings.simplefilter("error", RuntimeWarning)

        

class PPO():    
    """doc for ppo"""
    def __init__(self, state_dim, action_dim, action_low = -1.0, action_high = 1.0,
                 train_batch_size = 32, gamma = 0.99, actor_lr = 1e-4, critic_lr = 1e-3, lam = 0.95,
                 tau = 0.1, eps = 0.2, update_epoach = 10, trajectory_number = 10):
        self.train_batch_size = train_batch_size
        self.gamma, self.actor_lr, self.critic_lr = gamma, actor_lr, critic_lr
        self.global_step = 0
        self.tau, self.eps, self.lam = tau, eps, lam
        self.state_dim, self.action_dim = state_dim, action_dim
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'
        self.action_low, self.action_high = action_low, action_high
        self.actor_policy_net = CAC_a_fc_network(state_dim, action_dim, action_low, action_high).to(self.device)
        self.actor_target_net = CAC_a_fc_network(state_dim, action_dim, action_low, action_high).to(self.device)
        # self.actor_policy_net = CAC_a_sigma_fc_network(state_dim, action_dim, action_low, action_high).to(self.device)
        # self.actor_target_net = CAC_a_sigma_fc_network(state_dim, action_dim, action_low, action_high).to(self.device)
        #self.critic_policy_net = AC_v_fc_network(state_dim).to(self.device)
        #self.critic_target_net = AC_v_fc_network(state_dim).to(self.device)
        self.critic_net = AC_v_fc_network(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_policy_net.parameters(), self.actor_lr)
        # self.critic_optimizer = optim.Adam(self.critic_policy_net.parameters(), self.critic_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), self.critic_lr)
        self.hard_update(self.actor_target_net, self.actor_policy_net)
        # self.hard_update(self.critic_target_net, self.critic_policy_net)
        self.update_epoach = update_epoach 
        self.trajectory_number = trajectory_number
        self.trajectories = [[] for i in range(self.trajectory_number)]
        self.trajectory_pointer = 0
        self.critic_net.apply(self._weight_init)
        self.actor_policy_net.apply(self._weight_init)
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def _weight_init(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.01)
    
    # the fake training process, accmulate trajectories, keeping the API to be the same among different algorithms    
    def train(self, pre_state, action, reward, next_state, if_end):
        # collect trajactories
        self.trajectories[self.trajectory_pointer].append([pre_state, action, reward, next_state, if_end])
        if if_end:
            self.trajectory_pointer += 1
        if self.trajectory_pointer == self.trajectory_number:
            self.__train()
            self.trajectory_pointer = 0
            for i in range(self.trajectory_number):
                self.trajectories[i] = []   

    # the true training process       
    # trajectories is a list of trajectory, where each trajectory is a list of [s,a,r,s',if_end] tuples     
    def __train(self):
         
        trajectories = self.trajectories
        print("train epoach!")
        self.hard_update(self.actor_target_net, self.actor_policy_net)
        
        advantage_mem = []
        for traj in trajectories:
            states = [x[0] for x in traj]
            rewards = np.array([x[2] for x in traj])
            length = len(traj)
            states = torch.tensor(states, dtype = torch.float, device = self.device)
            with torch.no_grad():
                states_values = self.critic_net(states).detach().numpy().reshape(-1, 1)
                final_next_state = torch.tensor(traj[-1][-2], dtype = torch.float)
                vs = self.critic_net(final_next_state).detach().numpy()

            ret = []
            for r in rewards[::-1]:
                vs = r + self.gamma * vs
                ret.append(vs)
            ret.reverse()
            ret = np.array(ret)

            advantages = ret - states_values
            advantage_mem = [(traj[i][0], traj[i][1], advantages[i], ret[i]) for i in range(length)]
            
            # deltas = self.gamma * states_values[1:] + rewards[:-1] - states_values[:-1]
            # for i in range(length - 1):
            #     # gammas = np.array([self.gamma ** x for x in range(length -1 - i)])
            #     # lamdas = np.array([self.lam ** x for x in range(length - 1 - i)])
            #     # gae_advantage = np.sum(gammas * lamdas * deltas[i:])
            #     ret[i] = np.sum(gammas * rewards[i:-1]) + states_values[-1] * self.gamma * gammas[-1]
            #     advantage_mem.append((traj[i][0], traj[i][1], gae_advantage, ret))


        print('----------advantage memory built over----------')

        for _ in range(self.update_epoach):

            batch = random.sample(advantage_mem, self.train_batch_size)
            states_batch = torch.tensor([x[0] for x in batch], dtype = torch.float, device = self.device).view(self.train_batch_size,-1)
            action_batch = torch.tensor([x[1] for x in batch], dtype = torch.float, device = self.device).view(self.train_batch_size,-1)
            advantage = torch.tensor([x[2] for x in batch], dtype = torch.float)
            advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-5)
            old_prob = torch.sum(self.actor_target_net(states_batch).log_prob(action_batch), dim = 1)
            print("old_prob is", old_prob)
            new_prob = torch.sum(self.actor_policy_net(states_batch).log_prob(action_batch), dim = 1)
            aloss1 = torch.exp(new_prob - old_prob) * advantage
            aloss2 = torch.clamp(torch.exp(new_prob - old_prob), 1 - self.eps, 1 + self.eps) * advantage
            aloss = - torch.min(aloss1, aloss2)
            aloss = torch.mean(aloss)
            
            self.actor_optimizer.zero_grad()
            aloss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_policy_net.parameters(),1)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            return_batch = torch.tensor([x[-1] for x in batch], dtype = torch.float, device = self.device).view(self.train_batch_size,-1)
            value_pred = self.critic_net(states_batch)
            closs = (value_pred - return_batch)**2
            closs = torch.mean(closs)
            closs.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(),1)
            self.critic_optimizer.step()
    
    # use the policy net to choose the action with the highest Q value
    def action(self, s, sample = True): # use flag to suit other models' action interface
        s = torch.tensor(s, dtype=torch.float, device = self.device).unsqueeze(0)
        with torch.no_grad():
            m = self.actor_policy_net(s)
            a = np.clip(m.sample(), self.action_low, self.action_high) if sample else m.mean
            return a.numpy()[0]
    
        
    
        