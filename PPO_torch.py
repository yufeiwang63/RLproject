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
from torch_networks import AC_v_fc_network, CAC_a_fc_network
from helper_functions import SlidingMemory, PERMemory
import warnings

warnings.simplefilter("error", RuntimeWarning)

        

class PPO():    
    def __init__(self, state_dim, action_dim, action_low = -1.0, action_high = 1.0, mem_size = 40000, 
                 train_batch_size = 32, gamma = 0.99, actor_lr = 1e-4, critic_lr = 1e-3, 
                 tau = 0.1, eps = 0.2, update_epoach = 10, trajectory_number = 10):
        self.mem_size, self.train_batch_size = mem_size, train_batch_size
        self.gamma, self.actor_lr, self.critic_lr = gamma, actor_lr, critic_lr
        self.global_step = 0
        self.tau, self.eps = tau, eps
        self.state_dim, self.action_dim = state_dim, action_dim
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_mem = SlidingMemory(mem_size)
        self.device = 'cpu'
        self.action_low, self.action_high = action_low, action_high
        self.actor_policy_net = CAC_a_fc_network(state_dim, action_dim, action_low, action_high).to(self.device)
        self.actor_target_net = CAC_a_fc_network(state_dim, action_dim, action_low, action_high).to(self.device)
        self.critic_policy_net = AC_v_fc_network(state_dim).to(self.device)
        self.critic_target_net = AC_v_fc_network(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_policy_net.parameters(), self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_policy_net.parameters(), self.critic_lr)
        self.hard_update(self.actor_target_net, self.actor_policy_net)
        self.hard_update(self.critic_target_net, self.critic_policy_net)
        self.update_epoach = update_epoach 
        self.trajectory_number = trajectory_number
        self.trajectories = [[] for i in range(self.trajectory_number)]
        self.trajectory_pointer = 0
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    # the fake training process, accmulate trajectories, keeping the API to be the same among different algorithms    
    def train(self, pre_state, action, reward, next_state, if_end):
        # collect trajactories
        self.trajectories[self.trajectory_pointer].append([pre_state, action, reward, next_state, if_end])
        if if_end:
            self.trajectory_pointer += 1
        if self.trajectory_pointer == self.trajectory_number:
            self.__train()
            self.trajectory_pointer = 0


        # add to replay memory
        self.replay_mem.add(pre_state, action, reward, next_state, if_end)

        # update the value netwrok
        if self.replay_mem.num() >= self.train_batch_size:
            train_batch = self.replay_mem.sample(self.train_batch_size)

            # adjust dtype to suit the gym default dtype
            pre_state_batch = torch.tensor([x[0] for x in train_batch], dtype=torch.float, device = self.device) 
            action_batch = torch.tensor([x[1] for x in train_batch], dtype = torch.float, device = self.device) 
            # view to make later computation happy
            reward_batch = torch.tensor([x[2] for x in train_batch], dtype=torch.float, device = self.device).view(self.train_batch_size,1)
            next_state_batch = torch.tensor([x[3] for x in train_batch], dtype=torch.float, device = self.device)
            if_end = [x[4] for x in train_batch]
            if_end = torch.tensor(np.array(if_end).astype(float),device = self.device, dtype=torch.float).view(self.train_batch_size,1)
            
    
            # use the target_Q_network to get the target_Q_value
            with torch.no_grad():
                v_next_state = self.critic_target_net(next_state_batch).detach()
                v_target = self.gamma * v_next_state * (1 - if_end) + reward_batch
    
            v_pred = self.critic_policy_net(pre_state_batch)

            self.critic_optimizer.zero_grad()
            closs = (v_pred - v_target) ** 2 
            closs = closs.mean()
            closs.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_policy_net.parameters(),1)
            self.critic_optimizer.step()

            # update target network
            self.soft_update(self.critic_target_net, self.critic_policy_net, self.tau)

            

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
                states_values = self.critic_target_net(states).detach().numpy().reshape(-1)
            # print(length)
            # print(states_values)
            deltas = self.gamma * states_values[1:] + rewards[:-1] - states_values[:-1]
            # print(deltas)
            # print(traj[0][0])
            for i in range(length - 1):
                gammas = np.array([self.gamma ** x for x in range(length - 1 - i)])
                advantage = np.sum(gammas * deltas[i:])
                advantage_mem.append((traj[i][0], traj[i][1], advantage))


        print('----------advantage memory built over----------')

        for _ in range(self.update_epoach):

            obj = 0.0

            # sample from trajectories a batch of (s,a, r,s') pairs to construct the objective
            # for i in range(self.train_batch_size):
                # sample a (s,a,r,s') tuple
                # tj_idx = np.random.randint(0, len(trajectories))
                # chosen_tj = trajectories[tj_idx]
                # tp_idx = np.random.randint(0,len(chosen_tj))
                # chosen_tp = trajectories[tj_idx][tp_idx]

                # # compute advantage 
                # advantage = 0
                # for j in range(tp_idx, len(chosen_tj) - 1):
                #     with torch.no_grad():
                #         next_state = torch.tensor(chosen_tp[3], dtype = torch.float, device = self.device)
                #         current_state = torch.tensor(chosen_tp[0], dtype = torch.float, device = self.device)
                #         v_next_state = self.critic_target_net(next_state).detach()
                #         v_state = self.critic_target_net(current_state).detach()
                #     delta = chosen_tp[2] + self.gamma * v_next_state - v_state
                #     advantage += delta * self.gamma

                # # construct objective:
                # action = torch.tensor(chosen_tp[1], dtype = torch.float, device = self.device)
                # old_action_prob = self.actor_target_net(current_state).log_prob(action)
                # new_action_prob = self.actor_policy_net(current_state).log_prob(action)
                # aloss1 = new_action_prob / old_action_prob * advantage
                # aloss2 = torch.clamp(new_action_prob / old_action_prob, 1 - self.eps, 1 + self.eps) * advantage
                # aloss = - torch.min(aloss1, aloss2)
                # obj += aloss

            advantage_batch = random.sample(advantage_mem, self.train_batch_size)
            # print('advantage batch is :\n', advantage_batch)
            states_batch = torch.tensor([x[0] for x in advantage_batch], dtype = torch.float, device = self.device).view(self.train_batch_size,-1)
            action_batch = torch.tensor([x[1] for x in advantage_batch], dtype = torch.float, device = self.device).view(self.train_batch_size,-1)
            # print('state_batch is:\n', states_batch)
            # print('action batch is:\n',action_batch)
            advantage = torch.tensor([x[2] for x in advantage_batch], dtype = torch.float)
            old_prob = self.actor_target_net(states_batch).log_prob(action_batch)
            new_prob = self.actor_policy_net(states_batch).log_prob(action_batch)
            # print('old_prob is:\n', old_prob)
            # print(advantage)
            aloss1 = torch.exp(new_prob - old_prob) * advantage
            aloss2 = torch.clamp(torch.exp(new_prob - old_prob), 1 - self.eps, 1 + self.eps) * advantage
            obj = - torch.min(aloss1, aloss2)
            obj = torch.mean(obj)
            
            self.actor_optimizer.zero_grad()
            obj.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_policy_net.parameters(),1)
            self.actor_optimizer.step()
    
    # store the (pre_s, action, reward, next_state, if_end) tuples in the replay memory
    # just keep it here
    def perceive(self, pre_s, action, reward, next_state, if_end):
        self.replay_mem.append([pre_s, action, reward, next_state, if_end])
        if len(self.replay_mem) > self.mem_size:
            self.replay_mem.popleft()
        
    
    # use the policy net to choose the action with the highest Q value
    def action(self, s, sample = True): # use flag to suit other models' action interface
        s = torch.tensor(s, dtype=torch.float, device = self.device).unsqueeze(0)
        with torch.no_grad():
            m = self.actor_policy_net(s)
            a = np.clip(m.sample(), self.action_low, self.action_high) if sample else m.mean
            return a.numpy()[0]
    
        
    
        