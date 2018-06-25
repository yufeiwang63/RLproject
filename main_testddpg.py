# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 21:43:06 2018

@author: Wangyf
"""
import gym
import torch
import numpy as np
import objfunc

#from critics import *
#from actors import *
#from actor_critic import *
from helper_functions import *
from DQN_torch import DQN 
from NAF_torch import NAF
from DDPG_torch import DDPG
from AC_torch import AC
from CAC_torch import CAC
from PPO_torch import PPO
import matplotlib.pyplot as plt
import time


# env = gym.make('Pendulum-v0')
# env = gym.make('MountainCarContinuous-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('CartPole-v0')
# env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('LunarLander-v2')

dim = 1
window_size = 2
env = objfunc.make('quadratic', dim=dim, init_point=np.ones(dim) ,
                                window_size=window_size)

Replay_mem_size = 50000
Train_batch_size = 64
Actor_Learning_rate = 1e-3
Critic_Learning_rate = 1e-3
Gamma = 0.99
explore_rate = 10
tau = 0.1

# State_dim = env.observation_space.shape[0]
State_dim = dim + window_size + dim * window_size
print(State_dim)

# Action_dim = env.action_space.shape[0]
# Action_dim = env.action_space.n
Action_dim = dim
print(Action_dim)


print('----action range---')
#print(env.action_space.high)
#print(env.action_space.low)
#action_low = env.action_space.low[0].astype(float)
#action_high = env.action_space.high[0].astype(float)

ounoise = OUNoise(Action_dim, 8, 3, 0.9995)
gsnoise = GaussNoise(0.05, 0.02, 0.99995)


## featurization has been proved to be very important to the convergence of mountain car
# state_featurize = Featurize_state(env, True)
# After_featurize_state_dim = state_featurize.get_featurized_state_dim()

            

def play(agent, num_epoch, Epoch_step, show = False):
   
    final_value = []
    for epoch in range(1):
        pre_state = env.reset()
        for step in range(Epoch_step):
            if show:
                env.render()
            
            # action = agent.action(state_featurize.transfer(pre_state), False)
            action = agent.action(pre_state, False)

            next_state, reward, done, _ = env.step(action)
            if done or step == Epoch_step - 1:
                final_val = env.get_value()
                final_value.append(final_val)
                break
            pre_state = next_state
    return final_value


def train(agent, Train_epoch, Epoch_step, file_name = './res.dat'):        
    output_file = open(file_name, 'w')
    for epoch in range(Train_epoch):
        pre_state = env.reset()
        acc_reward = 0
        
        for step in range(Epoch_step):

            action = agent.action(pre_state)

            if action[0] != action[0]:
                raise('nan error!')

            next_state, reward, done, _ = env.step(action)
            acc_reward += reward
            
            # agent.train(state_featurize.transfer(pre_state), action, reward, state_featurize.transfer(next_state), done)
            agent.train(pre_state, action, reward, next_state, done)

            if done or step == Epoch_step - 1:
                #print('episoid: ', epoch + 1, 'step: ', step + 1, ' reward is', acc_reward,  file = output_file)
                #print('episoid: ', epoch + 1, 'step: ', step + 1, ' reward is', acc_reward, )
                print('episoid: ', epoch + 1, 'step: ', step + 1, ' final value: ', env.get_value())
                break
            
            pre_state = next_state
        
        if epoch % 100 == 0 and epoch > 0:
            final_value = play(agent, 10, 50, not True)
            print('--------------episode ', epoch,  'final_value: ', final_value, '---------------', file = output_file)
            print('--------------episode ', epoch,  'final value: ', final_value, '---------------')
            if np.mean(np.array(final_value)) < 0.05:
                print('----- using ', epoch, '  epochs')
                #agent.save_model()
                break
         
    return agent
            
    



naf = NAF(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
          Gamma, 1e-4, -0.3, 0.3, tau, gsnoise, False, False)  
#naf_addloss = NAF(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
#             Gamma, Critic_Learning_rate, -1.0, 1.0, tau, gsnoise, True)  
dqn = DQN(State_dim, Action_dim, Replay_mem_size, Train_batch_size,\
            Gamma, 1e-4, 0.1, True, True)  
ddpg = DDPG(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
             Gamma, 1e-3, 1e-3, -0.1, 0.1, 0.1, gsnoise, False) 

ac = AC(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
             Gamma, 1e-3, 1e-3, 0.1)

cac = CAC(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
             Gamma, 5e-5, 5e-5, 0.1, -0.3, 0.3, False)

cppo = PPO(State_dim, Action_dim,-1.0, 1.0, 2000, 64, Gamma, 1e-4, 1e-4, 0.3, 0.2, 100)

# agent = train(naf, 10000,300)
# agentnaf = train(naf, 3000, 300, r'./record/naf_lunar.txt')
# agentppo = train(cppo, 3000, 300, r'./record/ppo_lunar.txt')
# agentnaf_addloss = train(naf_addloss, 1500, 300, r'./record/naf_addloss_lunar.txt')
# agentddpg = train(ddpg, 3000, 300, r'./record/ddpg_lunar_PER.txt')
# agentnaf_addloss = train(naf_addloss, 1500, 300, r'D:\study\rl by david silver\Trainrecord\NAF_addloss.txt')
# agentnaf_ddpg = train(ddpg, 1500, 300, r'D:\study\rl by david silver\Trainrecord\ddpg_lunar.txt')
# agentac = train(ac, 3000, 300, r'./record/ac_lunar_land_continues.txt')
# agentcac = train(cac, 3000, 300, r'./record/cac_lunar_land_continues-PER.txt')
# agentdqn = train(dqn, 3000, 300, r'./record/dqn_lunar_dueling_PER_1e-3_0.3.txt')
agentNAF = train(ddpg, 10000, 50)

#print('after train')

#print(play(agentnaf,300, False))
#print(play(agentnaf_addloss,300, False))

