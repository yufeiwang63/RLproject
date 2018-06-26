import gym
import torch
import numpy as np
import argparse
import sys

from helper_functions import *
from DQN_torch import DQN 
from NAF_torch import NAF
from DDPG_torch import DDPG
from AC_torch import AC
from CAC_torch import CAC
from PPO_torch import PPO
import matplotlib.pyplot as plt
import time

# env = gym.make('MountainCarContinuous-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('CartPole-v0')
# env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('LunarLander-v2')
env = gym.make('Pendulum-v0')

argparser = argparse.ArgumentParser(sys.argv[0])
## parameters
argparser.add_argument('--lr', type=float, default=1e-3)
argparser.add_argument('--replay_size', type=int, default=20000)
argparser.add_argument('--batch_size', type=float, default=64)
argparser.add_argument('--gamma', type=float, default=0.99)
argparser.add_argument('--tau', type=float, default=0.1)
argparser.add_argument('--noise_type', type = str, default = 'gauss')
argparser.add_argument('--agent', type = str, default = 'ddpg')
##
args = argparser.parse_args()


Replay_mem_size = args.replay_size
Train_batch_size = args.batch_size
Actor_Learning_rate = args.lr
Critic_Learning_rate = args.lr
Gamma = args.gamma
tau = args.tau

State_dim = env.observation_space.shape[0]
print(State_dim)

Action_dim = env.action_space.shape[0]
# Action_dim = env.action_space.n
# Action_dim = dim
print(Action_dim)


print('----action range---')
print(env.action_space.high)
print(env.action_space.low)
action_low = env.action_space.low[0].astype(float)
action_high = env.action_space.high[0].astype(float)

ounoise = OUNoise(Action_dim, 8, 3, 0.9995)
gsnoise = GaussNoise(8, 0.5, 0.99995)
noise = gsnoise if args.noise_type == 'gauss' else ounoise

def play(agent, num_epoch, Epoch_step, show = False):
   
    acc_reward = 0
    for epoch in range(num_epoch):
        pre_state = env.reset()
        for step in range(Epoch_step):
            if show:
                env.render()
            
            # action = agent.action(state_featurize.transfer(pre_state), False)
            action = agent.action(pre_state, False)
            next_state, reward, done, _ = env.step(action)
            acc_reward += reward

            if done or step == Epoch_step - 1:
                break
            pre_state = next_state
    return acc_reward / num_epoch


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
                print('episoid: ', epoch + 1, 'step: ', step + 1, ' reward: ', acc_reward)
                break
            
            pre_state = next_state
        
        if epoch % 100 == 0 and epoch > 0:
            avg_reward = play(agent, 10, 300, not True)
            print('--------------episode ', epoch,  'avg_reward: ', avg_reward, '---------------', file = output_file)
            print('--------------episode ', epoch,  'avg_reward: ', avg_reward, '---------------')
            if avg_reward > -5:
                print('----- using ', epoch, '  epochs')
                #agent.save_model()
                break
         
    return agent
            
    



naf = NAF(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
          Gamma, Critic_Learning_rate, action_low, action_high, tau, noise, False, False)  
dqn = DQN(State_dim, Action_dim, Replay_mem_size, Train_batch_size,\
            Gamma, 1e-4, 0.1, True, True)  
ddpg = DDPG(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
             Gamma, Actor_Learning_rate, Critic_Learning_rate, action_low, action_high, tau, noise, False) 

ac = AC(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
             Gamma, 1e-3, 1e-3, 0.1)

cac = CAC(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
             Gamma, Actor_Learning_rate, Critic_Learning_rate, tau, action_low, action_high, 3, False)

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


if args.agent == 'ddpg':
    agent = train(ddpg, 10000, 200)
elif args.agent == 'naf':
    agent = train(naf, 10000, 200)
elif args.agent == 'cac':
    agent = train(cac, 10000, 200)

#print('after train')

#print(play(agentnaf,300, False))
#print(play(agentnaf_addloss,300, False))

