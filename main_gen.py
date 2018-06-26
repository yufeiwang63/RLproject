# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 21:43:06 2018

@author: Wangyf
"""
import gym
import torch
import numpy as np
import objfunc
import sys
import argparse

from helper_functions import *
from DQN_torch import DQN 
from NAF_torch import NAF
from DDPG_torch import DDPG
from AC_torch import AC
from CAC_torch import CAC
from PPO_torch import PPO
import matplotlib.pyplot as plt
import time

from cg import cg
from bb import sd
from newton import quasiNewton
from objfunc import Logistic, Quadratic, NeuralNet, Ackley
from dataset import LogisticDataset, NeuralNetDataset

## parameters
argparser = argparse.ArgumentParser(sys.argv[0])
argparser.add_argument('--lr', type=float, default=1e-4)
argparser.add_argument('--replay_size', type=int, default=30000)
argparser.add_argument('--batch_size', type=float, default=64)
argparser.add_argument('--gamma', type=float, default=0.99)
argparser.add_argument('--tau', type=float, default=0.1)
argparser.add_argument('--noise_type', type = str, default = 'gauss')
argparser.add_argument('--agent', type = str, default = 'ddpg')
argparser.add_argument('--action_low', type = float, default = -0.3)
argparser.add_argument('--action_high', type = float, default = 0.3)
argparser.add_argument('--dim', type = int, default = 3)
argparser.add_argument('--window_size', type = int, default = 10)
argparser.add_argument('--obj', type = str, default = 'quadratic')
args = argparser.parse_args()

### the env
dim = args.dim
window_size = args.window_size
init_point = np.arange(dim) / dim

# construct train and test objectives
num_train, num_test = 20, 10
env_train = []
env_test = []


if args.obj == 'quadratic':
    obj = Quadratic(dim)
    env = objfunc.make('quadratic', dim=dim, init_point=init_point ,
                                    window_size=window_size)

elif args.obj == 'logistic':
    X, Y = LogisticDataset(dim=dim)
    dim = dim + 1
    obj = Logistic(dim, X, Y)
    init_point = np.arange(dim) / dim
    env = objfunc.make('logistic', dim=dim, init_point=init_point, 
                        window_size=window_size, other_params=[X, Y])

elif args.obj == 'ackley':
    obj = Ackley(dim)
    init_point = np.array([7,8])
    env = objfunc.make('ackley', dim=dim, init_point=init_point, 
                        window_size=window_size)

elif args.obj == 'neural':
    d, h, p, lamda = 2, 2, 2, .0005
    kwargs = {'d' : d, 'h': h, 'p' : p, 'lamda' : lamda}
    dim = h * d + h + p * h + p
    init_point = np.arange(dim) / dim

    for k in range(num_train):
        X, Y = NeuralNetDataset(dim=d, seed=k)
        env_train.append(objfunc.make('neural', dim=dim, init_point=init_point,
                                      window_size=window_size, other_params=[X, Y], **kwargs))
    for k in range(num_test):
        X, Y = NeuralNetDataset(dim=d, seed=num_train+k)
        env_test.append(objfunc.make('neural', dim=dim, init_point=init_point,
                                      window_size=window_size, other_params=[X, Y], **kwargs))


### the params
Replay_mem_size = args.replay_size
Train_batch_size = args.batch_size
Actor_Learning_rate = args.lr
Critic_Learning_rate = args.lr
Gamma = args.gamma
Tau = args.tau
Action_low = args.action_low
Action_high = args.action_high

State_dim = dim + window_size + dim * window_size
print(State_dim)

Action_dim = dim
print(Action_dim)

ounoise = OUNoise(Action_dim, 8, 3, 0.9995)
gsnoise = GaussNoise(2, 0.05, 0.99995)
noise = gsnoise if args.noise_type == 'gauss' else ounoise
            

def play(agent, num_epoch, Epoch_step, test_count, show = False):
   
    final_value = []
    for epoch in range(1):

        global env_test
        env = env_test[test_count % num_test]

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

        global env_train
        env = env_train[epoch % num_train]

        pre_state = env.reset()
        acc_reward = 0
        
        for step in range(Epoch_step):
            
            # print('pre:', pre_state)
            action = agent.action(pre_state)
            # print('action:', action)

            if action[0] != action[0]:
                raise('nan error!')

            next_state, reward, done, _ = env.step(action)
            acc_reward += reward
            # print('next:', next_state)

            if step == Epoch_step - 1:
                done = True

            # agent.train(state_featurize.transfer(pre_state), action, reward, state_featurize.transfer(next_state), done)
            agent.train(pre_state, action, reward, next_state, done)

            if done:
                #print('episode: ', epoch + 1, 'step: ', step + 1, ' reward is', acc_reward,  file = output_file)
                #print('episode: ', epoch + 1, 'step: ', step + 1, ' reward is', acc_reward, )
                print('episode: ', epoch + 1, 'step: ', step + 1, ' final value: ', env.get_value())
                break
            
            pre_state = next_state
        
        if epoch % 100 == 0 and epoch > 0:
            final_value = play(agent, 1, 20, epoch // 100)
            print('--------------episode ', epoch,  'final_value: ', final_value, '---------------', file = output_file)
            print('--------------episode ', epoch,  'final value: ', final_value, '---------------')

            obj = NeuralNet(dim, env.func.X, env.func.Y, **kwargs)

            cg_x, cg_y, _, cg_iter, _ = cg(obj, x0 = init_point, maxiter=20)
            print('CG method: optimal value: {0}, iterations {1}'.format(cg_y, cg_iter))
            sd_x, sd_y, _, sd_iter, _ = sd(obj, x0=init_point, maxiter=20)
            print('SD method: optimal value: {0}, iterations {1}'.format(sd_y, sd_iter))
            # bfgs_x, bfgs_y, _, bfgs_iter, _ = quasiNewton(Logistic(dim, X, Y), x0=init_point, maxiter=20)
            # print('bfgs method:\n optimal point: {0}, optimal value: {1}, iterations {2}'.format(bfgs_x, bfgs_y, bfgs_iter))

            if np.mean(np.array(final_value)) < -1.8:
                print('----- using ', epoch, '  epochs')
                #agent.save_model()
                break
            time.sleep(1)
         
    return agent
            

naf = NAF(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
          Gamma, Critic_Learning_rate, Action_low, Action_high, Tau, noise, False, False)  

ddpg = DDPG(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
             Gamma, Actor_Learning_rate, Critic_Learning_rate, Action_low, Action_high, Tau, noise, False) 

cac = CAC(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
             Gamma, Actor_Learning_rate, Critic_Learning_rate,  Action_low, Action_high, Tau, False)

cppo = PPO(State_dim, Action_dim,Action_low, Action_high, Replay_mem_size, Train_batch_size, Gamma, 
            Actor_Learning_rate, Critic_Learning_rate, Tau, trajectory_number=100, update_epoach=20)


if args.agent == 'naf':
    agent = train(naf, 20000, 20)
elif args.agent == 'ddpg':
    agent = train(ddpg, 20000, 20)
elif args.agent == 'cac':
    agent = train(cac, 20000, 20)
elif args.agent == 'ppo':
    agent = train(cppo, 20000, 20)

#print('after train')

#print(play(agentnaf,300, False))
#print(play(agentnaf_addloss,300, False))

