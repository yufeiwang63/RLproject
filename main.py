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
argparser.add_argument('--batch_size', type=int, default=128)
argparser.add_argument('--gamma', type=float, default=1)
argparser.add_argument('--tau', type=float, default=0.1)
argparser.add_argument('--noise_type', type = str, default = 'ounoise')
argparser.add_argument('--agent', type = str, default = 'ddpg')
argparser.add_argument('--action_low', type = float, default = -0.3)
argparser.add_argument('--action_high', type = float, default = 0.3)
argparser.add_argument('--dim', type = int, default = 3)
argparser.add_argument('--window_size', type = int, default = 10)
argparser.add_argument('--obj', type = str, default = 'quadratic')
argparser.add_argument('--sigma', type = float, default = 1)
argparser.add_argument('--debug', type = str, default = sys.argv[1:])
argparser.add_argument('--max_iter', type = int, default = 50)
argparser.add_argument('--max_epoch', type = int, default = 30000)
args = argparser.parse_args()

### the env
dim = args.dim
window_size = args.window_size
init_point = np.arange(dim) / (dim / 2)

if args.obj == 'quadratic':
    env = objfunc.make('quadratic', dim=dim, init_point=init_point ,
                                    window_size=window_size)

elif args.obj == 'logistic':
    X, Y = LogisticDataset(dim=dim)
    dim += 1
    init_point = np.arange(dim) / dim
    env = objfunc.make('logistic', dim=dim, init_point=init_point, 
                        window_size=window_size, other_params=[X, Y])

elif args.obj == 'ackley':
    init_point = np.array([7,8])
    env = objfunc.make('ackley', dim=dim, init_point=init_point, 
                        window_size=window_size)

elif args.obj == 'neural':
    d, h, p, lamda = 2, 2, 2, .0005
    kwargs = {'d' : d, 'h': h, 'p' : p, 'lamda' : lamda}
    dim = h * d + h + p * h + p
    init_point = np.arange(dim) / dim

    X, Y = NeuralNetDataset(dim=d)
    env = objfunc.make('neural', dim=dim, init_point=init_point,
                        window_size=window_size, other_params=[X, Y], **kwargs)

### the params
Replay_mem_size = args.replay_size
Train_batch_size = args.batch_size
Actor_Learning_rate = args.lr
Critic_Learning_rate = args.lr
Gamma = args.gamma
Tau = args.tau
Action_low = args.action_low
Action_high = args.action_high
max_iter = args.max_iter
max_epoch = args.max_epoch

State_dim = dim + window_size + dim * window_size
print(State_dim)

Action_dim = dim
print(Action_dim)

ounoise = OUNoise(Action_dim, 8, 2, 0.9995)
gsnoise = GaussNoise(2, 0.1, 0.99995)
noise = gsnoise if args.noise_type == 'gauss' else ounoise

# record the test objective values of RL algorithms    
# RL_value = np.zeros((max_epoch, max_iter))
log_file = open('./' + str(args.agent) + '_' + str(args.obj) + '_' + str(init_point) + '_' + str(args.debug) + '.txt', 'w')

def play(agent, test_count, Epoch_step, show = False):
   
    print('debug info: ', args.debug)
    global log_file

    # record the value and point at each iteration
    val_record = []
    point_record = []
    for epoch in range(1):
        pre_state = env.reset()

        for step in range(Epoch_step):
            if show:
                env.render()
            
            # action = agent.action(state_featurize.transfer(pre_state), False)
            action = agent.action(pre_state, False)

            next_state, reward, done, _ = env.step(action)
            val_record.append(-reward)
            point_record.append(next_state[:dim])

            if done or step == Epoch_step - 1:
                final_val = env.get_value()
                break
            pre_state = next_state

    for item in val_record:
        print('%.5f' % item, end =' , ')
    print()
    print(' '.join(map(str, val_record)), file = log_file, end = '\n')
    for point in point_record:
        print(' '.join(map(str, point)), file = log_file, end = ',')
    log_file.write('\n')
    log_file.flush()

    return final_val


def train(agent, Train_epoch, Epoch_step):        
    for epoch in range(Train_epoch):
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
                print('episode: ', epoch + 1, 'step: ', step + 1, ' final value: ', env.get_value())
                break
            
            pre_state = next_state
        
        if epoch % 100 == 0 and epoch > 0:
            final_value = play(agent, epoch // 100, max_iter, not True)
            print('--------------episode ', epoch,  'final value: ', final_value, '---------------')
            if np.mean(np.array(final_value)) < -1.8:
                print('----- using ', epoch, '  epochs')
                #agent.save_model()
                break
            time.sleep(1)
         
    return agent
            

naf = NAF(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
          Gamma, Critic_Learning_rate, Action_low, Action_high, Tau, noise, flag = False, if_PER = False)  

ddpg = DDPG(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
             Gamma, Actor_Learning_rate, Critic_Learning_rate, Action_low, Action_high, Tau, noise, if_PER=False) 

cac = CAC(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
             Gamma, Actor_Learning_rate, Critic_Learning_rate,  Action_low, Action_high, Tau, sigma=args.sigma, if_PER=False)

cppo = PPO(State_dim, Action_dim,Action_low, Action_high, Replay_mem_size, Train_batch_size, Gamma, 
            Actor_Learning_rate, Critic_Learning_rate, Tau, trajectory_number=100, update_epoach=50)


if args.obj == 'quadratic':
    obj = Quadratic(dim)
elif args.obj == 'logistic':
    obj = Logistic(dim, X, Y)
elif args.obj == 'ackley':
    obj = Ackley(dim)
elif args.obj == 'neural':
    obj = NeuralNet(dim, X, Y, **kwargs)


cg_x, cg_y, _, cg_iter, _ = cg(obj, x0 = init_point, maxiter=max_iter)
print('CG method:\n optimal point: {0}, optimal value: {1}, iterations {2}'.format(cg_x, cg_y, cg_iter))
sd_x, sd_y, _, sd_iter, _ = sd(obj, x0=init_point, maxiter=max_iter)
print('SD method:\n optimal point: {0}, optimal value: {1}, iterations {2}'.format(sd_x, sd_y, sd_iter))
bfgs_x, bfgs_y, _, bfgs_iter, _ = quasiNewton(obj, x0=init_point, maxiter=max_iter)
print('bfgs method:\n optimal point: {0}, optimal value: {1}, iterations {2}'.format(bfgs_x, bfgs_y, bfgs_iter))


if args.agent == 'naf':
    agent = train(naf, max_epoch, max_iter)
elif args.agent == 'ddpg':
    agent = train(ddpg, max_epoch, max_iter)
elif args.agent == 'cac':
    agent = train(cac, max_epoch, max_iter)
elif args.agent == 'ppo':
    agent = train(cppo, max_epoch, max_iter)

log_file.close()

#print('after train')

#print(play(agentnaf,300, False))
#print(play(agentnaf_addloss,300, False))

