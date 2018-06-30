# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 16:43:56 2018

@author: Wangyf
"""

import sklearn.pipeline
import sklearn.preprocessing
import numpy as np
import random
import matplotlib.pyplot as plt
import objfunc as obj

from sklearn.kernel_approximation import RBFSampler
from collections import deque
from dataset import LogisticDataset, NeuralNetDataset


obj_list = ['ackley', 'logistic', 'neural']
init_list = ['[7 8]', '[2.33 6.66]', '[ 0.01 10.  ]', '[0.         0.08333333 0.16666667 0.25       0.33333333 0.41666667? 0.5        0.58333333 0.66666667 0.75       0.83333333 0.91666667]']

obj_idx = 1
init_idx = 3

filelist = ["./data/sd_%s3.txt"%(obj_list[obj_idx]),
            "./data/mom_%s3.txt"%(obj_list[obj_idx]),
            "./data/cg_%s3.txt"%(obj_list[obj_idx]),
            "./data/bfgs_%s3.txt"%(obj_list[obj_idx]),
            "./data/general_ddpg_logistic3.txt"]

fig = 'value'

dim = 1
X, Y = LogisticDataset(dim=dim, seed=23)

dim += 1
init_point = np.arange(dim) / dim
fun = obj.Logistic(dim, X, Y)

def dataprocessing(filename):
    avgnum = 7
    linecnt = 0
    linetot = avgnum * 2
    n_iter = 50

    mean = np.zeros(n_iter + 1)
    std = np.zeros(n_iter + 1)
    mean[0] = fun.f(init_point)
    # traj = []

    if 'sd' in filename or 'cg' in filename or 'bfgs' in filename or 'mom' in filename:
        with open(filename, 'r') as file:
            for line in file:
                linecnt += 1
                if linecnt % 2 == 1:
                    mean[1:] = np.array(line.split(' '), dtype=float)
                else:
                    traj = np.array(line.replace(',', ' ').split(' ')[:-1], dtype=float)
                    traj = traj.reshape((-1, dim)).T

        return mean, std, traj

    else:
        with open(filename, 'r') as file:
            for line in file:
                linecnt += 1
                if linecnt <= linetot - avgnum * 2:
                    continue

                if linecnt % 2 == 1:
                    value = np.array(line.split(' '), dtype=float)
                    mean[1:] = mean[1:] + value
                    std[1:] = std[1:] + value ** 2

                else:
                    traj = np.array(line.replace(',', ' ').split(' ')[:-1], dtype=float)
                    traj = traj.reshape((-1, dim)).T

        mean[1:] = mean[1:] / avgnum
        std[1:] = std[1:] / avgnum - mean[1:] ** 2
        std[np.abs(std) < 1e-8] = 0
        std = np.sqrt(std)
        # print('mean = ', mean, 'std = ', std, 'traj = ', traj)
        return mean, std, traj


def plotvalue(filelist=None):
    if type(filelist) == list:
        values = []
        for file in filelist:
            values.append(dataprocessing(file))

    if filelist is None:
        values = np.array([np.sin(np.linspace(0, np.pi, 50)),
                           np.cos(np.linspace(0, np.pi, 50)),
                           np.sqrt(np.linspace(0, np.pi, 50))])

    color = ['blue', 'brown', "red", "indigo", "green", "green"]
    label = ['Gradient descent', 'Momentum', 'Conjugate gradient',
             'L-BFGS algorithm', 'LTO with OU']

    for k in range(len(label)):
        n_iter = len(values[k][0])
        x = np.arange(0, n_iter)

        mean, std = values[k][0], values[k][1]
        y_lower = mean - std * .5
        y_upper = mean + std * .5
        plt.plot(x, mean, label=label[k], color=color[k])
        if k >= 4:
            plt.fill_between(x, y_lower, y_upper, color=color[k], alpha=.35, linewidth=0)

    plt.xlabel('Iteration')
    plt.ylabel('Objective value')

    legend = plt.legend(loc='upper right', shadow=True)#, fontsize='large')

    plt.xticks(range(0, n_iter + 1, n_iter // 10))
    # plt.show()
    plt.savefig('./fig/logistic3_val.png')


def plottraj(filelist=None):
    if type(filelist) == list:
        values = []
        for file in filelist:
            values.append(dataprocessing(file))

    if filelist is None:
        theta = np.arange(0, 2 * np.pi, np.pi / 50)
        values = np.array([np.cos(theta), np.sin(theta)])

    color = ['blue', 'brown', "red", "indigo", "green", "green"]
    label = ['Gradient descent', 'Momentum', 'Conjugate gradient',
             'L-BFGS algorithm', 'LTO with OU']

    for k in range(len(label)):
        n_iter = 50
        traj = values[k][2]
        x = traj[0, :-1]
        y = traj[1, :-1]
        # print(traj.shape)

        if k == 0:
            # Contour plot
            X = np.linspace(-1.8, 3.8, n_iter)
            Y = np.linspace(-4.8, 0.8, n_iter)
            X, Y = np.meshgrid(X, Y)

            Z = np.zeros((n_iter, n_iter))
            for i in range(n_iter):
                for j in range(n_iter):
                    Z[i][j] = fun.f(np.array([X[i][j], Y[i][j]]))

            plt.contourf(X, Y, Z, 50, cmap='GnBu')
            plt.colorbar()

        # Trajectory plot
        u = traj[0, 1:] - traj[0, :-1]
        v = traj[1, 1:] - traj[1, :-1]
        # print(traj)
        # print(u, v)

        q = plt.quiver(x, y, u, v, units='xy', scale=1, color=color[k], headwidth=2)
        plt.quiverkey(q, X = .13 + .52 * (k % 2), Y = 1.13 - .05 * (k // 2),
                         U = .6, label=label[k], labelpos='E')

    # Show plot
    plt.axis('equal')
    # plt.xticks(np.arange(np.min(x) - 1, np.max(x) + 1))
    # plt.yticks(np.arange(np.min(y) - 1, np.max(y) + 1))
    plt.show()
    # plt.savefig('./fig/logistic3_traj.png')


class Featurize_state():
    def __init__(self, env, no_change = False):
        self.no_change = no_change
        if no_change == True:
            self.After_featurize_state_dim = env.observation_space.shape[0]
            return 
        
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        self.featurizer.fit(observation_examples)
        self.After_featurize_state_dim = 400
        
    def get_featurized_state_dim(self):
        return self.After_featurize_state_dim
        
    def transfer(self, state):
        if self.no_change:
            return state
        
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]
        #return state
        


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    def __init__(self, action_dimension, initial_scale = 1, final_scale = 0.2, decay = 0.9995, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = initial_scale
        self.final_scale = final_scale
        self.decay = decay
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu

    def decaynoise(self):
        self.scale *= self.decay
        self.scale = max(self.scale, self.final_scale)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        res = self.state * self.scale
        return res[0]
    
    def noisescale(self):
        return self.scale
    
class GaussNoise():
    def __init__(self, initial_var = 10, final_var = 0, decay = 0.995):
        self.var = initial_var
        self.final_var = final_var
        self.decay = decay
        
    def decaynoise(self):
        self.var *= self.decay
        self.var = max(self.final_var, self.var)
        
    def noise(self, var = None):
        return np.random.normal(0, self.var) if var is None else np.random.normal(0, var) 
    
    def noisescale(self):
        return self.var


class SlidingMemory():
    
    def __init__(self, mem_size):
        self.mem = deque()
        self.mem_size = mem_size
        
    def add(self, state, action, reward, next_state, if_end):
        self.mem.append([state, action, reward, next_state, if_end])
        if len(self.mem) > self.mem_size:
            self.mem.popleft()
            
    def num(self):
        return len(self.mem)
    
    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)
    
    def clear(self):
        self.mem.clear()




class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )
        self.number = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        
        if idx >= self.capacity - 1:
            return idx
        
        left = 2 * idx + 1
        right = left + 1

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, data, p):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        self.number = min(self.number + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        
        return (idx, self.tree[idx], self.data[dataIdx])
    
    def num(self):
        return self.number

    

class PERMemory():
    
    def __init__(self, mem_size, alpha = 0.8, beta = 0.8, eps = 1e-2):
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.mem_size = mem_size
        self.mem = SumTree(mem_size)
        
    def add(self, state, action, reward, next_state, if_end):
        # here use reward for initial p, instead of maximum for initial p
        p = 1000
        self.mem.add([state, action, reward, next_state, if_end], p)
        
    def update(self, batch_idx, batch_td_error):
        for idx, error in zip(batch_idx, batch_td_error):
            p = (error + self.eps)  ** self.alpha 
            self.mem.update(idx, p)
        
    def num(self):
        return self.mem.num()
    
    def sample(self, batch_size):
        
        data_batch = []
        idx_batch = []
        p_batch = []
        
        segment = self.mem.total() / batch_size
        #print(self.mem.total())
        #print(segment * batch_size)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            #print(s < self.mem.total())
            idx, p, data = self.mem.get(s)
            data_batch.append(data)
            idx_batch.append(idx)
            p_batch.append(p)
        
        p_batch = (1.0/ np.array(p_batch) /self.mem_size) ** self.beta
        p_batch /= max(p_batch)
        
        self.beta = min(self.beta * 1.00005, 1)
    
        return (data_batch, idx_batch, p_batch)
        
if __name__ == '__main__':      
    if fig == 'value':
        plotvalue(filelist)
    else:
        plottraj(filelist)
