# -*- coding: utf-8 -*-

import numpy as np
import random
import torch
import math
import torch.nn.functional as F
from torch import nn



class ObjectiveEnvironment(object):
    """
    ObjectiveEnvironment class

    Constructor accepts a function object with callable f
    for objective value and g for gradient.
    """
    def __init__(self, func, initPoint, windowSize):
        super(ObjectiveEnvironment, self).__init__()
        self.func = func
        self.dim = func.dim
        self.initPoint = initPoint
        self.windowSize = windowSize

    def reset(self, windowSize=25):
        self.currentIterate = self.initPoint
        self.nIterate = 0

        currentValue = self.func.f(self.currentIterate)
        currentGradient = self.func.g(self.currentIterate)
            

        self.historyValue = np.zeros(self.windowSize)
        self.historyValue[0] = currentValue

        self.historyChange = np.zeros(self.windowSize)
        self.historyGradient = np.zeros(self.windowSize * self.dim)
        self.historyGradient[0:self.dim] = currentGradient

        initState = np.concatenate((self.currentIterate, self.historyChange,
                                    self.historyGradient))

        # initState = np.concatenate((self.historyChange,
        #                             self.historyGradient))
        return initState

    def get_value(self):
        return self.func.f(self.currentIterate)

    def step(self, update):
        self.nIterate += 1

        # negative_gradient = -self.func.g(self.currentIterate)
        # reward = -np.linalg.norm(update - negative_gradient, ord=2) 

        self.currentIterate = self.currentIterate + update

        currentValue = self.func.f(self.currentIterate)
        currentGradient = self.func.g(self.currentIterate)
        done = False
        # print('step:', self.currentIterate, currentValue)
        # print('step:', currentGradient)

        if (math.isinf(currentValue)):
            currentState = np.concatenate((self.currentIterate, self.historyChange,
                                           self.historyGradient))

            return currentState, -1000000, done, None

        if self.nIterate < self.windowSize:

            self.historyValue[self.nIterate] = currentValue
            self.historyGradient[self.nIterate * self.dim :
                                (self.nIterate + 1) * self.dim] = currentGradient

        else:
            self.historyValue[:-1] = self.historyValue[1:]
            self.historyValue[-1] = currentValue
            # print('cur value is:', currentValue)
            self.historyChange = currentValue - self.historyValue

            self.historyGradient[:-self.dim] = self.historyGradient[self.dim:]
            self.historyGradient[-self.dim:] = currentGradient

            if abs(self.historyChange[-2]) < 1e-8: # stopping criterion
                done = True

        # reward = currentValue

        currentState = np.concatenate((self.currentIterate, self.historyChange,
                                       self.historyGradient))

        # currentState = np.concatenate((self.historyChange,
        #                                self.historyGradient))

        return currentState, -currentValue, done, None


def make(str='quadratic', dim=3, init_point=None, window_size=25, other_params = [], **kwargs):

    if init_point is None:
        init_point = np.ones(dim)

    if str == 'quadratic':
        return ObjectiveEnvironment(Quadratic(dim), init_point, window_size)
    
    elif str == 'logistic':
        return ObjectiveEnvironment(Logistic(dim, other_params[0], other_params[1]), init_point, window_size)

    elif str == 'ackley':
        return ObjectiveEnvironment(Ackley(dim), init_point, window_size)

    elif str == 'neural':
        return ObjectiveEnvironment(NeuralNet(dim, other_params[0], other_params[1], **kwargs), init_point, window_size)


class Quadratic(object):
    """docstring for Quadratic"""
    def __init__(self, dim):
        super(Quadratic, self).__init__()
        self.dim = dim

    def f(self, x):
        # x_torch = torch.tensor(x, dtype = torch.float, requires_grad = True)
        # val = torch.sum(x_torch ** 2)
        # val.backward()
        # self.grad = x_torch.grad
        # print('val item: ',val.item())
        # return val.item()
        return np.dot(x, x)

    def g(self, x):
        # print('grad: ', self.grad.data.numpy())
        # return self.grad.data.numpy()
        return 2 * x


class Logistic(object):
    """ doc for Logistic """
    def __init__(self, dim, X, Y, lbd = 5e-4):
        self.X = torch.tensor(X, dtype = torch.double)
        self.Y = torch.tensor(Y, dtype = torch.double)
        self.dim = dim
        self.lbd = lbd

    def f(self, W):
        W_torch = torch.tensor(W, dtype = torch.double, requires_grad = True)
        val = - torch.mean(self.Y * torch.log(1 / (1 + (torch.exp(-torch.matmul(self.X, W_torch)))) + 1e-10) \
            + (1 - self.Y) * torch.log( 1e-10 + 1 - (1 / (1 + torch.exp(-torch.matmul(self.X, W_torch)))))) \
            + 0.5 * self.lbd * torch.sum(W_torch * W_torch)
        # val = - torch.mean(self.Y * (1 / (1 + (torch.exp(-torch.matmul(self.X, W_torch))))) \
        #     + (1 - self.Y) * (1 - (1 / (1 + torch.exp(-torch.matmul(self.X, W_torch)))))) \
        #     + 0.5 * self.lbd * torch.sum(W_torch * W_torch)
        return val.item()

    def g(self, W):
        W_torch = torch.tensor(W, dtype = torch.double, requires_grad = True)
        val = - torch.mean(self.Y * torch.log(1e-10 + 1 / (1 + (torch.exp(-torch.matmul(self.X, W_torch))))) \
            + (1 - self.Y) * torch.log(1e-10 + 1 - (1 / (1 + torch.exp(-torch.matmul(self.X, W_torch)))))) \
            + 0.5 * self.lbd * torch.sum(W_torch * W_torch)
        # val = - torch.mean(self.Y * (1 / (1 + (torch.exp(-torch.matmul(self.X, W_torch))))) \
        #     + (1 - self.Y) * (1 - (1 / (1 + torch.exp(-torch.matmul(self.X, W_torch)))))) \
        #     + 0.5 * self.lbd * torch.sum(W_torch * W_torch)
        val.backward()
        return W_torch.grad.data.numpy()


class Ackley(object):
    """doc for Ackley function"""
    def __init__(self, dim = 2):
        self.dim = dim

    def f(self, x):
        x_ = x[0]
        y_ = x[1]
        val = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x_ ** 2 + y_ ** 2))) \
           - np.exp(0.5 * (np.cos(2 * math.pi * x_) + np.cos(2 * y_ * math.pi))) + np.exp(1) + 20
        return val

    def g(self, x):
        X_torch = torch.tensor(x, dtype = torch.float, requires_grad = True)
        x = torch.sum(X_torch * torch.tensor([1,0], dtype = torch.float))
        y = torch.sum(X_torch * torch.tensor([0,1], dtype = torch.float))
        val = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * torch.sum(X_torch ** 2))) - \
           torch.exp(0.5 * (torch.cos(2 * math.pi * x) + torch.cos(2 * math.pi * y))) + np.exp(1) + 20
        val.backward()
        return X_torch.grad.data.numpy()


class NeuralNet(object):
    """docstring for NeuralNet"""
    def __init__(self, dim, X, Y, d=2, h=2, p=2, lamda=.0005):
        super(NeuralNet, self).__init__()
        self.dim = dim
        self.X = torch.tensor(X, dtype=torch.double)
        self.Y = torch.tensor(Y, dtype=torch.long)
        self.arch = {'d' : d, 'h' : h, 'p' : p, 'lamda' : lamda} # network architecture parameters

    def f(self, param):
        param = torch.tensor(param, dtype=torch.double, requires_grad=True)
        d, h, p, lamda = self.arch['d'], self.arch['h'], self.arch['p'], self.arch['lamda']

        W = param[0 : h*d].view(h, d)
        b = param[h*d : h*d+h]
        U = param[h*d+h : h*d+h+p*h].view(p, h)
        c = param[h*d+h+p*h : ]

        X, Y = self.X, self.Y
        fc1 = F.relu(torch.matmul(X, W) + b)
        
        temp = torch.matmul(fc1, U) + c
        temp = temp - torch.mean(temp)
        fc2 = torch.exp(temp)

        numerator = fc2.gather(1, Y.view(-1, 1)).squeeze()
        val = -torch.mean(torch.log(1e-6 + numerator / torch.sum(fc2, dim=1)))
        reg = (lamda / 2) * (torch.norm(W) ** 2 + torch.norm(U) ** 2)
        loss = val + reg

        return loss.item()

    def g(self, param):
        param = torch.tensor(param, dtype=torch.double, requires_grad=True)
        d, h, p, lamda = self.arch['d'], self.arch['h'], self.arch['p'], self.arch['lamda']

        W = param[0 : h*d].view(h, d)
        b = param[h*d : h*d+h]
        U = param[h*d+h : h*d+h+p*h].view(p, h)
        c = param[h*d+h+p*h : ]

        X, Y = self.X, self.Y
        fc1 = F.relu(torch.matmul(X, W) + b)

        temp = torch.matmul(fc1, U) + c
        temp = temp - torch.mean(temp)
        fc2 = torch.exp(temp)

        numerator = fc2.gather(1, Y.view(-1, 1)).squeeze()
        val = -torch.mean(torch.log(1e-6 + numerator / torch.sum(fc2, dim=1)))
        # print('val:', val)
        reg = (lamda / 2) * (torch.norm(W) ** 2 + torch.norm(U) ** 2)
        loss = val + reg
        # print('loss:', loss.data)

        loss.backward()
        # print('grad:', param.grad)
        return param.grad.data.numpy()
