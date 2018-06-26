# -*- coding: utf-8 -*-

import numpy as np
import random
import torch
import math


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
        self.currentIterate = self.currentIterate + update

        currentValue = self.func.f(self.currentIterate)
        currentGradient = self.func.g(self.currentIterate)
        done = False
        #print(self.currentIterate, currentValue)

        if(math.isinf(currentValue)):
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

        currentState = np.concatenate((self.currentIterate, self.historyChange,
                                       self.historyGradient))

        # currentState = np.concatenate((self.historyChange,
        #                                self.historyGradient))

        return currentState, -currentValue, done, None


def make(str='quadratic', dim=3, init_point=None, window_size=25, other_params = []):

    if init_point is None:
        init_point = np.zeros(dim)

    if str == 'quadratic':
        return ObjectiveEnvironment(Quadratic(dim), init_point, window_size)
    
    if str == 'logistic':
        return ObjectiveEnvironment(Logistic(dim, other_params[0], other_params[1]), init_point, window_size)


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


class Logistic():
    """ doc for Logistic """
    def __init__(self, dim,  X, Y, lbd = 5e-4):
        self.X = torch.tensor(X, dtype = torch.float)
        self.Y = torch.tensor(Y, dtype = torch.float)
        self.dim = dim
        self.lbd = lbd

    def f(self, W):
        W_torch = torch.tensor(W, dtype = torch.float, requires_grad = True)
        # val = - torch.mean(self.Y * torch.log(1 / (1 + (torch.exp(-torch.matmul(self.X, W_torch)))) + 1e-10) \
        #     + (1 - self.Y) * torch.log( 1e-10 + 1 - (1 / (1 + torch.exp(-torch.matmul(self.X, W_torch)))))) \
        #     + 0.5 * self.lbd * torch.sum(W_torch * W_torch)
        val = - torch.mean(self.Y * (1 / (1 + (torch.exp(-torch.matmul(self.X, W_torch))))) \
            + (1 - self.Y) * (1 - (1 / (1 + torch.exp(-torch.matmul(self.X, W_torch)))))) \
            + 0.5 * self.lbd * torch.sum(W_torch * W_torch)
        return val.item()

    def g(self,W):
        W_torch = torch.tensor(W, dtype = torch.float, requires_grad = True)
        # val = - torch.mean(self.Y * torch.log(1e-10 + 1 / (1 + (torch.exp(-torch.matmul(self.X, W_torch))))) \
        #     + (1 - self.Y) * torch.log(1e-10 + 1 - (1 / (1 + torch.exp(-torch.matmul(self.X, W_torch)))))) \
        #     + 0.5 * self.lbd * torch.sum(W_torch * W_torch)
        val = - torch.mean(self.Y * (1 / (1 + (torch.exp(-torch.matmul(self.X, W_torch))))) \
            + (1 - self.Y) * (1 - (1 / (1 + torch.exp(-torch.matmul(self.X, W_torch)))))) \
            + 0.5 * self.lbd * torch.sum(W_torch * W_torch)
        val.backward()
        return W_torch.grad.data.numpy()



'''
class Logistic(object):
    """docstring for Logistic"""
    def __init__(self, dim, seed):
        super(Logistic, self).__init__()
        self.dim = dim
        self.seed = seed
        self.nInstance = 100

    def generateData():
        gaussian = 


        self.x = []
        self.y = []

        for k in range(self.nInstance):
            s

    def f(x):



    def g(x):
'''
