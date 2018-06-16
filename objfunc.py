# -*- coding: utf-8 -*-

import numpy as np
import random


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
        return initState


    def step(self, update):
        self.nIterate += 1
        self.currentIterate = self.currentIterate + update

        currentValue = self.func.f(self.currentIterate)
        currentGradient = self.func.g(self.currentIterate)
        done = False
        print(self.currentIterate, currentValue)

        if self.nIterate < self.windowSize:

            self.historyValue[self.nIterate] = currentValue
            self.historyGradient[self.nIterate * self.dim :
                                (self.nIterate + 1) * self.dim] = currentGradient

        else:
            self.historyValue[:-1] = self.historyValue[1:]
            self.historyValue[-1] = currentValue
            self.historyChange = currentValue - self.historyValue

            self.historyGradient[:-self.dim] = self.historyGradient[self.dim:]
            self.historyGradient[-self.dim:] = currentGradient

            if abs(self.historyChange[-2]) < 1e-8: # stopping criterion
                done = True

        currentState = np.concatenate((self.currentIterate, self.historyChange,
                                       self.historyGradient))

        return currentState, -currentValue, done, None


def make(str='quadratic', dim=3, init_point=None, window_size=25):

    if init_point is None:
        init_point = np.zeros(dim)

    if str == 'quadratic':
        return ObjectiveEnvironment(Quadratic(dim), init_point, window_size)



class Quadratic(object):
    """docstring for Quadratic"""
    def __init__(self, dim):
        super(Quadratic, self).__init__()
        self.dim = dim

    def f(self, x):
        return np.dot(x, x)

    def g(self, x):
        return 2 * x


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
