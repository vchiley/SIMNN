#!/usr/bin/env python
"""
Implements optimizers for SIMNN
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import numpy as np


class Optimizer(object):
    '''
    Base Optimizer Class
    '''

    def __init__(self, name='optimizer'):
        assert isinstance(name, str), 'Name must be of type string'
        self.name = name
        self.grads = []

    def get_gradients(self, layers):
        '''
        Method to get the gradients of each layer

        :param layers: list of layers in the model
                       expected to be in the same order every time
        :type layers: list
        '''
        self.grads = []
        for layer in layers:
            self.grads += layer.param_grads

    def apply_gradients(self, alpha):
        '''
        Method to apply the gradients to the parameters.

        :param alpha: learning rate
        :type alpha: Number
        '''
        raise NotImplementedError

    def minimize(self, layers, alpha):
        self.get_gradients(layers)
        self.apply_gradients(alpha)


class StochasticGradientDecent(Optimizer):
    '''
    Stochastic gradient decent optimizer
    '''

    def __init__(self, name='sdg'):
        super(StochasticGradientDecent, self).__init__(name)

    def apply_gradients(self, alpha):
        '''
        Method to apply the gradients to the parameters.

        :param alpha: learning rate
        :type alpha: Number
        '''
        for var, grad in self.grads:
            var -= alpha * grad

            grad = 0


class MomentumOptimizer(Optimizer):
    '''
    Stochastic gradient decent optimizer with momentum

    :param momentum: momentum of optimizer
    :type momentum: Number
    :param nesterov: use nesterov momentum or regular momentum
    :type nesterov: Boolean
    :param name: optimizer name
    :type name: str
    '''

    def __init__(self, momentum=.9, nesterov=False, name='sdg'):
        super(MomentumOptimizer, self).__init__(name)
        self.m = momentum
        self.param_momentum = []
        if nesterov:
            self.param_momentum_prev = []

        self.nesterov = nesterov

    def get_gradients(self, layers):
        '''
        Method to get the gradients of each layer
        Method also builds the momentum tensors for each layers parameters

        :param layers: list of layers in the model
                       expected to be in the same order every time
        :type layers: list
        '''
        # build momentum tensors for each layers parameters
        if not self.param_momentum:
            for layer in layers:
                for param in layer.param_grads:
                    d_param = param[1]
                    self.param_momentum += [np.zeros(d_param.shape)]

                    if self.nesterov:
                        self.param_momentum_prev += [np.zeros(d_param.shape)]

        super(MomentumOptimizer, self).get_gradients(layers)

    def apply_gradients(self, alpha):
        '''
        Method to apply the gradients to the parameters.

        :param alpha: learning rate
        :type alpha: Number
        '''
        for i, ((var, grad), v) in enumerate(zip(self.grads, self.param_momentum)):
            if self.nesterov:
                v_prev = self.param_momentum_prev[i]
                v_prev = v
                v = self.m * v - alpha * grad
                var += -self.m * v_prev + (1 + self.m) * v

            else:
                v = self.m * v - alpha * grad
                var += v

            grad = 0

