#!/usr/bin/env python
"""
Impliments Layers for SIMNN
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import numpy as np
import warnings

from simnn.initializer import initializer


class Layer(object):

    def __init__(self, out_shape, activation=None, bias=False,
                 in_shape=None, init=.1, name='Layer'):

        if out_shape:
            assert isinstance(out_shape, int), 'out_shape must be a number'
        assert isinstance(name, str), 'Name must be of type string'

        self.out_shape = out_shape
        self.activation = activation
        self.bias = bias
        self.init = init
        self.in_shape = None
        self.x = None
        self.next_layer = None
        self.prev_layer = None
        self.name = name

        if self.activation:
            self.activation.out_shape = self.out_shape

    def __repr__(self):
        rep_str = '{}, '.format(self.name)
        rep_str += 'in_shape: {}, '.format(self.in_shape)
        rep_str += 'out_shape: {}, '.format(self.out_shape)
        rep_str += 'and has bias: {}, \n'.format(self.bias)

        return rep_str

    def config(self):
        # get inshape from previous layer
        if self.in_shape is None:
            self.in_shape = self.prev_layer.out_shape

    def allocate(self):
        '''
        allocate parameters of model
        '''
        pass


class Linear(Layer):

    def __init__(self, out_shape, activation=None, bias=True,
                 in_shape=None, init=.1, name='Linear Layer'):

        super(Linear, self).__init__(out_shape, activation=activation,
                                     bias=bias, in_shape=in_shape, init=init,
                                     name='Linear Layer')

    def allocate(self):
        # allocate parameters of layer
        self.W, self.b = initializer(self)

    def fprop(self, x):
        self.x = x.copy()

        self.y = self.x.dot(self.W)

        if self.bias:
            self.y += self.b

        return self.y

    def bprop(self, p_deltas, alpha):
        # create layers deltas i.e. transform deltas using linear layer
        self.deltas = p_deltas.dot(self.W.T)

        # update weights based on deltas
        self._param_update(p_deltas, alpha)

        # return deltas
        return self.deltas

    def _param_update(self, p_deltas, alpha):
        # compute Gradient
        self.d_W = self.x.T.dot(p_deltas)  # create weight gradient
        self.d_b = np.sum(p_deltas, axis=0)  # create bias gradient

        # update weights by taking gradient step
        self.W -= alpha * self.d_W
        # update bias by taking gradient step
        self.b -= alpha * self.d_b


class PM_BN(Layer):

    '''
    "Poor Man Batch Normilization Layer" only mean centers the data
    does not actually z-score the batch
    '''

    def __init__(self, out_shape, activation=None, bias=True,
                 in_shape=None, init=.1, name='Linear Layer'):

        super(PM_BN, self).__init__(out_shape, activation=activation,
                                    bias=True, in_shape=in_shape, init=init,
                                    name='PM_BN')

        self.in_shape = out_shape

        if bias is False:
            warnings.warn('Layer assumes a bias, Bias=False overridden')

    def __repr__(self):
        rep_str = '{}, '.format(self.name)

        return rep_str

    def allocate(self):
        # allocate parameters of layer
        _, self.b = initializer(self)

    def fprop(self, x):
        self.x = x.copy()

        self.y = self.x - np.mean(self.x, axis=0)

        if self.bias:
            self.y += self.b

        return self.y

    def bprop(self, p_deltas, alpha):
        # create layers deltas i.e. transform deltas using linear layer
        self.deltas = p_deltas

        # update weights based on deltas
        self._param_update(p_deltas, alpha)

        # return deltas
        return self.deltas

    def _param_update(self, p_deltas, alpha):
        # compute Gradient
        self.d_b = np.sum(p_deltas, axis=0)  # create bias gradient

        # update bias by taking gradient step
        self.b -= alpha * self.d_b
