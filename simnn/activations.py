#!/usr/bin/env python
"""
Implements activation functions for SIMNN
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import warnings
import numpy as np

from numbers import Number

from simnn.layers import Layer


class Activation(Layer):
    '''
    Base class for Activation objects

    :param out_shape: output shape
    :type out_shape: int
    :param in_shape: define input data shape
    :type in_shape: int
    :param name: layer name
    :type name: str
    '''

    def __init__(self, out_shape, in_shape, name):
        super(Activation, self).__init__(out_shape=out_shape,
                                         in_shape=out_shape, name=name)


class ReLU(Activation):
    '''
    Rectified Linear Unit Activation

    :param name: layer name
    :type name: str
    '''

    def __init__(self, name='ReLU'):

        super(ReLU, self).__init__(out_shape=None, in_shape=None,
                                   name=name)
        self.use_y = True

    def __repr__(self):
        rep_str = '{}\n'.format(self.name)
        return rep_str

    def fprop(self, x):
        '''
        fprop through the activation

        :param x: layer input
        :type x: np.ndarray
        '''
        assert isinstance(x, np.ndarray), 'must be a numpy vector'
        self.x = x.copy()

        self.y = x.copy()
        self.y[np.where(self.y < 0)] = 0

        return self.y

    def bprop(self, deltas, use_y=False):
        '''
        bprop through the activation

        :param deltas: propagating errors coming back
        :type deltas: np.ndarray
        :param use_y: defines if layer out should be used to get bprop
        :type use_y: bool
        '''
        if self.use_y:
            return np.sign(self.y).astype(np.float) * deltas

        x = self.x.copy()

        x[np.where(x < 0)] = 0

        return np.sign(x).astype(np.float) * deltas


class LogisticSigmoid(Activation):
    '''
    Logistic Sigmoid Activation

    Should be the last layer of a network with binary cross entropy cost

    :param name: layer name
    :type name: str
    '''

    def __init__(self, name='Sigmoid'):

        super(LogisticSigmoid, self).__init__(out_shape=None, in_shape=None,
                                               name=name)

        self.shortcut = False
        self.use_y = True

    def __repr__(self):
        rep_str = '{}, shortcut: {}\n'.format(self.name, self.shortcut)
        return rep_str

    def _fprop(self, x):
        '''
        Allows us to just use the function without caching the outputs
        '''
        return 1 / (1 + np.exp(-x))

    def fprop(self, x):
        '''
        fprop through the activation

        :param x: layer input
        :type x: np.ndarray
        '''
        self.x = x.copy()

        self.y = self._fprop(x)

        return self.y

    def bprop(self, deltas, use_y=None):
        '''
        bprop through the activation

        :param deltas: propagating errors coming back
        :type deltas: np.ndarray
        :param use_y: defines if layer out should be used to get bprop
        :type use_y: bool
        '''
        # if this is last layers activation with bin_cross_entropy loss
        if self.shortcut:
            return 1 * deltas

        if self.use_y:  # use y to compute dir, faster calculation
            return self.y * (1 - self.y) * deltas

        return self._fprop(self.x) * self._fprop(-self.x) * deltas


class Softmax(Activation):
    '''
    Softmax Activation

    Should be the last layer of a network with cross entropy cost

    :param name: layer name
    :type name: str
    '''

    def __init__(self, name='Softmax', n_stable=True):

        super(Softmax, self).__init__(out_shape=None, in_shape=None,
                                      name=name)
        self.n_stable = n_stable
        self.shortcut = False
        self.use_y = True

    def __repr__(self):
        rep_str = '{}, shortcut: {}, n_stable: {}\n'.format(self.name,
                                                            self.shortcut,
                                                            self.n_stable)
        return rep_str

    def _fprop(self, x):
        '''
        Allows us to just use the function without caching the outputs
        '''
        if self.n_stable:  # neumerically stable, but slightly slower
            x = x - np.max(x, axis=1)[:, np.newaxis]

        out = np.exp(x)

        return out / np.sum(out, axis=1)[:, np.newaxis]

    def fprop(self, x):
        '''
        fprop through the activation

        Numerically stable softmax function

        :param x: layer input
        :type x: np.ndarray
        '''
        assert isinstance(x, np.ndarray), 'must be a numpy vector'

        self.x = x.copy()

        self.y = self._fprop(x)

        return self.y

    def bprop(self, deltas, use_y=False):
        '''
        bprop through the activation

        :param deltas: propagating errors coming back
        :type deltas: np.ndarray
        :param use_y: defines if layer out should be used to get bprop
        :type use_y: bool
        '''
        if self.shortcut:
            return 1 * deltas
        if self.use_y:  # use y to compute dir, faster calculation
            return self.y * (1 - self.y) * deltas

        return self._fprop(self.x) * self._fprop(-self.x) * deltas


class Line(Activation):
    def __init__(self, a=1, b=0, name='Line'):

        super(Line, self).__init__(out_shape=None, in_shape=None,
                                   name=name)

        assert isinstance(a, Number), 'a must be a Number'
        assert isinstance(b, Number), 'b must be a Number'
        warnings.warn('Line is not a non-linearity', UserWarning)
        self.a = a
        self.b = b

    def __repr__(self):
        rep_str = '{}, '.format(self.name)
        rep_str = '{}X + {}\n, '.format(self.a, self.b)
        return rep_str

    def fprop(self, x):
        '''
        :param: x
        :type: np.ndarray
        '''
        assert isinstance(x, np.ndarray), 'must be a numpy vector'

        self.x = x.copy()

        self.y = self.a * x + self.b

        return self.y

    def bprop(self, deltas, use_y=False):
        return self.a * deltas


class Identity(Activation):
    def __init__(self, name='Identity'):
        super(Identity, self).__init__(name=name)
        warnings.warn('Identity is not a non-linearity', UserWarning)

    def __repr__(self):
        rep_str = '{}\n'.format(self.name)
        return rep_str
