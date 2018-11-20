#!/usr/bin/env python
"""
Impliments Layers for SIMNN
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import numpy as np

from simnn.initializer import initializer


class Layer(object):
    '''
    Base Layer object

    :param out_shape: output shape
    :type out_shape: int
    :param activation: activation type
    :type activation: Activation
    :param bias: boolean defining bias inclusion
    :type bias: bool
    :param in_shape: define input data shape
    :type in_shape: int
    :param init: parameter initialization method, std.dev. or norm or method
    :type init: str, Number
    :param name: layer name
    :type name: str
    '''

    def __init__(self, out_shape, activation=None, bias=False,
                 in_shape=None, init=.1, name='Layer', dtype=np.float32):

        if out_shape:
            assert isinstance(out_shape, int), 'out_shape must be a number'
        assert isinstance(name, str), 'Name must be of type string'

        self.out_shape = out_shape
        self.activation = activation
        self.bias = bias
        self.init = init
        self.dtype = dtype
        self.in_shape = in_shape
        self.x = None
        self.next_layer = None
        self.prev_layer = None
        self.name = name
        self.param_grads = []

        if self.activation:
            self.activation.out_shape = self.out_shape

    def __repr__(self):
        rep_str = '{}, '.format(self.name)
        rep_str += 'in_shape: {}, '.format(self.in_shape)
        rep_str += 'out_shape: {}, '.format(self.out_shape)
        rep_str += 'and has bias: {}, \n'.format(self.bias)

        return rep_str

    def config(self):
        '''
        configure layer
        '''
        # get inshape from previous layer
        if self.in_shape is None:
            self.in_shape = self.prev_layer.out_shape

    def allocate(self):
        '''
        allocate parameters of layer
        '''
        pass


class Linear(Layer):
    '''
    Linear (actually Affine) Layer object

    :param out_shape: output shape
    :type out_shape: int
    :param activation: activation type
    :type activation: Activation
    :param bias: boolean defining bias inclusion
    :type bias: bool
    :param in_shape: define input data shape
    :type in_shape: int
    :param init: parameter initialization method, std.dev. or norm or method
    :type init: str, Number
    :param name: layer name
    :type name: str
    '''

    def __init__(self, out_shape, activation=None, bias=True,
                 in_shape=None, init=.1, name='Linear Layer', dtype=np.float32):

        super(Linear, self).__init__(out_shape, activation=activation,
                                     bias=bias, in_shape=in_shape, init=init,
                                     name='Linear Layer', dtype=dtype)

    def allocate(self):
        '''
        allocate layer parameters
        '''
        self.W = initializer(self, dtype=self.dtype)
        if self.bias:
            self.b = np.zeros(self.out_shape, dtype=self.W.dtype)

    def fprop(self, x):
        '''
        fprop through the layer

        :param x: layer input
        :type x: np.ndarray
        '''
        self.x = x.copy()

        self.y = self.x.dot(self.W)

        if self.bias:
            self.y += self.b

        return self.y

    def bprop(self, deltas):
        '''
        bprop through the layer

        :param deltas: propagating errors coming back
        :type deltas: np.ndarray
        '''
        # create layers deltas i.e. transform deltas using linear layer
        grad_deltas = deltas.dot(self.W.T)

        # update weights based on deltas
        self.param_grads = self._param_grad(deltas)

        # return deltas
        return grad_deltas

    def _param_grad(self, deltas):
        '''
        update layer parameters based on deltas

        :param deltas: propagating errors coming back
        :type deltas: np.ndarray
        '''
        # compute Gradient
        self.d_W = self.x.T.dot(deltas)
        # create bias gradient
        if self.bias:
            self.d_b = np.sum(deltas, axis=0)
            return [(self.W, self.d_W), (self.b, self.d_b)]

        return [(self.W, self.d_W)]

class Mean_Center(Layer):

    '''
    "Poor Man Batch Normalization Layer" only mean centers the data
    does not actually z-score the batch

    :param out_shape: output shape
    :type out_shape: int
    :param activation: activation type
    :type activation: activation
    :param bias: boolean defining bias inclusion, this layer is dependent on it
    :type bias: bool
    :param in_shape: define input data shape
    :type in_shape: int
    :param init: parameter initialization method, std.dev. or norm or method
    :type init: str, Number
    :param name: layer name
    :type name: str
    '''

    def __init__(self, out_shape, init=.1, bias=True, name='Mean_Center Layer',
                 dtype=np.float32):

        super(Mean_Center, self).__init__(out_shape, in_shape=out_shape,
                                          bias=bias, init=init, name=name,
                                          dtype=dtype)

    def __repr__(self):
        rep_str = '{}\n'.format(self.name)
        return rep_str

    def allocate(self):
        '''
        allocate layer parameters
        '''
        self.b = np.zeros(self.out_shape, dtype=self.dtype)

    def fprop(self, x):
        '''
        fprop through the layer

        :param x: layer input
        :type x: np.ndarray
        '''
        self.x = x.copy()

        self.y = self.x - np.mean(self.x, axis=0)

        if self.bias:
            self.y += self.b  # learned undo of mean centering

        return self.y

    def bprop(self, deltas):
        '''
        bprop through the layer

        :param deltas: propagating errors coming back
        :type deltas: np.ndarray
        '''
        # create layers deltas i.e. transform deltas using linear layer
        grad_deltas = deltas - np.mean(deltas, axis=0)

        # update weights based on deltas
        self.param_grads = self._param_grad(deltas)

        # return deltas
        return grad_deltas

    def _param_grad(self, deltas):
        '''
        update layer parametres based on deltas

        :param deltas: propogating errors coming back
        :type deltas: np.ndarray
        '''
        # compute bias Gradient
        if self.bias:
            self.d_b = np.sum(deltas, axis=0)
            return [(self.b, self.d_b)]

        return []


class BatchNormalization(Layer):
    '''
    Batch Normalization Layer
    - No scale or offset

    :param out_shape: output shape
    :type out_shape: int
    :param activation: activation type
    :type activation: activation
    :param in_shape: define input data shape
    :type in_shape: int
    :param init: parameter initialization method, std.dev. or norm or method
    :type init: str, Number
    :param name: layer name
    :type name: str
    '''

    def __init__(self, out_shape, init=.1, epsilon=1e-32, axis=(0), bias=False,
                 name='BatchNormalization Layer', dtype=np.float32):

        self.epsilon = epsilon
        self.axis = axis

        super(BatchNormalization, self).__init__(out_shape, in_shape=out_shape,
                                                 init=init, name=name,
                                                 bias=bias, dtype=dtype)

    def __repr__(self):
        rep_str = '{}\n'.format(self.name)
        return rep_str

    def allocate(self):
        '''
        allocate layer parameters
        '''
        pass

    def fprop(self, x):
        '''
        fprop through the layer

        :param x: layer input
        :type x: np.ndarray
        '''
        self.x = x.copy()

        mu = np.mean(self.x, axis=self.axis, keepdims=True)
        var = np.var(self.x, axis=self.axis, keepdims=True)
        self.scale = np.sqrt(var + self.epsilon)

        # normalize
        self.y = (self.x - mu) / self.scale

        return self.y

    def bprop(self, deltas):
        '''
        bprop through the layer

        :param deltas: propagating errors coming back
        :type deltas: np.ndarray
        '''
        # create layers deltas i.e. transform deltas using linear layer
        u = np.mean(deltas, axis=self.axis)
        v = np.mean(deltas * self.y, axis=self.axis)
        grad_deltas = (deltas - u - v * self.y) / self.scale

        # update weights based on deltas
        self.param_grads = self._param_grads(deltas)

        # return deltas
        return grad_deltas

    def _param_grads(self, deltas):
        '''
        update layer parametres based on deltas

        :param deltas: propogating errors coming back
        :type deltas: np.ndarray
        '''

        return []


class BatchNormalization1D(BatchNormalization):

    '''
    Batch Normalization layer for fully connected / dense / linear layers 
    - No scale or offset

    :param out_shape: output shape
    :type out_shape: int
    :param activation: activation type
    :type activation: activation
    :param in_shape: define input data shape
    :type in_shape: int
    :param init: parameter initialization method, std.dev. or norm or method
    :type init: str, Number
    :param name: layer name
    :type name: str
    '''

    def __init__(self, out_shape, init=.1, epsilon=1e-32, bias=True,
                 name='BatchNormalization1D Layer', dtype=np.float32):

        super(BatchNormalization1D, self).__init__(out_shape,
                                                   axis=(0), epsilon=epsilon,
                                                   init=init, name=name,
                                                   bias=bias, dtype=dtype)
