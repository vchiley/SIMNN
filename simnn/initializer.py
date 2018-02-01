#!/usr/bin/env python
"""
Impliments initializer for SIMNN layers
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import numpy as np
from numbers import Number


def initializer(layer):
    if isinstance(layer.init, Number):
        '''
        if layer.init is an integer representing standard dev of a normal
        distribution
        '''
        w = layer.init * np.random.randn(layer.in_shape, layer.out_shape)
        b = 0
        if layer.bias:
            b = layer.init * np.random.randn(layer.out_shape)

        return w, b

    elif layer.init == 'lecun_normal':
        return lecun_normal(layer)

    elif layer.init == 'xavier_normal':
        return xavier_normal(layer)

    elif layer.init == 'he_normal':
        return he_normal(layer)

    elif layer.init == 'lecun_uniform':
        return lecun_uniform(layer)

    elif layer.init == 'xavier_uniform':
        return xavier_uniform(layer)

    elif layer.init == 'he_uniform':
        return he_uniform(layer)

    else:
        raise NotImplementedError('Initializer not implemented')


def lecun_normal(layer):
    fan_in = layer.in_shape
    std_dev = 1 / np.sqrt(fan_in)
    w = std_dev * np.random.randn(layer.in_shape, layer.out_shape)
    b = 0
    if layer.bias:
        b = std_dev * np.random.randn(layer.out_shape)

    return w, b


def xavier_normal(layer):
    fan_in = layer.in_shape
    fan_out = layer.out_shape

    std_dev = np.sqrt(2 / (fan_in + fan_out))

    w = std_dev * np.random.randn(layer.in_shape, layer.out_shape)
    b = 0
    if layer.bias:
        b = std_dev * np.random.randn(layer.out_shape)

    return w, b


def he_normal(layer):
    fan_in = layer.in_shape
    std_dev = 2 / np.sqrt(fan_in)
    w = std_dev * np.random.randn(layer.in_shape, layer.out_shape)
    b = 0
    if layer.bias:
        b = std_dev * np.random.randn(layer.out_shape)

    return w, b


def lecun_uniform(layer):
    fan_in = layer.in_shape
    lim = np.sqrt(3 / fan_in)
    w = np.random.uniform(low=-lim, high=lim,
                          size=(layer.in_shape, layer.out_shape))
    b = 0
    if layer.bias:
        b = np.random.uniform(low=-lim, high=lim,
                              size=(layer.out_shape))

    return w, b


def xavier_uniform(layer):
    fan_in = layer.in_shape
    fan_out = layer.out_shape
    lim = np.sqrt(6 / (fan_in + fan_out))
    w = np.random.uniform(low=-lim, high=lim,
                          size=(layer.in_shape, layer.out_shape))
    b = 0
    if layer.bias:
        b = np.random.uniform(low=-lim, high=lim,
                              size=(layer.out_shape))

    return w, b


def he_uniform(layer):
    fan_in = layer.in_shape
    lim = np.sqrt(6 / fan_in)
    w = np.random.uniform(low=-lim, high=lim,
                          size=(layer.in_shape, layer.out_shape))

    b = 0
    if layer.bias:
        b = np.random.uniform(low=-lim, high=lim,
                              size=(layer.out_shape))

    return w, b
