#!/usr/bin/env python
"""
Impliments cost functions for SIMNN
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import numpy as np


class CrossEntropy(object):
    def __init__(self, name='CrossEntropyCost', ep_clip=1e-64):
        assert isinstance(name, str), 'Name must be of type string'
        self.name = name
        self.ep_clip = ep_clip

    def __repr__(self):
        rep_str = '{}\n'.format(self.name)
        return rep_str

    def fprop(self, t, y):
        assert isinstance(t, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert y.shape == t.shape

        y = np.clip(y, self.ep_clip, 1.)  # ensure log(0) doesn't happen

        self.y = - np.sum(t * np.log(y)) / len(y)

        return self.y

    def bprop(self, t, y):
        '''
        Uses shortcut, assumes previous layer is softmax layer
        '''
        return y - t


class BinaryCrossEntropy(object):
    def __init__(self, name='BinaryCrossEntropy', ep_clip=1e-64):
        assert isinstance(name, str), 'Name must be of type string'
        self.name = name
        self.ep_clip = ep_clip

    def __repr__(self):
        rep_str = '{}\n'.format(self.name)
        return rep_str

    def fprop(self, t, y):
        assert isinstance(t, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert y.shape == t.shape

        y = np.clip(y, self.ep_clip, 1.)  # ensure log(0) doesn't happen

        self.y = - np.sum((t * np.log(y) + (1 - t) * np.log(1 - y)) / len(y))

        return self.y

    def bprop(self, t, y):
        '''
        Uses shortcut, assumes previous layer is logistic sigmoid layer
        '''
        return y - t
