#!/usr/bin/env python
"""
Impliments cost functions for SIMNN
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import numpy as np


class Cost(object):
    '''
    Base Cost Class

    :param name: layer name
    :type name: str
    '''

    def __init__(self, name):
        assert isinstance(name, str), 'Name must be of type string'
        # super(Cost, self).__init__()
        self.name = name

    def fprop(self, t, y):
        pass

    def bprop(self, t, y):
        pass


class CrossEntropy(Cost):
    '''
    Cross Entropy Cost

    Should have softmax as network output activation

    :param name: layer name
    :type name: str
    :param ep_clip: Clip output values so log(0) dosent occure
    :type ep_clip: Number
    '''

    def __init__(self, name='CrossEntropyCost', ep_clip=1e-64):
        super(CrossEntropy, self).__init__(name)
        self.ep_clip = ep_clip

    def __repr__(self):
        rep_str = '{}'.format(self.name)
        return rep_str

    def fprop(self, t, y):
        '''
        Computes the cost

        :param t: network targets
        :type t: np.ndarray
        :param y: network outputs
        :type y: np.ndarray
        '''
        assert isinstance(t, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert y.shape == t.shape
        assert np.in1d(t, np.array([0, 1])).all(), 'assumes one-hot bin labels'

        y = np.clip(y, self.ep_clip, 1.)  # ensure log(0) doesn't happen

        self.y = - np.sum(t * np.log(y)) / len(y)

        return self.y

    def bprop(self, t, y):
        '''
        Creates deltas for bprop
        Uses shortcut, assumes previous layer has softmax activation

        :param t: network targets
        :type t: np.ndarray
        :param y: network outputs
        :type y: np.ndarray
        '''
        return y - t


class BinaryCrossEntropy(Cost):
    '''
    Binary Cross Entropy Cost

    Should have logistic sigmoid as network output activation

    :param name: layer name
    :type name: str
    :param ep_clip: Clip output values so log(0) dosent occure
    :type ep_clip: Number
    '''

    def __init__(self, name='BinaryCrossEntropy', ep_clip=1e-64):
        super(BinaryCrossEntropy, self).__init__(name)
        self.ep_clip = ep_clip

    def __repr__(self):
        rep_str = '{}'.format(self.name)
        return rep_str

    def fprop(self, t, y):
        '''
        Computes the cost

        :param t: network targets
        :type t: np.ndarray
        :param y: network outputs
        :type y: np.ndarray
        '''
        assert isinstance(t, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert y.shape == t.shape
        assert np.in1d(t, np.array([0, 1])).all(), 'assumes binary labels'

        y = np.clip(y, self.ep_clip, 1.)  # ensure log(0) doesn't happen

        self.y = - np.sum((t * np.log(y) + (1 - t) * np.log(1 - y)) / len(y))

        return self.y

    def bprop(self, t, y):
        '''
        Creates deltas for bprop
        Uses shortcut, assumes previous layer has logistic sigmoid activation

        :param t: network targets
        :type t: np.ndarray
        :param y: network outputs
        :type y: np.ndarray
        '''
        return y - t
