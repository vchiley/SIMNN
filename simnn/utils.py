#!/usr/bin/env python
"""
Utilities for SIMNN
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import sys
import numpy as np


def one_hot(y, m=None):
    '''
    creates a one hot vector from list of classes

    assumes classes \in 1...m
    '''
    assert isinstance(y, np.ndarray)
    if m:
        assert isinstance(m, int), 'expecting an integer'
    else:
        m = len(set(y))

    y_out = np.zeros((y.shape[0], m), dtype=int)  # generate onehot shape
    # populate one-hot vector
    for i, yi in enumerate(y):
        y_out[i, yi] = 1

    return y_out


def shuffle_dataset(X, Y):
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)

    idx = np.arange(len(Y))  # get indices for itterable object
    np.random.shuffle(idx)  # suffle indices
    # return examples at idx
    return X[idx], Y[idx]


def print_epochs(p_str):
    sys.stdout.write('\r' + p_str)
    sys.stdout.flush()
