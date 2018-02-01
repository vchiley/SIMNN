#!/usr/bin/env python
"""
Utilities for dataset loading
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import numpy as np
from numbers import Number


def extract_data_label(X, Y, label, b_subset=None, e_subset=None):
    X_label_idx = np.where(Y[b_subset:e_subset] == label)
    X_label = X[X_label_idx]
    Y_label = label * np.ones(len(X_label), dtype=np.int)
    return X_label, Y_label


def combine_subsets(list_tup_XY, shuffle=False):
    assert isinstance(list_tup_XY, list)
    assert all(isinstance(tup, tuple) for tup in list_tup_XY)
    X, Y = [], []
    for tup in list_tup_XY:
        X.append(tup[0])
        Y.append(tup[1])

    X, Y = np.vstack(X), np.hstack(Y)
    return X, Y


def make_homogenious(X):
    ones = np.ones(len(X))[:, np.newaxis]
    return np.hstack([X, ones])


def d_range(x, r_min=-1, r_max=1):
    '''
    call _d_range on a list of datasets
    '''
    assert isinstance(x, list), 'list expected'

    for i, x_i in enumerate(x):
        x[i] = _d_range(x_i, r_min=r_min, r_max=r_max)

    return x


def _d_range(x, r_min=-1, r_max=1):
    '''
    Place x values into a specified range
    '''
    assert isinstance(r_min, Number), 'Number expected'
    assert isinstance(r_max, Number), 'Number expected'
    assert r_min < r_max, 'minimum of the range must be less than the max'

    x = x - np.min(x)

    x = x / (np.max(x) / (r_max - r_min))

    return x + r_min


def train_val_split(data, frac=.1):
    '''
    Splits the data into a training and validation set.
    Data is expected to be an itterable.
    frac is the fraction of data wanted for the validation
    '''
    assert isinstance(data, tuple), ' data must be a tuple containing (X, Y)'
    assert 0 < frac < 1, 'select and appropriet fraction of the data'
    X, Y = data

    pivot = int(len(Y) * (1 - frac))

    X_train, X_val = X[:pivot], X[pivot:]
    Y_train, Y_val = Y[:pivot], Y[pivot:]

    return ((X_train, Y_train), (X_val, Y_val))
