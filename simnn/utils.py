#!/usr/bin/env python
"""
Utilities for SIMNN

Disclaimer:
    get_im2col_indices, im2col_indices, col2im_indices taken from Stanford's
    deep learning course: CS231n
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


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    '''
    An implementation of im2col based on some fancy indexing
    '''
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    '''
    An implementation of col2im based on fancy indexing and np.add.at
    '''
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
