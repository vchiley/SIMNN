#!/usr/bin/env python
"""
Impliments activation functions for SIMNN
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import unittest
import numpy as np

from simnn.utils import *
from dataset.utils import _d_range
from dataset.mnist.mnist import load_mnist_data


class ConvUtilTests(unittest.TestCase):
    '''
    Test and understand how im2col and col2im work
    '''

    def test_im2col(self):
        '''
        TODO
        '''
        # Extract data
        ((X_train, _), (_, _)) = load_mnist_data('dataset/mnist/')
        X_train = X_train[:64]  # take a subset of the training data

        [X_train] = _d_range([X_train])

        N, D = X_train.shape
        H = int(np.sqrt(D))
        X_train = X_train.reshape(N, 1, H, H)

        # set up the filtration parameters
        h_filter, w_filter, padding, stride = 1, 1, 0, 1

        X_col = im2col_indices(X_train, h_filter, w_filter, padding=padding, stride=stride)
        X = col2im_indices(X_col, X_train.shape, h_filter, w_filter, padding=padding, stride=stride)

        assert all(r == _r for r, _r in zip(X_train.shape, X.shape))
        assert np.equal(X_train, X).all()

    def test_im2col2(self):
        '''
        TODO
        '''
        # Extract data
        ((X_train, _), (_, _)) = load_mnist_data('dataset/mnist/')
        X_train = X_train[:64]  # take a subset of the training data

        [X_train] = _d_range([X_train])

        N, D = X_train.shape
        H = int(np.sqrt(D))
        X_train = X_train.reshape(N, 1, H, H)

        # set up the filtration parameters
        h_filter, w_filter, padding, stride = 2, 2, 0, 1

        X_col = im2col_indices(X_train, h_filter, w_filter, padding=padding, stride=stride)
        X = col2im_indices(X_col, X_train.shape, h_filter, w_filter, padding=padding, stride=stride)

        assert all(r == _r for r, _r in zip(X_train.shape, X.shape))
        assert np.isclose(4 * X_train[:, :, 1:-1, 1:-1], X[:, :, 1:-1, 1:-1]).all()

    def test_im2col3(self):
        '''
        TODO
        '''
        # Extract data
        ((X_train, _), (_, _)) = load_mnist_data('dataset/mnist/')
        X_train = X_train[:64]  # take a subset of the training data

        [X_train] = _d_range([X_train])

        N, D = X_train.shape
        H = int(np.sqrt(D))
        X_train = X_train.reshape(N, 1, H, H)

        # set up the filtration parameters
        h_filter, w_filter, padding, stride = 3, 3, 1, 1

        X_col = im2col_indices(X_train, h_filter, w_filter, padding=padding, stride=stride)
        X = col2im_indices(X_col, X_train.shape, h_filter, w_filter, padding=padding, stride=stride)

        assert all(r == _r for r, _r in zip(X_train.shape, X.shape))
        assert np.isclose(9 * X_train[:, :, 1:-1, 1:-1], X[:, :, 1:-1, 1:-1]).all()

        


if __name__ == '__main__':
    unittest.main()
