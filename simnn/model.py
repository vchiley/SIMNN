#!/usr/bin/env python
"""
Impliments model object for SIMNN
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import time
import numpy as np
from numbers import Number

from simnn.activations import *
from simnn.costs import *
from simnn.utils import print_epochs


class Model(object):
    """docstring for Model"""

    def __init__(self, layers, dataset, cost, class_task=False, name='Model'):
        self.name = name
        # define if this is a classification task
        self.class_task = class_task

        # extract dataset
        x, t = dataset

        self.in_shape = x.shape[1]

        # define and initialize layers
        self.layers = layers

        # flatten activation layer with affine layers
        for i, layer in enumerate(self.layers):
            if layer.activation is not None:
                self.layers[i:i + 1] = [layer, layer.activation]

        self.out_shape = self.layers[-1].out_shape

        # Define previous layers for network
        for i, layer in enumerate(self.layers[1:]):
            layer.prev_layer = self.layers[i]
        # Define next layers for network
        for i, layer in enumerate(self.layers[:-1]):
            layer.next_layer = self.layers[i + 1]

        # get input shape
        self.layers[0].in_shape = self.in_shape
        # define shape and initialize weights for each layer
        for layer in self.layers:
            layer.config()

        for layer in self.layers:
            layer.allocate()

        # define cost
        self.cost = cost
        self._init_stat_holders()

        self.X_val = None
        self.X = None

        self._set_shortcut()

    def __repr__(self):
        rep_str = '{}, '.format(self.name)
        rep_str += 'in_shape: {}, '.format(self.in_shape)
        rep_str += 'out_shape: {}, '.format(self.out_shape)
        rep_str += '\nwith layers:\n'
        for layer in self.layers:
            rep_str += '{}'.format(layer)
        rep_str += 'and cost: {}'.format(self.cost)

        return rep_str

    def _init_stat_holders(self):
        # for entire epoch
        self.acc_e, self.v_acc_e = [], []  # accuracies
        self.cost_e, self.v_cost_e = [], []  # costs

        # for each training iteration
        self.acc_i = []  # accuracies
        self.cost_i = []  # costs

    def _set_shortcut(self):
        if isinstance(self.cost, CrossEntropy):
            if isinstance(self.layers[-1], Softmax):
                self.layers[-1].shortcut = True

        if isinstance(self.cost, BinaryCrossEntropy):
            if isinstance(self.layers[-1], Logistic_Sigmoid):
                self.layers[-1].shortcut = True

    def _error_rate(self, t, y):
        t_c = t.argmax(axis=1)
        y_c = y.argmax(axis=1)

        return np.sum(t_c != y_c) / len(t_c)

    def _accuracy_rate(self, t, y):
        return 1 - self._error_rate(t, y)

    def _check_data(self, X, Y):
        assert isinstance(X, np.ndarray), 'Input must be a numpy array'
        assert isinstance(Y, np.ndarray), 'Input must be a numpy array'
        assert len(X) == len(Y), 'each example in X must have a label'

    def _learn_anneal(self, t):
        self.nu = self.initial_learn / (1 + t / self.T)

    def _b_idx_gen(self, all_idx, b_size):
        '''
        Yield successive b_size-sized chunks from all_idxs
        adopted from:
        https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        '''
        for i in range(0, len(all_idx), b_size):
            yield all_idx[i:i + b_size]

    def _ep_fit(self, verbose):
        assert -1 <= self.b_size <= len(self.X), 'b_size out of range'
        if self.b_size == -1:
            # get all the trainset indecies
            all_idx = np.arange(len(self.X))
            if self.shuffle:
                np.random.shuffle(all_idx)  # shuffle dataset

            x, t = self.X[all_idx], self.target[all_idx]

            y = self.net_fprop(x)  # forward pass
            self.net_bprop(t, y)  # update weights
        else:
            # batched learning
            if len(self.X) % self.b_size != 0:
                warnings.warn('Training set not split equally by b_size')

            all_idx = np.arange(len(self.X))
            if self.shuffle:
                np.random.shuffle(all_idx)  # shuffle dataset

            # loop through batches from the trainset
            b_idxing = self._b_idx_gen(all_idx, self.b_size)
            for b_it, b_idx in enumerate(b_idxing):
                x, t = self.X[b_idx], self.target[b_idx]

                y = self.net_fprop(x)  # forward pass
                # b_prop errors and update weights
                self.net_bprop(t, y)

    def net_fprop(self, x):
        # forward pass through the network
        self.layers[0].fprop(x)
        for layer in self.layers[1:]:
            layer.fprop(layer.prev_layer.y)
        return self.layers[-1].y

    def net_bprop(self, target, y):
        # backwards pass with errors
        error = self.cost.bprop(target, y)
        for layer in self.layers[::-1]:
            error = layer.bprop(error, self.nu)

    def _epoch_stats(self, ep, verbose=0):
        y = self.net_fprop(self.X)
        # get accuracy for training data
        if self.class_task:
            acc = self._accuracy_rate(self.target, y)
            self.acc_e += [acc]  # store over time
        cost = self.cost.fprop(self.target, y)
        self.cost_e += [cost]

        if self.X_val is not None:
            y = self.net_fprop(self.X_val)
            # get accuracy for validation data
            if self.class_task:
                v_acc = self._accuracy_rate(self.target_val, y)
                self.v_acc_e += [v_acc]  # store over time
            v_cost = self.cost.fprop(self.target_val, y)
            self.v_cost_e += [v_cost]

        # print out epoch training point
        if verbose:
            t_time = time.time() - self.train_b_time
            verbose_str = 'Epoch: {}'.format(ep)
            verbose_str += ', time: {:.4}s'.format(t_time)
            verbose_str += ', with train cost {:.4}'.format(cost)
            if self.class_task:
                verbose_str += ', Train Acc {:.4}'.format(acc)
                if self.X_val is not None:
                    verbose_str += ', Val Acc {:.4}'.format(v_acc)

            print_epochs(verbose_str)

    def fit(self, dataset, num_epochs, val_set=None, initial_learn=1e-3,
            aneal_T=30, shuffle=True, b_size=-1, verbose=True):
        self.X, self.target = dataset
        self._check_data(self.X, self.target)

        if val_set:
            self.X_val, self.target_val = val_set
            self._check_data(self.X_val, self.target_val)

        assert isinstance(b_size, int), 'expecting an integer'
        assert isinstance(shuffle, bool), 'expecting a bool'
        assert isinstance(initial_learn, Number)
        assert isinstance(aneal_T, Number)

        self.b_size = b_size
        self.shuffle = shuffle
        self.initial_learn = initial_learn
        self.T = aneal_T

        self.train_b_time = time.time()
        for epoch in range(num_epochs):

            self._learn_anneal(epoch)

            self._ep_fit(verbose)

            self._epoch_stats(epoch, verbose)
