#!/usr/bin/env python
"""
Impliments model object for SIMNN
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import time
import warnings
import numpy as np
from numbers import Number

from simnn.activations import *
from simnn.costs import *
from simnn.utils import print_epochs


class Model(object):
    '''
    Neural Network Model

    :param layers: list of layers for the model
    :type layers: list
    :param dataset: training examples and labels
    :type dataset: tuple
    :param cost: cost of network
    :type cost: Cost
    :param class_task: defines if this is a classification task
    :type class_task: bool
    :param bin_class_task: defines if this is a binary classification task
    :type bin_class_task: bool
    :param name: give model a name
    :type name: str
    '''

    def __init__(self, layers, dataset, cost, optimizer,
                 bin_class_task=False, class_task=False, name='Model'):
        self.name = name
        # define if this is a classification task
        self.class_task = class_task
        self.bin_class_task = bin_class_task

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

        self.early_stop = False

        self.optimizer = optimizer

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
        '''
        initialize containers to hold training statistics

        called during model initialization
        '''
        # for entire epoch
        self.acc_e, self.v_acc_e = [], []  # accuracies
        self.cost_e, self.v_cost_e = [], []  # costs

        # for each training iteration
        self.acc_i = []  # accuracies
        self.cost_i = []  # costs

    def _set_shortcut(self):
        '''
        set shortcut for network cost / final layer options

        called during model initialization
        '''
        if isinstance(self.cost, CrossEntropy):
            if isinstance(self.layers[-1], Softmax):
                self.layers[-1].shortcut = True

        if isinstance(self.cost, BinaryCrossEntropy):
            if isinstance(self.layers[-1], LogisticSigmoid):
                self.layers[-1].shortcut = True

    def _early_stop_acc(self, metric, eps, n):
        '''
        if metric doesnt improve, stop the network
        error rate for one hot label classification

        :param metric: network metric, usually cost, must be decreasing
        :type metric: np.ndarray, list
        :param eps: epsilon for early stopping
        :type eps: Number
        '''
        dif = np.array(metric[-(n + 1):-1]) - np.array(metric[-n:])
        if all(dif <= -eps):
            self.early_stop = True
        else:
            self.early_stop = False

    def _error_rate(self, t, y):
        '''
        error rate for one hot label classification

        :param t: target labels
        :type t: np.ndarray
        :param y: network output
        :type y: np.ndarray
        '''
        t_c = t.argmax(axis=1)
        y_c = y.argmax(axis=1)

        return np.sum(t_c != y_c) / len(t_c)

    def _error_rate_bin(self, t, y):
        '''
        error rate for binary classification

        :param t: target labels
        :type t: np.ndarray
        :param y: network output
        :type y: np.ndarray
        '''
        return np.sum(t != np.round(y)) / len(t)

    def _accuracy_rate(self, t, y):
        '''
        accuracy rate for one hot label classification

        :param t: target labels
        :type t: np.ndarray
        :param y: network output
        :type y: np.ndarray
        '''
        return 1 - self._error_rate(t, y)

    def _accuracy_rate_bin(self, t, y):
        '''
        accuracy rate for binary classification

        :param t: target labels
        :type t: np.ndarray
        :param y: network output
        :type y: np.ndarray
        '''
        return 1 - self._error_rate_bin(t, y)

    def _check_data(self, X, Y):
        '''
        check data data passed to the network

        :param X: training examples
        :type X: np.ndarray
        :param Y: labels for training data
        :type Y: np.ndarray
        '''
        assert isinstance(X, np.ndarray), 'Input must be a numpy array'
        assert isinstance(Y, np.ndarray), 'Input must be a numpy array'
        assert len(X) == len(Y), 'each example in X must have a label'

    def _learn_anneal(self, ep):
        '''
        Anneals the learning rate

        :param ep: all the indicies of the dataset
        :type ep: np.ndarray
        '''
        self.nu = self.initial_learn / (1 + ep / self.T)

    def _b_idx_gen(self, all_idx, b_size):
        '''
        Yield successive b_size-sized chunks from all_idxs
        adopted / modification of:
        https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks

        :param all_idx: all the indicies of the dataset
        :type all_idx: np.ndarray
        :param b_size: batch size to break up the idx into
        :type b_size: int
        '''
        for i in range(0, len(all_idx), b_size):
            yield all_idx[i:i + b_size]

    def _ep_fit(self, verbose):
        '''
        fit the network for one epoch network

        :param verbose: print out training stats for epoch or not
        :type verbose: bool
        '''
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
        '''
        fprop through the network

        :param x: training examples / network inputs
        :type x: np.ndarray
        '''
        self.layers[0].fprop(x)
        for layer in self.layers[1:]:
            layer.fprop(layer.prev_layer.y)
        return self.layers[-1].y

    def net_bprop(self, target, y):
        '''
        bprop through the network and optimize parameters

        :param target: training targets
        :type target: np.ndarray
        :param y: network output from net_fprop
        :type y: np.ndarray
        '''
        error = self.cost.bprop(target, y)
        for layer in self.layers[::-1]:
            error = layer.bprop(error)

        self.optimizer.minimize(self.layers, self.nu)


    def _epoch_stats(self, ep, verbose=0):
        '''
        capture statistics of epoch

        :param ep: epoch number
        :type ep: tuple
        :param verbose: print epoch statistics or not
        :type verbose: bool
        '''
        y = self.net_fprop(self.X)
        # get accuracy for training data
        if self.class_task:
            if self.bin_class_task:
                acc = self._accuracy_rate_bin(self.target, y)
                self.acc_e += [acc]  # store over time
            else:
                acc = self._accuracy_rate(self.target, y)
                self.acc_e += [acc]  # store over time
        cost = self.cost.fprop(self.target, y)
        self.cost_e += [cost / len(y)]

        if self.X_val is not None:
            y = self.net_fprop(self.X_val)
            # get accuracy for validation data
            if self.class_task:
                if self.bin_class_task:
                    v_acc = self._accuracy_rate_bin(self.target_val, y)
                    self.v_acc_e += [v_acc]  # store over time
                else:
                    v_acc = self._accuracy_rate(self.target_val, y)
                    self.v_acc_e += [v_acc]  # store over time
            v_cost = self.cost.fprop(self.target_val, y)
            self.v_cost_e += [v_cost / len(y)]

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
            aneal_T=30, shuffle=True, b_size=-1, verbose=True,
            e_stop=False, e_stop_n=3, early_stop_eps=1e-32, min_epochs=1):
        '''
        fit the Neural Network Model to the dataset for a number of epochs.

        :param dataset: training examples and labels
        :type dataset: tuple
        :param num_epochs: number of epochs to train the network
        :type num_epochs: int
        :param val_set: validation set examples and labels
        :type val_set: tuple
        :param initial_learn: initial learning rate
        :type initial_learn: Number
        :param aneal_T: annealing parameter
        :type aneal_T: int
        :param shuffle: shuffle data after each epoch
        :type shuffle: bool
        :param b_size: batch size over which to train, -1 = full batch training
        :type b_size: int
        :param verbose: print training informations or not
        :type verbose: bool
        :param e_stop: stop network if cost has converged
        :type e_stop: bool
        :param e_stop_n: number of epochs over which to see convergence
        :type e_stop_n: int
        :param early_stop_eps: early stop parameter
        :type early_stop_eps: Number
        :param min_epochs: min number of epochs to train before early stop
        :type min_epochs: int
        '''
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

            # check for early stop
            if e_stop and epoch > e_stop_n + 1:
                if self.X_val is not None:
                    self._early_stop_acc(self.v_cost_e,
                                         early_stop_eps, e_stop_n)
                else:
                    self._early_stop_acc(self.cost_e, early_stop_eps, e_stop_n)
                if self.early_stop and epoch >= min_epochs:
                    print('\nStopping Early!!!!')
                    return
