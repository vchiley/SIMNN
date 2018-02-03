#!/usr/bin/env python
"""
Impliments activation functions for SIMNN
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import unittest
import numpy as np

from simnn import Model
from simnn import Linear
from simnn import ReLU, Softmax, Logistic_Sigmoid
from simnn import CrossEntropy
from simnn import one_hot

# sys.path.append('../')
from dataset.utils import _d_range
from dataset.mnist.mnist import load_mnist_data


def _numerical_grad(self, data, weight_def, epsilon):
    '''
    It's a numerical approximation of the gradient

    neumerical gradient should be withing \epsilon^2 of the real gradient
    with exessively small \epsilon computational roundoff error makes the
    error larger. If running tests with numerical grad checker, sometimes it
    will fail because initializations of the networks are random and
    probabilistically test failure occures, but rarely

    :param data: training example and label, one training example
    :type data: tuple
    :param weight_def: tuple describing weight to check: (layer, bias, idx)
    :type weight_def: tuple
    :param epsilon: epsilon for numerical gradient approximation
    :type epsilon: Number
    '''
    x, t = data
    assert len(t) == 1
    assert isinstance(weight_def, tuple), 'weight_def is tuple description'
    assert len(weight_def) == 3, '(in or hid, bias or not, (idx in array))'

    # unpack input directing us to the input
    layer, bias, idx = weight_def
    assert isinstance(bias, bool)
    assert not layer % 2, 'layer must have params, not just be an activation'

    if bias:
        w = self.layers[layer].b
    else:
        w = self.layers[layer].W

    # change the weight to positive
    w[idx] += epsilon
    y = self.net_fprop(x)
    c_p = self.cost.fprop(t, y)

    # change the weight to negative
    w[idx] -= 2 * epsilon
    y = self.net_fprop(x)
    c_n = self.cost.fprop(t, y)

    # find numerical approximation of gradient
    dw_n = (c_p - c_n) / (2 * epsilon)

    # reset weight
    w[idx] += epsilon

    # calculate actual Gradient and pull it at a particular index
    y = self.net_fprop(x)  # forward pass
    self.net_bprop(t, y)  # update weights

    if bias:
        dw = self.layers[layer].d_b[idx].copy()
    else:
        dw = self.layers[layer].d_W[idx].copy()

    dw_error = np.abs(dw_n - dw)

    return dw_error


class bpropTests(unittest.TestCase):
    '''
    Unit tests for checking the backpropogation of gradients through the model

    Great way to check new layers. Create the layer and check its grad prop
    '''

    def testOne(self):
        '''
        neumerically approximate gradient and check network gradients
        were correctly computed

        Check Softmax with shortcut
        '''
        # Extract data
        ((X_train, Y_train), (_0, _1)) = load_mnist_data('dataset/mnist/')

        # put data values \in [-1, 1]
        [X_train] = _d_range([X_train])

        t_train = one_hot(Y_train, m=10)

        dataset = (X_train, t_train)

        b_size = 1
        # take out only a few samples for gradent check
        X_check, Y_check = X_train[0:b_size], t_train[0:b_size]

        Model._numerical_grad = _numerical_grad
        Model.nu = 1e-3

        # define model structure
        layers = [Linear(out_shape=10, activation=Softmax(), bias=True,
                         init='lecun_normal')]

        # instantiate model
        model = Model(layers, dataset, CrossEntropy(), class_task=True)

        model.fit((X_check, Y_check), 3, verbose=False)

        configs = [(0, False, (0, 0)),
                   (0, False, (256, 9)),
                   (0, True, (0)),
                   (0, True, (4))]
        eps = np.arange(.0001, .2, .0001)
        dw_error = []
        for config in configs:
            dw_e = []
            for ep in eps:
                e = model._numerical_grad((X_check, Y_check),
                                          config, ep)
                dw_e += [e]
            dw_error += [dw_e]

        dw_error = np.array(dw_error)

        self.assertTrue((dw_error < (eps**2)).all)

    def testTwo(self):
        '''
        neumerically approximate gradient and check network gradients
        were correctly computed

        Check Logistic_Sigoid
        '''
        # Extract data
        ((X_train, Y_train), (_0, _1)) = load_mnist_data('dataset/mnist/')

        # put data values \in [-1, 1]
        [X_train] = _d_range([X_train])

        t_train = one_hot(Y_train, m=10)

        dataset = (X_train, t_train)

        b_size = 1
        # take out only a few samples for gradent check
        X_check, Y_check = X_train[0:b_size], t_train[0:b_size]

        Model._numerical_grad = _numerical_grad
        Model.nu = 1e-3

        # define model structure
        layers = [Linear(out_shape=64, activation=Logistic_Sigmoid(),
                         bias=True, init='lecun_normal'),
                  Linear(out_shape=10, activation=Softmax(), bias=True,
                         init='lecun_normal')]

        # instantiate model
        model = Model(layers, dataset, CrossEntropy(), class_task=True)

        model.fit((X_check, Y_check), 3, verbose=False)

        configs = [(0, False, (0, 0)),
                   (0, False, (60, 62)),
                   (2, False, (0, 0)),
                   (2, False, (4, 6)),
                   (0, False, (10, 26)),
                   (2, False, (7, 3)),
                   (0, True, (0)),
                   (0, True, (20)),
                   (2, True, (0)),
                   (2, True, (5))]
        eps = np.arange(.0001, .2, .0001)
        dw_error = []
        for config in configs:
            dw_e = []
            for ep in eps:
                e = model._numerical_grad((X_check, Y_check),
                                          config, ep)
                dw_e += [e]
            dw_error += [dw_e]

        dw_error = np.array(dw_error)

        self.assertTrue((dw_error < (eps**2)).all)

    def testThree(self):
        '''
        neumerically approximate gradient and check network gradients
        were correctly computed

        Check ReLU
        '''
        # Extract data
        ((X_train, Y_train), (_0, _1)) = load_mnist_data('dataset/mnist/')

        # put data values \in [-1, 1]
        [X_train] = _d_range([X_train])

        t_train = one_hot(Y_train, m=10)

        dataset = (X_train, t_train)

        b_size = 1
        # take out only a few samples for gradent check
        X_check, Y_check = X_train[0:b_size], t_train[0:b_size]

        Model._numerical_grad = _numerical_grad
        Model.nu = 1e-3

        # define model structure
        layers = [Linear(out_shape=64, activation=ReLU(),
                         bias=True, init='lecun_normal'),
                  Linear(out_shape=10, activation=Softmax(), bias=True,
                         init='lecun_normal')]

        # instantiate model
        model = Model(layers, dataset, CrossEntropy(), class_task=True)

        model.fit((X_check, Y_check), 3, verbose=False)

        configs = [(0, False, (0, 0)),
                   (0, False, (60, 62)),
                   (2, False, (0, 0)),
                   (2, False, (4, 6)),
                   (0, False, (10, 26)),
                   (2, False, (7, 3)),
                   (0, True, (0)),
                   (0, True, (20)),
                   (2, True, (0)),
                   (2, True, (5))]
        eps = np.arange(.0001, .2, .0001)
        dw_error = []
        for config in configs:
            dw_e = []
            for ep in eps:
                e = model._numerical_grad((X_check, Y_check),
                                          config, ep)
                dw_e += [e]
            dw_error += [dw_e]

        dw_error = np.array(dw_error)

        self.assertTrue((dw_error < (eps**2)).all)

    def testFour(self):
        '''
        neumerically approximate gradient and check network gradients
        were correctly computed

        Check ReLU and Logistic_Sigmoid in a deeper network
        '''
        # Extract data
        ((X_train, Y_train), (_0, _1)) = load_mnist_data('dataset/mnist/')

        # put data values \in [-1, 1]
        [X_train] = _d_range([X_train])

        t_train = one_hot(Y_train, m=10)

        dataset = (X_train, t_train)

        b_size = 1
        # take out only a few samples for gradent check
        X_check, Y_check = X_train[0:b_size], t_train[0:b_size]

        Model._numerical_grad = _numerical_grad
        Model.nu = 1e-3

        # define model structure
        layers = [Linear(out_shape=64, activation=ReLU(), bias=True,
                         init='lecun_normal'),
                  Linear(out_shape=64, activation=Logistic_Sigmoid(),
                         bias=True, init='lecun_normal'),
                  Linear(out_shape=10, activation=Softmax(), bias=True,
                         init='lecun_normal')]

        # instantiate model
        model = Model(layers, dataset, CrossEntropy(), class_task=True)

        model.fit((X_check, Y_check), 3, verbose=False)

        configs = [(0, False, (0, 0)),
                   (0, False, (60, 62)),
                   (2, False, (0, 0)),
                   (2, False, (4, 6)),
                   (0, False, (10, 26)),
                   (2, False, (7, 61)),
                   (0, True, (0)),
                   (0, True, (20)),
                   (2, True, (0)),
                   (2, True, (5)),
                   (4, False, (0, 0)),
                   (4, False, (9, 4)),
                   (4, True, (0)),
                   (4, True, (8))]
        eps = np.arange(.0001, .2, .0001)
        dw_error = []
        for config in configs:
            dw_e = []
            for ep in eps:
                e = model._numerical_grad((X_check, Y_check),
                                          config, ep)
                dw_e += [e]
            dw_error += [dw_e]

        dw_error = np.array(dw_error)

        self.assertTrue((dw_error < (eps**2)).all)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
