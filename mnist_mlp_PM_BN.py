#!/usr/bin/env python
"""
Example network usage, with mnist dataset and Mean Centering Layer (PM_BN)
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

import argparse

from simnn import Model
from simnn import Linear, PM_BN
from simnn import ReLU, Softmax
from simnn import CrossEntropy

from simnn.utils import one_hot

from dataset.mnist.mnist import load_mnist_data
from dataset.utils import *


if __name__ == '__main__':
    '''
    train and test mlp network on the mnist dataset
    '''
    parser = argparse.ArgumentParser(description='Train and test mlp network on the mnist dataset.')
    parser.add_argument('--verbose', action='store_true', default=True, help='Set Verbose')
    parser.add_argument('--mnist_dir', type=str, help='directory of mnist data')

    args = parser.parse_args()

    verbose = args.verbose
    mnist_dir = args.mnist_dir

    ((X_train, Y_train), (X_test, Y_test)) = load_mnist_data(mnist_dir)

    assert X_train.shape == (60000, 784), 'Data not loaded correctly'
    assert len(X_train) == len(Y_train), 'Data not loaded correctly'
    assert X_test.shape == (10000, 784), 'Data not loaded correctly'
    assert len(X_test) == len(Y_test), 'Data not loaded correctly'

    # put data values \in [-1, 1]
    [X_train, X_test] = d_range([X_train, X_test])

    # create training / validatin data split
    ((X_train, Y_train), (X_val, Y_val)) = train_val_split((X_train, Y_train),
                                                           1 / 6)

    t_train = one_hot(Y_train, m=10)
    t_test = one_hot(Y_test, m=10)
    t_val = one_hot(Y_val, m=10)

    dataset = (X_train, t_train)
    val_set = (X_val, t_val)

    # Define network layers
    nh = 64
    num_epochs = 128
    # define model structure
    layers = [Linear(out_shape=nh, activation=ReLU(), bias=True,
                     init='lecun_normal'),
              PM_BN(nh),
              Linear(out_shape=10, activation=Softmax(), bias=True,
                     init='lecun_normal')]

    # instantiate model
    model = Model(layers, dataset, CrossEntropy(), class_task=True)

    if verbose:
        print('Model Architecture:\n', model)
        v_str = 'Training Network for {}'.format(num_epochs)
        v_str += ', with early stop enabled'
        print(v_str)

    # fit model to datas
    model.fit(dataset, num_epochs=num_epochs, val_set=val_set,
              initial_learn=1e-3, aneal_T=30, shuffle=True,
              b_size=128, verbose=verbose, early_stop_eps=1e-32,
              min_epochs=32, e_stop_n=4, e_stop=True)

    # print results
    print("Final Training Accuracy Rate: {:.5}%".format(model.acc_e[-1]))
    print("Final Validation Accuracy Rate: {:.5}%".format(model.v_acc_e[-1]))

    print("Final Training Cost: {:.5}".format(model.cost_e[-1]))
    print("Final Validation Cost: {:.5}".format(model.v_cost_e[-1]))

    test_accuracy = model._accuracy_rate(t_test, model.net_fprop(X_test))
    print("Final Test Set Accuracy Rate: {:.5}%".format(test_accuracy))
