import numpy as np


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

    def __repr__(self):
        rep_str = '{}, '.format(self.name)
        rep_str += 'in_shape: {}, '.format(self.in_shape)
        rep_str += 'out_shape: {}, '.format(self.out_shape)
        rep_str += '\nwith layers:\n'
        for layer in self.layers:
            rep_str += '{}'.format(layer)
        rep_str += 'and cost: {}'.format(self.cost)

        return rep_str

    def _error_rate(self, t, y):
        t_c = t.argmax(axis=1)
        y_c = y.argmax(axis=1)

        return np.sum(t_c != y_c) / len(t_c)

    def fit(self, dataset, num_epochs, nu=1e-5, prnt=True):
        x, t = dataset

        costs = []
        err_rate = []
        for epoch in range(num_epochs):

            # forward pass through the network
            self.layers[0].fprop(x)
            for layer in self.layers[1:]:
                layer.fprop(layer.prev_layer.out)
            self.y = self.layers[-1].out

            costs.append(self.cost.fprop(t, self.y, epsilon=1e-9))

            # backwards pass with errors
            error = self.cost.bprop(t, self.y)
            for layer in self.layers[::-1]:
                error = layer.bporp(error, nu)

            # get error rate for classification problems
            if self.class_task:
                err_rate.append(self._error_rate(t, self.y))

            if prnt:
                print_str = 'Epoch {} of {}'.format(epoch, num_epochs)
                print_str += ', cost: {}'.format(costs[-1])
                print(print_str)

        return self.y, costs, err_rate
