import numpy as np


# layer types
class Linear(object):

    def __init__(self, out_shape, activation, bias=False,
                 in_shape=None, init=.1, name='Linear Layer'):

        assert isinstance(out_shape, int), 'alpha must be a number'

        self.out_shape = out_shape
        self.activation = activation
        self.bias = bias
        self.init = init
        self.in_shape = None
        self.x = None
        self.next_layer = None
        self.prev_layer = None
        self.name = name

    def __repr__(self):
        rep_str = '{}, '.format(self.name)
        rep_str += 'in_shape: {}, '.format(self.in_shape)
        rep_str += 'out_shape: {}, '.format(self.out_shape)
        rep_str += 'with bias: {}, '.format(self.bias)
        rep_str += 'and activation: {}\n'.format(self.activation)

        return rep_str

    def config(self):
        # get inshape from previous layer
        if self.in_shape is None:
            self.in_shape = self.prev_layer.out_shape

    def allocate(self):
        self.W = self.init * np.random.randn(self.in_shape, self.out_shape)
        if self.bias:
            self.W_bias = self.init * np.random.randn(1, self.out_shape)

    def fprop(self, x):
        self.x = x

        self.lin_out = self.x.dot(self.W)

        self.out = self.activation.fprop(self.lin_out)

        # add bias is asked
        if self.bias:
            self.out += self.W_bias

        return self.out

    def bporp(self, error, alpha):

        if self.next_layer is None:
            self.deltas = error  # difference based on loss ????
            self.deltas_avg = self.deltas

        else:
            d_activation = self.activation.bprop(self.lin_out)

            # delta based on non-linearity and prev deltas
            self.deltas_avg = self.next_layer.deltas.dot(self.next_layer.W.T)

            self.deltas = d_activation * self.deltas_avg

        self._weight_update(alpha)

        # return deltas
        return self.deltas

    def _weight_update(self, alpha):
        # compute Gradient
        d_W = self.x.T.dot(self.deltas)  # create weight gradient

        # update weights by taking gradient step
        self.W -= alpha * d_W

        if self.bias:
            # create weight gradient
            d_W_bias = np.sum(self.deltas_avg, axis=0)

            # update weights by taking gradient step
            self.W_bias -= alpha * d_W_bias
