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

    def bporp(self, p_deltas, alpha):
        # bprop deltas through activation
        d_activation = self.activation.bprop(self.lin_out)
        p_deltas = d_activation * p_deltas

        # create layers deltas
        self.deltas = p_deltas.dot(self.W.T)

        # update weights based on deltas
        self._weight_update(p_deltas, alpha)

        # return deltas
        return self.deltas

    def _weight_update(self, p_deltas, alpha):
        # compute Gradient
        d_W = self.x.T.dot(p_deltas)  # create weight gradient

        # update weights by taking gradient step
        self.W -= alpha * d_W
