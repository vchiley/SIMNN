import numpy as np


# layer types
class Layer(object):

    def __init__(self, out_shape, activation=None, bias=False,
                 in_shape=None, init=.1, name='Layer'):

        if out_shape:
            assert isinstance(out_shape, int), 'out_shape must be a number'
        assert isinstance(name, str), 'Name must be of type string'

        self.out_shape = out_shape
        self.activation = activation
        self.bias = bias
        self.init = init
        self.in_shape = None
        self.x = None
        self.next_layer = None
        self.prev_layer = None
        self.name = name

        if self.activation:
            self.activation.out_shape = self.out_shape

    def __repr__(self):
        rep_str = '{}, '.format(self.name)
        rep_str += 'in_shape: {}, '.format(self.in_shape)
        rep_str += 'out_shape: {}, '.format(self.out_shape)
        rep_str += 'and has bias: {}, \n'.format(self.bias)

        return rep_str

    def config(self):
        # get inshape from previous layer
        if self.in_shape is None:
            self.in_shape = self.prev_layer.out_shape

    def allocate(self):
        '''
        allocate parameters of model
        '''
        pass


class Linear(Layer):

    def __init__(self, out_shape, activation=None, bias=False,
                 in_shape=None, init=.1, name='Linear Layer',
                 regularization_weight=None):

        super(Linear, self).__init__(out_shape, activation=activation,
                                     bias=bias, in_shape=in_shape, init=init,
                                     name='Linear Layer')
        self.regularization_weight = regularization_weight

    def allocate(self):
        # allocate parameters of layer
        self.W = self.init * np.random.randn(self.in_shape, self.out_shape)
        if self.bias:
            self.W_bias = self.init * np.random.randn(1, self.out_shape)

    def fprop(self, x):
        self.x = x

        self.lin_out = self.x.dot(self.W)

        self.y = self.activation.fprop(self.lin_out)

        # add bias is asked
        if self.bias:
            self.y += self.W_bias

        return self.y

    def bprop(self, p_deltas, alpha):
        # create layers deltas i.e. transform deltas using linear layer
        self.deltas = p_deltas.dot(self.W.T)

        # update weights based on deltas
        self._param_update(p_deltas, alpha)

        # return deltas
        return self.deltas

    def _param_update(self, p_deltas, alpha):
        # compute Gradient
        d_W = self.x.T.dot(p_deltas)  # create weight gradient
        if self.regularization_weight:
            d_W -= self.regularization_weight / alpha * self.W

        # update weights by taking gradient step
        self.W -= alpha * d_W
