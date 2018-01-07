import numpy as np
import warnings

from numbers import Number

from simnn.layers import Layer


class Sigmoid(Layer):
    def __init__(self, name='Sigmoid'):

        super(Sigmoid, self).__init__(out_shape=None, activation=None,
                                      bias=None, in_shape=None, init=None,
                                      name=name)

    def __repr__(self):
        rep_str = '{}\n'.format(self.name)
        return rep_str

    def fprop(self, x):
        '''
        sigmoid function implimentation

        :param: x - vector on which to perform sigmoid
        :type: np.ndarray
        '''
        self.x = x

        self.out = 1 / (1 + np.exp(-self.x))

        return self.out

    def bprop(self, p_deltas, alpha):
        return self.fprop(self.x) * self.fprop(-self.x) * p_deltas


class Softmax(Layer):
    def __init__(self, name='Softmax', n_stable=True):

        super(Softmax, self).__init__(out_shape=None, activation=None,
                                      bias=None, in_shape=None, init=None,
                                      name=name)
        self.n_stable = n_stable

    def __repr__(self):
        rep_str = '{}, n_stable: {}\n'.format(self.name, self.n_stable)
        return rep_str

    def fprop(self, x):
        '''
        Neumerically stable softmax of the vector x

        :param: x - vector on which to perform softmax
        :type: np.ndarray
        '''
        assert isinstance(x, np.ndarray), 'must be a numpy vector'

        self.x = x

        if self.n_stable:  # neumerically stable, but slightly slower
            x = x - np.max(x, axis=1)[:, np.newaxis]

        out = np.exp(x)

        self.out = out / np.sum(out, axis=1)[:, np.newaxis]

        return self.out

    def bprop(self, p_deltas, alpha):
        return 1 * p_deltas


class ReLU(Layer):
    def __init__(self, name='ReLU'):

        super(ReLU, self).__init__(out_shape=None, activation=None,
                                   bias=None, in_shape=None, init=None,
                                   name=name)

    def __repr__(self):
        rep_str = '{}\n'.format(self.name)
        return rep_str

    def fprop(self, x):
        '''
        :param: x
        :type: np.ndarray
        '''
        assert isinstance(x, np.ndarray), 'must be a numpy vector'
        self.x = x
        x[x < 0] = 0

        self.out = x

        return self.out

    def bprop(self, p_deltas, alpha):
        return (self.x > 0).astype(np.float) * p_deltas


class Line(Layer):
    def __init__(self, a=1, b=0, name='Line'):

        super(Line, self).__init__(out_shape=None, activation=None,
                                   bias=None, in_shape=None, init=None,
                                   name=name)

        assert isinstance(a, Number), 'a must be a Number'
        assert isinstance(b, Number), 'b must be a Number'
        warnings.warn('Line is not a non-linearity', UserWarning)
        self.a = a
        self.b = b

    def __repr__(self):
        rep_str = '{}, '.format(self.name)
        rep_str = '{}X + {}\n, '.format(self.a, self.b)
        return rep_str

    def fprop(self, x):
        '''
        :param: x
        :type: np.ndarray
        '''
        assert isinstance(x, np.ndarray), 'must be a numpy vector'

        self.x = x

        self.out = self.a * x + self.b

        return self.out

    def bprop(self, p_deltas, alpha):
        return self.a * p_deltas


class Identity(Line):
    def __init__(self, name='Identity'):
        super(Identity, self).__init__(name=name)
        warnings.warn('Identity is not a non-linearity', UserWarning)

    def __repr__(self):
        rep_str = '{}\n'.format(self.name)
        return rep_str
