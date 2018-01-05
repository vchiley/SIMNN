import numpy as np
import warnings

from numbers import Number


class Sigmoid(object):
    def __init__(self, name='Sigmoid'):
        assert isinstance(name, str), 'Name must be of type string'
        self.name = name

    def __repr__(self):
        rep_str = '{}'.format(self.name)
        return rep_str

    def fprop(self, x):
        '''
        sigmoid function implimentation

        :param: x - vector on which to perform sigmoid
        :type: np.ndarray
        '''
        return 1 / (1 + np.exp(-x))

    def bprop(self, y):
        return self.function_call(y) * self.function_call(-y)


class Softmax(object):
    def __init__(self, name='Softmax', n_stable=True):
        assert isinstance(name, str), 'Name must be of type string'
        self.name = name
        self.n_stable = n_stable

    def __repr__(self):
        rep_str = '{}, n_stable: {}'.format(self.name, self.n_stable)
        return rep_str

    def fprop(self, x):
        '''
        Neumerically stable softmax of the vector x

        :param: x - vector on which to perform softmax
        :type: np.ndarray
        '''
        assert isinstance(x, np.ndarray), 'must be a numpy vector'

        if self.n_stable:  # neumerically stable, but slightly slower
            x = x - np.max(x, axis=1)[:, np.newaxis]

        out = np.exp(x)
        return out / np.sum(out, axis=1)[:, np.newaxis]

    def bprop(self, y):
        return 1


class ReLU(object):
    def __init__(self, name='ReLU'):
        assert isinstance(name, str), 'Name must be of type string'
        self.name = name

    def __repr__(self):
        rep_str = '{}'.format(self.name)
        return rep_str

    def fprop(self, x):
        '''
        :param: x
        :type: np.ndarray
        '''
        assert isinstance(x, np.ndarray), 'must be a numpy vector'
        self.x = x
        self.x[self.x < 0] = 0

        return self.x

    def bprop(self, y):
        return (self.x > 0).astype(np.int)


class Line(object):
    def __init__(self, a=1, b=0, name='Line'):
        assert isinstance(name, str), 'Name must be of type string'
        assert isinstance(a, Number), 'a must be a Number'
        assert isinstance(b, Number), 'b must be a Number'
        warnings.warn('Line is not a non-linearity', UserWarning)
        self.name = name
        self.a = a
        self.b = b

    def __repr__(self):
        rep_str = '{}, '.format(self.name)
        rep_str = '{}X + {}, '.format(self.a, self.b)
        return rep_str

    def fprop(self, x):
        '''
        :param: x
        :type: np.ndarray
        '''
        assert isinstance(x, np.ndarray), 'must be a numpy vector'

        return self.a * x + self.b

    def bprop(self, y):
        return self.a


class Identity(Line):
    def __init__(self, name='Identity'):
        super(Identity, self).__init__(name='Identity')
        warnings.warn('Identity is not a non-linearity', UserWarning)

    def __repr__(self):
        rep_str = '{}'.format(self.name)
        return rep_str
