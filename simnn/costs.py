import numpy as np


class CrossEntropy(object):
    def __init__(self, name='CrossEntropyCost'):
        assert isinstance(name, str), 'Name must be of type string'
        self.name = name

    def __repr__(self):
        rep_str = '{}\n'.format(self.name)
        return rep_str

    def fprop(self, t, y, epsilon=1e-9):
        assert isinstance(t, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert y.shape == t.shape

        y = np.clip(y, epsilon, 1.)  # ensure log(0) doesn't happen

        self.y = - np.sum(t * np.log(y))

        return self.y

    def bprop(self, t, y):
        return y - t
