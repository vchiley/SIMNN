import numpy as np


def one_hot(y):
    '''
    creates a one hot vector from list of classes

    assumes classes \in 1...m
    '''
    assert isinstance(y, np.ndarray)
    m = len(set(y))
    y_out = np.zeros((y.shape[0], m), dtype=int)
    for i, yi in enumerate(y):
        y_out[i, yi] = 1
    return y_out
