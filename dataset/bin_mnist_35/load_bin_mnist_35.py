import os
import numpy as np


def load_data(f_names=None):

    if f_names is None:
        f_names = ['new_train3.txt', 'new_train5.txt']

    assert isinstance(f_names, list), 'f_names must be a list'

    x_arr = []
    y_arr = []
    for fname in f_names:
        label = fname[fname.index(os.path.splitext(fname)[1]) - 1]

        x_arr.append(np.loadtxt(fname))
        y_arr.append(int(label) * np.ones((len(x_arr[-1]))))

    return (np.concatenate(x_arr, axis=0), np.concatenate(y_arr).astype(int))
