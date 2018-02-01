import os
import numpy as np
from urllib.request import urlretrieve
import gzip

'''
code modified from:
https://stackoverflow.com/questions/43149272/cannot-get-mnist-database-through-anaconda-jupyter

Used for importing mnist dataset
'''


def download(filename, source='http://yann.lecun.com/exdb/mnist/',
             save_folder=None):
    print('Downloading {}'.format(save_folder + filename))
    urlretrieve(source + filename, save_folder + filename)


def load_mnist_images(filename, save_folder=None):
    if not os.path.exists(save_folder + filename):
        download(filename, save_folder=save_folder)
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(save_folder + filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    data = data.reshape(-1, 28 * 28)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / 255.


def load_mnist_labels(filename, save_folder=None):
    if not os.path.exists(save_folder + filename):
        download(filename, save_folder=save_folder)
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(save_folder + filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data


def load_mnist_data(save_folder=None):
    X_train = load_mnist_images('train-images-idx3-ubyte.gz',
                                save_folder=str(save_folder))
    Y_train = load_mnist_labels('train-labels-idx1-ubyte.gz',
                                save_folder=str(save_folder))
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz',
                               save_folder=str(save_folder))
    Y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz',
                               save_folder=str(save_folder))
    return ((X_train, Y_train), (X_test, Y_test))


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist_data(save_folder='')

    print('Training Labels Shape: {}, Data Shape: {}'.format(y_train.shape,
                                                             X_train.shape))
    print('Testing Labels Shape: {}, Data Shape: {}'.format(y_test.shape,
                                                            X_test.shape))
