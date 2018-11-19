#!/usr/bin/env python
"""
SIMNN Library
"""
__author__ = 'Vitaliy Chiley'
__date__ = '01/2018'

from simnn.activations import LogisticSigmoid, Softmax, ReLU, Line, Identity
from simnn.costs import CrossEntropy, BinaryCrossEntropy
from simnn.layers import Linear, Mean_Center, BatchNormalization1D, BatchNormalization
from simnn.model import Model
from simnn.utils import one_hot
from simnn.optimizers import StochasticGradientDecent, MomentumOptimizer