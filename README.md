# SIMNN
Simple Neural Network (SIMNN) is a personal project in which I implement a Simple Neural Network library based on my understanding of Neural Networks. This understanding was mostly aquired from Professor Manuela Vasconcelos in ECE 271B, Discriminant Learning and using NN libraries.  
This will not be an extensive or optimized library which can be run with GPU's, this is rather a simple NN implementation which is easily human readable. Written in python using numpy for simplicity.  
An example of how to use it can be found as an ipython notebook in example.ipynb

SIMNN has implemented:
- Neural Network Model
- Layers
	- Linear (Actually Affine)
	- PM_BN: 'Poor Mans Batch Normalization'
- Activations
	- Sigmoid
	- Softmax
	- ReLU
- Costs
	- CrossEntropy
	- BinaryCrossEntropy
- Initialization Types
	- lecun_normal
	- xavier_normal
	- he_normal
	- lecun_uniform
	- xavier_uniform
	- he_uniform

# Usage
To instantiate a network, you must first define a list of layer objects for the network architecture:

'''

	from simnn import Linear
	from simnn import ReLU, Softmax

	layers = [Linear(out_shape=nh, activation=ReLU(), bias=True,
                     init='lecun_normal'),
              Linear(out_shape=10, activation=Softmax(), bias=True,
                     init='lecun_normal')]
'''

To instantiate Neural Network Model and fit to data:

'''

	from simnn import Model, CrossEntropy

	model = Model(layers, dataset, CrossEntropy())

    # fit model to datas
    model.fit(dataset, num_epochs=num_epochs, b_size=128, verbose=verbose)
'''

Other parameters can be passed into the network as well

Usage exampels in Jupyter Notebooks (all using the mnist dataset) can be seen in: 1. 'logistic_regression_3_and_5_mnist.ipynb' 2. 'mnist_mlp.ipynb', 3. 'mnist_mlp_dev.ipynb'

A usage example in a python script can be seen in: 'mnist_mlp_PM_BN.py'

# Network Testing
To test the network run 'python tests.py'.
In 'tests.py' a numerical gradient checker is implimented which checks weight gradients for different layers as compared to a numerical gradient approximation of the gradient. The Jupyter Notebook, 'bprop_numerical_grad_check.ipynb', describs the method used.

# Enviornment
An environment.yml file is provided which describes my current environment. If you cannot run my code for some reason, please follow the [instructions for creating a venv using anaconda](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) from an `environment.yml` file
