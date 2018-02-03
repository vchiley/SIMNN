# DeepWater Systems, Oil Spill Classifier

In an dual effort, Zoe and I have created the following Deep Learning architectuer to classify oil spills on the ocean floor.

## Network Performance
The network is set to run for a maximum of 128 but will early stop enabled, stopping the network once it has converged.
The network achieves 90%+ performance within the first epoch and eventually converges to about 97%+ on the validation and test sets

## Network Description

Our final had the following layer definitions:
```python
nh = 64
# define model structure
layers = [Linear(out_shape=nh, activation=ReLU(), init='lecun_normal'),
          PM_BN(nh),
          Linear(out_shape=10, activation=Softmax(), init='lecun_normal')]
```

### PM_BN

The `PM_BN` layer is what we call a 'Poor Mans Batch Normalization' layer. It mean centers the data like Batch Normalization but does not perform a complete z-score of the batch like Convensoinal Batch Normalization. The architecture with `PM_BN` was used because of its ability to converge faster and its better performance with smaller network architectures. Having a large hidden layer a network without `PM_BN` yeilds comprable results as a network which has the layer.

### Network Parameters

The Network has 784 x 64 + 64 parameters in the first layer, 64 parameters in the PM_BN layer, and 64 x 10 + 10 in the last layer. This is a total of 784 x 64 + 64 + 64 + 64 x 10 + 10 = 50954 tunnable parameters which meets early stops conditions in about 50 epochs, but usually converges to accuracies of 97% on the val set within 25 epochs.

## Result Replication 
The results can be replicated using: `mnist_mlp_PM_BN.py`

To run example: `python mnist_mlp_PM_BN.py --verbose --mnist_dir /path_to/mnist_dataset/`

## Experiment Replication

Experimentation was performed, can be viewed and replicated in the following Jupyter Notebook: `mnist_mlp_dev.ipynb`

## Usage and Tests

Classifier created using SIMNN, see [SIMNN](https://github.com/vchiley/SIMNN) for documentation, usage examples and network testing paradigm. 

## Enviornment
An `environment.yml` file is provided which describes my current environment. If you cannot run my code for some reason, please follow the [instructions for creating a venv using anaconda](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) from an `environment.yml` file
