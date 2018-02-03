# DeepWater Systems, Oil Spill Classifier

In an dual effort, Zoe and I have created the following Deep Learning architectuer to classify oil spills on the ocean floor.

## Final Network Description

Our final had the following layer definitions:
```python
nh = 64
# define model structure
layers = [Linear(out_shape=nh, activation=ReLU(), init='lecun_normal'),
          PM_BN(nh),
          Linear(out_shape=10, activation=Softmax(), init='lecun_normal')]
```

Note: the `PM_BN` layer is what we call a 'Poor Mans Batch Normalization' layer. It mean centers the data like Batch Normalization but does not perform a complete z-score. The architecture with `PM_BN` was used because of its ability to converge faster and better on smaller networks. Having a large hidden layer a network without `PM_BN` yeilds comprable results as having the layer.

## Result Replication 
The results can be replicated using: `mnist_mlp_PM_BN.py`

To run example: `python mnist_mlp_PM_BN.py --verbose --mnist_dir /path_to/mnist_dataset/`

## Experiment Replication

Experimentation was performed, can be viewed and replicated in the following Jupyter Notebook: `mnist_mlp_dev.ipynb`

## Usage and Tests

Classifier created using SIMNN, see [SIMNN](https://github.com/vchiley/SIMNN) for documentation, usage examples and network testing paradigm. 

## Enviornment
An `environment.yml` file is provided which describes my current environment. If you cannot run my code for some reason, please follow the [instructions for creating a venv using anaconda](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) from an `environment.yml` file
