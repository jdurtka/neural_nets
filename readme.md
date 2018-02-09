# neural_nets
Machine learning class project: neural networks with backpropagation

## Getting Started

There is no installation, just create a directory and download the repository.

### Prerequisites

The neural_nets project uses *pandas* and *numpy* for math operations, as well as a whole host of ordinary system libraries (*sys*, *logging*, *argparse*, *os*, *time*, *random*, *collections*, *pickle*). Python 3.5.2 was used for development; the project has not been tested with older versions of Python.

## Description

The neural_nets project implements a feedforward multi-layer neural network of arbitrary size and a radial-basis function network of arbitrary size. These are used to model continuous functions with multi-dimensional inputs and outputs, and trained with gradient descent through backpropagation. Some meta-algorithms are also implemented, such as different forms of gradient descent (batch, stochastic, and mini-batch) and k-fold cross-validation. An application is included, which is used as an interface to these algorithms.

## Running the application

The application relies on the assumption that all features (input and target) are real-valued, and ideally of a small magnitude (no scaling is implemented, so values should be approximately between -1 and 1, although nothing explicitly prohibits larger values). A further assumption made by the application is that there is only a single target variable to be predicted (although the neural network implemented supports arbitrary outputs).

Assuming the directories *runs* and *datasets* exist, and that there is a dataset in CSV format that meets the expectations described, one can run the program using the following command:

```
python train_nets.py -i datasets/my_dataset.csv -od runs/my_output_directory/ -vvvvv -epochs 25000
```

Note that the application will create the directory *runs/my_output_directory/* but that the directory *runs* needs to already exist.

By default, 5-fold validation is implemented. The number of epochs prescribed (in the above example, *25,000*) is divided by the number of training folds (so in this case, each fold is trained for *5,000* epochs). The application is hard-coded to train over the following configurations of networks:

* No hidden layers
* One hidden layer with either 10 or 100 nodes
* Two hidden layers, each with either 10 or 100 nodes
* RBF network with either 2, 3, or 4 radial basis functions

## Authors

* **Mitch Baker** - *Initial algorithm research and results analysis*
* **James Durtka** - *Design and programming*
* **Hongchuan "Rocky" Wang** - *Input data analysis and experimental design*