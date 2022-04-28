# Analyzing the Effect of Various Activation Functions on Diverse Datasets - A Survey

This is an attempt made to understand the performance of various Activation Functions on different datasets. For this we have used ResNet-18 model, the datasets used are CIFAR-10, MNIST and CheXpert. In this repository, we provide the installation steps, packages required, data preparation and code for all the experiments we ran.

We have categorised the Activation Functions based on their characteristics as :

## Classical activation functions

1. Sigmoid = nn.Sigmoid()

2. Hard Sigmoid Function = nn.Hardsigmoid()

3. Hyperboilic Tangent = nn.Tanh()

4. Hard Hyperbolic Function = nn.Hardtanh(-2, 2)

## Rectified activation functions

1. Rectified Linear Unit Function [ReLU] = nn.ReLU(inplace=TRUE)

2. LeakyReLu (LReLU) = nn.LeakyReLU(0.1)

3. Parametric ReLU (PReLU) = nn.PReLU()

4. Randomized ReLU (RReLU) = nn.RReLU(0.1, 0.3)

## Exponential activation functions

1. Exponential Linear Unit Function [ELU] = nn.ELU()

2. Scaled Exponential Linear Units (SELU) = nn.SELU() 

3. Continuously Differentiable Exponential Linear Unit Function [CELU] = nn.CELU() 

## Hybrid activation functions

1. Swish = nn.SiLU()

2. Hardswish = nn.Hardswish()

3. Mish = nn.Mish()

## Others

Gaussian Error Linear Unit Function [GELU] = nn.GELU()


## Installation
Install [Pytorch](https://pytorch.org/get-started/locally/) 

Install the following Python dependencies (with `pip install`):
    timm
    wandb
    torchxrayvision
    sklearn

We have also provided requirements.txt file. You can just install using this `pip install -r requirements.txt`

## Training

### Data preparation
Download the respective datasets and add in data folder under ActivationFunctions. The links for datasets are provided here.

1. [CIFAR-10] (https://www.cs.toronto.edu/~kriz/cifar.html) 
2. [MNIST] (http://yann.lecun.com/exdb/mnist/)
3. CheXpert 

Then set the parameters required to train in .json files under params folder and run.

### Training
` python run.py `