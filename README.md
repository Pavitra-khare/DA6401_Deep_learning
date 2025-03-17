# Deep Learning (CS6910) Assignment-1

## Repository Overview

This repository contains implementations for training a feedforward neural network using only NumPy. The network is designed to work efficiently with datasets like MNIST and Fashion MNIST, offering customization for activation functions, loss functions, and various optimization methods.

[GitHub Repository](https://github.com/Pavitra-khare/DA6401_Deep_learning/tree/main)  
[Weights & Biases Report](https://api.wandb.ai/links/3628-pavitrakhare-indian-institute-of-technology-madras/6l8em45m)

## Repository Contents

### `train.py`
This script allows training a feedforward neural network using specified hyperparameters. It accepts command-line arguments to customize training settings. If executed without arguments, it runs with optimized default parameters. 

### `DL_Assignment1_ALL_QUES.ipynb`
This Jupyter Notebook contains all the assignment questions along with their implementations and outputs.

### `DL_Assignment1_sweepsLog.ipynb`
This file logs all the output from running parameter sweeps using Weights & Biases.

## Features
- Configurable neural network design
- Support for activation functions: Sigmoid, Tanh, Identity, ReLU
- Loss functions: Mean Squared Error and Cross Entropy
- Optimizers: SGD, Momentum, NAG, RMSprop, Adam, Nadam
- Weight initialization methods: Xavier and Random
- Logging and visualization using Weights & Biases

## Best Parameters
- **Epochs**: 10  
- **Batch size**: 32  
- **Loss function**: Cross Entropy  
- **Optimizer**: Nadam  
- **Learning rate**: 1e-3  
- **Weight decay constant**: 0.5  
- **Weight initialization**: Xavier  
- **Number of hidden layers**: 3  
- **Hidden layer size**: 128  
- **Activation function**: ReLU  

## Best Results Obtained
- **Train Accuracy**: 91.25%
- **Validation Accuracy**: 88.94%
- **Test Accuracy**: 87.57%

## Command-Line Arguments

| Name          | Default Value   | Description |
|--------------|----------------|-------------|
| `-wp`, `--wandb_project`  | myprojectname  | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity`  | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset`  | fashion_mnist  | choices: ["mnist", "fashion_mnist"] |
| `-e`, `--epochs`  | 1  | Number of epochs to train neural network. |
| `-b`, `--batch_size`  | 4  | Batch size used to train neural network. |
| `-l`, `--loss`  | cross_entropy  | choices: ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer`  | sgd  | choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] |
| `-lr`, `--learning_rate`  | 0.1  | Learning rate used to optimize model parameters |
| `-m`, `--momentum`  | 0.5  | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta`  | 0.5  | Beta used by rmsprop optimizer |
| `-beta1`, `--beta1`  | 0.5  | Beta1 used by adam and nadam optimizers. |
| `-beta2`, `--beta2`  | 0.5  | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon`  | 0.000001  | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay`  | 0.0  | Weight decay used by optimizers. |
| `-w_i`, `--weight_init`  | random  | choices: ["random", "Xavier"] |
| `-nhl`, `--num_layers`  | 1  | Number of hidden layers used in feedforward neural network. |
| `-sz`, `--hidden_size`  | 4  | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation`  | sigmoid  | choices: ["identity", "sigmoid", "tanh", "ReLU"] |


## Training a Model
To train a model, run:
```bash
python train.py --wandb_entity myname --wandb_project myprojectname
```

This will start training using the specified parameters or defaults if no arguments are provided.
