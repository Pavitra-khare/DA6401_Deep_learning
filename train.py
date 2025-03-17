import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.datasets import fashion_mnist
import wandb
import argparse

#arguments taken from command line
parser = argparse.ArgumentParser()
parser.add_argument("-wp","--wandb_project",help="Wandb project name",default="DA6401_ass_1_run")
parser.add_argument("-we","--wandb_entity",help="Wandb Entity name.",default="3628-pavitrakhare")
parser.add_argument("-d","--dataset",help="choices: ['mnist', 'fashion_mnist']",choices=['mnist', 'fashion_mnist'],default="fashion_mnist")
parser.add_argument("-e","--epochs",help="Number of epochs.",default=10)
parser.add_argument("-b","--batch_size",help="batch size in which data needs to be divided.",default=32)
parser.add_argument("-l","--loss",help="choices: ['cross_entropy','mean_squared_error']",choices=['cross_entropy','mean_squared_error'],default='cross_entropy')
parser.add_argument("-o","--optimizer",help="choices: [ 'rmsprop', 'adam', 'nadam','sgd', 'momentum', 'nag']",choices=['sgd','momentum','nag','rmsprop','adam','nadam'],default='nadam')
parser.add_argument("-lr","--learning_rate",help="Learning rate ",default='1e-3')
parser.add_argument("-m","--momentum",help="momentum and nag optimizers momentum.",default=0.5)
parser.add_argument("-beta","--beta",help="rmsprop optimizer beta",default=0.5)
parser.add_argument("-beta1","--beta1",help="adam and nadam optimizers uses beta1.",default=0.5)
parser.add_argument("-beta2","--beta2",help="adam and nadam optimizers uses beta2.",default=0.5)
parser.add_argument("-eps","--epsilon",help="Epsilon used by optimizers.",default=0.001)
parser.add_argument("-w_d","--weight_decay",help="Weight decay.",default=0)
parser.add_argument("-w_i","--weight_init",help="choices: ['random', 'Xavier']",choices=['random','Xavier'],default='Xavier')
parser.add_argument("-nhl","--num_layers",help="Number of hidden layers used in feedforward neural network.",default=3)
parser.add_argument("-sz","--hidden_size",help="Number of hidden neurons in a layer.",default=128)
parser.add_argument("-a","--activation",help="choices: ['identity', 'sigmoid', 'tanh', 'ReLU']",choices=['identity','sigmoid','tanh','ReLU'],default='ReLU')
args = parser.parse_args()

#defining activation function
def sigmoid(x):
    
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    
    return np.tanh(x)

def ReLU(x):
    
    return np.maximum(0, x)

def softmax(x):
    
    x_shifted = x - np.max(x, axis=1, keepdims=True)  # Improve numerical stability
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def identity(x):
    
    return x

def cross_entropy(y_hat, y):
    
    epsilon = 1e-9  # Avoid log(0) errors
    return -np.mean(np.sum(y * np.log(y_hat + epsilon), axis=1))

def mean_squared_error(y_hat, y):
  
    return np.mean((y - y_hat) ** 2)

def activation_functions(x, fn_label="sigmoid"):
    
    activations = {
        "ReLU": ReLU,
        "sigmoid": sigmoid,
        "tanh": tanh,
        "softmax": softmax,
        "identity": identity
    }
    return activations.get(fn_label, lambda x: "error")(x)

def activation_derivative(x, fn_label="sigmoid"):
    
    derivatives = {
        "ReLU": lambda x: np.where(x > 0, 1, 0),
        "tanh": lambda x: 1.0 - np.tanh(x) ** 2,
        "sigmoid": lambda x: sigmoid(x) * (1 - sigmoid(x)),
        "identity": lambda x: np.ones_like(x)
    }
    return derivatives.get(fn_label, lambda x: "error")(x)



def map_data_with_classes(labels):
    
    num_samples = len(labels)
    num_classes = max(labels) + 1
    one_hot_matrix = np.zeros((num_samples, num_classes))

    for idx, label in enumerate(labels):
        one_hot_matrix[idx][label] = 1

    return one_hot_matrix

#load the data----------------------------

if args.dataset == 'fashion_mnist':
    (train_X,train_Y),(test_X,test_Y) = fashion_mnist.load_data()
else:
    (train_X,train_Y),(test_X,test_Y) = mnist.load_data()


train_X, test_X = train_X / 255.0, test_X / 255.0

needed_y_train, needed_y_test = train_Y, test_Y

# dataset train-test split
trainX, val_X, trainy, valy = train_test_split(train_X, train_Y, test_size=0.1, random_state=40)


trainX = trainX.reshape(trainX.shape[0], -1)
testX = test_X.reshape(test_X.shape[0], -1)
valX = val_X.reshape(val_X.shape[0], -1)

# Adjust dataset size to be multiples of 128
batch_size = 128
trainX, testX, valX = (arr[:(len(arr) // batch_size) * batch_size] for arr in [trainX, testX, valX])
trainy, test_Y, valy = (arr[:(len(arr) // batch_size) * batch_size] for arr in [trainy, test_Y, valy])

# Convert class labels into one-hot encoded format
trainy = map_data_with_classes(trainy)
testy = map_data_with_classes(test_Y)
valiy = map_data_with_classes(valy)

# Determine input and output layer sizes
input_layer_size = trainX.shape[1]
output_layer_size = trainy.shape[1]

# function to initialize the weights and biases

def initialize_weights_and_biases(layers, number_hidden_layers=1, init_type='random'):
    
    weights, biases = [], []
    
    for i in range(number_hidden_layers + 1):
        input_dim, output_dim = layers[i]["input_size"], layers[i]["output_size"]

        if init_type == 'random':
            w = np.random.randn(output_dim, input_dim) * 0.01
            b = np.zeros((output_dim, 1))
        else:
            bound = np.sqrt(6 / (input_dim + output_dim))
            w = np.random.uniform(-bound, bound, (output_dim, input_dim))
            b = np.random.uniform(-bound, bound, (output_dim, 1))
        

        weights.append(w)
        biases.append(b)

    return weights, biases



def train_accuracy(batch_testy, y_predicted, trainy):
    
    correct_count = 0

    for i in range(len(batch_testy)):
        for j in range(len(batch_testy[i])):
            actual_label = np.argmax(batch_testy[i][j])
            predicted_label = np.argmax(y_predicted[i][j])

            if predicted_label == actual_label:
                correct_count += 1

    return correct_count / len(trainy)

# calculate validation and test accuracy
def test_accuracy(testX, testy, weights, biases, number_hidden_layers, activation_function, output_function):
    
    _, activations = forward_propagation(testX, weights, biases, number_hidden_layers, activation_function, output_function)
    y_pred = activations[-1]  # Get final layer activations
    y_predicted = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices

    # Convert testy to class indices if one-hot encoded
    if testy.ndim > 1 and testy.shape[1] > 1:
        testy = np.argmax(testy, axis=1)

    # Ensure both lists are of the same length
    min_len = min(len(y_predicted), len(testy))
    return np.mean(y_predicted[:min_len] == testy[:min_len])

# calculating regularizing term that is to be added to loss
def calculate_regularizing_term(y, weight_decay_const, number_hidden_layers, weights):
    
    total_weight_sum = sum(np.sum(weight ** 2) for weight in weights)
    return (weight_decay_const / (2 * len(y))) * total_weight_sum

# calculating validation loss
def val_loss(valX, valy, weights, biases, number_hidden_layers, activation_function, output_function, loss_function):
    

    # Perform forward propagation
    _, h = forward_propagation(valX, weights, biases, number_hidden_layers, activation_function, output_function)

    # Retrieve the predicted values
    y_pred = h[-1]

    # Compute loss based on the chosen function
    if loss_function == 'cross_entropy':
        error = cross_entropy(y_pred, valy)
    elif loss_function == 'mean_squared_error':
        error = mean_squared_error(y_pred, valy)
    else:
        raise ValueError("Invalid loss function. Choose 'cross_entropy' or 'mean_squared_error'.")

    return error



def forward_propagation(input_x, W, B, hidden_layers, activ_label, op_label):
    

    # Initialize lists for activations and hidden states
    a, h = [], []

    # Reshape input if necessary
    input_x = input_x.reshape(len(input_x), -1)

    # Compute activations for first hidden layer
    first_activation = np.dot(W[0], input_x.T) + B[0]
    a.append(first_activation)
    h.append(activation_functions(first_activation, activ_label))

    # Forward propagate through hidden layers
    for i in range(1, hidden_layers):
        layer_activation = np.dot(W[i], h[i-1]) + B[i]
        a.append(layer_activation)
        h.append(activation_functions(layer_activation, activ_label))

    # Compute activations for the output layer
    output_activation = np.dot(W[hidden_layers], h[-1]) + B[hidden_layers]
    final_output = activation_functions(output_activation.T, op_label).T
    a.append(output_activation)
    h.append(final_output)

    # Ensure all activations are transposed for consistency
    a = [activation.T for activation in a]
    h = [hidden.T for hidden in h]

    return a, h

def backward_propagation(batch_trainy, batch_trainX, y_hat, activations, hidden_states, weights, num_hidden_layers, derivative_function='sigmoid'):
    

    weight_gradients, bias_gradients, activation_gradients, hidden_gradients = {}, {}, {}, {}

    # Reshape batch_trainy to ensure correct dimensions
    batch_trainy = batch_trainy.reshape(batch_trainy.shape[0], batch_trainy.shape[1])

    epsilon = 1e-8  # Small value to prevent division errors
    last_activation_key = f'a{num_hidden_layers + 1}'
    last_hidden_key = f'h{num_hidden_layers + 1}'

    activation_gradients[last_activation_key] = -(batch_trainy - y_hat)
    hidden_gradients[last_hidden_key] = -(batch_trainy / (y_hat + epsilon))

    num_samples = len(batch_trainX)

    # Backpropagation from output layer to first hidden layer
    for layer in range(num_hidden_layers + 1, 1, -1):
        weight_key = f'W{layer}'
        bias_key = f'b{layer}'
        activation_key = f'a{layer}'
        prev_activation_key = f'a{layer - 1}'
        prev_hidden_key = f'h{layer - 1}'

        # Compute weight gradient
        weight_gradients[weight_key] = np.dot(activation_gradients[activation_key].T, hidden_states[layer - 2])

        # Apply L2 regularization
        weight_gradients[weight_key] += (args.weight_decay * weights[layer - 1])
        weight_gradients[weight_key] /= num_samples

        # Compute bias gradient
        bias_gradients[bias_key] = activation_gradients[activation_key]

        # Compute hidden gradients and activation gradients
        hidden_gradients[prev_hidden_key] = np.dot(weights[layer - 1].T, activation_gradients[activation_key].T)
        activation_gradients[prev_activation_key] = np.multiply(hidden_gradients[prev_hidden_key], activation_derivative(activations[layer - 2].T, derivative_function))
        activation_gradients[prev_activation_key] = activation_gradients[prev_activation_key].T

    # Compute gradients for the first layer (no hidden gradients needed)
    weight_gradients['W1'] = np.dot(activation_gradients['a1'].T, batch_trainX)
    bias_gradients['b1'] = activation_gradients['a1']

    # Normalize biases across samples
    for layer in range(1, len(bias_gradients) + 1):
        bias_key = f'b{layer}'
        bias_gradients[bias_key] = np.mean(bias_gradients[bias_key], axis=0).reshape(-1, 1)

    return weight_gradients, bias_gradients

def gradient_descent(trainX, trainy, number_hidden_layers=1, hidden_layer_size=4, eta=0.1, initial_weights='random', activation_function='sigmoid', epochs=1, output_function='softmax', mini_batch_size=4, loss_function='cross_entropy', weight_decay_const=0, wandb_flag=False):
    INPUT_KEY = 'input_size'
    OUTPUT_KEY = 'output_size'
    FUN_KEY = "function"
    layers = [{INPUT_KEY: input_layer_size, OUTPUT_KEY: hidden_layer_size, FUN_KEY: activation_function}]
    
    for _ in range(number_hidden_layers - 1):
        layers.append({INPUT_KEY: hidden_layer_size, OUTPUT_KEY: hidden_layer_size, FUN_KEY: activation_function})
    
    layers.append({INPUT_KEY: hidden_layer_size, OUTPUT_KEY: output_layer_size, FUN_KEY: output_function})
    
    # Initialize model parameters
    weights, biases = initialize_weights_and_biases(layers, number_hidden_layers, initial_weights)
    
    x_total = len(trainX)
    num_batches = x_total // mini_batch_size
    mini_batch_trainX = np.array_split(trainX, num_batches)
    mini_batch_trainy = np.array_split(trainy, num_batches)
    
    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []
    h = None
    
    for epoch in range(epochs):
        total_train_loss = 0
        predictions = []
        
        for batch_idx in range(len(mini_batch_trainX)):
            activations, h = forward_propagation(mini_batch_trainX[batch_idx], weights, biases, number_hidden_layers, activation_function, output_function)
            predictions.append(h[-1])
            
            if loss_function == 'cross_entropy':
                total_train_loss += cross_entropy(h[-1], mini_batch_trainy[batch_idx])
            elif loss_function == 'mean_squared_error':
                total_train_loss += mean_squared_error(h[-1], mini_batch_trainy[batch_idx])
            else:
                raise ValueError('Invalid loss function specified')
            
            del_W, del_b = backward_propagation(mini_batch_trainy[batch_idx], mini_batch_trainX[batch_idx], h[-1], activations, h, weights, number_hidden_layers, activation_function)
            
            for idx in range(len(weights)):
                weights[idx] -= eta * del_W[f'W{idx + 1}']
                biases[idx] -= eta * del_b[f'b{idx + 1}']
        
        reg_train_loss = calculate_regularizing_term(trainy, weight_decay_const, number_hidden_layers, weights)
        final_train_loss = total_train_loss / num_batches + reg_train_loss
        
        val_loss_value = val_loss(valX, valiy, weights, biases, number_hidden_layers, activation_function, output_function, loss_function)
        reg_val_loss = calculate_regularizing_term(valiy, weight_decay_const, number_hidden_layers, weights)
        final_val_loss = val_loss_value + reg_val_loss
        
        print(f"Epoch {epoch + 1}: Validation Loss = {final_val_loss}")
        
        train_loss_list.append(final_train_loss)
        val_loss_list.append(final_val_loss)
        
        train_acc = train_accuracy(mini_batch_trainy, predictions, trainy)
        train_acc_list.append(train_acc)
        
        val_acc = test_accuracy(valX, valy, weights, biases, number_hidden_layers, activation_function, output_function)
        val_acc_list.append(val_acc)
        
        if wandb_flag:
            wandb.log({"loss": final_train_loss, "val_loss": final_val_loss, "accuracy": train_acc, "val_accuracy": val_acc, "epoch": epoch})
    
    return h[-1], weights, biases, [train_loss_list, val_loss_list, train_acc_list, val_acc_list]




def momentum_based_gradient_descent(trainX, trainy, number_hidden_layers=1, hidden_layer_size=4, eta=0.1, initial_weights='random', activation_function='sigmoid', epochs=1, output_function='softmax', mini_batch_size=4, loss_function='cross_entropy', weight_decay_const=0, wandb_flag=False, m_beta=0.9):
    
    # Define layer configurations
    layers = []
    input_layer = {'input_size': trainX.shape[1], 'output_size': hidden_layer_size, 'activation': activation_function}
    layers.append(input_layer)

    for _ in range(number_hidden_layers - 1):
        hidden_layer = {'input_size': hidden_layer_size, 'output_size': hidden_layer_size, 'activation': activation_function}
        layers.append(hidden_layer)

    output_layer = {'input_size': hidden_layer_size, 'output_size': trainy.shape[1], 'activation': output_function}
    layers.append(output_layer)

    # Initialize weights and biases
    weights, biases = initialize_weights_and_biases(layers,number_hidden_layers, initial_weights)

    # Split data into mini-batches
    num_batches = len(trainX) // mini_batch_size
    mini_batches_X = np.array_split(trainX, num_batches)
    mini_batches_y = np.array_split(trainy, num_batches)

    # Initialize momentum terms
    momentum_weights = [np.zeros_like(w) for w in weights]
    momentum_biases = [np.zeros_like(b) for b in biases]

    # Lists to store training and validation metrics
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        predictions = []

        for batch_X, batch_y in zip(mini_batches_X, mini_batches_y):
            # Forward propagation
            activations, hidden_states = forward_propagation(batch_X, weights, biases, number_hidden_layers, activation_function, output_function)
            predictions.append(hidden_states[-1])

            # Compute loss
            if loss_function == 'cross_entropy':
                batch_loss = cross_entropy(hidden_states[-1], batch_y)
            elif loss_function == 'mean_squared_error':
                batch_loss = mean_squared_error(hidden_states[-1], batch_y)
            else:
                raise ValueError("Unsupported loss function")

            epoch_loss += batch_loss

            # Backward propagation
            grad_weights, grad_biases = backward_propagation(batch_y, batch_X, hidden_states[-1], activations, hidden_states, weights, number_hidden_layers, activation_function)

            # Update weights and biases with momentum
            for i in range(len(weights)):
                # Access gradients using keys (e.g., 'W1', 'b1', 'W2', 'b2', etc.)
                keyW = 'W' + str(i + 1)
                keyB = 'b' + str(i + 1)

                # Momentum update for weights
                momentum_weights[i] = momentum_weights[i] * (args.momentum) + grad_weights[keyW] * eta
                weights[i] -= momentum_weights[i]

                # Momentum update for biases
                momentum_biases[i] = momentum_biases[i] * (args.momentum) + grad_biases[keyB] * eta
                biases[i] -= momentum_biases[i]

        # Calculate training accuracy and loss
        train_acc = train_accuracy(mini_batches_y,predictions,trainy)
        reg_term_train = calculate_regularizing_term(trainy,weight_decay_const,number_hidden_layers ,weights)
        avg_train_loss = epoch_loss / num_batches + reg_term_train

        # Calculate validation accuracy and loss
        val_acc = test_accuracy(valX, valiy, weights, biases, number_hidden_layers, activation_function, output_function)
        val_loss_value = val_loss(valX, valiy, weights, biases, number_hidden_layers, activation_function, output_function, loss_function)
        reg_term_val = calculate_regularizing_term(valiy,weight_decay_const ,number_hidden_layers, weights)
        val_loss_value += reg_term_val

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss_value)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Log metrics if wandb is enabled
        if wandb_flag:
            wandb.log({
                "loss": avg_train_loss,
                "val_loss": val_loss_value,
                "accuracy": train_acc,
                "val_accuracy": val_acc,
                "epoch": epoch
            })

        print(f"Epoch: {epoch + 1}, Validation Loss: {val_loss_value}")

    # Return final predictions, weights, biases, and metrics
    return hidden_states[-1], weights, biases, [train_losses, val_losses, train_accuracies, val_accuracies]





def nestrov_accelerated_gradient_descent(trainX, trainy, number_hidden_layers=1, hidden_layer_size=4, eta=0.1,
                                         initial_weights='random', activation_function='sigmoid', epochs=1,
                                         output_function='softmax', mini_batch_size=4, loss_function='cross_entropy',
                                         weight_decay_const=0, wandb_flag=False):
    """
    Implements Nesterov Accelerated Gradient Descent (NAG) for neural network optimization.
    """
    # Define network layers dynamically
    INPUT_KEY = 'input_size'
    OUTPUT_KEY = 'output_size'
    FUN_KEY = "function"
    layers = [{INPUT_KEY: input_layer_size, OUTPUT_KEY: hidden_layer_size, FUN_KEY: activation_function}]
    layers.extend([{INPUT_KEY: hidden_layer_size, OUTPUT_KEY: hidden_layer_size, FUN_KEY: activation_function}
                   for _ in range(number_hidden_layers - 1)])
    layers.append({INPUT_KEY: hidden_layer_size, OUTPUT_KEY: output_layer_size, FUN_KEY: output_function})

    # Initialize weights and biases
    weights, biases = initialize_weights_and_biases(layers, number_hidden_layers, initial_weights)

    # Determine batch processing details
    num_samples = len(trainX)
    num_batches = num_samples // mini_batch_size
    mini_batches_X = np.array_split(trainX, num_batches)
    mini_batches_y = np.array_split(trainy, num_batches)

    # Initialize past gradients for momentum update
    momentum_weights = [np.zeros_like(w) for w in weights]
    momentum_biases = [np.zeros_like(b) for b in biases]

    # Lists to track training progress
    loss_train, loss_val, acc_train, acc_val = [], [], [], []

    for epoch in range(epochs):
        total_loss = 0
        y_predictions = []

        for batch_X, batch_y in zip(mini_batches_X, mini_batches_y):
            # Compute lookahead weights and biases
            lookahead_W = [weights[i] - (args.momentum * momentum_weights[i]) for i in range(len(weights))]
            lookahead_B = [biases[i] - (args.momentum * momentum_biases[i]) for i in range(len(biases))]

            activations, h_states = forward_propagation(batch_X, lookahead_W, lookahead_B, number_hidden_layers, activation_function, output_function)
            y_predictions.append(h_states[-1])

            # Compute loss
            total_loss += cross_entropy(h_states[-1], batch_y) if loss_function == 'cross_entropy' else mean_squared_error(h_states[-1], batch_y)

            # Compute gradients
            grad_W, grad_B = backward_propagation(batch_y, batch_X, h_states[-1], activations, h_states, lookahead_W, number_hidden_layers, activation_function)

            # Apply Nesterov update
            for i in range(len(weights)):
                momentum_weights[i] = (args.momentum * momentum_weights[i]) + (eta * grad_W[f'W{i+1}'])
                momentum_biases[i] = (args.momentum * momentum_biases[i]) + (eta * grad_B[f'b{i+1}'])

                weights[i] -= momentum_weights[i]
                biases[i] -= momentum_biases[i]

        # Compute regularized loss
        reg_term_train = calculate_regularizing_term(trainy, weight_decay_const, number_hidden_layers, weights)
        avg_train_loss = (total_loss / num_batches) + reg_term_train

        # Validation loss calculation
        val_loss_value = val_loss(valX, valiy, weights, biases, number_hidden_layers, activation_function, output_function, loss_function)
        val_loss_value += calculate_regularizing_term(valiy, weight_decay_const, number_hidden_layers, weights)

        print(f"Epoch {epoch + 1}: Validation Loss = {val_loss_value:.4f}")

        # Compute accuracy metrics
        train_acc = train_accuracy(mini_batches_y, y_predictions, trainy)
        val_acc = test_accuracy(valX, valy, weights, biases, number_hidden_layers, activation_function, output_function)

        # Store metrics
        loss_train.append(avg_train_loss)
        loss_val.append(val_loss_value)
        acc_train.append(train_acc)
        acc_val.append(val_acc)

        # Log metrics if W&B logging is enabled
        if wandb_flag:
            wandb.log({"loss": avg_train_loss, "val_loss": val_loss_value, "accuracy": train_acc, "val_accuracy": val_acc, "epoch": epoch + 1})

    return h_states[-1], weights, biases, [loss_train, loss_val, acc_train, acc_val]


def rmsprop(trainX, trainy, number_hidden_layers=1, hidden_layer_size=4, eta=0.1, initial_weights='random',
            activation_function='sigmoid', epochs=1, output_function='softmax', mini_batch_size=4,
            loss_function='cross_entropy', weight_decay_const=0, wandb_flag=False):
    INPUT_KEY = 'input_size'
    OUTPUT_KEY = 'output_size'
    FUN_KEY = "function"
    layers = [{INPUT_KEY: input_layer_size, OUTPUT_KEY: hidden_layer_size, FUN_KEY: activation_function}]
    layers.extend([{INPUT_KEY: hidden_layer_size, OUTPUT_KEY: hidden_layer_size, FUN_KEY: activation_function}
                   for _ in range(number_hidden_layers - 1)])
    layers.append({INPUT_KEY: hidden_layer_size, OUTPUT_KEY: output_layer_size, FUN_KEY: output_function})

    # Initialize weights and biases
    weights, biases = initialize_weights_and_biases(layers, number_hidden_layers, initial_weights)

    # Determine batch processing details
    num_samples = len(trainX)
    num_batches = num_samples // mini_batch_size
    mini_batches_X = np.array_split(trainX, num_batches)
    mini_batches_y = np.array_split(trainy, num_batches)

    # RMSProp accumulators
    v_weights = [np.zeros_like(w) for w in weights]
    v_biases = [np.zeros_like(b) for b in biases]

    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []
    for epoch in range(epochs):
        epoch_loss = 0
        y_preds = []

        for batch_X, batch_y in zip(mini_batches_X, mini_batches_y):
            activations, h_states = forward_propagation(batch_X, weights, biases, number_hidden_layers, activation_function, output_function)
            y_preds.append(h_states[-1])

            # Compute loss
            epoch_loss += cross_entropy(h_states[-1], batch_y) if loss_function == 'cross_entropy' else mean_squared_error(h_states[-1], batch_y)

            # Backpropagation
            del_W, del_b = backward_propagation(batch_y, batch_X, h_states[-1], activations, h_states, weights, number_hidden_layers, activation_function)

            # RMSProp weight update
            for i in range(len(weights)):
                v_weights[i] = (args.beta) * v_weights[i] + (1 - args.beta) * (del_W[f'W{i+1}'] ** 2)
                v_biases[i] = args.beta * v_biases[i] + (1 - args.beta) * (del_b[f'b{i+1}'] ** 2)

                weights[i] -= eta * del_W[f'W{i+1}'] / (np.sqrt(v_weights[i] + args.epsilon))
                biases[i] -= eta * del_b[f'b{i+1}'] / (np.sqrt(v_biases[i] + args.epsilon))

        # Compute loss & accuracy
        train_loss = (epoch_loss / num_batches) + calculate_regularizing_term(trainy, weight_decay_const, number_hidden_layers, weights)
        val_loss_value = val_loss(valX, valiy, weights, biases, number_hidden_layers, activation_function, output_function, loss_function)
        val_loss_value += calculate_regularizing_term(valiy, weight_decay_const, number_hidden_layers, weights)

        print(f"Epoch {epoch+1}: Validation Loss = {val_loss_value:.4f}")

        # Accuracy Calculation
        train_acc_list.append(train_accuracy(mini_batches_y, y_preds, trainy))
        val_acc_list.append(test_accuracy(valX, valy, weights, biases, number_hidden_layers, activation_function, output_function))
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss_value)

        if wandb_flag:
            wandb.log({"loss": train_loss, "val_loss": val_loss_value, "accuracy": train_acc_list[-1], "val_accuracy": val_acc_list[-1], "epoch": epoch+1})

    return h_states[-1], weights, biases, [train_loss_list, val_loss_list, train_acc_list, val_acc_list]

def adam(trainX, trainy, number_hidden_layers=1, hidden_layer_size=4, eta=0.1, initial_weights='random',
         activation_function='sigmoid', epochs=1, output_function='softmax', mini_batch_size=4,
         loss_function='cross_entropy', weight_decay_const=0, wandb_flag=False):
    INPUT_KEY = 'input_size'
    OUTPUT_KEY = 'output_size'
    FUN_KEY = "function"
    layers = [{INPUT_KEY: input_layer_size, OUTPUT_KEY: hidden_layer_size, FUN_KEY: activation_function}]
    layers.extend([{INPUT_KEY: hidden_layer_size, OUTPUT_KEY: hidden_layer_size, FUN_KEY: activation_function}
                   for _ in range(number_hidden_layers - 1)])
    layers.append({INPUT_KEY: hidden_layer_size, OUTPUT_KEY: output_layer_size, FUN_KEY: output_function})

    # Initialize weights and biases
    weights, biases = initialize_weights_and_biases(layers, number_hidden_layers, initial_weights)

    # Determine batch processing details
    num_samples = len(trainX)
    num_batches = num_samples // mini_batch_size
    mini_batches_X = np.array_split(trainX, num_batches)
    mini_batches_y = np.array_split(trainy, num_batches)

    # Adam accumulators
    v_weights, v_biases = [np.zeros_like(w) for w in weights], [np.zeros_like(b) for b in biases]
    m_weights, m_biases = [np.zeros_like(w) for w in weights], [np.zeros_like(b) for b in biases]

    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []
    time_step = 0

    for epoch in range(epochs):
        epoch_loss = 0
        y_preds = []

        for batch_X, batch_y in zip(mini_batches_X,mini_batches_y):
            time_step += 1
            activations, h_states = forward_propagation(batch_X, weights, biases, number_hidden_layers, activation_function, output_function)
            y_preds.append(h_states[-1])

            # Compute loss
            epoch_loss += cross_entropy(h_states[-1], batch_y) if loss_function == 'cross_entropy' else mean_squared_error(h_states[-1], batch_y)

            # Backpropagation
            del_W, del_b = backward_propagation(batch_y, batch_X, h_states[-1], activations, h_states, weights, number_hidden_layers, activation_function)

            # Adam weight update
            for i in range(len(weights)):
                m_weights[i] = args.beta1 * m_weights[i] + (1 - args.beta1) * del_W[f'W{i+1}']
                m_biases[i] = args.beta1 * m_biases[i] + (1 - args.beta1) * del_b[f'b{i+1}']

                v_weights[i] = args.beta2 * v_weights[i] + (1 - args.beta2) * (del_W[f'W{i+1}'] ** 2)
                v_biases[i] = args.beta2 * v_biases[i] + (1 - args.beta2) * (del_b[f'b{i+1}'] ** 2)

                # Bias correction
                m_hat_w, v_hat_w = m_weights[i] / (1 - args.beta1 ** time_step), v_weights[i] / (1 - args.beta2 ** time_step)
                m_hat_b, v_hat_b = m_biases[i] / (1 - args.beta1 ** time_step), v_biases[i] / (1 - args.beta2 ** time_step)

                weights[i] -= eta * m_hat_w / (np.sqrt(v_hat_w) + args.epsilon)
                biases[i] -= eta * m_hat_b / (np.sqrt(v_hat_b) + args.epsilon)

        # Compute loss & accuracy
        train_loss = (epoch_loss / num_batches) + calculate_regularizing_term(trainy, weight_decay_const, number_hidden_layers, weights)
        val_loss_value = val_loss(valX, valiy, weights, biases, number_hidden_layers, activation_function, output_function, loss_function)
        val_loss_value += calculate_regularizing_term(valiy, weight_decay_const, number_hidden_layers, weights)

        print(f"Epoch {epoch+1}: Validation Loss = {val_loss_value:.4f}")

        # Store accuracy
        train_acc_list.append(train_accuracy(mini_batches_y, y_preds, trainy))
        val_acc_list.append(test_accuracy(valX, valy, weights, biases, number_hidden_layers, activation_function, output_function))

        if wandb_flag:
            wandb.log({"loss": train_loss, "val_loss": val_loss_value, "accuracy": train_acc_list[-1], "val_accuracy": val_acc_list[-1], "epoch": epoch+1})

    return h_states[-1], weights, biases, [train_loss_list, val_loss_list, train_acc_list, val_acc_list]

def nadam(trainX, trainy, number_hidden_layers=1, hidden_layer_size=4, eta=0.1, initial_weights='random',
          activation_function='sigmoid', epochs=1, output_function='softmax', mini_batch_size=4,
          loss_function='cross_entropy', weight_decay_const=0, wandb_flag=False):
    INPUT_KEY = 'input_size'
    OUTPUT_KEY = 'output_size'
    FUN_KEY = "function"
    # Define network layers dynamically
    layers = [{INPUT_KEY: input_layer_size, OUTPUT_KEY: hidden_layer_size, FUN_KEY: activation_function}]
    layers.extend([{INPUT_KEY: hidden_layer_size, OUTPUT_KEY: hidden_layer_size, FUN_KEY: activation_function}
                   for _ in range(number_hidden_layers - 1)])
    layers.append({INPUT_KEY: hidden_layer_size, OUTPUT_KEY: output_layer_size, FUN_KEY: output_function})

    # Initialize weights and biases
    weights, biases = initialize_weights_and_biases(layers, number_hidden_layers, initial_weights)

    # Determine batch processing details
    num_samples = len(trainX)
    num_batches = num_samples // mini_batch_size
    mini_batches_X = np.array_split(trainX, num_batches)
    mini_batches_y = np.array_split(trainy, num_batches)

    # Initialize moving averages for Nadam
    v_weights, v_biases = [np.zeros_like(w) for w in weights], [np.zeros_like(b) for b in biases]
    m_weights, m_biases = [np.zeros_like(w) for w in weights], [np.zeros_like(b) for b in biases]

    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []
    beta_1, beta_2, epsilon = 0.9, 0.999, 1e-3
    t = 0  # Step counter

    for epoch in range(epochs):
        total_train_loss = 0
        y_preds = []

        for X_batch, y_batch in zip(mini_batches_X,mini_batches_y):
            t += 1  # Increment time step

            # Compute bias-corrected moving averages
            v_hat_weights = [v / (1 - args.beta2 ** t) for v in v_weights]
            v_hat_biases = [v / (1 - args.beta2 ** t) for v in v_biases]

            m_hat_weights = [m / (1 - args.beta1 ** t) for m in m_weights]
            m_hat_biases = [m / (1 - args.beta1 ** t) for m in m_biases]

            # Compute "lookahead" weights & biases
            lookahead_weights = [weights[i] - (m_hat_weights[i] / np.sqrt(v_hat_weights[i] + epsilon)) * eta
                                 for i in range(len(weights))]
            lookahead_biases = [biases[i] - (m_hat_biases[i] / np.sqrt(v_hat_biases[i] + epsilon)) * eta
                                for i in range(len(biases))]

            # Forward propagation
            activations, outputs = forward_propagation(X_batch, lookahead_weights, lookahead_biases, number_hidden_layers, activation_function, output_function)
            y_preds.append(outputs[-1])

            # Compute loss
            loss_func = cross_entropy if loss_function == 'cross_entropy' else mean_squared_error
            total_train_loss += loss_func(outputs[-1], y_batch)

            # Backpropagation
            grad_W, grad_B = backward_propagation(y_batch, X_batch, outputs[-1], activations, outputs, lookahead_weights, number_hidden_layers, activation_function)

            # Update moving averages & apply Nadam update rule
            for i in range(len(weights)):
                v_weights[i] = args.beta2 * v_weights[i] + (1 - args.beta2) * (grad_W[f'W{i+1}'] ** 2)
                v_biases[i] = args.beta2 * v_biases[i] + (1 - args.beta2) * (grad_B[f'b{i+1}'] ** 2)

                m_weights[i] = args.beta1 * m_weights[i] + (1 - args.beta1) * grad_W[f'W{i+1}']
                m_biases[i] = args.beta1 * m_biases[i] + (1 - args.beta1) * grad_B[f'b{i+1}']

                v_hat_w, v_hat_b = v_weights[i] / (1 - args.beta2 ** t), v_biases[i] / (1 - args.beta2 ** t)
                m_hat_w, m_hat_b = m_weights[i] / (1 - args.beta1 ** t), m_biases[i] / (1 - args.beta1 ** t)

                weights[i] -= (m_hat_w * eta) / np.sqrt(v_hat_w + epsilon)
                biases[i] -= (m_hat_b * eta) / np.sqrt(v_hat_b + epsilon)

        # Compute losses with regularization
        reg_train = calculate_regularizing_term(trainy, weight_decay_const, number_hidden_layers, weights)
        avg_train_loss = total_train_loss / num_batches + reg_train
        val_loss_value = val_loss(valX, valiy, weights, biases, number_hidden_layers, activation_function, output_function, loss_function)
        val_loss_value += calculate_regularizing_term(valiy, weight_decay_const, number_hidden_layers, weights)

        print(f"Epoch {epoch + 1}: Validation Loss = {val_loss_value:.4f}")

        # Compute accuracy
        train_acc_list.append(train_accuracy(mini_batches_y, y_preds, trainy))
        val_acc_list.append(test_accuracy(valX, valy, weights, biases, number_hidden_layers, activation_function, output_function))

        # Store metrics
        train_loss_list.append(avg_train_loss)
        val_loss_list.append(val_loss_value)

        if wandb_flag:
            wandb.log({
                "loss": avg_train_loss,
                "val_loss": val_loss_value,
                "accuracy": train_acc_list[-1],
                "val_accuracy": val_acc_list[-1],
                "epoch": epoch + 1
            })

    return outputs[-1], weights, biases, [train_loss_list, val_loss_list, train_acc_list, val_acc_list]



def train(trainX, trainy, textX, testy, number_hidden_layers, hidden_layer_size, eta, init_type, activation_function,
          epochs, mini_batch_size, loss_function, optimizer, output_function, weight_decay_const, wandb_flag=False):
    
    wdc = weight_decay_const

    optimizer_functions = {
        'sgd': gradient_descent,
        'momentum': momentum_based_gradient_descent,
        'nag': nestrov_accelerated_gradient_descent,
        'rmsprop': rmsprop,
        'adam': adam,
        'nadam': nadam
    }

    if optimizer not in optimizer_functions:
        raise ValueError(f"Invalid optimizer: {optimizer}. Choose from {list(optimizer_functions.keys())}.")

    hL, weights, biases, plot_list = optimizer_functions[optimizer](
        trainX, trainy, number_hidden_layers, hidden_layer_size, eta, init_type, activation_function, epochs,
        output_function, mini_batch_size, loss_function, wdc, wandb_flag
    )

    return [weights, biases, number_hidden_layers, activation_function, output_function]

# Define expected types for each argument
expected_types = {
    "num_layers": int,
    "hidden_size": int,
    "learning_rate": float,
    "epochs": int,
    "batch_size": int,
    "weight_decay": float
}

# Convert arguments to the correct type if needed
for arg, dtype in expected_types.items():
    if isinstance(getattr(args, arg), str):
        setattr(args, arg, dtype(getattr(args, arg)))


wandb.login(key="2ddfedd72c75efe3f8e05402fe38a36933f8d1ba")
run = wandb.init(project=args.wandb_project)#,entity=args.wandb_entity)
params = train(trainX=trainX,
    trainy=trainy,
    textX=testX,
    testy=needed_y_test,
    number_hidden_layers=args.num_layers,
    hidden_layer_size=args.hidden_size,
    eta=args.learning_rate,
    init_type=args.weight_init,
    activation_function=args.activation,
    epochs=args.epochs,
    mini_batch_size=args.batch_size,
    loss_function=args.loss,
    optimizer=args.optimizer,
    output_function='softmax',
    weight_decay_const=args.weight_decay,
    wandb_flag=True)

test_ac = test_accuracy(testX,needed_y_test,params[0],params[1],params[2],params[3],params[4])

print("Test accuracy on the model = ", test_ac*100,'%')
wandb.save()
wandb.finish()
