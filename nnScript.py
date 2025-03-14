import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1 / (1 + np.exp(-z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loadin the data set

    # stacking training as well as testing images for  10 classes 0-9

    train_data = np.vstack([mat['train' + str(i)] for i in range (10)])
    test_data = np.vstack([mat['test' + str(i)] for i in range (10)])

    # now will generate lable 0 -9

    train_labels = np.hstack([i * np.ones(mat['train' + str(i)].shape[0]) for i in range (10)])
    test_labels = np.hstack([i * np.ones(mat['test' + str(i)].shape[0]) for i in range (10)])

    # here will normalize my data  basically witll scale the values of pixel et 0 to 1
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0

    # shuffle training data
    np.random.seed(42)
    shuffled_indices = np.random.permutation(train_data.shape[0])
    
    #shuffling my training data

    train_size = int(0.8333 * train_data.shape[0])
    train_indices = shuffled_indices[: train_size]
    val_indices = shuffled_indices[train_size:]

    train_data, val_data = train_data[train_indices], train_data[val_indices]

    train_labels_split = train_labels[train_indices]      
    val_labels_split = train_labels[val_indices] 

  # feature selection: remove feacture(pixels) with zero variance

    valid_features = np.std(train_data, axis=0) > 0
    train_data =train_data[:, valid_features]
    val_data = val_data[:, valid_features]
    test_data = test_data[:, valid_features]
  
 # ensuring lables are integers

    train_labels_split = train_labels_split.astype(np.int32)
    val_labels_split = val_labels_split.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    

    print('preprocess done')

    return train_data, train_labels_split, val_data, val_labels_split, test_data, test_labels


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    #if training_data.shape[1] != w1.shape[1] - 1:  
     #   bias_input = np.ones((training_data.shape[0], 1))
     #   training_data = np.hstack((training_data, bias_input))  
    
    # adding bias  to trianing data 
    bias_input = np.ones((training_data.shape[0], 1))     # this will crate a bias column 
    training_data = np.hstack((training_data, bias_input))  # now adding bias to training_data

    # forward propagation
    hidden_layer_input = np.dot(training_data, w1.T)
    hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))

    hidden_bias = np.ones((hidden_layer_output.shape[0], 1))
    hidden_layer_output = np.hstack((hidden_layer_output, hidden_bias))

    output_layer_input = np.dot(hidden_layer_output, w2.T)
    output_layer_output = 1 / (1 + np.exp(-output_layer_input))
    
    #y = np.zeros((training_data.shape[0], n_class))
    #i = 0
    #while i < training_data.shape[0]:
        # y[i, int(training_label[i])] = 1
        # i += 1
    if training_label.shape[0] != training_data.shape[0]:
        raise ValueError(f"shape mismatch: training_data={ training_data.shape}, training _label={training_label.shape}")
    

    # one -hot encoding is done ..
    y =np.zeros((training_data.shape[0],n_class))    # using numpy for more effiecent one hot encoding
    y[np.arange(training_data.shape[0]), training_label.astype(int)] = 1
    #training_label = np.round(training_label).astype(int)
    #y[np.arange(training_data.shape[0]), training_label] =1


    #loss = -np.sum(y * np.log(output_layer_output) + (1 - y) * np.log(1 - output_layer_output))   
    #loss /= training_data.shape[0]
     
    epsilon = 1e-10 
    loss = -np.sum(y * np.log(output_layer_output + epsilon) + (1 - y) * np.log(1 - output_layer_output + epsilon))


    reg_term = (lambdaval / (2 * training_data.shape[0])) * (np.sum(w1 ** 2) + np.sum(w2 ** 2))
    obj_val = loss + reg_term

    # will perform back propagation here
    delta_output = output_layer_output - y
    grad_w2 = np.dot(delta_output.T, hidden_layer_output) / training_data.shape[0]
    grad_w2[:, :-1] += (lambdaval / training_data.shape[0]) *w2[:, :-1]

    delta_hidden = np.dot(delta_output, w2[:, :-1]) * (hidden_layer_output[:, :-1] *  (1 - hidden_layer_output[:, :-1]))
    grad_w1 = np.dot( delta_hidden.T, training_data) / training_data.shape[0]
    grad_w1[:, :-1] += (lambdaval / training_data.shape[0]) * w1[:, :-1] 
    #grad_w1 += (lambdaval / training_data.shape[0]) * w1
    #grad_w1[:, -1] = 0

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weobj_val ights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    bias = np.ones((data.shape[0],1))
    data = np.hstack((data, bias))    # here i am ppend a bias 

    hidden_layer_input = np.dot(data, w1.T)
    hidden_layer_output = 1 /(1 + np.exp(-hidden_layer_input))

    hidden_bias = np.ones((hidden_layer_output.shape[0],1))
    hidden_layer_output =  np.hstack((hidden_layer_output, hidden_bias))

    output_layer_input = np.dot(hidden_layer_output, w2.T)
    output_layer_output = 1 /(1 + np.exp(-output_layer_input))

    labels = np.argmax(output_layer_output, axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""
if __name__ == "__main__":
    
        
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 50

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 0.1

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset

    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset

    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

params = {
    "selected_features": list(range(n_input)),  # Assuming all features are selected
    "optimal_n_hidden": n_hidden,
    "w1": w1,
    "w2": w2,
    "optimal_lambda": lambdaval
}

with open("params.pickle", "wb") as f:
    pickle.dump(params, f)

print("\nParameters successfully saved in params.pickle!")

