'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1/(1 + np.exp(-z))
    
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval =args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, n_input +1))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, n_hidden +1))

    # here i am adding bias proprtly to our training data
   # bias_input = np.ones((training_data.shape[0], 1))
    training_data_bias = np.hstack((training_data, np.ones((training_data.shape[0],1))))

# Here will start our  forward propagation.

    hidden_layer_input = np.dot(training_data_bias, w1.T)
    #hidden_layer_output = 1/ (1 + np.exp(-hidden_layer_input))
    hidden_layer_output = sigmoid(hidden_layer_input)
    hidden_layer_output_bias = np.hstack((hidden_layer_output, np.ones((hidden_layer_output.shape[0], 1))))


    #hidden_bias = np.ones((hidden_layer_output.shape[0], 1))
    #hidden_layer_output_bias = np.hstack((hidden_layer_output, hidden_bias))

    output_layer_input = np.dot(hidden_layer_output_bias, w2.T)
    output_layer_output = sigmoid(output_layer_input)

   # one-hot encoding lables

    y = np.zeros((training_data.shape[0], n_class))
    y[np.arange(training_data.shape[0]), training_label.astype(int)] = 1


   # loss computation..
    epsilon = 1e-10
    loss = -np.sum(y * np.log(output_layer_output + epsilon) + (1 - y) * np.log(1- output_layer_output + epsilon))
    reg_term = (lambdaval / (2 * training_data.shape[0])) * (np.sum(w1 ** 2) + np.sum(w2 ** 2))
    obj_val = (loss / training_data.shape[0]) + reg_term

    #  now will perform backpropagartion..

    delta_output = output_layer_output - y
    grad_w2 = np.dot(delta_output.T, hidden_layer_output_bias) / training_data.shape[0]
    grad_w2[:, :-1] += (lambdaval / training_data.shape[0]) * w2[:, :-1]

    delta_hidden = np.dot(delta_output, w2[:, :-1]) * hidden_layer_output * (1 - hidden_layer_output)
    grad_w1 = np.dot(delta_hidden.T,training_data_bias) / training_data.shape[0]
    grad_w1[:, :-1] += (lambdaval / training_data.shape[0]) * w1[:, :-1]

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()))
    return (obj_val, obj_grad)


    #def sigmoid(z):
       # return 1 /(1 + np.exp(-z))

# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    data = np.hstack((data, np.ones((data.shape[0],1))))
    hidden_layer_output = sigmoid(np.dot(data, w1.T))
    hidden_layer_output = np.hstack((hidden_layer_output, np.ones((hidden_layer_output.shape[0],1))))
    output_layer_output = sigmoid(np.dot(hidden_layer_output, w2.T))
    return np.argmax(output_layer_output, axis=1)

    #hidden_layer_output = sigmoid(np.dot(data,w1.T))
    #hidden_bias = np.ones((hidden_layer_output.shape[0], 1))
    #hidden_layer_output = np.hstack((hidden_layer_output, hidden_bias))

    #output_layer_output = sigmoid( np.dot(hidden_layer_output, w2.T))
    #abels = np.argmax(output_layer_output, axis =1)
    #return labels
    
# Do not change this

def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

train_data, train_label,validation_data,validation_label,test_data, test_label = preprocess()

lambdaval_list = [0,5,10,20,30,40,50,60]

hidden_units_list = [4, 8, 12, 16, 20]

results = {}

for n_hidden in hidden_units_list:
    results[n_hidden] = []
    for lambdaval in lambdaval_list:
        print(f"\nTraining with hidden Units = {n_hidden}, {lambdaval}")
        




"""**************Neural Network Script Starts here********************************"""
#train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
#n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
#n_hidden = 256
# set the number of nodes in output unit
#n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(train_data.shape[1], n_hidden);
initial_w2 = initializeWeights(n_hidden, 2)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter

args = (train_data.shape[1], n_hidden, 2, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
#opts = {'maxiter' :50}    # Preferred value.

start_time = time.time()

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options={'maxiter': 50})
training_time = time.time() - start_time

params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (train_data.shape[1] + 1)].reshape( (n_hidden, (train_data.shape[1] + 1)))
w2 = params[(n_hidden * (train_data.shape[1] + 1)):].reshape((2, (n_hidden + 1)))


# compute accuracy 

train_acc = 100 * np.mean((nnPredict(w1,w2,train_data)== train_label).astype(float))
val_acc = 100 * np.mean((nnPredict(w1, w2, validation_data) == validation_label).astype(float))
test_acc = 100 * np.mean((nnPredict(w1,w2, test_data) == test_label).astype(float))


print(f"Training Time: {training_time:.2f}s | train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%")

results[n_hidden].append((lambdaval, training_time, val_acc))
#Test the computed parameters
#predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
#print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
#predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
#print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
#predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
#print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')




plt.figure(figsize=(10,5))

for n_hidden in results:
    lambdas = [x[0] for x in results [n_hidden]]
    val_accs =[x[2] for x in results[n_hidden]]
    plt.plot(lambdas, val_accs, marker='o', label =f"{n_hidden} Hidden Units")


plt.xlabel("regularization parameter ") 
plt.ylabel("validation Accuracy (%)")
plt.title("effect of lambda on validation accuracy")

plt.legend()
plt.grid()
plt.show()

