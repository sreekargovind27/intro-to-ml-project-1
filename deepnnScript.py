'''
Comparing single layer MLP with deep MLP (using PyTorch)
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
# Create model
# Add more hidden layers to create deeper networks
# Remember to connect the final hidden layer to the out_layer
def create_multilayer_perceptron(n_hidden_layers=3):

    class DeepNN(nn.Module):
        def __init__(self, n_hidden_layers):
            super().__init__()
             
            # Network Parameters
            n_input = 2376  # data input
            n_hidden = 256  # 1st layer number of features
            n_classes = 2   # will do binary classification for me 

            # now here i am creating a hidden layer 
            self.hidden_layers = nn.ModuleList([nn.Linear(n_input, n_hidden)])

            # need to create additional  hidden layers dynamically
            for _ in range(n_hidden_layers - 1):
                self.hidden_layers.append(nn.Linear(n_hidden, n_hidden))

             # Output layer
            self.out_layer = nn.Linear(n_hidden, n_classes)

        def forward(self, x):
            for layer in self.hidden_layers:
                x = F.relu(layer(x))
            x = self.out_layer(x)
            return F.log_softmax(x, dim=1)

    return DeepNN(n_hidden_layers)

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = np.squeeze(labels)
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]

    class dataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    trainset = dataset(train_x, train_y)
    validset = dataset(valid_x, valid_y)
    testset = dataset(test_x, test_y)

    return trainset, validset, testset


def train(dataloader, model, loss_fn, optimizer):
    #size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")


def test(dataloader, model, loss_fn):
    #size = len(dataloader.dataset)
   # num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
     #test_loss /= num_batches
    accuracy = (correct / len(dataloader.dataset))* 100
    avg_loss = test_loss /len(dataloader) 
    return accuracy, avg_loss
    #correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Parameters
learning_rate = 0.0001
training_epochs = 50
batch_size = 100

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# load data
trainset, validset, testset = preprocess()
train_dataloader =DataLoader(trainset, batch_size=batch_size, shuffle=True)
valid_dataloader =DataLoader(validset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Construct model
#model = create_multilayer_perceptron().to(device)

# Define loss and openptimizer
#cost = nn.CrossEntropyLoss()
#ptimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# load data
#trainset, validset, testset = preprocess()
#train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
#valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False)
#test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


# Training cycle
for n_layers in [3,5,7]:
    print(f"\ntraining Deepnn with{n_layers} hidden layers")

    model = create_multilayer_perceptron(n_layers).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    for t in range(training_epochs):
        train(train_dataloader, model,loss_fn, optimizer)

    train_time = time.time() - start_time

    train_acc, train_loss = test(train_dataloader, model, loss_fn)
    val_acc, val_loss = test(valid_dataloader, model, loss_fn)
    test_acc,  test_loss = test(test_dataloader, model, loss_fn)

    print(f"\n Results for {n_layers} Hidden Layers:")
    print(f" Training Time: {train_time:.2f} sec")
    print(f"Train Accuracy: {train_acc:.2f}%, TrainLoss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%, Test_loss:{test_loss:.4f}\n")


#save model
torch.save(model.state_dict(), f"deepnn_{n_layers}_layers.pth")
print(f"model with {n_layers} layers saved as deepnn_{n_layers}_layers.pth")
        
hidden_layers = [3, 5, 7]
train_accuracies = [94.32, 95.18, 96.76]
val_accuracies = [85.41, 87.12, 89.01]
test_accuracies = [84.92, 86.55, 88.64]
training_times = [30.15, 40.32, 49.68]

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(hidden_layers, train_accuracies, marker='o', linestyle='-', label="Train Accuracy")
plt.plot(hidden_layers, val_accuracies, marker='s', linestyle='-', label="Validation Accuracy")
plt.xlabel("Number of Hidden Layers")
plt.ylabel("Accuracy (%)")
plt.title("Hidden Layers vs. Accuracy")
plt.legend()
plt.grid()


plt.subplot(1,2,2)
plt.plot(hidden_layers, training_times, marker='o', linestyle='-', color='r', label="Training Time")
plt.ylabel("Training Time (seconds)")
plt.title("Hidden Layers vs Training Time")
plt.legend()
plt.grid()


plt.tight_layout()
plt.show()






