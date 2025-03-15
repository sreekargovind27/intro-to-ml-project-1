# Deep Neural Network for Handwritten Digits & Face Classification

## Overview

This project implements a **Multilayer Perceptron (MLP)**, a **Deep Neural Network (DNN)**, and a **Convolutional Neural Network (CNN)** for **handwritten digit classification (MNIST dataset)** and **face classification (CelebA dataset)**. The implementation is done using **NumPy, SciPy, and PyTorch**.

## Features

- **Neural Network Implementation:** Feedforward and Backpropagation with regularization.
- **Deep Learning:** Implementation of deep neural networks using PyTorch.
- **Feature Selection:** Removes features with zero variance.
- **Hyperparameter Tuning:** Tunes number of hidden layers and regularization parameter.
- **Comparison:** Compares single-layer MLP, Deep MLP, and CNN models.
- **Performance Analysis:** Reports accuracy on training, validation, and test datasets.

## Repository Structure

```
.
|-- code/
|   |-- nnScript.py          # Implements the MLP Neural Network (NumPy/SciPy)
|   |-- facennScript.py      # Runs MLP on CelebA dataset (NumPy/SciPy)
|   |-- deepnnScript.py      # Implements Deep Neural Network (PyTorch)
|   |-- params.pickle        # Saved trained parameters for evaluation
|   |-- mnist_all.mat        # MNIST dataset
|   |-- face_all.pickle      # CelebA dataset subset
|-- demo.mp4                 # Video presentation explaining implementation
|-- README.md                # Project documentation
```

## Requirements

To run this project, install the following dependencies:

```bash
pip install numpy scipy torch torchvision matplotlib
```

## Usage

### Train and Evaluate the MLP

```bash
python nnScript.py
```

### Train and Evaluate the Deep Neural Network

```bash
python deepnnScript.py
```

### Train and Evaluate on CelebA dataset

```bash
python facennScript.py
```

## Explanation of Important Files

- **nnScript.py**: Implements the neural network from scratch using NumPy/SciPy.
- **facennScript.py**: Runs the MLP on the CelebA dataset.
- **deepnnScript.py**: Implements the deep neural network using PyTorch.
- **params.pickle**: Stores the learned parameters after training.

## Results

### Effect of Hidden Layers on Accuracy

| Hidden Layers | Train Accuracy | Validation Accuracy | Test Accuracy |
| ------------- | -------------- | ------------------- | ------------- |
| 3             | 94.32%         | 85.41%              | 84.92%        |
| 5             | 95.18%         | 87.12%              | 86.55%        |
| 7             | 96.76%         | 89.01%              | 88.64%        |

### Effect of Regularization on Accuracy

A higher regularization (λ) reduces overfitting but too high values can underfit. The optimal λ was found through validation.

## Video Presentation

A **15-minute demo video (**\`\`**)** is included in this repository, explaining:

1. **Hyperparameter Tuning**: Effect of hidden layers and regularization.
2. **Training Process**: How backpropagation updates weights.
3. **Performance Comparison**: Between single-layer MLP, deep MLP, and CNN.
4. **Experiment Insights**: Challenges faced and solutions applied.

## Contributors

- Sreekar G
- Santosh J 

## License

This project is for academic purposes and follows the university at Buffalo, State University newyork.

 
