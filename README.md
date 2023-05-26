# pytorch_for_tabular_data

## PyTorch Binary Classification, Multiclass Classification, and Regression
This repository contains jupyter notebooks demonstrating the use of PyTorch for binary classification, multiclass classification, and regression tasks. The datasets used in these examples are generated using the make_blobs, make_moon, and make_regression functions from the sklearn library.

## Contents
The repository consists of the following files:

pytorch_for_binary_classification.ipynb: This Jupyter Notebook demonstrates how to perform binary classification using PyTorch. It provides a step-by-step guide on loading the dataset, preprocessing the data, creating a neural network model using PyTorch, training the model, and evaluating its performance.

multiclass_classification_pytorch.ipynb: This Jupyter Notebook illustrates the process of multiclass classification using PyTorch. It explains how to handle datasets with multiple classes, preprocess the data, build a neural network model with PyTorch, train the model, and assess its accuracy.

regression_pytorch.ipynb: This Jupyter Notebook showcases regression using PyTorch. It demonstrates how to generate a regression dataset using the make_regression function from sklearn, preprocess the data, design a neural network model using PyTorch, train the model, and evaluate its performance.

regression_pytorch_with_sequential.ipynb: This Jupyter Notebook showcases regression using PyTorch. It demonstrates how to generate a regression dataset using the make_regression function from sklearn, preprocess the data, design a neural network model using PyTorch, train the model, and evaluate its performance. This jupyter notebook uses the sequential method for creating the pytorch model

requirements.txt: This file lists the Python packages required to run the Jupyter Notebooks. You can install them using pip with the command pip install -r requirements.txt.

## How to Use PyTorch Instead of Keras
While Keras is a popular deep learning framework, PyTorch offers a flexible and dynamic approach to building and training neural networks. Here are some key steps to consider when transitioning from Keras to PyTorch:

### Model Definition: 
In Keras, models are defined using a sequential or functional API. However, in PyTorch, you can define your model by creating a class that inherits from the torch.nn.Module class. This allows for more flexibility in model architecture and customization.

### Tensor Operations: 
Both Keras and PyTorch provide tensor operations, but PyTorch's tensor operations closely resemble NumPy's syntax. You can easily perform mathematical operations on tensors in PyTorch, making it convenient for complex computations.

### Training Loop: 
In Keras, training loops are handled automatically when calling the fit function. In PyTorch, you have more control over the training loop. You can iterate over your dataset, perform forward and backward passes, update the model's parameters, and evaluate the model's performance.

### Device Management: 
PyTorch allows you to choose between running your computations on the CPU or GPU. You can explicitly move tensors and models to the GPU using the .to(device) method, where device can be either "cuda" or "cpu". This enables faster training and inference on GPUs if available.

By exploring the provided Jupyter Notebooks, you will gain a deeper understanding of how PyTorch can be used for binary classification, multiclass classification, and regression tasks, and how it differs from using Keras.

## Requirements
To run the Jupyter Notebooks, make sure you have the following packages installed:

PyTorch,
NumPy,
Matplotlib,
Scikit-learn,
