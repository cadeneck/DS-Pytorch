
# Deep Learning Project: Implementing Neural Networks with Numpy, PyTorch, and Convolutional Neural Networks (CNNs)

## Introduction
This project is a comprehensive exploration into building and training neural networks using Numpy, PyTorch, and PyTorch's CNN capabilities. The project is structured into three main parts, each focusing on a different aspect of neural network implementation:

1. **Numpy Artificial Neural Network (ANN):** A simple neural network with one hidden layer implemented using Numpy, showcasing the basics of neural network operations without the need for a deep learning framework.
2. **PyTorch ANN:** Utilizing PyTorch, a popular deep learning library, to implement a three-layer neural network with ReLU activation. This part emphasizes the use of PyTorch's dynamic computation graph and built-in functions for more efficient and straightforward model construction.
3. **PyTorch Convolutional Neural Network (CNN):** Expanding upon the PyTorch ANN by introducing convolutional layers. This network is designed for image classification tasks, demonstrating the power of CNNs in capturing spatial hierarchies in data.

Each segment provides hands-on experience with key concepts in deep learning, from manual implementation of forward and backward passes to leveraging advanced neural network architectures.

## Features
- **Numpy ANN:** Implementation of forward and backward passes, activation functions, and training loops from scratch.
- **PyTorch ANN:** Simplified model construction using `nn.Sequential`, training, and evaluation with PyTorch's built-in modules.
- **PyTorch CNN:** Advanced model construction with convolutional layers, pooling, batch normalization, and dropout for improved image classification performance.

## Getting Started
### Prerequisites
- Python 3.x
- Numpy
- PyTorch
- Matplotlib (for plotting training/validation metrics)

### Installation
1. Install Python dependencies:
    ```sh
    pip install numpy torch matplotlib
    ```
2. Clone the repository to your local machine:
    ```sh
    git clone https://github.com/your-username/your-repo-name.git
    ```

## Usage
### Numpy ANN
- To initialize and train the Numpy-based ANN, use the following commands:
    ```python
    from numpy_ann import NumpyANN
    ann = NumpyANN(input_size=784, hidden_size=64, output_size=10, lr=0.01)
    ```

### PyTorch ANN
- Initialize and train the PyTorch-based ANN as follows:
    ```python
    from pytorch_ann import PyTorchANN
    pytorch_ann = PyTorchANN(input_size=784, hidden_size=64, output_size=10)
    ```

### PyTorch CNN
- For the CNN model, use:
    ```python
    from pytorch_cnn import PyTorchCNN
    pytorch_cnn = PyTorchCNN(num_classes=10)
    ```
