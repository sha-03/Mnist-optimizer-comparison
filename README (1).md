# MNIST Optimizer Comparison

Comparing SGD, Adam, RMSprop, Adagrad and Nadam optimizers on a neural network trained on the MNIST handwritten digits dataset, using TensorFlow / Keras.

## Overview

This project trains the same fully connected neural network with five different optimizers and compares their test accuracy to find the best one. The best optimizer is then used to train a final model.

**Model architecture**

| Layer | Units | Activation | Dropout |
|-------|-------|------------|---------|
| Input | 784 (28 x 28 flattened) | - | - |
| Dense | 64 | ReLU | 0.2 |
| Dense | 32 | ReLU | 0.2 |
| Dense | 16 | ReLU | 0.2 |
| Output | 10 | Softmax | - |

**Training setup:** sparse categorical cross-entropy loss, 10 epochs, pixel values normalized to 0-1.

## Files

| File | Description |
|------|-------------|
| `optimizer_comparison.py` | Trains the model with SGD, Adam, RMSprop, Adagrad and Nadam, then prints the best optimizer |
| `nadam_mnist_model.py` | Trains the final model with Nadam and shows loss and accuracy in a table |
| `requirements.txt` | Python dependencies |

## Getting Started

```bash
git clone https://github.com/sha-03/mnist-optimizer-comparison.git
cd mnist-optimizer-comparison
pip install -r requirements.txt

python optimizer_comparison.py
python nadam_mnist_model.py
```

## Tech Stack

Python, TensorFlow / Keras, NumPy, Pandas

## Author

**Sham Anand** - [GitHub](https://github.com/sha-03) | [LinkedIn](https://linkedin.com/in/sham-anand)
