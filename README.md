# neural-network-challenge-1

# README: Understanding Bias and Activation in Artificial Neural Networks (ANN)

This guide demonstrates how bias and activation functions work in a basic **Artificial Neural Network (ANN)** using Python and **NumPy**.

---

## 🧠 **Concept Overview**

1. **Bias**:
   - A bias term is added to the weighted sum of the inputs before passing through the activation function.
   - It allows the neural network to shift the activation threshold, making it more flexible and capable of learning complex patterns.

2. **Activation Function**:
   - An activation function (like **Sigmoid** or **ReLU**) determines if a neuron should "fire" (produce an output).
   - The **Sigmoid** function is commonly used for binary classification problems as its output is between 0 and 1.

3. **Without Bias**:
   - When there's no bias, the activation function is limited to being centered around the origin (zero), which makes it less flexible.
   - The model might struggle to learn patterns that require the activation function to be shifted.

---

## ⚙️ **Python Code Example**

### **1. Single Neuron Output**

This basic example demonstrates the weighted sum and output of a neuron using **ReLU** as the activation function.

```python
import numpy as np

# Inputs
x = np.array([0.5, 1.0])   # Example inputs
w = np.array([0.4, -0.7])  # Weights
b = 0.2                    # Bias

# Weighted sum
z = np.dot(x, w) + b

# Activation function (ReLU in this case)
output = max(0, z)

print("Output:", output)
```

---

