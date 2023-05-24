# MinimalGrad
MinimalGrad is an open-source automatic differentiation library for deep learning that is designed to be lightweight and efficient, with a focus on simplicity and ease of use. It was developed by a team of students at Eötvös Loránd University as part of Advanced software technology class.

# Features
MinimalGrad implements reverse-mode automatic differentiation, which allows it to efficiently compute gradients of arbitrary computational graphs. It supports a variety of numerical data types and mathematical operations, and can be easily integrated into existing deep learning models.

Some of the key features of MinimalGrad include:

- Simple and easy-to-use API
- Lightweight and efficient implementation
- Support for a variety of numerical data types
- Support for a variety of mathematical operations
- Integration with existing deep learning models

# Installation
To install MinimalGrad, simply run the following command:
``Not working yet``
```
pip install minimalgrad 
```

# Usage
To use MinimalGrad, simply import the library and create a computation graph using the available operations. Here's a simple example:

```python
from MinimalGrad.core import Tensor

# Define variables
x = Tensor([2, 3])
y = Tensor([5, 7])

# Define computation graph
z = x + y
w = z * z

# Compute gradients
w.backward()

# Print gradients
print(x.grad)  # [14, 20]
print(y.grad)  # [14, 20]
```
