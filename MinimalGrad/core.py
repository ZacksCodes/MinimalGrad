import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, dtype=np.float32):
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None

    @property
    def shape(self):
        return self.data.shape

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = np.ones_like(self.data, dtype=self.data.dtype)

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self._grad_fn is not None:
            self._grad_fn(self.grad)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = np.array(other, dtype=self.data.dtype)
        return Tensor(np.add(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = np.array(other, dtype=self.data.dtype)
        return Tensor(np.multiply(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)

    def sum(self, axis=None):
        return Tensor(np.sum(self.data, axis=axis, dtype=self.data.dtype), requires_grad=self.requires_grad)

    def mean(self, axis=None):
        return Tensor(np.mean(self.data, axis=axis, dtype=self.data.dtype), requires_grad=self.requires_grad)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = np.array(other, dtype=self.data.dtype)
        return Tensor(np.divide(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
    
    def __matmul__(self, other):
        if isinstance(other, (int, float)):
            other = np.array(other, dtype=self.data.dtype)
        return Tensor(np.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
    
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            other = np.array(other, dtype=self.data.dtype)
        return Tensor(np.power(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
    
    def broadcast_to(self, shape):
        if self.shape == shape:
            return self
        if len(shape) < len(self.shape):
            raise ValueError("Shape must have at least as many dimensions as the tensor")
        diff = len(shape) - len(self.shape)
        axis = tuple(range(diff)) + tuple([i + diff for i in range(len(self.shape))])
        return Tensor(np.broadcast_to(self.data, shape), requires_grad=self.requires_grad).transpose(axis)




