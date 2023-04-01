import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
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
            grad = self.__class__(np.ones_like(self.data))

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
            other = self.__class__(np.array(other))
        return self.__class__(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = self.__class__(np.array(other))
        return self.__class__(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

    def sum(self, axis=None):
        return self.__class__(self.data.sum(axis=axis), requires_grad=self.requires_grad)

    def mean(self, axis=None):
        return self.__class__(self.data.mean(axis=axis), requires_grad=self.requires_grad)
