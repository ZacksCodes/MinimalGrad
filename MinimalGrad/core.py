import numpy as np
from numba import njit


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op='',_dtype=np.float32):
        self.data = np.array(data, dtype=_dtype)
        self.requires_grad = requires_grad
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self.dim = self.data.ndim
        self._op = _op  # the op that produced this node, for graphviz

    def to_array(self):
        return self.data

    def backward(self):
        topological = []
        visited = set()

        def build_topological_order(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topological_order(child)
                topological.append(node)

        build_topological_order(self)

        self.grad = 1
        for node in reversed(topological):
            node._backward()

    def broadcast_to(self, shape):
        if self.shape == shape:
            return self
        if len(shape) < len(self.shape):
            raise ValueError("Shape must have at least as many dimensions as the tensor")
        diff = len(shape) - len(self.shape)
        axis = tuple(range(diff)) + tuple([i + diff for i in range(len(self.shape))])
        return Tensor(np.broadcast_to(self.data,
                                      shape, dtype=self.data.dtype), requires_grad=self.requires_grad).transpose(axis)

    @property
    def shape(self):
        return self.data.shape

    def sum(self, axis=None):
        return Tensor(np.sum(self.data, axis=axis, dtype=self.data.dtype), requires_grad=self.requires_grad, _op='Sum')

    def mean(self, axis=None):
        return Tensor(np.mean(self.data, axis=axis, dtype=self.data.dtype), requires_grad=self.requires_grad,
                      _op='Mean')

    def transpose(self, axes=None):
        return Tensor(np.transpose(self.data, axes=axes), requires_grad=self.requires_grad, _op='Transpose')

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=self.data.dtype))
        result = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad,
                        _children=(self, other), _op='+')

        def _backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = _backward

        return result

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=self.data.dtype))
        result = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad,
                        _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = _backward

        return result

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=self.data.dtype))
        return self * other ** -1

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=self.data.dtype))
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Supporting only int and float power for now."
        result = Tensor(self.data ** power, _children=(self,), _op="power", _dtype=self.data.dtype)

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * result.grad

        result._backward = _backward

        return result

    def relu(self):
        out = Tensor(0 if self.data < 0 else self.data, _children=(self,), _op='ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def __getitem__(self, index):
        return Tensor(self.data[index], requires_grad=self.requires_grad)

    def __setitem__(self, index, value):
        self.data[index] = value

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    @staticmethod
    @njit
    def determinant(x):
        return np.linalg.det(x)

    @staticmethod
    @njit
    def inverse(x):
        return np.linalg.inv(x)
