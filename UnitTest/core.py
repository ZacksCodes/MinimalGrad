import numpy as np
import unittest

from MinimalGrad.core import Tensor


class TestTensor(unittest.TestCase):
    def test_creation(self):
        # Test tensor creation with a NumPy array
        data = np.array([[1, 2, 3], [4, 5, 6]])
        tensor = Tensor(data)
        self.assertTrue(np.array_equal(tensor.to_array(), data))

        # Test tensor creation with a list
        data = [[1, 2], [3, 4]]
        tensor = Tensor(data)
        self.assertTrue(np.array_equal(tensor.to_array(), np.array(data)))

        # Test tensor creation with a scalar
        data = 5
        tensor = Tensor(data)
        self.assertTrue(np.array_equal(tensor.to_array(), np.array(data)))

    def test_arithmetic_operations(self):
        # Test tensor addition with another tensor
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[2, 4], [6, 8]])
        tensor1 = Tensor(data1)
        tensor2 = Tensor(data2)
        tensor3 = tensor1 + tensor2
        expected_result = np.array([[3, 6], [9, 12]])
        self.assertTrue(np.array_equal(tensor3.to_array(), expected_result))

        # Test tensor multiplication with another tensor
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[2, 4], [6, 8]])
        tensor1 = Tensor(data1)
        tensor2 = Tensor(data2)
        tensor3 = tensor1 * tensor2
        expected_result = np.array([[2, 8], [18, 32]])
        self.assertTrue(np.array_equal(tensor3.to_array(), expected_result))

        # Test tensor division with a scalar value
        data = np.array([[2, 4], [6, 8]])
        tensor = Tensor(data)
        scalar = 2
        tensor2 = tensor / scalar
        expected_result = np.array([[1, 2], [3, 4]])
        self.assertTrue(np.array_equal(tensor2.to_array(), expected_result))

    def test_gradients(self):
        # Test tensor gradient computation with a simple function
        x1 = Tensor(2, requires_grad=True)
        x2 = Tensor(-2, requires_grad=True)
        w1 = Tensor(-3, requires_grad=True)
        w2 = Tensor(6, requires_grad=True)
        bias = Tensor(0.6546542324, requires_grad=True)
        f = x1*w1 + x2*w2 + bias
        f.backward()
        print("x1.grad : ", x1.grad)
        print("w1.grad : ", w1.grad)
        print("x2.grad : ", x2.grad)
        print("w2.grad : ", w2.grad)
        print("f.grad  : ", f.grad)
        print(f.data)
        self.assertTrue(x1.grad, -3)
        self.assertTrue(x2.grad, 1)


if __name__ == '__main__':
    unittest.main()
