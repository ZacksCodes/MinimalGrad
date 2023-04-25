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
        x = Tensor([2, 3], requires_grad=True)
        y = Tensor([5, 7], requires_grad=True)
        z = x * y
        z.backward(Tensor(np.array([1, 1])))
        self.assertTrue(np.array_equal(x.grad, np.array([5, 7])))
        self.assertTrue(np.array_equal(y.grad, np.array([2, 3])))


if __name__ == '__main__':
    unittest.main()
