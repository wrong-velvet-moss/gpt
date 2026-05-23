import numpy as np
from numpy.typing import NDArray


class Solution:
    """
    Activation functions are what give neural networks gives
    the ability to learn complex patterns. Without them,
    a neural network is just a series of linear transformations.
    No matter how many layers you stack, the output would still
    be a linear function of the input.
    """
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)"
        return np.round(1 /(1 + np.exp(-z)), 5)
            

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        ReLU replaced sigmoid because ReLU gradient is f'(x) = 1.
        It does not suffer from the vanishing gradient problem in which
        sigmoid suffers from.

        sigmoid = vanishing gradient results to slow learning.
        """
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        return np.maximum(0, z)
