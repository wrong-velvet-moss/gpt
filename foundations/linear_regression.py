import numpy as np
from numpy.typing import NDArray

class Solution:
    """
    A single neuron without an activation function.
    A neural network is simply many of these stacked together.
    """

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the forward pass Y_hat = X \cdot W
        """
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        return np.round(np.dot(X, weights) , 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        """
        Mean Squared Error, computes error between predictions and ground truth
        """
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        squared_error = pow(model_prediction - ground_truth, 2)
        mean_squared_error = np.mean(squared_error)
        return np.round(mean_squared_error, 5)
