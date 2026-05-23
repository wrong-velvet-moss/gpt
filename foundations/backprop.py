import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self, z: float) -> float:
        return 1 / (1 + np.exp(-z))
            

    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        """
        Single neuron with sigmoid activation and MSE loss.

        Forward pass:
            z = w · x + b
            y_hat = sigmoid(z) = 1 / (1 + exp(-z))

        Loss (mean squared error, with 1/2 for clean gradients):
            L = 1/2 * (y_hat - y)^2

        Gradients (via chain rule: dL/dw = dL/dy_hat * dy_hat/dz * dz/dw):
            dL/dy_hat = (y_hat - y)
            dy_hat/dz = y_hat * (1 - y_hat)        # sigmoid derivative
            dz/dw_i   = x_i
            dz/db     = 1

        Therefore:
            dL/dw_i = (y_hat - y) * y_hat * (1 - y_hat) * x_i
            dL/db   = (y_hat - y) * y_hat * (1 - y_hat)
        """
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        forward = np.dot(x, w) + b 
        y_hat = self.sigmoid(forward)

        loss = 0.5 * pow(y_hat - y_true, 2)
        dl_dw = np.round((y_hat - y_true) * y_hat * (1 - y_hat) * x, 5)
        dl_db = np.round((y_hat - y_true) * y_hat * (1 - y_hat), 5)
        return (dl_dw, dl_db)
    