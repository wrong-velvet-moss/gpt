import numpy as np
from typing import List


class Solution:
    def relu(self, z: float) -> float:
        return np.maximum(0, z)

    def relu_deriv(self, z): 
        return (z > 0).astype(float)


    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        epsilon = 0.0

        z1 = x @ np.transpose(W1) + b1
        a1 = self.relu(z1)
        z2 = a1 @ np.transpose(W2) + b2
        y_pred = z2 

        n = len(y_true)
        # mae 
        loss = np.mean((y_pred - y_true) ** 2)

        dz2 = (2.0 / n) * (y_pred - y_true)
        db2 = dz2
        dw2 = np.outer(dz2, a1)

        da1 = dz2 @W2
        dz1 = da1 * self.relu_deriv(z1)
        db1 = dz1 
        dw1 = np.outer(dz1, x)

        return {
            "loss": round(float(loss), 4),
            "dW1": (np.round(dw1, 4) + 0.0).tolist(),
            "db1": (np.round(db1, 4) + 0.0).tolist(),
            "dW2": (np.round(dw2, 4) + 0.0).tolist(),
            "db2": (np.round(db2, 4) + 0.0).tolist(),
        }
