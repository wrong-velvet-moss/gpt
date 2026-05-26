import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        """
        A MLP neutral network with one or more hidden layers.
        Each layer performs a linear transformation followed by a non linear activation in this case ReLU.

        h = ReLU(hW + b)
        h is the input layer l
        W is the weight matrix
        b is the bias vector

        Why we need multiple layers? A single linear layer can only learn linear decision boundaries.
        """
        h = x
        for i in range(len(weights)):
            # linear transform
            h = h @ weights[i] + biases[i]
            if i < len(weights) - 1:
                h = np.maximum(0, h) # relu on the hideen layers
            
        
        return np.round(h, 5)