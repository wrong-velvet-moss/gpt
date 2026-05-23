import numpy as np
from numpy.typing import NDArray


class Solution:
    """
    Softmax gives your probabilities, the model's 
    confidence for each class. But how do you tell
    the model how wrong it was. 

    Cross entropy compares the predicted distribution to the actual
    distribution produces a single number. 

    A high value is when the model is confidently wrong.
    A low value is when the model is confidently right.

    The loss function = -ln(p)

    Ideally, the lower the loss function the better. A perfrect
    prediction gives a loss of 0.

    A loss function is behind almost every classifier and 
    token predictions in GPT.
    """

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        """
        For two classes, 
        L = -1/n sum_i=1^n [y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i)]
        where y_i is the true label (0 or 1) and p_i is the predicted probability.
        """
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    
        loss_function = (y_true * np.log(y_pred)) + ((1 - y_true) * np.log(1 - y_pred))
        loss = -np.mean(loss_function)
        return np.round(loss, 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        loss_function = np.sum(y_true * np.log(y_pred), axis=1)
        loss = -np.mean(loss_function)
        return np.round(loss, 4)
