import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        https://arxiv.org/abs/1607.06450

        When we stack many layers in MLP, numbers flowing through the network can grow huge or
        shrink to zero. 

        Layer Normalisation recenteres each layers output so the values stay in a range.
        It normalises each input independently across its features.

        Layer norma normalises across the feature dimension, each sample statistics come from its
        own features. whereas batch dimension, each features statistics come from the batch.

        Layer norm becomes independent of batch size. 

        We can apply an affline transformation with learned parameters gamma (scale) and beta (shift)
        out_i = gamma * x_hat + beta .

        The network might need activations with a non zero mean or non zero variance.
        """
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)

        eps = 1e-5
        mu = np.mean(x)
        var = np.var(x)
        x_hat = (x- mu) / np.sqrt(var + eps)
        out = gamma * x_hat + beta
        return np.round(out, 5)