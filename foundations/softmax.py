import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:        
        """
        Softmax converts a vector of raw scores called logits into a probability
        distribution. The output values are all positive and sum to 1, making them
        interpretable as probabilities.

        This is how models like GPT decide which token comes next. The final layer 
        outputs a score for every possible token, and softmax turns those scores
        into probabilities. 
        """

        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        shifted = z - np.max(z)
        exp_z = np.exp(shifted)
        return np.round(exp_z / np.sum(exp_z), 4)
