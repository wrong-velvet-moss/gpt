import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        # embeddings: (vocab_size, embed_dim) matrix
        # token_ids: 1D array of integer token IDs
        # Return the embedding vectors for the given token IDs
        # return np.round(your_answer, 5)
        look_up = [None] * len(token_ids)
        for i, token_id in enumerate(token_ids):
            look_up[i] = embeddings[token_id]
        return np.round(np.array(look_up), 5)