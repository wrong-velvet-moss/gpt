import torch
import torch.nn
from torchtyping import TensorType

# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        """
        Convert an M x N tensor to shape (M * N // 2) x 2.
        Flatten the elements in the row, then fold into two columns.
        """
        # Reshape (M, N) tensor to (M*N/2, 2)
        # Use torch.reshape(tensor, new_shape)
        M, N = to_reshape.shape
        reshaped = torch.reshape(to_reshape, (M * N // 2, 2))
        return torch.round(reshaped, decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        """
        a 2-D tensore whose columns will be averaged reduced along axis 0.

        Return the per column mean.
        """
        # Compute column-wise mean (average across rows)
        # Use torch.mean(tensor, dim=0)
        avg = torch.mean(to_avg, dim=0)
        return torch.round(avg, decimals=4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        """
        Concatenate join tensors along a dimension.
        We concatenate two tensore along dim=1 produce (n x 2m).
        Tensors must agree on the row dimensions.

        """
        # Join two tensors side-by-side along dim=1
        # Use torch.cat((a, b), dim=1)
        concatenate = torch.cat((cat_one, cat_two), dim=1)
        return torch.round(concatenate, decimals=4)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        """
        Compute the mean squared error. between prediction and target
        """
        # Compute Mean Squared Error between prediction and target
        # Use torch.nn.functional.mse_loss(prediction, target)
        loss = torch.nn.functional.mse_loss(prediction, target)
        return torch.round(loss, decimals=4)
