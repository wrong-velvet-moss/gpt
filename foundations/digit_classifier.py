import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        """Raw pixel values arrive as a flat 784 row element vector (28x28).
        
        Hidden layer projects to 512 features.
        ReLu introduces non linearity and droput 20% regualrises.
        The final 10 sigmoid
        """
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.first_linear = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.projection = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        """ Takes a 28x28 MNIST image and predicts which digit (0-9) it represents

        The architecutre is a two layer mlp.
        Linear 784 ->512) projects pixel space into a learned 512 dimensiion
        ReLuu introduces on linearity 
        Dropout randomly zeros 20% of activations during training. Forces
        information across neurons. During evaluation, dropout is turned off
        Linear (512 -> 10) projects to each digit
        Sigmoid squashes the output into a probablity from 0 to 1.
        """
        torch.manual_seed(0)

        # (784 x 512) x (512, 10) = ()
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x = self.first_linear(images) # (batch, 784) -> (batch, 512)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.projection(x) # (batch, 512) -> (batch, 10)
        x = self.sigmoid(x) 
        return torch.round(x, decimals=4)

