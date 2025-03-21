import torch.nn as nn
import torch

class PowerSystemNN(nn.Module):
    """
    Neural network model for electric power system branch overload prediction.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize the neural network model.

        Parameters:
            input_dim (int): The number of input features.
            output_dim (int): The number of output classes.
        """
        super(PowerSystemNN, self).__init__()

       # Fully connected layers
        self.fc1 = nn.Linear(input_dim, 120)
        self.bn1 = nn.BatchNorm1d(120)  # Batch normalization for stability
        self.fc2 = nn.Linear(120, 100)
        self.bn2 = nn.BatchNorm1d(100)  # Batch normalization for stability
        self.fc3 = nn.Linear(100, output_dim)  

        # Activation function & dropout
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(0.2)  # Dropout for regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Parameters:
            x (Tensor): The input tensor to the neural network.

        Returns:
            Tensor: The output of the network.
        """
        x = self.act(self.bn1(self.fc1(x)))
        x = self.dropout(x)  # Apply dropout
        x = self.act(self.bn2(self.fc2(x)))
        x = self.fc3(x)  # Output layer
        return x