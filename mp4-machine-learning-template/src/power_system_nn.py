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
        self.fc2 = nn.Linear(120, 84)
        self.bn2 = nn.BatchNorm1d(84)  # Batch normalization for stability
        self.fc3 = nn.Linear(84, output_dim)  

        # Activation function & dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # Dropout for regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Parameters:
            x (Tensor): The input tensor to the neural network.

        Returns:
            Tensor: The output of the network.
        """
        x = self.relu(self.bn1(self.fc1(x)))  # Fully connected -> BatchNorm -> ReLU
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.bn2(self.fc2(x)))  # Fully connected -> BatchNorm -> ReLU
        x = self.dropout(x)  # Apply dropout again
        x = self.fc3(x)  # Output layer (no activation)
        return x