import torch.nn as nn
import torch

class PowerSystemNN(nn.Module):
    """
    Neural network model for electric power system branch overload prediction.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize neural network model.

        Parameters:
            input_dim (int): The number of input features.
            output_dim (int): The number of output classes.
        """
        super(PowerSystemNN, self).__init__()

        # Define number of layers
        self.fc1 = nn.Linear(input_dim, 120) # First fully connected layer. Number of neurons found by trial and error. From assignment template repository [1]. Last modified 03/23/2025.
        self.bn1 = nn.BatchNorm1d(120)  # First batch normalization. From PyTorch tutorial [2]. Last modified 03/23/2025.
        self.fc2 = nn.Linear(120, 100)  # Second fully connected layer. Number of layers found by trial and error. From assignment template repository [1]. Last modified 03/23/2025.
        self.bn2 = nn.BatchNorm1d(100)  # Second batch normalization. From PyTorch tutorial [2]. Last modified 03/23/2025.
        self.fc3 = nn.Linear(100, output_dim) # Last fully connected layer. Number of neurons found by trial and error. From assignment template repository [1]. Last modified 03/23/2025.

        # Activation function & dropout
        self.act = nn.Tanh() # Activation function chose from trial and error. From assignment template repository [1]. PyTorch documentation [3]. Last modified 03/23/2025.
        self.dropout = nn.Dropout(0.2) # Dropout layer from PyTorch tutorial [2]. Last modified 03/23/2025.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Parameters:
            x (Tensor): The input tensor to the neural network.

        Returns:
            Tensor: The output of the network.
        """
        x = self.act(self.bn1(self.fc1(x))) # Run input through first hidden layer. From assignment template repository [1]. Last modified 03/23/2025.
        x = self.dropout(x)  # Apply dropout. Determined where by trial and error. From PyTorch tutorial [2]. Last modified 03/23/2025.
        x = self.act(self.bn2(self.fc2(x))) # Run through second hidden layer. From assignment template repository [1]. Last modified 03/23/2025.
        x = self.fc3(x)  # Output layer. Last modified 03/23/2025.
        return x
    
    # Sources
    # [1] Gholami, A., Shekari, T., "Intro to CPS Security - Mini Project 4", 2024. Available: https://github.com/tshekari3/cps_sec_mp4/tree/main
    # [2] PyTorch Developers, "Building Models with PyTorch", 2024. Available: https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
    # [3] PyTorch Developers, "Tanh", 2024. Available: https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh