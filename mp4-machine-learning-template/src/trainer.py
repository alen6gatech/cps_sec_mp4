import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_reader import DataReader


class Trainer:
    """
    Class responsible for training a neural network model.
    """
    def __init__(self, model: nn.Module) -> None:
        """
        Initializes the Trainer with a model.
        
        Parameters:
            model (nn.Module): The neural network model to be trained.
        """
        self.model = model
        
        # Define loss function based on output layer size
        if model.fc3.out_features == 1:
            self.criterion = nn.BCEWithLogitsLoss()  # Binary Classification
        else:
            self.criterion = nn.SmoothL1Loss()  # Regression

        # Create optimizer exactly as in the PyTorch tutorial
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.005)

    def train_model(self, data_reader) -> None:
        """
        Trains the model on data provided by the DataReader instance.

        Parameters:
            data_reader: An instance of DataReader containing the training data and labels.
        """
        
        # Create DataLoader for mini-batch processing
        train_dataset = TensorDataset(data_reader.X_tensor, data_reader.y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Training loop
        for data, target in train_loader:
            self.optimizer.zero_grad()  # Zero the gradient buffers
            output = self.model(data)   # Forward pass
            loss = self.criterion(output, target)  # Compute loss
            loss.backward()  # Backward pass
            self.optimizer.step()  # Update weights
            print(f"Loss: {loss.item()}")