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
        Initializes the Trainer with a model and an optional learning rate.
        
        Parameters:
            model (nn.Module): The neural network model to be trained.
        """
        
        self.model = model
        
        # Define loss function based on output layer size
        if model.fc3.out_features == 1:
            self.criterion = nn.BCEWithLogitsLoss()  # Binary Classification
        else:
            self.criterion = nn.SmoothL1Loss()  # More stable loss for regression

        # Define Adam optimizer with default learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.007)  # Lowered for stability

        # Learning Rate Decay (StepLR)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)  

    def train_model(self, data_reader) -> None:
        """
        Trains the model on data provided by the DataReader instance.

        Parameters:
            data_reader: An instance of DataReader containing the training data and labels.

        Returns:
            None
        """
        
        # Create DataLoader for mini-batch processing
        train_dataset = TensorDataset(data_reader.X_tensor, data_reader.y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        epochs: int = 70  # Increased for better convergence
        
         # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0  # Track epoch loss

            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()  # Reset gradients

                # Forward pass
                output = self.model(data)

                # Compute loss
                loss = self.criterion(output, target)

                # Backward pass & optimization
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                epoch_loss += loss.item()

            # Apply Learning Rate Decay
            self.scheduler.step()

            # Print average loss for the epoch
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")