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
        
        # Define the loss function. # From template
        self.criterion = nn.SmoothL1Loss()  # Define the appropriate loss function # From template
            
        # Define the optimizer. # From template
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01) # Initialize the optimizer with model parameters and learning rate # From template

        # Learning Rate Decay (StepLR)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)


    def train_model(self, data_reader) -> None:
        """
        Trains the model on data provided by the DataReader instance.

        Parameters:
            data_reader: An instance of DataReader containing the training data and labels.
        Returns:
            None
        """
        # Create DataLoader for mini-batch processing # From template
        train_dataset = TensorDataset(data_reader.X_tensor, data_reader.y_tensor)   # From template
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)   # From template, modified example
        
        epochs: int = 400 # Define the number of epochs to train the model for # From template
        
        # Training loop # From template
        for epoch in range(epochs):
            epoch_loss = 0.0  # Track epoch loss
            
            # Iterate over batches of data # From template
            for batch_idx, (data, target) in enumerate(train_loader):  # Use your DataLoader here # From template
                # Reset gradients via zero_grad() # From template
                self.optimizer.zero_grad()
                
                # Forward pass # From template
                output = self.model(data)

                # Compute loss # From template
                loss = self.criterion(output, target)

                # Backward pass and optimize via backward() and optimizer.step() # From template
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                epoch_loss += loss.item()

            # Apply Learning Rate Decay
            self.scheduler.step()

            # You can print the loss here to see how it decreases # From template
            # Print average loss for the epoch
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")