import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_reader import DataReader


class Trainer: # Creates new class. From assignment template repository [1]. Last modified 07/01/2024.
    """
    Class responsible for training a neural network model.
    """
    def __init__(self, model: nn.Module) -> None:   # Constructor for initializer method where "self" is an instance of Trainer,
													# and "model" is expected to be model from torch.nn.Module. The contstructior
													# has no output From assignment template repository [1]. Last modified 07/01/2024.
        """
        Initializes the Trainer with a model and an optional learning rate.
        
        Parameters:
            model (nn.Module): The neural network model to be trained.
        """
        self.model = model # Set input "model" as input's "self" (Trainer instance's) model from torch.nn.Module. From assignment template repository [1]. Last modified 07/01/2024.
        
        # Define the loss function.
        self.criterion = nn.SmoothL1Loss()  # Set input's "self" (Trainer instance's) loss function "criterion". From assignment template repository [1] and TODO [2]. Last modified 03/23/2025.
            
        # Define the optimizer. # From template
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.012) # Set input's "self" (Trainer instance's) "optimizer". From assignment template repository [1] and TODO [3]. Last modified 03/23/2025.

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
        
        epochs: int = 350 # Define the number of epochs to train the model for # From template
        
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

# References:
# [1] Gholami, A., Shekari, T., "Intro to CPS Security - Mini Project 4", 2024. Available: https://github.com/tshekari3/cps_sec_mp4/tree/main