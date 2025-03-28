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
        self.model = model # Set input "model" as input's "self" (Trainer instance's) model from torch.nn.Module. From assignment template repository [1]. Last modified 03/23/2025.
        
        # Define the loss function.
        self.criterion = nn.SmoothL1Loss()  # Set input's "self" (Trainer instance's) loss function "criterion". From assignment template repository [1], trial and error, and PyTorch documentation [2]. Last modified 03/23/2025.
            
        # Define the optimizer.
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.012) # Set input's "self" (Trainer instance's) "optimizer". From assignment template repository [1], trial and error, and PyTorch documentation [3]. Last modified 03/23/2025.

        # Decay Learning Rate
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5) # Gradually decrease learning rate for stable convergence. From example [4], PyTorch documentation [5], and tuned with trial and error.  Last modified 03/23/2025.

    def train_model(self, data_reader) -> None:
        """
        Trains the model on data provided by the DataReader instance.

        Parameters:
            data_reader: An instance of DataReader containing the training data and labels.
        Returns:
            None
        """
        # Create DataLoader for mini-batch processing # From template
        train_dataset = TensorDataset(data_reader.X_tensor, data_reader.y_tensor) # From assignment template repository [1]. Last modified 03/23/2025.
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True) # From assignment template repository [1], tuned by trial and error. Last modified 03/23/2025.
        
        epochs: int = 350 # From assignment template repository [1], tuned by trial and error. Last modified 03/23/2025.
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0  # Track epoch loss
            
            # Iterate over batches of data # From template
            for batch_idx, (data, target) in enumerate(train_loader):  # Use your DataLoader here # From template
                # Reset gradients via zero_grad()
                self.optimizer.zero_grad() # From assignment template repository [1]. Last modified 03/23/2025.
                
                # Forward pass
                output = self.model(data) # From assignment template repository [1] and PyTorch tutorial [5]. Last modified 03/23/2025.

                # Compute loss # From template
                loss = self.criterion(output, target) # From assignment template repository [1] and PyTorch tutorial [5]. Last modified 03/23/2025.

                # Backward pass and optimize via backward() and optimizer.step()
                loss.backward() # From assignment template repository [1] and PyTorch tutorial [5]. Last modified 03/23/2025.
                self.optimizer.step() # From assignment template repository [1] and PyTorch tutorial [5]. Last modified 03/23/2025.
                
                # Accumulate loss
                epoch_loss += loss.item() # From assignment template repository [1] and PyTorch tutorial [5]. Last modified 03/23/2025.
                
            # Apply Learning Rate Decay
            self.scheduler.step() # Gradually decrease learning rate for stable convergence. From example [4], PyTorch documentation [5], and tuned with trial and error.  Last modified 03/23/2025.

            # Print epoch number and average loss for the epoch
            print(f"Epoch: {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}") # From assignment template repository [1] and PyTorch tutorial [5]. Last modified 03/27/2025.

# References:
# [1] Gholami, A., Shekari, T., "Intro to CPS Security - Mini Project 4", 2024. Available: https://github.com/tshekari3/cps_sec_mp4/tree/main
# [2] PyTorch Developers, "SmoothL1Loss", 2024. Available: https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
# [3] PyTorch Developers, "Adam", 2024. Available: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
# [4] PyTorch Developers, "StepLR", 2024. Available: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
# [5] Chilamkurthy, S., "Transfer Learning for Computer Vision Tutorial", 2024. Available: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html