import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedNN(nn.Module):
    def __init__(self, device='cpu'):
        super(FullyConnectedNN, self).__init__()
        # TODO: try 64 neurons vs 128.
        self.fc1 = nn.Linear(in_features=12, out_features=64, device=device)  # Input layer
        self.fc7 = nn.Linear(in_features=64, out_features=1, device=device) # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc7(x))
        return x


"""Tests."""
# Create the neural network model
model = FullyConnectedNN()

# Print the model structure
print(model)