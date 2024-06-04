import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Toy model inspired by "Opening the Black Box of Deep Neural Networks via Information."
(https://arxiv.org/abs/1703.00810)
This is a MLP with decreasing number of neurons. (The authors wanted to capture "compression")
Originally they used tanh instead of relu as activations. 
"""


class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()

        # Define the fully connected layers
        # self.fcs = nn.ModuleList()
        # for i in range(4):
        #     if i == 0:
        #         self.fcs.append(nn.Linear(in_features=12, out_features=64))
        #     else:
        #         self.fcs.append(nn.Linear(in_features=4**(4-i), out_features=4**(4-i-1)))
        self.fc1 = nn.Linear(in_features=12, out_features=64)  # Input layer
        # self.fc2 = nn.Linear(in_features=64, out_features=32)
        # self.fc3 = nn.Linear(in_features=32, out_features=16)
        # self.fc4 = nn.Linear(in_features=16, out_features=8)
        # self.fc5 = nn.Linear(in_features=8, out_features=4)
        # self.fc6 = nn.Linear(in_features=4, out_features=2)
        self.fc7 = nn.Linear(in_features=64, out_features=1) # Output layer

    def forward(self, x):
        # Pass the input through the layers with ReLU activation for hidden layers
        # for i in range(3):
        #     x = F.relu(self.fcs[i](x))
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))

        # Use sigmoid activation for the output layer
        # x = torch.sigmoid(self.fcs[3](x))
        x = torch.sigmoid(self.fc7(x))
        return x


"""Tests."""
# Create the neural network model
model = FullyConnectedNN()

# Print the model structure
print(model)