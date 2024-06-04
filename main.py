from models import FullyConnectedNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: {}".format(device))

# TODO: Change the following for experiment.
algorithm = "Adam"

def load_data():
    X_train = np.loadtxt("X_train.txt")
    Y_train = np.loadtxt("Y_train.txt").astype(int)
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32, device=device)  # Note the type change here for BCELoss
    return X_train, Y_train


# Initialize the model, loss criterion, and optimizer
model = FullyConnectedNN()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
if algorithm == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=0.05)
elif algorithm == "Adagrad":
    optimizer = optim.Adagrad(model.parameters(), lr=0.05)
elif algorithm == "SGD-M":
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
elif algorithm == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=0.05, betas=(0.9, 0.999)) # is this a good parameter?

# Load training data
X_train, Y_train = load_data()

# DataLoader setup
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(dataset, batch_size=100, shuffle=True)

# Training the model
num_epochs = 200
losses = []
accuracies = []
# hessians = [] # One hessian per epoch, evaluated at the beginning of epoch.
max_eigenvals = []
log_R_med = [] # Log of the ratio max_eigenvalue / med_eigenvalue.

for epoch in range(num_epochs):
    epoch_losses = []
    correct = 0
    total = 0
    compute_hessian = True

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()  # Ensure output is same shape as labels
        loss = criterion(outputs, labels)

        # Calculate Hessian at beginning of each epoch.
        if compute_hessian:
            # Use autograd.grad to compute the first derivative
            first_derivative = autograd.grad(loss, model.parameters(), create_graph=True)
            first_derivative_flattened = torch.cat([grad.view(-1) for grad in first_derivative])

            # Compute the Hessian matrix
            hessian = []
            for grad in first_derivative: # Each entry: grad on weight1, bias1, etc.
                for g in grad.view(-1):  # Flatten the gradient
                    second_derivative = autograd.grad(g, model.parameters(), retain_graph=True, allow_unused=True)
                    second_derivative_flattened = torch.cat([sd.view(-1) for sd in second_derivative if sd is not None]).numpy()
                    hessian.append(second_derivative_flattened)

            hessian = torch.tensor(np.array(hessian))
            L = torch.linalg.eigvalsh(hessian)
            max_eigenvals.append(torch.max(L))
            log_R_med.append(torch.log(max_eigenvals[-1]) - torch.log(torch.median(L)))
            # hessians.append(hessian)
            compute_hessian = False

        loss.backward() # Create graph for higher-order derivatives
        optimizer.step()
        epoch_losses.append(loss.item())

        # Calculate accuracy manually
        predicted = (outputs > 0.5).float()  # Convert probabilities to 0 or 1
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    epoch_acc = 100 * correct / total
    losses.append(epoch_loss)
    accuracies.append(epoch_acc)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Max ev: {max_eigenvals[-1]}, Log R_med: {log_R_med[-1]}")

# Plotting
plt.figure(figsize=(12, 12))
plt.suptitle("Training using {}".format(algorithm))

plt.subplot(2, 2, 1)
plt.plot(losses, label='Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(accuracies, label='Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(max_eigenvals, label='Max eigenvalue')
plt.title('Max eigenvalue of Hessian vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Max eigenvalue')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(log_R_med, label='Log R_med')
plt.title('Log R_med vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Log R_med')
plt.legend()

plt.show()
