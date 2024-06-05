from models import FullyConnectedNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
device = torch.device('cpu')
print("device: {}".format(device))

# TODO: Change the following for experiment.
algorithm = "Adam"

def load_training_data():
    X_train = np.loadtxt("X_train.txt")
    Y_train = np.loadtxt("Y_train.txt").astype(int)
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32, device=device)  # Note the type change here for BCELoss
    return X_train, Y_train

def load_testing_data():
    X_test = np.loadtxt("X_test.txt")
    Y_test = np.loadtxt("Y_test.txt").astype(int)
    # Convert to tensors
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test = torch.tensor(Y_test, dtype=torch.float32, device=device)
    return X_train, Y_train


# Initialize the model, loss criterion, and optimizer
model = FullyConnectedNN(device=device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
if algorithm == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=0.01) # TODO: try 0.1 vs 0.01.
elif algorithm == "Adagrad":
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
elif algorithm == "SGD-M":
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
elif algorithm == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
elif algorithm == "RMSprop":
    optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)


# Load training data
X_train, Y_train = load_training_data()
X_test, Y_test = load_testing_data()

w = np.loadtxt("ground_truth_w.txt")
b = np.loadtxt("ground_truth_b.txt")
ground_truth_output = (X_test.numpy().dot(w) + b >= 0).astype(float)
ground_truth_correct = (ground_truth_output == Y_test.numpy()).sum()
ground_truth_accuracy = 100 * ground_truth_correct / Y_test.size(0)
print("Ground truth accuracy: {}".format(ground_truth_accuracy))

# DataLoader setup
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(dataset, batch_size=100, shuffle=True)

# Training the model
num_epochs = 400
losses = []
train_accuracies = []
test_accuracies = []
# hessians = [] # One hessian per epoch, evaluated at the beginning of epoch.
max_eigenvals = []
log_R_med = [] # Log of the ratio max_eigenvalue / med_eigenvalue.

print("start training")
for epoch in range(num_epochs):
    epoch_losses = []
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    compute_hessian = True

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        train_outputs = model(inputs).squeeze()  # Ensure output is same shape as labels
        loss = criterion(train_outputs, labels)

        # Calculate Hessian at beginning of each epoch.
        if compute_hessian:
            # Use autograd.grad to compute the first derivative
            first_derivative = autograd.grad(loss, model.parameters(), create_graph=True)
            first_derivative_flattened = torch.cat([grad.view(-1) for grad in first_derivative])

            # Compute the Hessian matrix
            hessian = torch.zeros((0, first_derivative_flattened.shape[0]), device=device)
            for grad in first_derivative: # Each entry: grad on weight1, bias1, etc.
                for g in grad.view(-1):  # Flatten the gradient
                    second_derivative = autograd.grad(g, model.parameters(), retain_graph=True, allow_unused=True)
                    second_derivative_flattened = torch.cat([sd.view(-1) for sd in second_derivative if sd is not None])
                    hessian = torch.cat((hessian, torch.unsqueeze(second_derivative_flattened, dim=0)), dim=0)

            L = torch.linalg.eigvalsh(hessian)
            max_eigenvals.append(torch.max(L))
            log_R_med.append(torch.log(max_eigenvals[-1]) - torch.log(torch.median(L)))
            # hessians.append(hessian)
            compute_hessian = False

        loss.backward() # Create graph for higher-order derivatives
        optimizer.step()
        epoch_losses.append(loss.item())

        # Calculate accuracy manually
        predicted = (train_outputs > 0.5).float()  # Convert probabilities to 0 or 1
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # Testing
        test_outputs = model(X_test).squeeze()
        predicted = (test_outputs > 0.5).float()
        test_total += Y_test.size(0)
        test_correct += (predicted == Y_test).sum().item()

    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(epoch_loss)
    train_accuracy = 100 * train_correct / train_total
    train_accuracies.append(train_accuracy)
    test_accuracy = 100 * test_correct / test_total
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train accuracy: {train_accuracy:.2f}%, Test accuracy: {test_accuracy:.2f}%, Max ev: {max_eigenvals[-1]}, Log R_med: {log_R_med[-1]}")

# Plotting
x = [i for i in range(num_epochs)]

plt.figure(figsize=(12, 12))
plt.suptitle("Training using {}".format(algorithm))

plt.subplot(2, 2, 1)
plt.plot(x, losses, label='Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, train_accuracies, label='Train accuracy')
plt.plot(x, test_accuracies, label='Test accuracy')
plt.axhline(y=ground_truth_accuracy, color='g', linestyle='--', label='Ground_truth') # why is it sometimes better than  ground truth?
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, max_eigenvals, label='Max eigenvalue')
plt.title('Max eigenvalue of Hessian vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Max eigenvalue')
plt.legend()

plt.subplot(2, 2, 4)
log_R_med = np.array([tensor.item() for tensor in log_R_med])
plt.plot(x, log_R_med, label='Log R_med')
plt.title('Log R_med vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Log R_med')

# Annotate infinity values with dots above the highest finite value
max_finite_value = np.max(log_R_med[np.isfinite(log_R_med)])
inf_y = max_finite_value + (max_finite_value * 0.1)  # Position slightly above the highest finite value
for i in range(len(x)):
    if np.isinf(log_R_med[i]):
        plt.plot(i, inf_y, 'ro')
plt.legend()

plt.show()

