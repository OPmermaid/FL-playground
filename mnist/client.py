import flwr as fl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
from dotenv import load_dotenv
# Get the path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define a simple PyTorch model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Flower client class
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        # Add lists to track metrics
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {key: torch.tensor(val) for key, val in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):        
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0
        correct = 0
        total = 0

        epochs = config.get("epochs", 1)  # Fetch 'epochs' with a default of 1
        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                epoch_loss = 0
                epoch_correct = 0
                epoch_total = 0
                for data, target in self.train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    # Accumulate loss and accuracy
                    epoch_loss += loss.item() * data.size(0)
                    _, predicted = output.max(1)
                    epoch_correct += predicted.eq(target).sum().item()
                    epoch_total += target.size(0)

                # Log progress
                avg_loss = epoch_loss / epoch_total
                avg_accuracy = epoch_correct / epoch_total
                pbar.set_description(
                    f"Client Training (Client {id(self)}) - Epoch {epoch + 1}/{epochs}, "
                    f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}"
                )

                total_loss += epoch_loss
                correct += epoch_correct
                total += epoch_total

        # Store metrics
        self.train_losses.append(total_loss / total)
        self.train_accuracies.append(correct / total)

        return self.get_parameters(), len(self.train_loader.dataset), {}



    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
                total += target.size(0)

        # Compute average loss and accuracy
        avg_loss = total_loss / total
        avg_accuracy = correct / total

        # Store evaluation metrics
        self.test_losses.append(avg_loss)
        self.test_accuracies.append(avg_accuracy)

        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.4f}")

        return avg_loss, total, {"accuracy": avg_accuracy}


    def plot_metrics(self):
            # Plot training and evaluation metrics
            plt.figure(figsize=(10, 5))

            # Plot Loss
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses, label="Train Loss")
            plt.plot(self.test_losses, label="Test Loss")
            plt.xlabel("Rounds")
            plt.ylabel("Loss")
            plt.title("Loss per Round")
            plt.legend()

            # Plot Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(self.train_accuracies, label="Train Accuracy")
            plt.plot(self.test_accuracies, label="Test Accuracy")
            plt.xlabel("Rounds")
            plt.ylabel("Accuracy")
            plt.title("Accuracy per Round")
            plt.legend()

            plt.tight_layout()
            plt.show()

# # Partition data based on client_id
# def partition_data(dataset, num_clients, client_id, sparse=True):
#     """
#     Split the dataset into partitions for clients.

#     - num_clients: Total number of clients
#     - client_id: ID of the current client
#     - sparse: Whether to sparsely sample the dataset (default=True)
#     """
#     indices = list(range(len(dataset)))
#     if sparse:
#         # Assign every Nth data point to each client
#         client_indices = indices[client_id::num_clients]
#     else:
#         # Split data into contiguous blocks
#         data_size = len(indices) // num_clients
#         start = client_id * data_size
#         end = start + data_size
#         client_indices = indices[start:end]

#     return Subset(dataset, client_indices)

# Load partitioned data
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root="./mnist/data", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="./mnist/data", train=False, download=True, transform=transform)

    return train_dataset, test_dataset

# Main function
if __name__ == "__main__":

    # Load .env file, and Get the server IP dynamically or from an environment variable
    load_dotenv(os.path.join(parent_dir, ".env"))
    server_ip = os.getenv("SERVER_IP")  # Default to localhost
    port = 8080

    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Load and split data
    train_dataset, test_dataset = load_data()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # Initialize model
    model = SimpleCNN().to(device)
    

    # Start Flower client
    client = FLClient(model, train_loader, test_loader, device)
    print(f"Connecting to server at {server_ip}:{port}...")
    fl.client.start_numpy_client(server_address=f"{server_ip}:{port}", client=client)

    # Plot metrics after training
    client.plot_metrics()

