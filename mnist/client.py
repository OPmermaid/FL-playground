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
    
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

    def get_parameters(self, config=None) -> list[np.ndarray]:
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

        epochs = config.get("epoch", 1)
        with tqdm(range(epochs)) as pbar:
            for epoch in tqdm(range(epochs), desc=f"Client Training (Client {id(self)})", leave=True):
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

                    epoch_loss += loss.item()
                    pred = output.argmax(dim=1)
                    epoch_correct += pred.eq(target).sum().item()
                    epoch_total += len(target)

            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total

            # print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(self.train_loader):.4f}, "
            #     f"Accuracy: {epoch_correct / epoch_total:.4f}")
            # Dynamically update the description with epoch, loss, and accuracy
            pbar.set_description(
                f"Client Training (Client {id(self)}) - Epoch {epoch + 1}/{epochs}, "
                f"Loss: {epoch_loss / len(self.train_loader):.4f}, "
                f"Accuracy: {epoch_correct / epoch_total:.4f}"
            )


        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)

        print(f"Final Train Loss: {avg_loss:.4f}, Final Train Accuracy: {accuracy:.4f}")
        return self.get_parameters(), len(self.train_loader.dataset), {}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)

        avg_loss = loss / len(self.test_loader)
        accuracy = correct / total
        self.test_losses.append(avg_loss)
        self.test_accuracies.append(accuracy)

        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return avg_loss, len(self.test_loader.dataset), {"accuracy": accuracy}



    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)

        avg_loss = loss / len(self.test_loader)
        accuracy = correct / total
        self.test_losses.append(avg_loss)
        self.test_accuracies.append(accuracy)

        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return avg_loss, len(self.test_loader.dataset), {"accuracy": accuracy}

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

# Partition data based on client_id
def partition_data(dataset, num_clients, client_id, sparse=True):
    """
    Split the dataset into partitions for clients.

    - num_clients: Total number of clients
    - client_id: ID of the current client
    - sparse: Whether to sparsely sample the dataset (default=True)
    """
    indices = list(range(len(dataset)))
    if sparse:
        # Assign every Nth data point to each client
        client_indices = indices[client_id::num_clients]
    else:
        # Split data into contiguous blocks
        data_size = len(indices) // num_clients
        start = client_id * data_size
        end = start + data_size
        client_indices = indices[start:end]

    return Subset(dataset, client_indices)

# Load partitioned data
def load_data(client_id, num_clients):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root="./mnist/data", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="./mnist/data", train=False, download=True, transform=transform)

    # Partition data for this client
    train_subset = partition_data(train_dataset, num_clients, client_id, sparse=True)
    test_subset = partition_data(test_dataset, num_clients, client_id, sparse=False)

    return train_subset, test_subset

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True, help="ID of the client (0, 1, ...)")
    parser.add_argument("--num_clients", type=int, required=True, help="Total number of clients")
    args = parser.parse_args()

    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load partitioned data
    train_dataset, test_dataset = load_data(args.client_id, args.num_clients)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # Initialize model
    model = SimpleCNN().to(device)
    client = FLClient(model, train_loader, test_loader, device)

    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

    # Plot metrics after training
    client.plot_metrics()

