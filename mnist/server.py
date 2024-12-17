import flwr as fl
import os
from dotenv import load_dotenv

# Get the path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
    on_fit_config_fn=lambda rnd: {"epochs": 5, "batch_size": 32},
)

# Start the Flower server with the custom strategy
if __name__ == "__main__":
    # Load the .env file
    load_dotenv(os.path.join(parent_dir, ".env"))

    # Dynamically get the server IP
    server_ip = os.getenv("SERVER_IP")
    port = 8080

    print(f"Flower server is starting...")
    print(f"Connect clients using server IP: {server_ip}:{port}")

    # Start the Flower server
    fl.server.start_server(
        server_address=f"{server_ip}:{port}",  # Use dynamically determined IP
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
