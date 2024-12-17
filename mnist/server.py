import flwr as fl

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
    on_fit_config_fn=lambda rnd: {"epoch": 5, "batch_size": 32},
)

# Start the Flower server with the custom strategy
if __name__ == "__main__":
    config = fl.server.ServerConfig(num_rounds=5)  # Use the ServerConfig class
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Address/port for the server
        strategy=strategy,  # Pass the strategy here
        config=config,  # Pass the ServerConfig object
    )
