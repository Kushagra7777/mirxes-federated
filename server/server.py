import flwr as fl
from flwr.server.strategy import FedAvg
import logging
import os
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FL Server")
    parser.add_argument("--server-address", default="0.0.0.0:8080", type=str, help="Server address")
    args = parser.parse_args()

    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,  # Train on all available clients
        fraction_evaluate=1.0,  # Evaluate on all available clients
        min_fit_clients=3,  # Minimum number of clients to train on
        min_evaluate_clients=3,  # Minimum number of clients to evaluate on
        min_available_clients=3,  # Minimum number of clients to start training
    )

    # Start server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()