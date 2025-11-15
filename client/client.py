import flwr as fl
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from model import SimpleMLP
from data_loader import load_data
from audit_log import log_event
import os
import argparse

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Client(fl.client.NumPyClient):
    def __init__(self, hospital_id, data_path):
        self.hospital_id = hospital_id
        self.data_path = data_path
        self.model = SimpleMLP(input_size=512, hidden_size=64, output_size=2)
        self.trainloader, self.valloader = load_data(data_path)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        log_event(self.hospital_id, "get_parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        log_event(self.hospital_id, "set_parameters")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        log_event(self.hospital_id, "fit_start")
        self.set_parameters(parameters)
        
        self.model.train()
        for _ in range(1):  # Train for 1 epoch
            for features, labels in self.trainloader:
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
        log_event(self.hospital_id, "fit_end")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        log_event(self.hospital_id, "evaluate")
        self.set_parameters(parameters)
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in self.valloader:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return total_loss, len(self.valloader.dataset), {"accuracy": accuracy}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FL Client")
    parser.add_argument("--hospital-id", type=str, required=True, help="Hospital ID")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data CSV")
    parser.add_argument("--server-address", default="localhost:8080", type=str, help="Server address")
    args = parser.parse_args()

    # Create client instance
    client = Client(hospital_id=args.hospital_id, data_path=args.data_path)

    # Start client
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )

if __name__ == "__main__":
    main()