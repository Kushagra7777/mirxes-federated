import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class CSVData(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        # Assume last column is the label
        self.features = df.iloc[:, :-1].values.astype(np.float32)
        self.labels = df.iloc[:, -1].values.astype(np.int64)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx])
        label = torch.tensor(self.labels[idx])
        return feature, label

def load_data(csv_path, batch_size=32):
    dataset = CSVData(csv_path)
    
    # Split into train and validation (80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return trainloader, valloader