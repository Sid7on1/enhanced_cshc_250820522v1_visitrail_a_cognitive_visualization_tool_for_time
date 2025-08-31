import logging
import os
import sys
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisiTrailDataset(Dataset):
    """
    Custom dataset class for VisiTrail data.
    
    Attributes:
    data (pd.DataFrame): Dataframe containing the dataset.
    labels (pd.Series): Series containing the labels.
    """
    def __init__(self, data: pd.DataFrame, labels: pd.Series):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        data_point = self.data.iloc[idx]
        label = self.labels.iloc[idx]
        return data_point, label

class VisiTrailModel(nn.Module):
    """
    Custom model class for VisiTrail data.
    
    Attributes:
    input_dim (int): Input dimension of the model.
    hidden_dim (int): Hidden dimension of the model.
    output_dim (int): Output dimension of the model.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(VisiTrailModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VisiTrailTrainer:
    """
    Trainer class for VisiTrail model.
    
    Attributes:
    model (VisiTrailModel): Model to be trained.
    device (torch.device): Device to train the model on.
    optimizer (torch.optim.Optimizer): Optimizer to use for training.
    loss_fn (torch.nn.Module): Loss function to use for training.
    """
    def __init__(self, model: VisiTrailModel, device: torch.device, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, data_loader: DataLoader):
        """
        Train the model on the given data loader.
        
        Args:
        data_loader (DataLoader): Data loader to train the model on.
        """
        self.model.train()
        total_loss = 0
        for batch in data_loader:
            data, labels = batch
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        logger.info(f'Training loss: {total_loss / len(data_loader)}')

    def evaluate(self, data_loader: DataLoader):
        """
        Evaluate the model on the given data loader.
        
        Args:
        data_loader (DataLoader): Data loader to evaluate the model on.
        
        Returns:
        accuracy (float): Accuracy of the model on the given data loader.
        """
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for batch in data_loader:
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / len(data_loader.dataset)
        logger.info(f'Evaluation accuracy: {accuracy}')
        return accuracy

def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the data from the given file path.
    
    Args:
    file_path (str): File path to load the data from.
    
    Returns:
    data (pd.DataFrame): Loaded data.
    labels (pd.Series): Loaded labels.
    """
    data = pd.read_csv(file_path)
    labels = data['label']
    data = data.drop('label', axis=1)
    return data, labels

def split_data(data: pd.DataFrame, labels: pd.Series, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.
    
    Args:
    data (pd.DataFrame): Data to split.
    labels (pd.Series): Labels to split.
    test_size (float): Proportion of the data to use for testing.
    
    Returns:
    train_data (pd.DataFrame): Training data.
    test_data (pd.DataFrame): Testing data.
    train_labels (pd.Series): Training labels.
    test_labels (pd.Series): Testing labels.
    """
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size, random_state=42)
    return train_data, test_data, train_labels, test_labels

def main():
    # Load the data
    file_path = 'data.csv'
    data, labels = load_data(file_path)
    
    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = split_data(data, labels)
    
    # Create data loaders
    train_dataset = VisiTrailDataset(train_data, train_labels)
    test_dataset = VisiTrailDataset(test_data, test_labels)
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create the model, device, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisiTrailModel(input_dim=train_data.shape[1], hidden_dim=128, output_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create the trainer
    trainer = VisiTrailTrainer(model, device, optimizer, loss_fn)
    
    # Train the model
    for epoch in range(10):
        logger.info(f'Epoch {epoch+1}')
        trainer.train(train_data_loader)
        accuracy = trainer.evaluate(test_data_loader)
        logger.info(f'Epoch {epoch+1} accuracy: {accuracy}')

if __name__ == '__main__':
    main()