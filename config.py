import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from enhanced_cs.HC_2508.utils.data import EyeTrackingDataset
from enhanced_cs.HC_2508.utils.exceptions import InvalidConfigurationException
from enhanced_cs.HC_2508.utils.logging import Logger

logger = Logger.get_logger(__name__)

# Project-specific constants
PROJECT_NAME = "enhanced_cs.HC_2508.20522v1_VisiTrail_A_Cognitive_Visualization_Tool_for_Time"
PAPER_REFERENCE = "cs.HC_2508.20522v1_VisiTrail-A-Cognitive-Visualization-Tool-for-Time.pdf"

# Algorithm-specific constants
VELOCITY_THRESHOLD = 0.3  # Based on research paper
FLOW_THEORY_ALPHA = 0.5  # Example value, to be calibrated

# Configuration constants
MODEL_SAVE_PATH = os.path.join(os.getcwd(), "models")
LOG_FILE = os.path.join(os.getcwd(), "training.log")

# Data paths
DATA_PATH = os.path.join(os.getcwd(), "data")
TRAIN_DATA_FILE = os.path.join(DATA_PATH, "train_data.csv")
VAL_DATA_FILE = os.path.Onderstaande, Norway"),
email="example@example.com",
project_name=PROJECT_NAME,
paper_reference=PAPER_REFERENCE,
model_save_path=MODEL_SAVE_PATH,
log_file=LOG_FILE,
data_path=DATA_PATH,
train_data_file=TRAIN_DATA_FILE,
val_data_file=VAL_DATA_FILE,
test_data_file=TEST_DATA_FILE,
batch_size=BATCH_SIZE,
num_workers=NUM_WORKERS,
pin_memory=PIN_MEMORY,
shuffle=SHUFFLE,
valid_split=VALID_SPLIT,
transform=TRANSFORM,
)


class ModelConfig:
    def __init__(
        self,
        model_type: str = "cnn",
        input_size: Tuple[int, int] = (64, 64),
        num_classes: int = 2,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
    ):
        self.model_type = model_type
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay


class TrainingConfig:
    def __init__(
        self,
        num_epochs: int = 50,
        checkpoint_interval: int = 5,
        early_stopping_patience: int = 10,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.num_epochs = num_epochs
        self.checkpoint_interval = checkpoint_interval
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(device)


def validate_config(config: Dict[str, Any]) -> None:
    # Perform comprehensive configuration validation
    # Raise InvalidConfigurationException with detailed error messages for any issues
    if not os.path.isdir(config["model_save_path"]):
        raise InvalidConfigurationException(
            "Model save path does not exist. Please create the directory."
        )
    if not os.path.isfile(config["train_data_file"]):
        raise InvalidConfigurationException("Training data file not found.")
    if not os.path.isfile(config["val_data_file"]):
        raise InvalidConfigurationException("Validation data file not found.")
    if config["batch_size"] < 1:
        raise InvalidConfigurationException("Batch size must be a positive integer.")
    if config["num_workers"] < 0:
        raise InvalidConfigurationException("Num workers must be a non-negative integer.")
    if not isinstance(config["pin_memory"], bool):
        raise InvalidConfigurationException("Pin memory must be a boolean value.")
    if not isinstance(config["shuffle"], bool):
        raise InvalidConfigurationException("Shuffle must be a boolean value.")
    if not 0 <= config["valid_split"] < 1:
        raise InvalidConfigurationException(
            "Validation split must be between 0 and 1 (exclusive)."
        )
    if not isinstance(config["transform"], transforms.Compose):
        raise InvalidConfigurationException(
            "Transform must be an instance of torchvision.transforms.Compose."
        )

    # Additional validation for model and training configs...


def load_config(config_file: str) -> Dict[str, Any]:
    # Load configuration from a file (e.g., JSON, YAML)
    # Return the loaded configuration as a dictionary
    pass


def save_config(config: Dict[str, Any], config_file: str) -> None:
    # Save the provided configuration to a file (e.g., JSON, YAML)
    # Use the specified config_file path for saving
    pass


def get_data_loaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    # Function to create and return DataLoaders for training, validation, and testing sets
    transform = config["transform"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    pin_memory = config["pin_memory"]
    shuffle = config["shuffle"]

    training_set = EyeTrackingDataset(
        data_file=config["train_data_file"], transform=transform
    )
    validation_set = EyeTrackingDataset(
        data_file=config["val_data_file"], transform=transform
    )
    test_set = EyeTrackingDataset(
        data_file=config["test_data_file"], transform=transform
    )

    train_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
    val_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def get_model(config: ModelConfig) -> nn.Module:
    # Function to create and return a model instance based on the provided configuration
    if config.model_type == "cnn":
        model = CNNModel(config.input_size, config.num_classes)
    else:
        raise InvalidConfigurationException(
            "Unsupported model type. Choose either 'cnn' or implement your own."
        )

    return model


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
) -> None:
    # Function to train the provided model using the given DataLoaders and training configuration
    device = config.device
    num_epochs = config.num_epochs
    checkpoint_interval = config.checkpoint_interval
    early_stopping_patience = config.early_stopping_patience

    # Move model to device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Implement early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, verbose=True
    )

    for epoch in range(num_epochs):
        # Training steps
        # ...

        # Validation steps
        # ...

        # Save checkpoint
        if epoch % checkpoint_interval == 0:
            # ...

    # Finalize training and save the best model
    # ...


if __name__ == "__main__":
    # Example usage
    base_config = load_config("config.yaml")
    validate_config(base_config)

    # Modify configuration for specific experiment
    experiment_config = {
        **base_config,
        "model_type": "cnn",
        "input_size": (64, 64),
        "num_classes": 2,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "num_epochs": 50,
        "checkpoint_interval": 5,
        "early_stopping_patience": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    validate_config(experiment_config)

    # Get data loaders
    data_loaders = get_data_loaders(experiment_config)

    # Get model instance
    model_config = ModelConfig(**experiment_config)
    model = get_model(model_config)

    # Train the model
    train_model(model, data_loaders["train"], data_loaders["val"], TrainingConfig(**experiment_config))