import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LossFunctionException(Exception):
    """Base exception class for loss function errors."""
    pass

class InvalidLossFunctionError(LossFunctionException):
    """Raised when an invalid loss function is specified."""
    pass

class LossFunction:
    """Base class for custom loss functions."""
    def __init__(self, name: str, **kwargs):
        """
        Initialize the loss function.

        Args:
        - name (str): The name of the loss function.
        - **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.config = kwargs

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.

        Args:
        - input (torch.Tensor): The input tensor.
        - target (torch.Tensor): The target tensor.

        Returns:
        - torch.Tensor: The computed loss.
        """
        raise NotImplementedError

class VelocityThresholdLoss(LossFunction):
    """Velocity threshold loss function."""
    def __init__(self, threshold: float = 0.5, **kwargs):
        """
        Initialize the velocity threshold loss function.

        Args:
        - threshold (float): The velocity threshold. Defaults to 0.5.
        - **kwargs: Additional keyword arguments.
        """
        super().__init__('velocity_threshold', threshold=threshold, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity threshold loss.

        Args:
        - input (torch.Tensor): The input tensor.
        - target (torch.Tensor): The target tensor.

        Returns:
        - torch.Tensor: The computed loss.
        """
        velocity = torch.abs(input - target)
        loss = torch.mean(torch.where(velocity > self.config['threshold'], velocity, torch.zeros_like(velocity)))
        return loss

class FlowTheoryLoss(LossFunction):
    """Flow theory loss function."""
    def __init__(self, alpha: float = 0.1, beta: float = 0.1, **kwargs):
        """
        Initialize the flow theory loss function.

        Args:
        - alpha (float): The alpha parameter. Defaults to 0.1.
        - beta (float): The beta parameter. Defaults to 0.1.
        - **kwargs: Additional keyword arguments.
        """
        super().__init__('flow_theory', alpha=alpha, beta=beta, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the flow theory loss.

        Args:
        - input (torch.Tensor): The input tensor.
        - target (torch.Tensor): The target tensor.

        Returns:
        - torch.Tensor: The computed loss.
        """
        loss = torch.mean(torch.where(input > target, self.config['alpha'] * (input - target), self.config['beta'] * (target - input)))
        return loss

class CompositeLoss(LossFunction):
    """Composite loss function."""
    def __init__(self, loss_functions: List[LossFunction], weights: List[float] = None, **kwargs):
        """
        Initialize the composite loss function.

        Args:
        - loss_functions (List[LossFunction]): The list of loss functions.
        - weights (List[float]): The weights for each loss function. Defaults to None.
        - **kwargs: Additional keyword arguments.
        """
        super().__init__('composite', loss_functions=loss_functions, weights=weights, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the composite loss.

        Args:
        - input (torch.Tensor): The input tensor.
        - target (torch.Tensor): The target tensor.

        Returns:
        - torch.Tensor: The computed loss.
        """
        if self.config['weights'] is None:
            weights = [1.0 / len(self.config['loss_functions'])] * len(self.config['loss_functions'])
        else:
            weights = self.config['weights']

        loss = torch.zeros_like(input)
        for i, loss_function in enumerate(self.config['loss_functions']):
            loss += weights[i] * loss_function.forward(input, target)
        return loss

def get_loss_function(name: str, **kwargs) -> LossFunction:
    """
    Get a loss function by name.

    Args:
    - name (str): The name of the loss function.
    - **kwargs: Additional keyword arguments.

    Returns:
    - LossFunction: The loss function instance.
    """
    loss_functions = {
        'velocity_threshold': VelocityThresholdLoss,
        'flow_theory': FlowTheoryLoss,
        'composite': CompositeLoss
    }

    if name not in loss_functions:
        raise InvalidLossFunctionError(f"Invalid loss function: {name}")

    return loss_functions[name](**kwargs)

def main():
    # Example usage
    input_tensor = torch.randn(10)
    target_tensor = torch.randn(10)

    velocity_threshold_loss = VelocityThresholdLoss()
    flow_theory_loss = FlowTheoryLoss()
    composite_loss = CompositeLoss([velocity_threshold_loss, flow_theory_loss])

    print("Velocity Threshold Loss:", velocity_threshold_loss.forward(input_tensor, target_tensor))
    print("Flow Theory Loss:", flow_theory_loss.forward(input_tensor, target_tensor))
    print("Composite Loss:", composite_loss.forward(input_tensor, target_tensor))

if __name__ == "__main__":
    main()