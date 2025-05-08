import os
import sys
import importlib.util
import inspect
import torch.nn as nn
import streamlit as st
from typing import Callable, Dict

def load_custom_losses(losses_dir="custom_losses"):
    """
    Dynamically load custom loss functions from Python files in the specified directory.
    
    Args:
        losses_dir (str): Directory containing custom loss Python files
        
    Returns:
        dict: Dictionary of loss names to loss functions
    """
    # Create the directory if it doesn't exist
    os.makedirs(losses_dir, exist_ok=True)
    
    # Dictionary to store discovered losses
    custom_losses = {}
    
    # Get list of Python files in the directory
    py_files = [f for f in os.listdir(losses_dir) if f.endswith('.py')]
    
    if not py_files:
        return custom_losses
    
    # Add the losses directory to the Python path
    if losses_dir not in sys.path:
        sys.path.append(os.path.abspath(losses_dir))
    
    # Load each Python file and extract loss functions
    for py_file in py_files:
        module_name = os.path.splitext(py_file)[0]
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                module_name, 
                os.path.join(losses_dir, py_file)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find classes that inherit from nn.Module and have 'Loss' in the name
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, nn.Module) and 
                    'loss' in name.lower()):
                    
                    # Add the loss to our dictionary
                    loss_name = f"{module_name}.{name}"
                    custom_losses[loss_name] = obj
        
        except Exception as e:
            st.warning(f"Error loading custom loss from {py_file}: {str(e)}")
    
    return custom_losses

def get_loss_info(loss_class):
    """
    Extract loss information from the class docstring.
    
    Args:
        loss_class: The loss class
        
    Returns:
        dict: Dictionary containing loss metadata
    """
    doc = loss_class.__doc__ or ""
    
    # Extract description from docstring
    description = doc.strip().split("\n")[0] if doc else "Custom loss function"
    
    return {
        "description": description
    }

def create_example_loss_file():
    """
    Create an example custom loss file in the custom_losses directory.
    """
    os.makedirs("custom_losses", exist_ok=True)
    
    example_code = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss function.
    
    A smoothed version of Huber loss that is differentiable everywhere.
    Works well for regression tasks as it dampens the effect of outliers.
    """
    
    def __init__(self):
        super(LogCoshLoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        """
        Calculate the log-cosh loss.
        
        Args:
            y_pred: Predicted values
            y_true: Target values
            
        Returns:
            torch.Tensor: The log-cosh loss value
        """
        diff = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(diff)))

class QuantileLoss(nn.Module):
    """
    Quantile Loss function.
    
    Useful for predicting intervals or when different penalties for
    over-prediction and under-prediction are needed.
    """
    
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
        
    def forward(self, y_pred, y_true):
        """
        Calculate the quantile loss.
        
        Args:
            y_pred: Predicted values
            y_true: Target values
            
        Returns:
            torch.Tensor: The quantile loss value
        """
        diff = y_true - y_pred
        return torch.mean(
            torch.max(self.quantile * diff, (self.quantile - 1) * diff)
        )
'''
    
    with open("custom_losses/example_losses.py", "w") as f:
        f.write(example_code)
    
    return "custom_losses/example_losses.py"

# If run directly, create an example loss file
if __name__ == "__main__":
    example_file = create_example_loss_file()
    print(f"Created example loss file: {example_file}")