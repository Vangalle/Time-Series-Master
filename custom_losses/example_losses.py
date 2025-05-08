import torch
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
