import numpy as np

def metric_directional_accuracy(y_true, y_pred):
    """Directional Accuracy - measures ability to predict the direction of change.
    
    Higher is better.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        float: Directional accuracy as a percentage
    """
    # Calculate the direction of change
    true_diff = np.diff(y_true, axis=0)
    pred_diff = np.diff(y_pred, axis=0)
    
    # Count correct direction predictions
    correct_direction = (true_diff * pred_diff) > 0
    
    # Return the percentage of correct direction predictions
    return np.mean(correct_direction) * 100

def metric_wape(y_true, y_pred):
    """Weighted Absolute Percentage Error (WAPE).
    Also known as Mean Absolute Scaled Error.
    
    Lower is better.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        float: WAPE value
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
