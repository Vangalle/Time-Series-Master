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

def metric_mape_dir(y_true, y_pred):
    """MAPE applied to direction data (MAPE_DIR)

    Lower is better.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        float: MAPE_DIR value    
    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true_mean = np.mean(np.abs(y_true))
    diff = y_pred - y_true
    locs = np.where(diff > 180)[0]
    y_true[locs] = 360 - y_true[locs]

    diff = np.where(diff > 180, diff - 360, diff)
    diff = np.where(diff < -180, diff + 360, diff)

    error = np.abs(diff)

    mape = np.mean(error / y_true_mean)

    return mape