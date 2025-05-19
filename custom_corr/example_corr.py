import numpy as np
from sklearn.feature_selection import mutual_info_regression

def robust_correlation(x, y):
    # Simple robust correlation using median
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return np.nan
    
    x_clean, y_clean = x[mask], y[mask]
    x_median = np.median(x_clean)
    y_median = np.median(y_clean)
    
    # Median-based covariance
    numerator = np.median((x_clean - x_median) * (y_clean - y_median))
    denominator = np.sqrt(np.median((x_clean - x_median)**2) * 
                         np.median((y_clean - y_median)**2))
    
    return numerator / denominator if denominator != 0 else np.nan

def distance_correlation(x, y):
    """Simple implementation of distance correlation"""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return np.nan
    
    x_clean, y_clean = x[mask], y[mask]
    n = len(x_clean)
    
    # Distance matrices
    a = np.abs(x_clean[:, np.newaxis] - x_clean)
    b = np.abs(y_clean[:, np.newaxis] - y_clean)
    
    # Centered distance matrices
    a_centered = a - a.mean(axis=0) - a.mean(axis=1)[:, np.newaxis] + a.mean()
    b_centered = b - b.mean(axis=0) - b.mean(axis=1)[:, np.newaxis] + b.mean()
    
    # Distance covariance and variances
    dcov = np.sqrt(np.mean(a_centered * b_centered))
    dvar_x = np.sqrt(np.mean(a_centered * a_centered))
    dvar_y = np.sqrt(np.mean(b_centered * b_centered))
    
    return dcov / np.sqrt(dvar_x * dvar_y) if (dvar_x * dvar_y) > 0 else 0

def mutual_info_sklearn(x, y):
    """
    Wrapper for sklearn's mutual_info_regression to work with pd.DataFrame.corr()
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:  # Need minimum samples
        return np.nan
    
    x_clean = x[mask].reshape(-1, 1)  # sklearn expects 2D array for X
    y_clean = y[mask]                 # sklearn expects 1D array for y
    
    # Calculate mutual information
    mi = mutual_info_regression(x_clean, y_clean, random_state=42)
    return mi[0]  # mutual_info_regression returns an array, take first element


