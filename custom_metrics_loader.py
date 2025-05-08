import os
import sys
import importlib.util
import inspect
import numpy as np
import streamlit as st
from typing import Callable, Dict, List, Any

def load_custom_metrics(metrics_dir="custom_metrics"):
    """
    Dynamically load custom metric functions from Python files in the specified directory.
    
    Args:
        metrics_dir (str): Directory containing custom metric Python files
        
    Returns:
        dict: Dictionary of metric names to metric functions
    """
    # Create the directory if it doesn't exist
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Dictionary to store discovered metrics
    custom_metrics = {}
    
    # Get list of Python files in the directory
    py_files = [f for f in os.listdir(metrics_dir) if f.endswith('.py')]
    
    if not py_files:
        return custom_metrics
    
    # Add the metrics directory to the Python path
    if metrics_dir not in sys.path:
        sys.path.append(os.path.abspath(metrics_dir))
    
    # Load each Python file and extract metric functions
    for py_file in py_files:
        module_name = os.path.splitext(py_file)[0]
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                module_name, 
                os.path.join(metrics_dir, py_file)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find functions that could be metrics
            for name, obj in inspect.getmembers(module):
                if (inspect.isfunction(obj) and 
                    name.startswith('metric_') or 
                    any(keyword in name.lower() for keyword in ['rmse', 'mae', 'mse', 'accuracy', 'error', 'score'])):
                    
                    # Extract parameter information
                    params = inspect.signature(obj).parameters
                    
                    # Check if function has right signature (y_true, y_pred)
                    if len(params) >= 2:
                        # Add the metric to our dictionary
                        metric_name = name.replace('metric_', '').replace('_', ' ').title()
                        custom_metrics[f"{module_name}.{metric_name}"] = obj
        
        except Exception as e:
            st.warning(f"Error loading custom metric from {py_file}: {str(e)}")
    
    return custom_metrics

def get_metric_info(metric_func):
    """
    Extract metric information from the function docstring.
    
    Args:
        metric_func: The metric function
        
    Returns:
        dict: Dictionary containing metric metadata
    """
    doc = metric_func.__doc__ or ""
    
    # Extract description and other metadata from docstring
    description = doc.strip().split("\n")[0] if doc else "Custom metric"
    
    # Determine metric type (higher is better or lower is better)
    metric_type = "lower_is_better"  # Default for most error metrics
    
    lower_keywords = ['error', 'loss', 'rmse', 'mse', 'mae']
    higher_keywords = ['accuracy', 'r2', 'score', 'auc', 'precision', 'recall', 'f1']
    
    func_name = metric_func.__name__.lower()
    
    if any(keyword in func_name for keyword in higher_keywords) or "higher is better" in doc.lower():
        metric_type = "higher_is_better"
    elif any(keyword in func_name for keyword in lower_keywords) or "lower is better" in doc.lower():
        metric_type = "lower_is_better"
    
    return {
        "description": description,
        "type": metric_type
    }

def create_example_metrics_file():
    """
    Create an example custom metrics file in the custom_metrics directory.
    """
    os.makedirs("custom_metrics", exist_ok=True)
    
    example_code = '''import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

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
'''
    
    with open("custom_metrics/example_metrics.py", "w") as f:
        f.write(example_code)
    
    return "custom_metrics/example_metrics.py"


# If run directly, create an example metrics file
if __name__ == "__main__":
    example_file = create_example_metrics_file()
    print(f"Created example metrics file: {example_file}")
