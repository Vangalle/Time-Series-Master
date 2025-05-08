import os
import sys
import importlib.util
import inspect
import torch.nn as nn
import streamlit as st

def load_custom_models(models_dir="custom_models"):
    """
    Dynamically load custom model classes from Python files in the specified directory.
    
    Args:
        models_dir (str): Directory containing custom model Python files
        
    Returns:
        dict: Dictionary of model names to model classes
    """
    # Create the directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Dictionary to store discovered models
    custom_models = {}
    
    # Get list of Python files in the directory
    py_files = [f for f in os.listdir(models_dir) if f.endswith('.py')]
    
    if not py_files:
        return custom_models
    
    # Add the models directory to the Python path
    if models_dir not in sys.path:
        sys.path.append(os.path.abspath(models_dir))
    
    # Load each Python file and extract model classes
    for py_file in py_files:
        module_name = os.path.splitext(py_file)[0]
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                module_name, 
                os.path.join(models_dir, py_file)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find classes that inherit from nn.Module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, nn.Module) and 
                    obj != nn.Module):
                    
                    # Add the model to our dictionary
                    custom_models[f"{module_name}.{name}"] = obj
        
        except Exception as e:
            st.warning(f"Error loading custom model from {py_file}: {str(e)}")
    
    return custom_models

def get_model_info(model_class):
    """
    Extract model information from the class docstring or class definition.
    
    Args:
        model_class: The model class
        
    Returns:
        dict: Dictionary containing model metadata
    """
    doc = model_class.__doc__ or ""
    
    # Try to parse model type from docstring
    model_type = "deep_learning"  # Default type
    if "linear model" in doc.lower():
        model_type = "linear"
    
    # Try to determine if this is a specific architecture
    architecture = "custom"
    if "lstm" in doc.lower() or "lstm" in model_class.__name__.lower():
        architecture = "lstm"
    elif "gru" in doc.lower() or "gru" in model_class.__name__.lower():
        architecture = "gru"
    elif "rnn" in doc.lower() or "rnn" in model_class.__name__.lower():
        architecture = "rnn"
    elif "transformer" in doc.lower() or "transformer" in model_class.__name__.lower():
        architecture = "transformer"
    
    return {
        "type": model_type,
        "architecture": architecture,
        "description": doc.strip().split("\n")[0] if doc else "Custom model"
    }

def create_example_model_file():
    """
    Create an example custom model file in the custom_models directory.
    """
    os.makedirs("custom_models", exist_ok=True)
    
    example_code = '''import torch
import torch.nn as nn

class CustomLSTM(nn.Module):
    """A custom LSTM model for time series forecasting.
    
    This model uses a stacked LSTM architecture with dropout 
    and a final linear layer for prediction.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(CustomLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
'''

    example_code += '''        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Decode the hidden state
        out = self.fc(context_vector)
        return out


class WavenetModel(nn.Module):
    """A Wavenet-inspired model for time series forecasting.
    
    This model uses dilated causal convolutions to capture long-range dependencies.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5, kernel_size=2):
        super(WavenetModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initial causal convolution
        self.causal_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=1
        )
        
        # Dilated convolution stack
        self.dilated_convs = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i
            
            # Dilated convolution
            self.dilated_convs.append(
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=dilation,
                    dilation=dilation
                )
            )
            
            # Skip connection
            self.skip_connections.append(
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=1
                )
            )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        )
        
    def forward(self, x):
        # Input shape: [batch, seq_len, features]
        # Convert to [batch, features, seq_len] for Conv1d
        x = x.transpose(1, 2)
        
        # Initial causal convolution
        x = self.causal_conv(x)
        
        # Dilated convolution stack with skip connections
        skip_outputs = []
        for i, (dilated_conv, skip_conv) in enumerate(zip(self.dilated_convs, self.skip_connections)):
            # Apply dilated convolution
            residual = x
            x = torch.tanh(dilated_conv(x))
            
            # Skip connection
            skip = skip_conv(x)
            skip_outputs.append(skip)
            
            # Residual connection
            x = x + residual
        
        # Sum skip connections
        x = sum(skip_outputs)
        
        # Output layer
        x = self.output_layer(x)
        
        # Convert back to [batch, seq_len, features]
        # We only care about the last prediction
        x = x.transpose(1, 2)[:, -1, :]
        
        return x


class SimpleLinearEnsemble(nn.Module):
    """linear model
    
    A simple ensemble of linear models for time series forecasting.
    """
    
    def __init__(self, input_dim, output_dim, ensemble_size=3):
        super(SimpleLinearEnsemble, self).__init__()
        
        self.flatten = nn.Flatten()
        
        # Create multiple linear models
        self.linear_models = nn.ModuleList([
            nn.Linear(input_dim, output_dim) 
            for _ in range(ensemble_size)
        ])
        
    def forward(self, x):
        x = self.flatten(x)
        
        # Get predictions from each model
        predictions = [model(x) for model in self.linear_models]
        
        # Average the predictions
        ensemble_prediction = torch.mean(torch.stack(predictions), dim=0)
        
        return ensemble_prediction
'''
    
    with open("custom_models/example_models.py", "w") as f:
        f.write(example_code)
    
    return "custom_models/example_models.py"

# If run directly, create an example model file
if __name__ == "__main__":
    example_file = create_example_model_file()
    print(f"Created example model file: {example_file}")
