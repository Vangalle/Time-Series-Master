import os
import sys
import importlib.util
import inspect
import torch
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

def load_model_for_prediction(model_path):
    # Load using PyTorch's deserializer
    model_data = torch.load(model_path, weights_only=False)
    
    # Get model configuration
    model_config = model_data['model_config']
    model_class_name = model_data['model_class_name']
    model_state = model_data['model_state']
    
    # Import the module and get class
    custom_models = load_custom_models()
    
    # If the model is in custom models
    if model_class_name in custom_models:
        model_class = custom_models[model_class_name]
        
        # Create a new instance of the model with saved configuration
        if model_config["type"] in ["deep_learning", "custom_deep_learning"]:
            # Extract parameters from config
            input_dim = len(model_data['input_vars'])
            output_dim = len(model_data['target_vars'])
            hidden_dim = model_config.get("hidden_dim", 64)
            num_layers = model_config.get("num_layers", 2)
            input_length = model_config.get("input_length", 10)
            output_length = model_config.get("output_length", 10)
            
            # Check if it's a transformer model
            if "d_model" in model_config and "num_heads" in model_config:
                model = model_class(
                    input_dim=input_dim,
                    d_model=model_config["d_model"],
                    nhead=model_config["num_heads"],
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    input_length=input_length,
                    output_length=output_length
                )
            else:
                # Regular RNN/LSTM/GRU model
                model = model_class(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=num_layers,
                    input_length=input_length,
                    output_length=output_length
                )
        else:
            # Linear model reconstruction
            input_dim = len(model_data['input_vars'])
            output_dim = len(model_data['target_vars'])
            model = model_class(input_dim, output_dim)
            
        # Load the saved weights
        model.load_state_dict(model_state)
        return model, model_data
    else:
        # Handle the case where the model isn't found in custom_models
        raise ValueError(f"Model class {model_class_name} not found in custom models")

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
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, input_length, output_length, dropout=0.5):
        super(CustomLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_length = input_length
        self.output_length = output_length
        self.output_dim = output_dim
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
        self.fc = nn.Linear(hidden_dim * input_length, output_dim * output_length)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1) # [Batch_size, Input_length, 1]
        context_vector = attention_weights * lstm_out # [Batch_size, Input_length, Hidden_dim]

        batch_size, _, _ = context_vector.size()
        x = context_vector.reshape(batch_size, -1)
        x = self.fc(x)
        x = x.reshape(batch_size, self.output_length, self.output_dim)

        # Decode the hidden state
        return x

'''
    
    with open("custom_models/example_models.py", "w") as f:
        f.write(example_code)
    
    return "custom_models/example_models.py"

# If run directly, create an example model file
if __name__ == "__main__":
    example_file = create_example_model_file()
    print(f"Created example model file: {example_file}")
