import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from datetime import datetime
import pickle
import math
import copy
from pathlib import Path
import platform
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Import custom model and metrics loaders
from custom_model_loader import load_custom_models, get_model_info
from custom_metrics_loader import load_custom_metrics, get_metric_info
from custom_loss_loader import load_custom_losses, get_loss_info

# Load custom losses
custom_losses = load_custom_losses()
loss_options = ["L2Loss", "L1Loss", "Huber"]
if custom_losses:
    loss_options.append("Custom Loss")

# Default model definitions
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, input_dim, d_model=64, hidden_dim=256, output_dim=1, 
                 num_layers=2, nhead=8, dropout=0.2, forecast_horizon=1,
                 use_projection=True, use_causal_mask=True):
        super(Transformer, self).__init__()
        
        # Input projection layer (if original dimension is small)
        self.use_projection = use_projection
        if use_projection:
            self.input_projection = nn.Linear(input_dim, d_model)
            model_dim = d_model
        else:
            model_dim = input_dim
            assert model_dim % nhead == 0, f"Model dimension ({model_dim}) must be divisible by num_heads ({nhead})"
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim, dropout=dropout)
        
        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers with support for multi-horizon forecasting
        self.forecast_horizon = forecast_horizon
        
        if forecast_horizon > 1:
            # For multi-step forecasting, we'll use separate prediction head
            self.output_layer = nn.Linear(model_dim, output_dim * forecast_horizon)
        else:
            # For single-step forecasting, simple linear layer
            self.output_layer = nn.Linear(model_dim, output_dim)
            
        self.use_causal_mask = use_causal_mask
        self.output_dim = output_dim
        
    def _generate_causal_mask(self, seq_len):
        # Lower triangular mask to prevent attending to future time steps
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        seq_len = x.size(1)
        batch_size = x.size(0)
        
        # Apply input projection if specified
        if self.use_projection:
            x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate causal mask if needed
        mask = self._generate_causal_mask(seq_len).to(x.device) if self.use_causal_mask else None
        
        # Apply transformer encoder
        out = self.transformer_encoder(x, mask=mask)
        
        # Extract features from the last time step
        last_hidden = out[:, -1, :]
        
        # Apply output layer
        if self.forecast_horizon > 1:
            # For multi-horizon forecasting
            predictions = self.output_layer(last_hidden)
            # Return flattened output to match other models
            return predictions
        else:
            # For single-step forecasting
            return self.output_layer(last_hidden)
        
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.flatten(x)
        out = self.linear(x)
        return out

# Function to prepare time series data
def prepare_time_series_data(data, input_vars, target_vars, input_length, output_length):
    """
    Prepare time series data for model training, handling datetime features appropriately.
    
    Args:
        data: DataFrame containing the data
        input_vars: List of input variable names
        target_vars: List of target variable names
        input_length: Length of input sequence
        output_length: Length of output sequence
        
    Returns:
        X, y, norm_params
    """
    # Create a copy of input data
    processed_data = data.copy()
    
    # Identify datetime columns
    datetime_cols = []
    numeric_cols = []
    
    for col in input_vars:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            datetime_cols.append(col)
        else:
            numeric_cols.append(col)
    
    # Process datetime columns
    datetime_features = {}
    
    for col in datetime_cols:
        # Extract datetime components as separate features
        dt_series = data[col]
        col_prefix = f"{col}_"
        
        # Add cyclical encoding for day of week (sin and cos transformation)
        day_of_week = dt_series.dt.dayofweek
        processed_data[col_prefix + 'day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        processed_data[col_prefix + 'day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Add cyclical encoding for month
        month = dt_series.dt.month
        processed_data[col_prefix + 'month_sin'] = np.sin(2 * np.pi * month / 12)
        processed_data[col_prefix + 'month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # Add cyclical encoding for hour
        if dt_series.dt.hour.nunique() > 1:
            hour = dt_series.dt.hour
            processed_data[col_prefix + 'hour_sin'] = np.sin(2 * np.pi * hour / 24)
            processed_data[col_prefix + 'hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Add year as a linear feature (normalized)
        year = dt_series.dt.year
        year_min = year.min()
        year_max = year.max()
        if year_max > year_min:
            processed_data[col_prefix + 'year'] = (year - year_min) / (year_max - year_min)
        
        # Add day of year as cyclical feature
        day_of_year = dt_series.dt.dayofyear
        processed_data[col_prefix + 'day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        processed_data[col_prefix + 'day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Track the new feature names
        new_features = [
            col_prefix + 'day_of_week_sin', col_prefix + 'day_of_week_cos',
            col_prefix + 'month_sin', col_prefix + 'month_cos',
            col_prefix + 'day_of_year_sin', col_prefix + 'day_of_year_cos'
        ]
        
        if dt_series.dt.hour.nunique() > 1:
            new_features.extend([col_prefix + 'hour_sin', col_prefix + 'hour_cos'])
        
        if year_max > year_min:
            new_features.append(col_prefix + 'year')
        
        datetime_features[col] = new_features
    
    # Replace datetime columns with their numeric representations
    transformed_input_vars = []
    for var in input_vars:
        if var in datetime_cols:
            transformed_input_vars.extend(datetime_features[var])
        else:
            transformed_input_vars.append(var)
    
    # Extract data for transformed input variables and targets
    X_data = processed_data[transformed_input_vars].values
    y_data = data[target_vars].values
    
    # Standardize the data (only numeric data)
    X_mean = X_data.mean(axis=0)
    X_std = X_data.std(axis=0)
    X_data = (X_data - X_mean) / (X_std + 1e-8)
    
    y_mean = y_data.mean(axis=0)
    y_std = y_data.std(axis=0)
    y_data = (y_data - y_mean) / (y_std + 1e-8)
    
    # Store normalization parameters and feature mappings for later use
    norm_params = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'datetime_features': datetime_features,
        'transformed_input_vars': transformed_input_vars
    }
    
    # Create sequences
    X, y = [], []
    for i in range(len(X_data) - input_length - output_length + 1):
        X.append(X_data[i:i+input_length])
        y.append(y_data[i+input_length:i+input_length+output_length])
    
    return np.array(X), np.array(y), norm_params

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
               device, epochs, patience, loss_chart, progress_bar, status_text, verbose=True):
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        training_history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        training_history['val_loss'].append(val_loss)
        
        # Update progress visualization
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        # Add new data point to the chart
        loss_chart.add_rows(pd.DataFrame({
            'Training Loss': [train_loss],
            'Validation Loss': [val_loss]
        }))
        
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    status_text.text(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return best_model, training_history

# Function to evaluate the model
def evaluate_model(model, test_loader, criterion, device, custom_metrics=None):
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())
    
    test_loss /= len(test_loader)
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    # Calculate additional metrics if provided
    metrics_results = {}
    if custom_metrics:
        for metric_name, metric_func in custom_metrics.items():
            try:
                metrics_results[metric_name] = metric_func(actuals, predictions)
            except Exception as e:
                st.warning(f"Error calculating metric {metric_name}: {str(e)}")
                metrics_results[metric_name] = float('nan')
    
    return test_loss, predictions, actuals, metrics_results

# Function to denormalize the predictions and actual values
def denormalize_data(data, mean, std):
    return data * std + mean

# Function to save model and results
def save_model_and_results(model_info, model_config, norm_params, 
                         predictions, actuals, input_vars, target_vars):
    # Create directory for models
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_data = {
        'model_name': model_config.get("model_name", "Unknown"),
        'model_config': model_config,
        'model_state': model_info.get("model_state"),
        'model_class': model_info.get("model_class_name"),
        'input_vars': input_vars,
        'target_vars': target_vars,
        'norm_params': norm_params,
        'timestamp': timestamp
    }
    
    model_path = f"models/{model_data['model_name']}_{timestamp}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Return data for saving results separately
    return model_path, timestamp

def add_scroll_to_top():
    """Add enhanced JavaScript to ensure scrolling to top of page."""
    js = '''
    <script>
        // Function to scroll to top
        function scrollToTop() {
            window.scrollTo({top: 0, behavior: 'instant'});
        }
        
        // Multiple event listeners for different scenarios
        window.addEventListener('load', scrollToTop);
        document.addEventListener('DOMContentLoaded', scrollToTop);
        
        // Timeout to ensure it runs after Streamlit's scripts
        setTimeout(scrollToTop, 100);
        setTimeout(scrollToTop, 500);
    </script>
    '''
    st.markdown(js, unsafe_allow_html=True)

def run():
    """
    Main function to run the model training component.
    This function will be called from the main application.
    """
    add_scroll_to_top()
    # Title and introduction
    st.title("Time Series Model Selection and Training")
    st.write("Select a model, configure parameters, and train your time series model.")
    
    # Check if data and variables are available
    if (st.session_state.data is None or 
        not st.session_state.input_vars or 
        not st.session_state.target_vars):
        
        st.warning("Please select data, input variables, and target variables first.")
        
        if st.button("Go to Data Selection"):
            st.session_state.page = "data_selection"
            st.rerun()
            
        return
    
    # Get data from session state
    data = st.session_state.data
    st.session_state.data_Rows = data.shape[0]
    st.session_state.data_Columns = data.shape[1]
    input_vars = st.session_state.input_vars
    target_vars = st.session_state.target_vars
    
    # Load custom models and metrics
    custom_models = load_custom_models()
    custom_metrics = load_custom_metrics()
    
    # Display current data and variables
    st.subheader("Current Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Data Source:** {st.session_state.file_name}")
        st.write(f"**Number of Rows:** {st.session_state.data_Rows}")
        st.write(f"**Number of Columns:** {st.session_state.data_Columns}")

    with col2:
        st.write(f"**Input Variables:** {', '.join(input_vars)}")
        st.write(f"**Target Variables:** {', '.join(target_vars)}")
    
    # Model selection and parameters section
    st.header("Model Selection")
    
    # Create tabs for different model categories
    # model_tabs = st.tabs(["Built-in Models", "Custom Models"])

    model_type = st.selectbox("Select Model Type", ["Built-in", "Custom"])
    
    if model_type == "Built-in":  # Built-in Models
        st.subheader("Built-in Model Selection")
        
        # Use nested tabs for deep learning vs linear
        # builtin_tabs = st.tabs(["Deep Learning Models", "Linear Models"])
        builtin_model_type = st.selectbox("Select Built-in Model Type", ["Deep Learning", "Linear"], key="builtin_model_type")
        
        if builtin_model_type == "Deep Learning":
            col1, col2 = st.columns(2)
            
            with col1:
                # Model type selection
                dl_model_type = st.selectbox(
                    "Select Model Type",
                    ["LSTM", "GRU", "RNN", "Transformer"]
                )
                
                # Model depth selection
                num_layers = st.number_input("Number of Layers", min_value=1, max_value=20, value=2, step=1)
                
                # Hidden dimension selection
                hidden_dim = st.number_input("Hidden Dimension", min_value=32, max_value=2048, value=64, step=32)

                # If Transformer model is selected, add num_heads parameter
                if dl_model_type == "Transformer":

                    d_model = st.number_input("Model Dimension", min_value=32, max_value=2048, value=64, step=32)

                    # Calculate valid divisors of d_model
                    divisors = [i for i in range(1, d_model + 1) if d_model % i == 0]
                    
                    # Default to a value close to 8 (common default) if possible
                    default_index = min(range(len(divisors)), key=lambda i: abs(divisors[i] - 8))
                    
                    num_heads = st.selectbox(
                        "Number of Attention Heads",
                        options=divisors,
                        index=default_index,
                        help="For Transformer models, number of heads must divide the hidden dimension evenly."
                    )
                else:
                    num_heads = 8  # Default value for other models (won't be used)
                
                # Input and output sequence length
                input_length = st.number_input("Input Sequence Length", min_value=1, max_value=500, value=12, step=3)

                output_length = st.number_input("Output Sequence Length", min_value=1, max_value=500, value=12, step=3)

                
            with col2:
                # Loss function selection
                loss_function = st.selectbox(
                    "Loss Function",
                    loss_options,
                    index=0
                )

                selected_custom_loss = None
                if loss_function == "Custom Loss":
                    selected_custom_loss = st.selectbox(
                        "Select Custom Loss",
                        list(custom_losses.keys()),
                        format_func=lambda x: x.split(".")[-1]  # Show class name only
                    )
                    
                    # Display loss information
                    if selected_custom_loss:
                        loss_class = custom_losses[selected_custom_loss]
                        loss_info = get_loss_info(loss_class)
                        st.info(f"**Description:** {loss_info['description']}")
                
                # Dataset splits
                train_ratio = st.number_input("Training Set Ratio", min_value=0.1, max_value=0.9, value=0.8, step=0.05,
                                              format="%.2f")

                val_ratio = st.number_input("Validation Set Ratio", min_value=0.0, max_value=0.5, value=0.10, step=0.05, 
                                            format="%.2f")
                
                test_ratio = 1.0 - train_ratio - val_ratio
                st.write(f"Test Set Ratio: {test_ratio:.2f}")
                
                # Early stopping parameters
                use_early_stopping = st.checkbox("Use Early Stopping", value=True)
                if use_early_stopping:
                    patience = st.number_input("Early Stopping Patience", min_value=1, max_value=100,
                                               value=15, step=1)
                else:
                    patience = 0
                
                # Learning rate scheduler
                use_lr_scheduler = st.checkbox("Use Learning Rate Scheduler", value=True)
                lr_scheduler_type = None
                if use_lr_scheduler:
                    lr_scheduler_type = st.selectbox(
                        "Learning Rate Scheduler",
                        ["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR"],
                        index=0
                    )
            
            # Training parameters
            st.subheader("Training Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, 
                                                format="%.4f", step=0.0001)
                
            with col2:
                batch_size = st.number_input("Batch Size", min_value=1, max_value=2048, value=64, step=64)
                
            with col3:
                epochs = st.number_input("Maximum Epochs", min_value=1, max_value=1000, value=100, step=20)
            
            selected_model = {
                "type": "deep_learning",
                "model_name": dl_model_type,
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
                "input_length": input_length,
                "output_length": output_length,
                "loss_function": loss_function,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "use_early_stopping": use_early_stopping,
                "patience": patience,
                "use_lr_scheduler": use_lr_scheduler,
                "lr_scheduler_type": lr_scheduler_type,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs
            }
            
            if dl_model_type == "Transformer":
                selected_model["d_model"] = d_model
                selected_model["num_heads"] = num_heads
            
        if builtin_model_type == "Linear":
            st.subheader("Linear Model Configuration")
            
            linear_model_type = st.selectbox(
                "Select Linear Model Type",
                ["Simple Linear Regression"]
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Input and output sequence length
                input_length = st.number_input("Input Sequence Length", min_value=1, max_value=500,
                                                value=12, step=3, key="linear_input_length")
                output_length = st.number_input("Output Sequence Length", min_value=1, max_value=500,
                                                value=12, step=3, key="linear_output_length")
                
                # Loss function selection
                loss_function = st.selectbox(
                    "Loss Function",
                    loss_options,
                    index=0,
                    key="linear_loss"
                )
                
                selected_custom_loss = None
                if loss_function == "Custom Loss":
                    selected_custom_loss = st.selectbox(
                        "Select Custom Loss",
                        list(custom_losses.keys()),
                        format_func=lambda x: x.split(".")[-1]  # Show class name only
                    )
                    
                    # Display loss information
                    if selected_custom_loss:
                        loss_class = custom_losses[selected_custom_loss]
                        loss_info = get_loss_info(loss_class)
                        st.info(f"**Description:** {loss_info['description']}")

            with col2:
                # Dataset splits
                train_ratio = st.number_input("Training Set Ratio", min_value=0.5, max_value=0.9, value=0.8, step=0.05, key="linear_train_ratio")
                val_ratio = st.number_input("Validation Set Ratio", min_value=0.0, max_value=0.3, value=0.1, step=0.05, key="linear_val_ratio")
                test_ratio = 1.0 - train_ratio - val_ratio
                st.write(f"Test Set Ratio: {test_ratio:.2f}")
            
            col1, col2, col3 = st.columns(3)

            with col1:
                # Learning rate and batch size
                learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.01, step=0.0001, 
                                                format="%.4f", key="linear_lr_input")
            
            with col2:
                batch_size = st.number_input("Batch Size", min_value=8, max_value=2048, value=64, step=64, key="linear_batch")
            
            with col3:
                # Training parameters
                epochs = st.number_input("Maximum Epochs", min_value=1, max_value=500, value=100, step=20, key="linear_epochs")
            
            selected_model = {
                "type": "linear",
                "model_name": linear_model_type,
                "input_length": input_length,
                "output_length": output_length,
                "loss_function": loss_function,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs
            }
    
    if model_type == "Custom":
        st.subheader("Custom Model Selection")
        
        if not custom_models:
            st.info("No custom models found. Add custom model files to the 'custom_models' directory.")
            st.write("You can create an example model file using the sidebar option.")
        else:
            # Select custom model
            custom_model_key = st.selectbox(
                "Select Custom Model",
                list(custom_models.keys()),
                format_func=lambda x: x.split(".")[-1]  # Show class name only
            )
            
            custom_model_class = custom_models[custom_model_key]
            model_info = get_model_info(custom_model_class)

            # Display model information
            st.write(f"**Model Type:** {model_info['type'].title()}")
            st.write(f"**Architecture:** {model_info['architecture'].title()}")
            st.write(f"**Description:** {model_info['description']}")
            
            # Configuration based on model type
            if model_info['type'] == "deep_learning":
                col1, col2 = st.columns(2)
                
                with col1:
                    # Model depth selection
                    num_layers = st.number_input("Number of Layers", min_value=1, max_value=20, value=2, step=1, 
                                                 key="custom_num_layers")
                    
                    # Hidden dimension selection
                    hidden_dim = st.number_input("Hidden Dimension", min_value=32, max_value=2048, value=64, step=32, 
                                                 key="custom_hidden_dim_input")

                    # If Transformer model is selected, add num_heads parameter
                    if model_info['architecture'].title() == "Transformer":

                        d_model = st.number_input("Model Dimension", min_value=32, max_value=2048, value=64, step=32)

                        # Calculate valid divisors of d_model
                        divisors = [i for i in range(1, d_model + 1) if d_model % i == 0]
                        
                        # Default to a value close to 8 (common default) if possible
                        default_index = min(range(len(divisors)), key=lambda i: abs(divisors[i] - 8))
                        
                        num_heads = st.selectbox(
                            "Number of Attention Heads",
                            options=divisors,
                            index=default_index,
                            help="For Transformer models, number of heads must divide the hidden dimension evenly."
                        )
                    else:
                        num_heads = 8  # Default value for other models (won't be used)

                    # Input and output sequence length
                    input_length = st.number_input("Input Sequence Length", min_value=1, max_value=500, value=12, 
                                                   step=3, key="custom_input_length")

                    output_length = st.number_input("Output Sequence Length", min_value=1, max_value=500, value=12, 
                                                    step=3, key="custom_output_length")
                    
                with col2:
                    # Loss function selection
                    loss_function = st.selectbox(
                        "Loss Function",
                        loss_options,
                        index=0,
                        key="custom_loss"
                    )

                    selected_custom_loss = None
                    if loss_function == "Custom Loss":
                        selected_custom_loss = st.selectbox(
                            "Select Custom Loss",
                            list(custom_losses.keys()),
                            format_func=lambda x: x.split(".")[-1]  # Show class name only
                        )
                        
                        # Display loss information
                        if selected_custom_loss:
                            loss_class = custom_losses[selected_custom_loss]
                            loss_info = get_loss_info(loss_class)
                            st.info(f"**Description:** {loss_info['description']}")
                    
                    # Dataset splits
                    train_ratio = st.number_input("Training Set Ratio", min_value=0.1, max_value=0.9, value=0.8, 
                                                  step=0.05, key="custom_train_ratio")
                    val_ratio = st.number_input("Validation Set Ratio", min_value=0.0, max_value=0.3, value=0.1,
                                                 step=0.05, key="custom_val_ratio")
                    test_ratio = 1.0 - train_ratio - val_ratio
                    st.write(f"Test Set Ratio: {test_ratio:.2f}")
                
                # Early stopping parameters
                use_early_stopping = st.checkbox("Use Early Stopping", value=True, key="custom_early_stopping")
                if use_early_stopping:
                    patience = st.number_input("Early Stopping Patience", min_value=1, max_value=100,
                                               value=15, step=1, key="custom_patience")
                else:
                    patience = 0
                
                # Learning rate scheduler
                use_lr_scheduler = st.checkbox("Use Learning Rate Scheduler", value=True, key="custom_lr_scheduler")
                lr_scheduler_type = None
                if use_lr_scheduler:
                    lr_scheduler_type = st.selectbox(
                        "Learning Rate Scheduler",
                        ["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR"],
                        index=0,
                        key="custom_scheduler_type"
                    )
                
                # Training parameters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, 
                                                    value=0.001, step=0.0001, format="%.4f", key="custom_learning_rate")
                    
                with col2:
                    batch_size = st.number_input("Batch Size", min_value=8, max_value=2048, value=64, 
                                                 step=64, key="custom_batch_size")
                    
                with col3:
                    epochs = st.number_input("Maximum Epochs", min_value=1, max_value=500, value=100, 
                                             step=20, key="custom_epochs")
                
                selected_model = {
                    "type": "custom_deep_learning",
                    "model_name": custom_model_key.split(".")[-1],
                    "model_class": custom_model_class,
                    "model_class_name": custom_model_key,
                    "num_layers": num_layers,
                    "hidden_dim": hidden_dim,
                    "input_length": input_length,
                    "output_length": output_length,
                    "loss_function": loss_function,
                    "train_ratio": train_ratio,
                    "val_ratio": val_ratio,
                    "test_ratio": test_ratio,
                    "use_early_stopping": use_early_stopping,
                    "patience": patience,
                    "use_lr_scheduler": use_lr_scheduler,
                    "lr_scheduler_type": lr_scheduler_type,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs
                }
            else:
                # Linear model configuration
                col1, col2 = st.columns(2)
                
                with col1:
                    # Input and output sequence length
                    input_length = st.number_input("Input Sequence Length", min_value=1, max_value=500, 
                                                   value=12, step=3, key="custom_linear_input_length")
                    output_length = st.number_input("Output Sequence Length", min_value=1, max_value=500, 
                                                    value=12, step=3, key="custom_linear_output_length")
                    
                    # Loss function selection
                    loss_function = st.selectbox(
                        "Loss Function",
                        loss_options,
                        index=0,
                        key="custom_linear_loss"
                    )

                    selected_custom_loss = None
                    if loss_function == "Custom Loss":
                        selected_custom_loss = st.selectbox(
                            "Select Custom Loss",
                            list(custom_losses.keys()),
                            format_func=lambda x: x.split(".")[-1]  # Show class name only
                        )
                        
                        # Display loss information
                        if selected_custom_loss:
                            loss_class = custom_losses[selected_custom_loss]
                            loss_info = get_loss_info(loss_class)
                            st.info(f"**Description:** {loss_info['description']}")
                    
                with col2:
                    # Dataset splits
                    train_ratio = st.number_input("Training Set Ratio", min_value=0.5, max_value=0.9, value=0.7, 
                                                  step=0.05, key="custom_linear_train_ratio")
                    val_ratio = st.number_input("Validation Set Ratio", min_value=0.0, max_value=0.3, value=0.15,
                                                 step=0.05, key="custom_linear_val_ratio")
                    test_ratio = 1.0 - train_ratio - val_ratio
                    st.write(f"Test Set Ratio: {test_ratio:.2f}")
                    
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001,
                                                     step=0.0001, format="%.4f", key="custom_linear_learning_rate")
                    
                with col2:
                    batch_size = st.number_input("Batch Size", min_value=8, max_value=2048, value=64,
                                                  step=64, key="custom_linear_batch_size")
                    
                with col3:
                    epochs = st.number_input("Maximum Epochs", min_value=1, max_value=500, value=100,
                                             step=20, key="custom_linear_epochs")
                
                selected_model = {
                    "type": "custom_linear",
                    "model_name": custom_model_key.split(".")[-1],
                    "model_class": custom_model_class,
                    "model_class_name": custom_model_key,
                    "input_length": input_length,
                    "output_length": output_length,
                    "loss_function": loss_function,
                    "train_ratio": train_ratio,
                    "val_ratio": val_ratio,
                    "test_ratio": test_ratio,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs
                }

    
    # Update model_params dictionary
    if loss_function == "Custom Loss" and selected_custom_loss:
        model_params["custom_loss"] = selected_custom_loss
    
    # Custom metrics selection
    st.header("Evaluation Metrics")
    
    # Built-in metrics
    builtin_metrics = ["R²", "MAPE", "RMSE", "MSE", "MAE"]
    selected_builtin_metrics = st.multiselect(
        "Select Built-in Metrics",
        builtin_metrics,
        default=builtin_metrics[:3]
    )
    
    # Custom metrics
    if custom_metrics:
        selected_custom_metrics = st.multiselect(
            "Select Custom Metrics",
            list(custom_metrics.keys()),
            default=[],
            format_func=lambda x: x.split(".")[-1]
        )
    else:
        selected_custom_metrics = []
        st.info("No custom metrics found. Add custom metric files to the 'custom_metrics' directory.")

    # Training section
    st.header("Model Training")
    
    if st.button("Train Model"):
        st.session_state.trained_model = None
        st.session_state.predictions = None
        st.session_state.ground_truth = None
        st.session_state.training_history = None
        st.session_state.best_model_state = None
        
        # Create containers for visualization
        training_container = st.container()
        with training_container:
            st.subheader("Training Progress")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Initialize the chart with empty data
                loss_chart = st.line_chart(pd.DataFrame({
                    'Training Loss': [],
                    'Validation Loss': []
                }), color=["#2B66C2", "#93C6F9"])
            
            with col2:
                progress_bar = st.progress(0)
                status_text = st.empty()
        
        try:
            # Prepare data
            status_text.text("Preparing data...")
            
            # Extract parameters
            model_params = selected_model
            input_length = model_params["input_length"]
            output_length = model_params["output_length"]
            
            # Prepare data
            X, y, norm_params = prepare_time_series_data(
                data, input_vars, target_vars, input_length, output_length
            )
            
            # Check if we have enough data
            if len(X) < 10:
                st.error(f"Not enough data for the selected sequence lengths. Try using shorter sequences.")
                return
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y.reshape(y.shape[0], -1))  # Flatten output for simplicity
            
            # Create dataset
            dataset = TensorDataset(X_tensor, y_tensor)
            
            # Split data
            train_ratio = model_params["train_ratio"]
            val_ratio = model_params["val_ratio"]
            test_ratio = model_params["test_ratio"]
            
            dataset_size = len(dataset)
            train_size = int(train_ratio * dataset_size)
            val_size = int(val_ratio * dataset_size)
            test_size = dataset_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )
            
            # Create data loaders
            batch_size = model_params["batch_size"]
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            progress_bar.progress(10)
            status_text.text("Creating model...")
            
            # Set device
            os_name = platform.system()
            if os_name == "Darwin":
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                    print("MPS is available")
                else:
                    device = torch.device("cpu")
                    print("MPS not available, using CPU")
            # 对于 Windows, 检查 CUDA 支持
            elif os_name == "Windows":
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    print("CUDA is available")
                else:
                    device = torch.device("cpu")
                    print("CUDA not available, using CPU")
            # 其他操作系统默认使用 CPU
            else:
                device = torch.device("cpu")
                print("Using CPU")

            # Ensure transformed_input_vars is properly defined
            transformed_input_vars = norm_params.get('transformed_input_vars', input_vars)
            input_feature_count = len(transformed_input_vars)

            # Log information about dimensions for debugging
            # st.info(f"Input variables count: {len(input_vars)}")
            # st.info(f"Transformed input variables count: {input_feature_count}")
            # st.info(f"Input length: {input_length}, Output length: {output_length}")
            
            # Initialize model based on selection
            if model_params["type"] in ["deep_learning", "custom_deep_learning"]:
                input_dim = input_feature_count
                hidden_dim = model_params["hidden_dim"]
                output_dim = len(target_vars) * output_length
                num_layers = model_params["num_layers"]
                
                if model_params["type"] == "deep_learning":
                    model_name = model_params["model_name"]
                    # st.info(f"Creating {model_name} model with input dim {input_dim}, hidden dim {hidden_dim}, output dim {output_dim}")
                    
                    if model_name == "LSTM":
                        model = LSTM(input_dim, hidden_dim, output_dim, num_layers)
                    elif model_name == "GRU":
                        model = GRU(input_dim, hidden_dim, output_dim, num_layers)
                    elif model_name == "RNN":
                        model = SimpleRNN(input_dim, hidden_dim, output_dim, num_layers)
                    elif model_name == "Transformer":
                        model = Transformer(input_dim, model_params["d_model"], hidden_dim, output_dim, 
                                            num_layers, nhead=model_params["num_heads"])
                else:
                    # Custom deep learning model
                    model_class = model_params["model_class"]
                    model = model_class(input_dim, hidden_dim, output_dim, num_layers)
            else:
                # Linear or custom linear model
                input_dim = input_feature_count * input_length
                output_dim = len(target_vars) * output_length
                
                st.info(f"Creating linear model with input dim {input_dim} and output dim {output_dim}")
                
                if model_params["type"] == "linear":
                    model = LinearModel(input_dim, output_dim)
                else:
                    # Custom linear model
                    model_class = model_params["model_class"]
                    model = model_class(input_dim, output_dim)
            
            model = model.to(device)
            
            # Set loss function
            if model_params["loss_function"] == "L2Loss":
                criterion = nn.MSELoss()
            elif model_params["loss_function"] == "L1Loss":
                criterion = nn.L1Loss()
            elif model_params["loss_function"] == "Huber":
                criterion = nn.SmoothL1Loss()
            elif model_params["loss_function"] == "Custom Loss":
                # Initialize the custom loss
                loss_class = custom_losses[model_params["custom_loss"]]
                criterion = loss_class()
            
            # Set optimizer
            optimizer = optim.Adam(model.parameters(), lr=model_params["learning_rate"])
            
            # Set scheduler
            scheduler = None
            if model_params.get("use_lr_scheduler", False):
                if model_params["lr_scheduler_type"] == "ReduceLROnPlateau":
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=5, verbose=True
                    )
                elif model_params["lr_scheduler_type"] == "StepLR":
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer, step_size=10, gamma=0.5
                    )
                elif model_params["lr_scheduler_type"] == "CosineAnnealingLR":
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=model_params["epochs"]
                    )
            
            progress_bar.progress(20)
            status_text.text("Training model...")
            
            # Train model
            epochs = model_params["epochs"]
            patience = model_params.get("patience", 15) if model_params.get("use_early_stopping", True) else float('inf')
            
            best_model_state, training_history = train_model(model, train_loader, val_loader, criterion, optimizer, 
                                                             scheduler, device, epochs, patience, loss_chart, progress_bar, 
                                                             status_text, verbose=False)
            
            # Mark training as complete
            status_text.success("Training complete!")

            # Update progress
            progress_bar.progress(70)
            status_text.text("Evaluating model...")
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Prepare selected metrics
            selected_metrics = {}
            
            # Add built-in metrics
            if "MSE" in selected_builtin_metrics:
                selected_metrics["MSE"] = lambda y_true, y_pred: mean_squared_error(y_true, y_pred)
            
            if "MAE" in selected_builtin_metrics:
                selected_metrics["MAE"] = lambda y_true, y_pred: mean_absolute_error(y_true, y_pred)
            
            if "RMSE" in selected_builtin_metrics:
                selected_metrics["RMSE"] = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
            
            if "R²" in selected_builtin_metrics:
                selected_metrics["R²"] = lambda y_true, y_pred: r2_score(y_true, y_pred)
            
            if "MAPE" in selected_builtin_metrics:
                selected_metrics["MAPE"] = lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred)
            
            # Add custom metrics
            for metric_key in selected_custom_metrics:
                metric_func = custom_metrics[metric_key]
                metric_name = metric_key.split(".")[-1]
                selected_metrics[metric_name] = metric_func
            
            # Evaluate on test set
            test_loss, predictions, actuals, metrics_results = evaluate_model(
                model, test_loader, criterion, device, selected_metrics
            )
            
            # Reshape predictions and actuals to original dimensions
            predictions = predictions.reshape(-1, output_length, len(target_vars))
            actuals = actuals.reshape(-1, output_length, len(target_vars))
            
            # Store results in session state
            st.session_state.trained_model = model
            st.session_state.predictions = predictions
            st.session_state.ground_truth = actuals
            st.session_state.training_history = training_history
            st.session_state.best_model_state = best_model_state
            st.session_state.norm_params = norm_params
            st.session_state.metrics_results = metrics_results
            
            # Store model configuration
            st.session_state.model_config = model_params
            
            # Save model and results
            progress_bar.progress(90)
            status_text.text("Saving model and results...")
            
            model_info = {
                "model_state": best_model_state,
                "model_class_name": model_params.get("model_class_name", model_params["model_name"])
            }
            
            model_path, timestamp = save_model_and_results(
                model_info, model_params, norm_params, 
                predictions, actuals, input_vars, target_vars
            )
            
            progress_bar.progress(100)
            status_text.text("Training complete!")
            
            st.success(f"Model trained successfully and saved as {model_path}")
            
        except Exception as e:
            st.error(f"Error during training: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Results section (only show if model has been trained)
    if st.session_state.trained_model is not None:
        st.header("Training Results")
        
        st.subheader("Training Analysis")

        # Get training history from session state instead of local variable
        training_history = st.session_state.training_history
        
        # Create a more detailed analysis plot
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs_range = range(1, len(training_history['train_loss']) + 1)
        ax.plot(epochs_range, training_history['train_loss'], label='Training Loss')
        ax.plot(epochs_range, training_history['val_loss'], label='Validation Loss')
        
        # Add min/max lines and annotations
        min_val_loss = min(training_history['val_loss'])
        min_val_epoch = training_history['val_loss'].index(min_val_loss) + 1
        ax.axhline(y=min_val_loss, color='r', linestyle='--', alpha=0.3)
        ax.axvline(x=min_val_epoch, color='r', linestyle='--', alpha=0.3)

        ax.plot(min_val_epoch, min_val_loss, 'o', markersize=3, 
            markerfacecolor='red',  # Distinctive green color
            markeredgecolor='darkred',
            markeredgewidth=0.5,
            label='Best Model')

        ax.text(min_val_epoch + 1, min_val_loss*0.95, 
            f'Best: {min_val_loss:.4f} (epoch {min_val_epoch})',
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History - Final Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Show metrics
        if hasattr(st.session_state, 'metrics_results'):
            st.subheader("Evaluation Metrics")
            
            metrics_results = st.session_state.metrics_results
            metrics_df = pd.DataFrame({
                'Metric': list(metrics_results.keys()),
                'Value': list(metrics_results.values())
            })
            
            st.table(metrics_df)
        
            # Show option to proceed to evaluation
            if st.button("Proceed to Model Evaluation"):
                st.session_state.page = "model_evaluation"
                st.rerun()

# If run directly outside the main application
if __name__ == "__main__":
    run()
