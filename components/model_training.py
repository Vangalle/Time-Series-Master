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

# Default model definitions
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, input_length, output_length, dropout=0.5):
        super(SimpleRNN, self).__init__()

        self.output_dim = output_dim
        self.output_length = output_length
        
        # Encoder RNN
        self.rnn1 = nn.RNN(input_dim, hidden_dim * output_dim, batch_first=True, dropout=dropout)
        self.rnn2 = nn.RNN(hidden_dim * output_dim, hidden_dim * output_dim, batch_first=True, dropout=dropout)
        if output_dim == 1:
            self.dense = nn.Linear(hidden_dim * output_dim, output_length)
        else:
            self.dense = nn.Linear(hidden_dim * output_dim * input_length, output_dim * output_length)
        
    def forward(self, x):
        
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        if self.output_dim == 1:
            x = self.dense(x[:, -1, :])
            x = x.unsqueeze(-1)
        else:
            batch_size, seq_length, hidden_dim = x.size()
            x = x.reshape(batch_size, seq_length * hidden_dim)
            x = self.dense(x)
            x = x. reshape(batch_size, self.output_length, self.output_dim)
        
        return x

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, input_length, output_length, dropout=0.5):
        super(LSTM, self).__init__()

        self.output_dim = output_dim
        self.output_length = output_length
        
        # Encoder LSTM
        self.lstm1 = nn.LSTM(input_dim, hidden_dim * output_dim, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dim * output_dim, hidden_dim * output_dim, batch_first=True, dropout=dropout)
        if output_dim == 1:
            self.dense = nn.Linear(hidden_dim * output_dim, output_length)
        else:
            self.dense = nn.Linear(hidden_dim * output_dim * input_length, output_dim * output_length)
        
    def forward(self, x):
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        if self.output_dim == 1:
            x = self.dense(x[:, -1, :])
            x = x.unsqueeze(-1)
        else:
            batch_size, seq_length, hidden_dim = x.size()
            x = x.reshape(batch_size, seq_length * hidden_dim)
            x = self.dense(x)
            x = x. reshape(batch_size, self.output_length, self.output_dim)
        
        return x

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, input_length, output_length, dropout=0.5):
        super(GRU, self).__init__()

        self.output_dim = output_dim
        self.output_length = output_length
        
        # Encoder GRU
        self.gru1 = nn.GRU(input_dim, hidden_dim * output_dim, batch_first=True, dropout=dropout)
        self.gru2 = nn.GRU(hidden_dim * output_dim, hidden_dim * output_dim, batch_first=True, dropout=dropout)
        if output_dim == 1:
            self.dense = nn.Linear(hidden_dim * output_dim, output_length)
        else:
            self.dense = nn.Linear(hidden_dim * output_dim * input_length, output_dim * output_length)
        
    def forward(self, x):
        
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        if self.output_dim == 1:
            x = self.dense(x[:, -1, :])
            x = x.unsqueeze(-1)
        else:
            batch_size, seq_length, hidden_dim = x.size()
            x = x.reshape(batch_size, seq_length * hidden_dim)
            x = self.dense(x)
            x = x. reshape(batch_size, self.output_length, self.output_dim)
        
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, hidden_dim, output_dim,
                 input_length, output_length, max_seq_length=5000, encoding_type='sinusoidal',
                 use_projection=True):
        
        self.input_length = input_length
        self.output_length = output_length

        super().__init__()
        
        # Ensure d_model is divisible by nhead
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by num_heads ({nhead})"
        
        # Feature projection
        self.use_projection = use_projection
        if use_projection:
            self.input_projection = nn.Linear(input_dim, d_model)
        else:
            # If not using projection, input_dim must equal d_model
            assert input_dim == d_model, f"When use_projection=False, input_dim ({input_dim}) must equal d_model ({d_model})"

        # Position encoding configuration
        self.encoding_type = encoding_type.lower()  # 'sinusoidal' or 'learned'
        self.d_model = d_model
        
        # Position embedding table (for learned embeddings)
        if encoding_type.lower() == 'learned':
            self.position_embedding = nn.Embedding(max_seq_length, d_model)
            nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)  # Common in transformer models
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer decoder (if needed)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection for multivariate prediction
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def _get_sinusoidal_encoding(self, position_indices, d_model):
        """
        Create sinusoidal positional encodings using vectorized operations.
        
        Args:
            position_indices: Tensor of shape [batch_size, seq_length]
            d_model: Dimension of the model
            
        Returns:
            Tensor of shape [batch_size, seq_length, d_model]
        """
        device = position_indices.device

        # Handle case where position_indices has 3 dimensions
        if position_indices.dim() == 3:
            # Extract just the first feature or use another appropriate strategy
            position_indices = position_indices[:, :, 0]
        
        # Create frequency tensor
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * 
                        -(math.log(10000.0) / d_model))
        
        # Expand dimensions for broadcasting
        pos = position_indices.unsqueeze(-1)  # [batch_size, seq_length, 1]
        div_term = div_term.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model/2]
        
        # Compute sine and cosine parts with broadcasting
        sin_part = torch.sin(pos * div_term)
        cos_part = torch.cos(pos * div_term)
        
        # Interleave sine and cosine components
        position_enc = torch.zeros(position_indices.shape[0], position_indices.shape[1], 
                                d_model, device=device)
        
        # Assign sin to even indices, cos to odd indices
        position_enc[:, :, 0::2] = sin_part
        position_enc[:, :, 1::2] = cos_part
        
        return position_enc
    
    def _generate_default_positions(self, batch_size, seq_length, device):
        # Generate sequential position indices [0, 1, 2, ..., seq_length-1]
        return torch.arange(0, seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

    def forward(self, features, position_indices=None, tgt=None):
        batch_size, _, _ = features.shape
        device = features.device
        
        # Project features if needed
        if self.use_projection:
            x = self.input_projection(features)  # [batch_size, self.input_length, d_model]
        else:
            x = features  # Already in correct dimension

        # Generate default position indices if not provided
        if position_indices is None:
            position_indices = self._generate_default_positions(batch_size, self.input_length, device)

        # Get position encodings
        if self.encoding_type == 'sinusoidal':
            pos_encoding = self._get_sinusoidal_encoding(position_indices, self.d_model)
        else:  # 'learned'
            # st.write(f"Shape of position_indices: {position_indices.shape}")
            if position_indices.dim() == 3:
                position_indices = position_indices[:, :, 0]
            pos_encoding = self.position_embedding(position_indices.long())
        
        # Add position encodings to input
        # st.write(f"X shape: {x.shape}, type: {x.dtype}")
        # st.write(f"Pos_encoding shape: {pos_encoding.shape}, type: {pos_encoding.dtype}")
        x = x + pos_encoding
        
        # Pass through transformer
        output = self.transformer_encoder(x)

        if tgt is not None:
            if self.use_projection:
                tgt = self.input_projection(tgt)
            else:
                tgt = tgt

            batch_size, _, _ = tgt.shape
            
            # Generate default position indices if not provided
            position_indices = self._generate_default_positions(batch_size, self.output_length, device)
            
            # Generate position encodings for target
            if self.encoding_type == 'sinusoidal':
                pos_encoding = self._get_sinusoidal_encoding(position_indices, self.d_model)
            else:  # 'learned'
                pos_encoding = self.position_embedding(position_indices.long())

            tgt = tgt + pos_encoding
            
            # Pass through transformer decoder
            output = self.transformer_decoder(tgt=tgt, memory=output) # [batch_size, self.output_length, d_model]

            # Project to output dimension for multivariate prediction
            predictions = self.output_projection(output)
            
            return predictions
        
        # Autoregressive mode (inference)
        else:
            # Start with last timestep from features as initial decoder input
            current_input = features[:, -1:, :]  
            
            # Container for predictions
            predictions = []
            
            # Generate predictions step by step
            for i in range(self.output_length):
                # Project to embedding space if needed
                if self.use_projection:
                    decoder_input = self.input_projection(current_input)
                else:
                    decoder_input = current_input
                    
                # Create position encoding for current timestep
                position_idx = torch.tensor([[self.input_length + i]], device=device).expand(batch_size, 1)
                
                if self.encoding_type == 'sinusoidal':
                    pos_enc = self._get_sinusoidal_encoding(position_idx, self.d_model)
                else:  # 'learned'
                    pos_enc = self.position_embedding(position_idx.long())
                    
                decoder_input = decoder_input + pos_enc
                
                # Run through decoder with encoded inputs as memory
                decoder_output = self.transformer_decoder(tgt=decoder_input, memory=output)
                
                # Apply output projection directly to decoder output
                next_prediction = self.output_projection(decoder_output)
                
                # Store prediction
                predictions.append(next_prediction)
                
                # Use prediction as input for next timestep
                current_input = next_prediction
            
            # Concatenate all predictions along time dimension
            final_predictions = torch.cat(predictions, dim=1)
            
            return final_predictions

class EncoderTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, hidden_dim, output_dim,
                 input_length, output_length, max_seq_length=5000, encoding_type='sinusoidal',
                 use_projection=True):
        
        self.input_length = input_length
        self.output_length = output_length
        self.output_dim = output_dim

        super().__init__()
        
        # Ensure d_model is divisible by nhead
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by num_heads ({nhead})"
        
        # Feature projection
        self.use_projection = use_projection
        if use_projection:
            self.input_projection = nn.Linear(input_dim, d_model)
        else:
            # If not using projection, input_dim must equal d_model
            assert input_dim == d_model, f"When use_projection=False, input_dim ({input_dim}) must equal d_model ({d_model})"

        # Position encoding configuration
        self.encoding_type = encoding_type.lower()  # 'sinusoidal' or 'learned'
        self.d_model = d_model
        
        # Position embedding table (for learned embeddings)
        if encoding_type.lower() == 'learned':
            self.position_embedding = nn.Embedding(max_seq_length, d_model)
            nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)  # Common in transformer models
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection for multivariate prediction
        self.output_projection = nn.Linear(d_model * input_length, output_dim * output_length)

    def _get_sinusoidal_encoding(self, position_indices, d_model):
        """
        Create sinusoidal positional encodings using vectorized operations.
        
        Args:
            position_indices: Tensor of shape [batch_size, seq_length]
            d_model: Dimension of the model
            
        Returns:
            Tensor of shape [batch_size, seq_length, d_model]
        """
        device = position_indices.device

        # Handle case where position_indices has 3 dimensions
        if position_indices.dim() == 3:
            # Extract just the first feature or use another appropriate strategy
            position_indices = position_indices[:, :, 0]
        
        # Create frequency tensor
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * 
                        -(math.log(10000.0) / d_model))
        
        # Expand dimensions for broadcasting
        pos = position_indices.unsqueeze(-1)  # [batch_size, seq_length, 1]
        div_term = div_term.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model/2]
        
        # Compute sine and cosine parts with broadcasting
        sin_part = torch.sin(pos * div_term)
        cos_part = torch.cos(pos * div_term)
        
        # Interleave sine and cosine components
        position_enc = torch.zeros(position_indices.shape[0], position_indices.shape[1], 
                                d_model, device=device)
        
        # Assign sin to even indices, cos to odd indices
        position_enc[:, :, 0::2] = sin_part
        position_enc[:, :, 1::2] = cos_part
        
        return position_enc
    
    def _generate_default_positions(self, batch_size, seq_length, device):
        # Generate sequential position indices [0, 1, 2, ..., seq_length-1]
        return torch.arange(0, seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

    def forward(self, features, position_indices=None, tgt=None):
        batch_size, _, _ = features.shape
        device = features.device
        
        # Project features if needed
        if self.use_projection:
            x = self.input_projection(features)  # [batch_size, self.input_length, d_model]
        else:
            x = features  # Already in correct dimension

        # Generate default position indices if not provided
        if position_indices is None:
            position_indices = self._generate_default_positions(batch_size, self.input_length, device)

        # Get position encodings
        if self.encoding_type == 'sinusoidal':
            pos_encoding = self._get_sinusoidal_encoding(position_indices, self.d_model)
        else:  # 'learned'
            # st.write(f"Shape of position_indices: {position_indices.shape}")
            if position_indices.dim() == 3:
                position_indices = position_indices[:, :, 0]
            pos_encoding = self.position_embedding(position_indices.long())
        
        # Add position encodings to input
        x = x + pos_encoding
        
        # Pass through transformer
        output = self.transformer_encoder(x)

        batch_size, _, _ = output.shape
        output = output.reshape(batch_size, -1)

        # Project to output dimension for multivariate prediction
        predictions = self.output_projection(output)
        predictions = predictions.reshape(batch_size, self.output_length, self.output_dim)
        
        return predictions

class MultivariatePosEncTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, hidden_dim, output_dim,
                 max_seq_length=5000, encoding_type='sinusoidal', use_projection=True):
        super().__init__()
        
        # Basic validations
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by num_heads ({nhead})"
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_projection = use_projection
        self.encoding_type = encoding_type
        
        # Input projection
        if use_projection:
            self.input_projection = nn.Linear(input_dim, d_model)
        
        # Variable-specific position embeddings
        if encoding_type == 'learned':
            # Create separate embedding tables for each variable
            self.position_embeddings = nn.ModuleList([
                nn.Embedding(max_seq_length, d_model // input_dim)
                for _ in range(input_dim)
            ])
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
    
    def _get_variable_sinusoidal_encoding(self, position_indices, batch_size, seq_length):
        """Create separate sinusoidal encodings for each variable"""
        device = position_indices.device
        
        # Calculate the dimension allocated to each variable
        var_dim = self.d_model // self.input_dim
        
        # Initialize the full position encoding tensor
        full_position_enc = torch.zeros(batch_size, seq_length, self.d_model, device=device)
        
        # Generate different encodings for each variable
        for var_idx in range(self.input_dim):
            # Calculate offset for this variable in the embedding dimension
            start_dim = var_idx * var_dim
            end_dim = start_dim + var_dim
            
            # Create sinusoidal encoding for this variable
            div_term = torch.exp(torch.arange(0, var_dim, 2, device=device) * 
                               -(math.log(10000.0) / var_dim))
            
            # Apply potentially different position indices for each variable
            var_positions = position_indices
            
            # For each batch and position
            for b in range(batch_size):
                for i in range(seq_length):
                    pos = var_positions[b, i].item()
                    
                    # Handle odd-sized dimensions
                    actual_dim = min(var_dim, end_dim - start_dim)
                    actual_even_dims = actual_dim - (actual_dim % 2)
                    
                    # Apply sinusoidal encoding
                    if actual_even_dims > 0:
                        full_position_enc[b, i, start_dim:start_dim+actual_even_dims:2] = torch.sin(pos * div_term[:actual_even_dims//2])
                        full_position_enc[b, i, start_dim+1:start_dim+actual_even_dims:2] = torch.cos(pos * div_term[:actual_even_dims//2])
                    
                    # Handle the remaining dimension if odd
                    if actual_dim % 2 == 1 and actual_dim > 0:
                        full_position_enc[b, i, end_dim-1] = torch.sin(pos * div_term[-1])
            
        return full_position_enc
    
    def _generate_default_positions(self, batch_size, seq_length, device):
        # Default sequential positions
        return torch.arange(0, seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
    
    def forward(self, features, position_indices=None, variable_positions=None):
        batch_size, seq_length, _ = features.shape
        device = features.device
        
        # Project input features if needed
        if self.use_projection:
            x = self.input_projection(features)
        else:
            x = features
        
        # Generate default position indices if not provided
        if position_indices is None:
            position_indices = self._generate_default_positions(batch_size, seq_length, device)
        
        # Generate position encodings
        if self.encoding_type == 'sinusoidal':
            # Create variable-specific sinusoidal encodings
            pos_encoding = self._get_variable_sinusoidal_encoding(
                position_indices, batch_size, seq_length)
        else:  # 'learned'
            # Initialize full position encoding tensor
            pos_encoding = torch.zeros(batch_size, seq_length, self.d_model, device=device)
            
            # Calculate var dimension
            var_dim = self.d_model // self.input_dim
            
            # Apply learned embeddings for each variable
            for var_idx in range(self.input_dim):
                var_pos = variable_positions[var_idx] if variable_positions else position_indices
                embedding = self.position_embeddings[var_idx](var_pos)
                start_dim = var_idx * var_dim
                end_dim = start_dim + var_dim
                pos_encoding[:, :, start_dim:end_dim] = embedding
        
        # Add position encodings to input
        x = x + pos_encoding
        
        # Pass through transformer
        output = self.transformer_encoder(x)
        
        # Project to output dimension
        predictions = self.output_projection(output)
        
        return predictions

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim, input_length, output_length,):
        super(LinearModel, self).__init__()
        self.output_dim = output_dim
        self.output_length = output_length

        if output_dim == 1:
            self.dense = nn.Linear(input_dim, output_length)
        else:
            self.dense = nn.Linear(input_dim * input_length, output_dim * output_length)
        
    def forward(self, x):
        
        if self.output_dim == 1:
            x = self.dense(x[:, -1, :])
            x = x.unsqueeze(-1)
        else:
            batch_size, seq_length, hidden_dim = x.size()
            x = x.reshape(batch_size, seq_length * hidden_dim)
            x = self.dense(x)
            x = x. reshape(batch_size, self.output_length, self.output_dim)
        return x

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
        X_data: Input data for the model 
        X_pos: Position Encoding data
        y_data: Target data
        norm_params: Normalization parameters
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
        # Get the datetime series
        dt_series = data[col]
        col_prefix = f"{col}_"
        
        # Get unique datetime values and sort them
        unique_dt = np.sort(dt_series.unique())
        
        # If there are no datetime values or only one value, add a default index
        if len(unique_dt) <= 1:
            processed_data[col_prefix + 'continuous_index'] = 0
            datetime_features[col] = [col_prefix + 'continuous_index']
            continue
        
        # Calculate time differences in seconds
        time_diffs_seconds = np.array([(unique_dt[i+1] - unique_dt[i]).astype('timedelta64[s]').astype(int) 
                                    for i in range(len(unique_dt)-1)])
        
        # Get the smallest non-zero time difference
        min_diff_seconds = min(td for td in time_diffs_seconds if td > 0)
        
        # Determine the time unit based on the smallest difference
        if min_diff_seconds < 60:  # Less than a minute
            unit_divisor = 1
        elif min_diff_seconds < 3600:  # Less than an hour
            unit_divisor = 60
        elif min_diff_seconds < 86400:  # Less than a day
            unit_divisor = 3600
        else:
            unit_divisor = 86400
        
        # Create an array of continuous indices
        continuous_indices = np.zeros(len(unique_dt), dtype=int)
        
        # Calculate the continuous index
        for i in range(1, len(unique_dt)):
            time_diff_seconds = (unique_dt[i] - unique_dt[i-1]).astype('timedelta64[s]').astype(int)
            units_to_advance = max(1, round(time_diff_seconds / unit_divisor))
            continuous_indices[i] = continuous_indices[i-1] + units_to_advance
        
        # Create a DataFrame with unique datetime values and their continuous indices
        dt_index_df = pd.DataFrame({
            'datetime': unique_dt,
            'continuous_index': continuous_indices
        })
        
        # Convert the original datetime series to a DataFrame and merge with dt_index_df
        temp_df = pd.DataFrame({'datetime': dt_series})
        temp_df = pd.merge(temp_df, dt_index_df, on='datetime', how='left')
        
        # Add the continuous index to processed_data
        processed_data[col_prefix + 'continuous_index'] = temp_df['continuous_index']
        
        # Track the new feature name
        datetime_features[col] = [col_prefix + 'continuous_index']
    
    # Replace datetime columns with their numeric representations
    transformed_input_vars = []
    datetime_embedding_cols = []
    for var in input_vars:
        if var in datetime_cols:
            datetime_embedding_cols.extend(datetime_features[var])
        else:
            transformed_input_vars.append(var)
    
    # Extract data for transformed input variables and targets
    # st.write(f"columns: {transformed_input_vars}")
    X_data = processed_data[transformed_input_vars].values
    X_pos = processed_data[datetime_embedding_cols].values
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
    
    return X_data, X_pos, y_data, norm_params

# Function to train the model
# Modify the train_model function to accept position data
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
               device, epochs, patience, loss_chart, progress_bar, status_text, 
               uses_positional_encoding=False, verbose=True):
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
        for batch in train_loader:
            if uses_positional_encoding:
                inputs, positions, targets = [b.to(device) for b in batch]
            else:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with or without positions
            if uses_positional_encoding:
                outputs = model(inputs, positions, targets)
            else:
                outputs = model(inputs)
                
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        
        train_loss /= len(train_loader)
        training_history['train_loss'].append(train_loss)
        
        # Validation phase
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if uses_positional_encoding:
                    inputs, positions, targets = [b.to(device) for b in batch]
                    outputs = model(inputs, positions)
                else:
                    inputs, targets = batch
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
def evaluate_model(model, test_loader, criterion, device, target_vars, uses_positional_encoding=False, custom_metrics=None):
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            if uses_positional_encoding:
                inputs, positions, targets = [b.to(device) for b in batch]
            else:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass with or without positions
            if uses_positional_encoding:
                outputs = model(inputs, positions)
            else:
                outputs = model(inputs)
                
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())
    
    test_loss /= len(test_loader)
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    num_targets = actuals.shape[-1]
    
    # Calculate additional metrics if provided
    metrics_results = {target_var: {} for target_var in target_vars}
    if custom_metrics:
        for metric_name, metric_func in custom_metrics.items():
            try:
                # Initialize array to store metric values for each target
                target_metrics = np.zeros(num_targets)
                
                # Calculate metric for each target variable
                for i in range(num_targets):
                    target_pred = predictions[:, :, i].flatten()
                    target_actual = actuals[:, :, i].flatten()
                    target_metrics[i] = metric_func(target_actual, target_pred)
                
                # # Average the metrics across all targets
                # metrics_results[metric_name] = np.mean(target_metrics)
                
                for i in range(num_targets):
                    metrics_results[f"{target_vars[i]}"][f"{metric_name}"] = target_metrics[i]
                    # metrics_results[f"{metric_name} of {target_vars[i]}"] = target_metrics[i]

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

    # Load custom losses
    custom_losses = load_custom_losses()
    loss_options = ["L2Loss", "L1Loss", "Huber"]
    if custom_losses:
        loss_options.append("Custom Loss")
    
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
                    ["LSTM", "GRU", "RNN", "Transformer", "EncoderTransformer"],
                )
                
                # Model depth selection
                num_layers = st.number_input("Number of Layers", min_value=1, max_value=20, value=2, step=1)
                
                # Hidden dimension selection
                hidden_dim = st.number_input("Hidden Dimension", min_value=32, max_value=2048, value=64, step=32,
                                             help="For Transformer models, it's the FFN dimension and is " \
                                             "often set to 4 times the d_model.")

                # If Transformer model is selected, add num_heads parameter
                if dl_model_type == "Transformer" or dl_model_type == "EncoderTransformer":

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
                    if st.session_state.datetime_column == True:
                        st.session_state.uses_positional_encoding = st.checkbox(
                            "Use Positional Encoding",
                            value=True,
                            help="Use positional encoding for Transformer models."
                        )

                        if st.session_state.uses_positional_encoding:
                            st.session_state.positional_encoding_type = st.selectbox(
                                "Positional Encoding Type",
                                ["Sinusoidal", "Learned"],
                                index=0,
                                help="Select the type of positional encoding to use."
                            )
                    else:
                        st.session_state.uses_positional_encoding = False
                        st.session_state.positional_encoding_type = "sinusoidal"

                else:
                    st.session_state.uses_positional_encoding = False
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
                try:
                    # Calculate minimum required samples for validation and test sets
                    min_samples_needed = input_length + output_length
                    min_ratio_needed = min_samples_needed / data.shape[0]
                    
                    # Maximum training ratio ensuring validation and test sets have minimum samples
                    max_train_ratio = 1.0 - (2 * min_ratio_needed)
                    
                    if max_train_ratio < 0.1:
                        st.error(f"Not enough data for proper splitting with current parameters.")
                        st.info(f"Data: {data.shape[0]} rows | Min samples needed per split: {min_samples_needed}")
                        st.stop()  # Stop execution to prevent further issues
                    
                    # Training ratio input with adaptive default value
                    default_train = min(0.8, max_train_ratio)  # Try for 80% but respect maximum
                    train_ratio = st.number_input(
                        "Training Set Ratio", 
                        min_value=0.1, 
                        max_value=float(max_train_ratio), 
                        value=default_train,
                        step=0.05 if max_train_ratio > 0.2 else 0.01,
                        format="%.2f",
                        help = f"Maximum value: {max_train_ratio:.2f} (ensures validation and test sets have enough samples)"
                    )
                    
                    # Calculate remaining ratio for validation and test
                    remaining_ratio = 1.0 - train_ratio
                    
                    # Validation ratio input with proper constraints
                    max_val_ratio = remaining_ratio - min_ratio_needed  # Ensure test set gets minimum
                    default_val = min(0.1, max_val_ratio)  # Try for 10% but respect maximum
                    val_ratio = st.number_input(
                        "Validation Set Ratio", 
                        min_value=float(min_ratio_needed), 
                        max_value=float(max_val_ratio),
                        value=default_val,
                        step=0.05 if max_val_ratio > 0.2 else 0.01,
                        format="%.2f",
                        help = f"Maximum value: {max_val_ratio:.2f} (ensures test set has enough samples)"
                    )
                    
                    # Calculate and display test ratio
                    test_ratio = 1.0 - train_ratio - val_ratio
                    st.write(f"Test Set Ratio: {test_ratio:.2f}")

                except Exception as e:
                    st.error(f"Dataset splitting error: {str(e)}")
                    st.warning("Please adjust input/output lengths or check your dataset dimensions.")
                
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
            
            if dl_model_type == "Transformer" or dl_model_type == "EncoderTransformer":
                selected_model["d_model"] = d_model
                selected_model["num_heads"] = num_heads
            
        if builtin_model_type == "Linear":
            st.subheader("Linear Model Configuration")

            st.session_state.uses_positional_encoding = False
            
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
                try:
                    # Calculate minimum required samples for validation and test sets
                    min_samples_needed = input_length + output_length
                    min_ratio_needed = min_samples_needed / data.shape[0]
                    
                    # Maximum training ratio ensuring validation and test sets have minimum samples
                    max_train_ratio = 1.0 - (2 * min_ratio_needed)
                    
                    if max_train_ratio < 0.1:
                        st.error(f"Not enough data for proper splitting with current parameters.")
                        st.info(f"Data: {data.shape[0]} rows | Min samples needed per split: {min_samples_needed}")
                        st.stop()  # Stop execution to prevent further issues
                    
                    # Training ratio input with adaptive default value
                    default_train = min(0.8, max_train_ratio)  # Try for 80% but respect maximum
                    train_ratio = st.number_input(
                        "Training Set Ratio", 
                        min_value=0.1, 
                        max_value=float(max_train_ratio), 
                        value=default_train,
                        step=0.05 if max_train_ratio > 0.2 else 0.01,
                        format="%.2f",
                        key="linear_train_ratio",
                        help = f"Maximum value: {max_train_ratio:.2f} (ensures validation and test sets have enough samples)"
                    )
                    
                    # Calculate remaining ratio for validation and test
                    remaining_ratio = 1.0 - train_ratio
                    
                    # Validation ratio input with proper constraints
                    max_val_ratio = remaining_ratio - min_ratio_needed  # Ensure test set gets minimum
                    default_val = min(0.1, max_val_ratio)  # Try for 10% but respect maximum
                    val_ratio = st.number_input(
                        "Validation Set Ratio", 
                        min_value=float(min_ratio_needed), 
                        max_value=float(max_val_ratio),
                        value=default_val,
                        step=0.05 if max_val_ratio > 0.2 else 0.01,
                        format="%.2f",
                        key="linear_val_ratio",
                        help=f"Maximum value: {max_val_ratio:.2f} (ensures test set gets minimum samples)"
                    )
                    
                    # Calculate and display test ratio
                    test_ratio = 1.0 - train_ratio - val_ratio
                    st.write(f"Test Set Ratio: {test_ratio:.2f}")

                except Exception as e:
                    st.error(f"Dataset splitting error: {str(e)}")
                    st.warning("Please adjust input/output lengths or check your dataset dimensions.")
            
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
                        if st.session_state.datetime_column == True:
                            st.session_state.uses_positional_encoding = st.checkbox(
                                "Use Positional Encoding",
                                value=True,
                                help="Use positional encoding for Transformer models."
                            )
                            if st.session_state.uses_positional_encoding:
                                st.session_state.positional_encoding_type = st.selectbox(
                                    "Positional Encoding Type",
                                    ["Sinusoidal", "Learned"],
                                    index=0,
                                    help="Select the type of positional encoding to use."
                                )
                        else:
                            st.session_state.uses_positional_encoding = False
                            st.session_state.positional_encoding_type = "sinusoidal"
                    else:
                        st.session_state.uses_positional_encoding = False
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
                    
                    try:
                        # Calculate minimum required samples for validation and test sets
                        min_samples_needed = input_length + output_length
                        min_ratio_needed = min_samples_needed / data.shape[0]
                        
                        # Maximum training ratio ensuring validation and test sets have minimum samples
                        max_train_ratio = 1.0 - (2 * min_ratio_needed)
                        
                        if max_train_ratio < 0.1:
                            st.error(f"Not enough data for proper splitting with current parameters.")
                            st.info(f"Data: {data.shape[0]} rows | Min samples needed per split: {min_samples_needed}")
                            st.stop()  # Stop execution to prevent further issues
                        
                        # Training ratio input with adaptive default value
                        default_train = min(0.8, max_train_ratio)  # Try for 80% but respect maximum
                        train_ratio = st.number_input(
                            "Training Set Ratio", 
                            min_value=0.1, 
                            max_value=float(max_train_ratio), 
                            value=default_train,
                            step=0.05 if max_train_ratio > 0.2 else 0.01,
                            format="%.2f",
                            key="custom_train_ratio",
                            help = f"Maximum value: {max_train_ratio:.2f} (ensures validation and test sets have enough samples)"
                        )
                        
                        # Calculate remaining ratio for validation and test
                        remaining_ratio = 1.0 - train_ratio
                        
                        # Validation ratio input with proper constraints
                        max_val_ratio = remaining_ratio - min_ratio_needed  # Ensure test set gets minimum
                        default_val = min(0.1, max_val_ratio)  # Try for 10% but respect maximum
                        val_ratio = st.number_input(
                            "Validation Set Ratio", 
                            min_value=float(min_ratio_needed), 
                            max_value=float(max_val_ratio),
                            value=default_val,
                            step=0.05 if max_val_ratio > 0.2 else 0.01,
                            format="%.2f",
                            key="custom_val_ratio",
                            help=f"Maximum value: {max_val_ratio:.2f} (ensures test set gets minimum samples)"
                        )
                        
                        # Calculate and display test ratio
                        test_ratio = 1.0 - train_ratio - val_ratio
                        st.write(f"Test Set Ratio: {test_ratio:.2f}")

                    except Exception as e:
                        st.error(f"Dataset splitting error: {str(e)}")
                        st.warning("Please adjust input/output lengths or check your dataset dimensions.")
                
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
                st.session_state.uses_positional_encoding = False
                
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
                    try:
                        # Calculate minimum required samples for validation and test sets
                        min_samples_needed = input_length + output_length
                        min_ratio_needed = min_samples_needed / data.shape[0]
                        
                        # Maximum training ratio ensuring validation and test sets have minimum samples
                        max_train_ratio = 1.0 - (2 * min_ratio_needed)
                        
                        if max_train_ratio < 0.1:
                            st.error(f"Not enough data for proper splitting with current parameters.")
                            st.info(f"Data: {data.shape[0]} rows | Min samples needed per split: {min_samples_needed}")
                            st.stop()  # Stop execution to prevent further issues
                        
                        # Training ratio input with adaptive default value
                        default_train = min(0.8, max_train_ratio)  # Try for 80% but respect maximum
                        train_ratio = st.number_input(
                            "Training Set Ratio", 
                            min_value=0.1, 
                            max_value=float(max_train_ratio), 
                            value=default_train,
                            step=0.05 if max_train_ratio > 0.2 else 0.01,
                            format="%.2f",
                            key="custom_linear_train_ratio",
                            help = f"Maximum value: {max_train_ratio:.2f} (ensures validation and test sets have enough samples)"
                        )
                        
                        # Calculate remaining ratio for validation and test
                        remaining_ratio = 1.0 - train_ratio
                        
                        # Validation ratio input with proper constraints
                        max_val_ratio = remaining_ratio - min_ratio_needed  # Ensure test set gets minimum
                        default_val = min(0.1, max_val_ratio)  # Try for 10% but respect maximum
                        val_ratio = st.number_input(
                            "Validation Set Ratio", 
                            min_value=float(min_ratio_needed), 
                            max_value=float(max_val_ratio),
                            value=default_val,
                            step=0.05 if max_val_ratio > 0.2 else 0.01,
                            format="%.2f",
                            key="custom_linear_val_ratio",
                            help=f"Maximum value: {max_val_ratio:.2f} (ensures test set gets minimum samples)"
                        )
                        
                        # Calculate and display test ratio
                        test_ratio = 1.0 - train_ratio - val_ratio
                        st.write(f"Test Set Ratio: {test_ratio:.2f}")

                    except Exception as e:
                        st.error(f"Dataset splitting error: {str(e)}")
                        st.warning("Please adjust input/output lengths or check your dataset dimensions.")
                    
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
            X_data, X_pos, y_data, norm_params = prepare_time_series_data(
                data, input_vars, target_vars, input_length, output_length
            )

            # st.write(f"Data shape: {X_data.shape}, {X_pos.shape}, {y_data.shape}")
            
            # Split the raw data chronologically
            total_points = len(X_data)
            train_ratio = model_params["train_ratio"]
            val_ratio = model_params["val_ratio"]
            test_ratio = model_params["test_ratio"]

            # Calculate indices for data splits
            train_end = int(total_points * train_ratio)
            val_end = train_end + int(total_points * val_ratio)

            # Split the raw data
            train_X_data = X_data[:train_end]
            train_X_pos = X_pos[:train_end]
            train_y_data = y_data[:train_end]

            val_X_data = X_data[train_end:val_end]
            val_X_pos = X_pos[train_end:val_end]
            val_y_data = y_data[train_end:val_end]

            test_X_data = X_data[val_end:]
            test_X_pos = X_pos[val_end:]
            test_y_data = y_data[val_end:]

            # Create sequences with different strides
            # Training data: stride=1
            train_X, train_pos, train_y = [], [], []
            for i in range(len(train_X_data) - input_length - output_length + 1):
                train_X.append(train_X_data[i:i+input_length])
                train_pos.append(train_X_pos[i:i+input_length])
                train_y.append(train_y_data[i+input_length:i+input_length+output_length])

            # Validation data: stride=output_length
            val_X, val_pos, val_y = [], [], []
            for i in range(0, len(val_X_data) - input_length - output_length + 1, output_length):
                val_X.append(val_X_data[i:i+input_length])
                val_pos.append(val_X_pos[i:i+input_length])
                val_y.append(val_y_data[i+input_length:i+input_length+output_length])

            # Test data: stride=output_length
            test_X, test_pos, test_y = [], [], []
            for i in range(0, len(test_X_data) - input_length - output_length + 1, output_length):
                test_X.append(test_X_data[i:i+input_length])
                test_pos.append(test_X_pos[i:i+input_length])
                test_y.append(test_y_data[i+input_length:i+input_length+output_length])

            # Print info about how many sequences we created
            st.info(f"Created {len(train_X)} training sequences, {len(val_X)} validation sequences, and {len(test_X)} test sequences")

            # Convert to numpy arrays
            train_X = np.array(train_X)
            train_pos = np.array(train_pos)
            train_y = np.array(train_y)
            val_X = np.array(val_X)
            val_pos = np.array(val_pos)
            val_y = np.array(val_y)
            test_X = np.array(test_X)
            test_pos = np.array(test_pos)
            test_y = np.array(test_y)

            # Convert to PyTorch tensors
            train_X_tensor = torch.FloatTensor(train_X)
            train_pos_tensor = torch.FloatTensor(train_pos)
            train_y_tensor = torch.FloatTensor(train_y)

            # st.write(f"Training data shape: {train_X_tensor.shape}, {train_pos_tensor.shape}, {train_y_tensor.shape}")

            val_X_tensor = torch.FloatTensor(val_X)
            val_pos_tensor = torch.FloatTensor(val_pos)
            val_y_tensor = torch.FloatTensor(val_y)

            # st.write(f"Validation data shape: {val_X_tensor.shape}, {val_pos_tensor.shape}, {val_y_tensor.shape}")

            test_X_tensor = torch.FloatTensor(test_X)
            test_pos_tensor = torch.FloatTensor(test_pos)
            test_y_tensor = torch.FloatTensor(test_y)

            # st.write(f"Test data shape: {test_X_tensor.shape}, {test_pos_tensor.shape}, {test_y_tensor.shape}")

            # Create datasets
            if st.session_state.uses_positional_encoding:
                train_dataset = TensorDataset(train_X_tensor, train_pos_tensor, train_y_tensor)
                val_dataset = TensorDataset(val_X_tensor, val_pos_tensor, val_y_tensor)
                test_dataset = TensorDataset(test_X_tensor, test_pos_tensor, test_y_tensor)
            else:
                train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
                val_dataset = TensorDataset(val_X_tensor, val_y_tensor)
                test_dataset = TensorDataset(test_X_tensor, test_y_tensor)

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
                output_dim = len(target_vars)
                num_layers = model_params["num_layers"]
                
                if model_params["type"] == "deep_learning":
                    model_name = model_params["model_name"]
                    # st.info(f"Creating {model_name} model with input dim {input_dim}, hidden dim {hidden_dim}, output dim {output_dim}")
                    
                    if model_name == "LSTM":
                        model = LSTM(input_dim, hidden_dim, output_dim, num_layers, input_length, output_length)
                    elif model_name == "GRU":
                        model = GRU(input_dim, hidden_dim, output_dim, num_layers, input_length, output_length)
                    elif model_name == "RNN":
                        model = SimpleRNN(input_dim, hidden_dim, output_dim, num_layers, input_length, output_length)
                    elif model_name == "Transformer":
                        # st.write(st.session_state.positional_encoding_type)
                        model = Transformer(input_dim, model_params["d_model"], model_params["num_heads"], num_layers, 
                                            hidden_dim, output_dim, input_length, output_length,
                                            encoding_type=st.session_state.positional_encoding_type)
                    elif model_name == "EncoderTransformer":
                        model = EncoderTransformer(input_dim, model_params["d_model"], model_params["num_heads"], num_layers,
                                            hidden_dim, output_dim, input_length, output_length,
                                            encoding_type=st.session_state.positional_encoding_type)
                else:
                    # Custom deep learning model
                    model_class = model_params["model_class"]
                    model = model_class(input_dim, hidden_dim, output_dim, num_layers)
            else:
                # Linear or custom linear model
                input_dim = input_feature_count
                output_dim = len(target_vars)
                
                st.info(f"Creating linear model with input dim {input_dim} and output dim {output_dim}")
                
                if model_params["type"] == "linear":
                    model = LinearModel(input_dim, output_dim, input_length, output_length)
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
            
            best_model_state, training_history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                                                             device, epochs, patience, loss_chart, progress_bar, status_text, 
                                                             uses_positional_encoding=st.session_state.uses_positional_encoding, 
                                                             verbose=False)
            
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
                model, test_loader, criterion, device, st.session_state.target_vars,
                uses_positional_encoding=st.session_state.uses_positional_encoding, 
                custom_metrics=selected_metrics
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
            for target_var in target_vars:
                with st.expander(f"Metrics for {target_var}"):
                    target_df = pd.DataFrame({
                        'Metric': list(metrics_results[target_var].keys()),
                        'Value': list(metrics_results[target_var].values())
                    })
                    st.table(target_df)
            
            col1, col2 = st.columns(2)
            with col1:
                # Save configuration
                if st.button("Save Model Configuration"):
                    model_config = st.session_state.model_config
                    config_file = f"configs/config_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(config_file, "w") as f:
                        json.dump(model_config, f, indent=4)
                    st.success(f"Model configuration saved as {config_file}")

            with col2:
            # Show option to proceed to evaluation
                if st.button("Proceed to Model Evaluation"):
                    st.session_state.page = "model_evaluation"
                    st.rerun()

# If run directly outside the main application
if __name__ == "__main__":
    run()
