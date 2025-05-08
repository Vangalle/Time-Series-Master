import torch
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
