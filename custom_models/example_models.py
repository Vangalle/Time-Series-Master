import torch
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
