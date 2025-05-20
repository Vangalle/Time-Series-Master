# Time Series Master

Your ultimate platform for time series prediction or testing your time series prediction model.

## Overview

Time Series Master is a comprehensive, user-friendly application designed to simplify the process of time series forecasting. With an intuitive interface and powerful capabilities, this platform allows users to rapidly deploy various time series models without writing code, while still providing the flexibility for advanced users to implement custom solutions.

## Youtube Overview

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/n1J1X5Fs_Yg/0.jpg)](https://www.youtube.com/watch?v=n1J1X5Fs_Yg)

## Key Features

- **Streamlined Workflow**: Integrated data loading, model selection, training, and evaluation in a cohesive interface
- **Multiple Model Support**: Built-in implementations of LSTM, GRU, RNN, Transformer, and linear regression models
- **Extensible Architecture**: Easily add custom models and evaluation metrics
- **Interactive Data Visualization**: Explore and visualize your time series data before modeling
- **Advanced Training Options**: Configure model depth, hidden dimensions, early stopping, and learning rate scheduling
- **Comprehensive Evaluation**: Assess model performance through various metrics and visual comparisons
- **Result Export**: Save predictions and evaluation results for further analysis

## Installation

### Prerequisites

- Python 3.8 or higher: Download from python.org and ensure you select "Add Python to PATH" during installation.
- Git (optional, for cloning the repository): Download from git-scm.com and use the default installation options.
- Pip (Python package installer)
- PyTorch: Based on your system and GPU, download the [CPU or GPU version from PyTorch's official website](https://pytorch.org). The project do not include related installtion commands.

### Windows

Please open Command Prompt or PowerShell and execute the following commands sequentially:

```bash
# Clone the repository
git clone https://github.com/Vangalle/time-series-master.git
cd time-series-master

# Create a virtual environment
python -m venv Time-Series-Master

# Activate the virtual environment
# For Command Prompt:
Time-Series-Master\Scripts\activate

# For PowerShell (if you encounter execution policy restrictions):
# First run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# Then run: .\Time-Series-Master\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt

# Launch the application
streamlit run main_app.py
```

#### Troubleshooting

- **"Python is not recognized as an internal or external command"**: This indicates Python is not added to your PATH. Reinstall Python and ensure you check the "Add Python to PATH" option.

- **PowerShell execution policy errors**: If you encounter restrictions when activating the virtual environment in PowerShell, execute `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` before attempting activation.

- **Package installation failures**: Ensure you have an active internet connection and that your Python version is compatible with the required packages.

#### Alternative Installation Method

If you prefer not to use the command line, you may:

1. Download the project as a ZIP file from GitHub
2. Extract the contents to your preferred location
3. Open the extracted folder
4. Install Python if not already installed
5. Right-click inside the folder while holding Shift and select "Open PowerShell window here"
6. Follow the commands from the "Activate the virtual environment" step onwards

For additional assistance, please refer to the project documentation or submit an issue on the GitHub repository.

### macOS

```bash
# Clone the repository (or download and extract the ZIP file)
git clone https://github.com/Vangalle/time-series-master.git
cd time-series-master

# Create and activate a virtual environment (recommended)
python3 -m venv Time-Series-Master
source Time-Series-Master/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run main_app.py
```

### Linux

```bash
# Clone the repository (or download and extract the ZIP file)
git clone https://github.com/Vangalle/time-series-master.git
cd time-series-master

# Create and activate a virtual environment (recommended)
python3 -m venv Time-Series-Master
source Time-Series-Master/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run main_app.py
```

### Verification

After launching the application, your default web browser should automatically open to `http://localhost:8501`. If not, you can manually navigate to this address to access the Time Series Master interface.

### Troubleshooting

- **PyTorch Installation**: If you encounter issues with PyTorch installation, visit [PyTorch's official website](https://pytorch.org/get-started/locally/) to get the installation command specific to your system configuration.

- **GPU Support**: For accelerated training with GPU:
  - Windows/Linux: Install CUDA Toolkit and appropriate drivers for your NVIDIA GPU
  - macOS: MPS acceleration is automatically used if available on Apple Silicon devices

- **Package Conflicts**: If you encounter dependency conflicts, try creating a fresh virtual environment before installation.

## Dependencies

The application requires the following main libraries:
- streamlit
- pandas
- numpy
- matplotlib
- torch (PyTorch)
- scikit-learn

## Usage Guide

### Navigation

The application is divided into four main sections, accessible via the sidebar:

1. **Home**: Overview and introduction to the platform
2. **Data Selection**: Upload and prepare your time series data
3. **Model Training**: Configure and train forecasting models
4. **Model Evaluation**: Assess model performance and export results

### Data Selection

In this section, you can:

- Upload CSV or Excel files containing time series data
- Preview the data and view summary statistics
- Select input variables (features) and target variables (what you want to predict)
- Visualize relationships between variables with various plot types
- Save your data configuration for model training

The platform automatically handles datetime features, converting them into appropriate numerical representations for time series modeling.

### Model Training

Once your data is prepared, you can:

- Select from built-in models (LSTM, GRU, RNN, Transformer, linear regression)
- Configure model architecture (layers, hidden dimensions)
- Set input and output sequence lengths
- Choose training parameters (learning rate, batch size, epochs)
- Enable early stopping and learning rate scheduling
- Train your model with visualization of training progress

All trained models are automatically saved for later evaluation.

### Model Evaluation

After training, you can:

- Calculate performance metrics (MSE, MAE, RMSE, R², and custom metrics)
- Visualize predictions vs. actual values
- Analyze error distributions
- Export predictions to CSV or Excel for further analysis

## Adding Custom Extensions

### Custom Models

1. Create a Python file in the `custom_models` directory
2. Define a class that inherits from `torch.nn.Module`
3. **(important!) Include docstring with the keyword "linear model" if it's a linear model (otherwise it's treated as a deep learning model)**
4. **(important!) Include docstring or class name with the keyword "transformer" if it's using multi-head attention.**
5. Implement `__init__` and `forward` methods following PyTorch conventions

Example model structure:
```python
class CustomLSTM(nn.Module):
    """A custom LSTM model for time series forecasting.
    
    This model uses a stacked LSTM architecture with attention mechanism.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(CustomLSTM, self).__init__()
        # Model definition here
        
    def forward(self, x):
        # Forward pass implementation
        return output
```

### Custom Metrics

1. Create a Python file in the `custom_metrics` directory
2. Define functions with the signature `metric_name(y_true, y_pred)`
3. Include docstring with "higher is better" or "lower is better" to indicate metric direction
4. Return a single numerical value representing the metric

Example metric:
```python
def metric_rmse(y_true, y_pred):
    """Root Mean Squared Error (RMSE).
    
    Lower is better.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
```

### Custom Correlation Function

- Add the function to the custom_corr/example_corr.py

Example corr function:
```python
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
```

## Directory Structure

```
Time_Series_Master/
├── main_app.py                 # Main application entry point
├── custom_model_loader.py      # Custom model loading utility
├── custom_metrics_loader.py    # Custom metrics loading utility
├── custom_loss_loader.py       # Custom loss function loading utility
├── components/                 # Application components
│   ├── data_loader.py          # Data selection component
│   ├── model_training.py       # Model training component
│   ├── model_evaluation.py     # Model evaluation component
│   └── statistical_models.py   # Model evaluation component
├── custom_models/              # Directory for custom model definitions
│   └── example_models.py       # Example custom models (auto-generated)
├── custom_metrics/             # Directory for custom metric definitions
│   └── example_metrics.py      # Example custom metrics (auto-generated)
├── custom_losses/              # Directory for custom metric definitions
│   └── example_losses.py       # Example custom metrics (auto-generated)
├── custom_corr/                # Directory for custom correlation function definitions
│   └── example_corr.py         # Example custom metrics (predefined)
├── configs/                    # Saved configurations
├── models/                     # Saved trained models
├── exports/                    # Exported predictions
└── plots/                      # Saved visualizations
```

## Example Workflow

1. **Load Data**: Upload a CSV file with historical time series data
2. **Prepare Variables**: Select relevant features as inputs and targets for prediction
3. **Choose Model**: Select an LSTM model with 2 layers and 128 hidden dimensions
4. **Configure Training**: Set sequence length, batch size, and enable early stopping
5. **Train Model**: Train the model and monitor the loss curves
6. **Evaluate Performance**: Analyze predictions against actual values using various metrics
7. **Export Results**: Save predictions for external reporting or analysis

## Troubleshooting

- **Missing components**: Ensure all component files are in the correct directories
- **Import errors**: Check that all dependencies are correctly installed
- **Model errors**: Verify that custom models follow PyTorch conventions
- **Memory issues**: For large datasets, consider reducing batch size or using a subset of data
