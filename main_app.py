import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import importlib
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Time Series Master",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application structure
APP_DIR = Path(__file__).parent
COMPONENTS_DIR = APP_DIR / "components"
CUSTOM_MODELS_DIR = APP_DIR / "custom_models"
CUSTOM_METRICS_DIR = APP_DIR / "custom_metrics"
CUSTOM_LOSSES_DIR = APP_DIR / "custom_losses"

# Create required directories
os.makedirs(COMPONENTS_DIR, exist_ok=True)
os.makedirs(CUSTOM_MODELS_DIR, exist_ok=True)
os.makedirs(CUSTOM_METRICS_DIR, exist_ok=True)
os.makedirs(CUSTOM_LOSSES_DIR, exist_ok=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_rows' not in st.session_state:
    st.session_state.data_Rows = None
if 'data_columns' not in st.session_state:
    st.session_state.data_Columns = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'input_vars' not in st.session_state:
    st.session_state.input_vars = []
if 'input_vars_default' not in st.session_state:
    st.session_state.input_vars_default = []
if 'target_vars' not in st.session_state:
    st.session_state.target_vars = []
if 'target_vars_default' not in st.session_state:
    st.session_state.target_vars_default = []
if 'config' not in st.session_state:
    st.session_state.config = {}
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'ground_truth' not in st.session_state:
    st.session_state.ground_truth = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'datetime_column' not in st.session_state:
    st.session_state.datetime_column = False
if 'uses_positional_encoding' not in st.session_state:
    st.session_state.uses_positional_encoding = False
if 'positional_encoding_type' not in st.session_state:
    st.session_state.positional_encoding_type = "Sinusoidal"
if 'is_statistical_model' not in st.session_state:
    st.session_state.is_statistical_model = None

if 'persistent_model_type' not in st.session_state:
    st.session_state.persistent_model_type = "Built-in"
if 'input_length' not in st.session_state:
    st.session_state.input_length = 12
if 'output_length' not in st.session_state:
    st.session_state.output_length = 12
if 'loss_function' not in st.session_state:
    st.session_state.loss_function = "L2Loss"
if 'seletced_custom_loss' not in st.session_state:
    st.session_state.seletced_custom_loss = 0
if 'train_ratio' not in st.session_state:
    st.session_state.train_ratio = None
if 'val_ratio' not in st.session_state:
    st.session_state.val_ratio = None
if 'persistent_builtin_model_type' not in st.session_state:
    st.session_state.persistent_builtin_model_type = "Deep Learning"
if 'persistent_dl_model_type' not in st.session_state:
    st.session_state.persistent_dl_model_type = "LSTM"
if 'persistent_linear_model_type' not in st.session_state:
    st.session_state.persistent_linear_model_type = "Direct Linear Projection Network"
if 'learning_rate' not in st.session_state:
    st.session_state.learning_rate = 0.001
if 'batch_size' not in st.session_state:
    st.session_state.batch_size = 64
if 'epochs' not in st.session_state:
    st.session_state.epochs = 200
if 'num_layers' not in st.session_state:
    st.session_state.num_layers = 2
if 'hidden_dim' not in st.session_state:
    st.session_state.hidden_dim = 64
if 'use_early_stopping' not in st.session_state:
    st.session_state.use_early_stopping = True
if 'patience' not in st.session_state:
    st.session_state.patience = 15
if 'lr_scheduler_index' not in st.session_state:
    st.session_state.lr_scheduler_index = 0
if 'd_model' not in st.session_state:
    st.session_state.d_model = 64
if 'positional_encoding_index' not in st.session_state:
    st.session_state.positional_encoding_index = 0

# Functions to navigate between pages
def nav_to(page):
    st.session_state.page = page

# Import components dynamically
def import_component(component_name):
    """Import a component module dynamically."""
    try:
        # Check if the file exists in the components directory
        component_file = COMPONENTS_DIR / f"{component_name}.py"
        if not component_file.exists():
            st.error(f"Component file '{component_name}.py' not found in components directory.")
            return None
        
        # Add components directory to path if not already there
        components_path = str(COMPONENTS_DIR.absolute())
        if components_path not in sys.path:
            sys.path.append(components_path)
        
        # Import the module
        module = importlib.import_module(component_name)
        return module
    
    except Exception as e:
        st.error(f"Error importing component '{component_name}': {str(e)}")
        return None

def add_scroll_to_top():
    # JavaScript to scroll to top on page load
    js = '''
    <script>
        window.addEventListener('load', function() {
            window.scrollTo(0, 0);
        });
    </script>
    '''
    st.markdown(js, unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("Time Series Master")
    
    # Navigation buttons
    st.sidebar.header("Navigation")
    
    if st.sidebar.button("Home", key="nav_home"):
        nav_to("home")
    
    if st.sidebar.button("Data Selection", key="nav_data"):
        nav_to("data_selection")
    
    if st.sidebar.button("Model Training", key="nav_model"):
        if st.session_state.data is not None and st.session_state.input_vars and st.session_state.target_vars:
            nav_to("model_training")
        else:
            st.sidebar.warning("Please select data and variables first.")
    
    if st.sidebar.button("Model Evaluation", key="nav_eval"):
        nav_to("model_evaluation")
        # if st.session_state.trained_model is not None:
        #     nav_to("model_evaluation")
        # else:
        #     st.sidebar.warning("Please train a model first.")
    
    if st.session_state.page != "data_selection":
        
        # Custom model and metrics management
        st.sidebar.markdown("<hr style='margin: 0.5em 0'>", unsafe_allow_html=True)
        st.sidebar.header("Advanced Options")
        
        with st.sidebar.expander("Custom Models"):
            st.write("Add your custom model .py files to the 'custom_models' directory.")
            
            if st.button("Create Example Model File"):
                from custom_model_loader import create_example_model_file
                example_file = create_example_model_file()
                st.success(f"Created example model file: {example_file}")
            
            if st.button("Refresh Custom Models"):
                st.rerun()
        
        with st.sidebar.expander("Custom Metrics"):
            st.write("Add your custom metric .py files to the 'custom_metrics' directory.")
            
            if st.button("Create Example Metrics File"):
                from custom_metrics_loader import create_example_metrics_file
                example_file = create_example_metrics_file()
                st.success(f"Created example metrics file: {example_file}")
            
            if st.button("Refresh Custom Metrics"):
                st.rerun()
        
        with st.sidebar.expander("Custom Losses"):
            st.write("Add your custom loss .py files to the 'custom_losses' directory.")
            
            if st.button("Create Example Loss File"):
                from custom_loss_loader import create_example_loss_file
                example_file = create_example_loss_file()
                st.success(f"Created example loss file: {example_file}")
            
            if st.button("Refresh Custom Losses"):
                st.rerun()

# Main content area
if st.session_state.page == "home":
    st.title("Welcome to Time Series Master")
    
    st.markdown("""
    ## An Easy-to-Use Time Series Forecasting Platform
    
    This application provides a user-friendly interface for time series forecasting with multiple model options.
    
    ### Features
    
    - **Data Selection**: Upload time series data and select input/target variables
    - **Model Training**: Choose from various models including RNN, LSTM, GRU, Transformer, and linear models
    - **Custom Models**: Add your own model implementations
        - Deep Learning Models Input Parameters:
            - Non Transformer: [input_dim, hidden_dim, output_dim, num_layers, input_length, output_length]
            - Transformer: [input_dim, d_model, num_heads, num_layers, hidden_dim, output_dim, input_length, 
                           output_length, encoding_type=st.session_state.positional_encoding_type]
    - **Custom Metrics**: Define and use your own evaluation metrics
        - Function Name should start with `metric_`
    - **Custom Losses** Define and use your own loss function
    - **Custom Correlation Function** Define and use your own correlation function to apply them to feature selction.
    - **Model Evaluation**: Evaluate model performance and export predictions
    
    ### Getting Started
    
    1. Click on **Data Selection** in the sidebar to load your data and select variables
    2. Proceed to **Model Training** to configure and train a forecasting model
    3. Use **Model Evaluation** to assess performance and export results
    
    ### Custom Extensions
    
    - To add custom models, place Python files in the `custom_models` directory
    - To add custom metrics, place Python files in the `custom_metrics` directory
    - To add custom losses, place Python files in the `custom_losses` directory
    """)
    
    # Example data section
    with st.expander("Don't have data? Use example data"):
        st.write("You can load example data from the Data Selection page.")

elif st.session_state.page == "data_selection":
    # Load and run the data selection component
    data_selection = import_component("data_loader")
    if data_selection:
        # We're dynamically importing and running the data_loader component
        # The component will access and modify session state directly
        data_selection.run()
    else:
        st.error("Failed to load data selection component.")
        
        # Fallback: Basic file uploader
        st.subheader("Basic File Upload (Fallback)")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                # Detect file type and read accordingly
                file_extension = uploaded_file.name.split(".")[-1].lower()
                
                if file_extension == "csv":
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                # Update session state
                st.session_state.data = data
                st.session_state.file_name = uploaded_file.name
                
                # Show file info
                st.success(f"File loaded: {uploaded_file.name}")
                st.info(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
                
                # Data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Simple variable selection
                st.subheader("Variable Selection")
                
                input_vars = st.multiselect("Select Input Variables", data.columns)
                target_vars = st.multiselect("Select Target Variables", 
                                            [col for col in data.columns if col not in input_vars])
                
                if st.button("Save Selection"):
                    st.session_state.input_vars = input_vars
                    st.session_state.target_vars = target_vars
                    st.session_state.config = {
                        "file_name": uploaded_file.name,
                        "input_variables": input_vars,
                        "target_variables": target_vars
                    }
                    st.success("Selection saved")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")

elif st.session_state.page == "model_training":
    add_scroll_to_top()
    # Load and run the model training component
    model_training = import_component("model_training")
    if model_training:
        # We're dynamically importing and running the model_training component
        # The component will access and modify session state directly
        model_training.run()
    else:
        st.error("Failed to load model training component.")
        
        # Display a simple message
        st.subheader("Model Training (Not Available)")
        st.write("The model training component could not be loaded.")
        st.write("Please ensure that the 'model_training.py' file exists in the components directory.")

elif st.session_state.page == "model_evaluation":
    add_scroll_to_top()
    # Load and run the model evaluation component
    model_evaluation = import_component("model_evaluation")
    model_evaluation.run()
    # if model_evaluation:
    #     # We're dynamically importing and running the model_evaluation component
    #     # The component will access and modify session state directly
    #     model_evaluation.run()
    # else:
    #     st.error("Failed to load model evaluation component.")
        
    #     # Display a simple message if the trained model exists
    #     if st.session_state.trained_model is not None:
    #         st.subheader("Model Evaluation (Not Available)")
    #         st.write("The model evaluation component could not be loaded.")
    #         st.write("However, a trained model exists in the session state.")
            
    #         if st.session_state.predictions is not None and st.session_state.ground_truth is not None:
    #             st.subheader("Basic Performance Metrics")
                
    #             # Calculate basic metrics
    #             mse = np.mean((st.session_state.predictions.flatten() - st.session_state.ground_truth.flatten()) ** 2)
    #             mae = np.mean(np.abs(st.session_state.predictions.flatten() - st.session_state.ground_truth.flatten()))
                
    #             col1, col2 = st.columns(2)
    #             col1.metric("Mean Squared Error", f"{mse:.4f}")
    #             col2.metric("Mean Absolute Error", f"{mae:.4f}")
    #     else:
    #         st.warning("No trained model available for evaluation.")
