import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Import custom metrics loader
from custom_metrics_loader import load_custom_metrics, get_metric_info

# Function to denormalize the predictions and actual values
def denormalize_data(data, mean, std):
    return data * std + mean

def smape_calculator(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Parameters:
    y_true (np.array): Array of true values
    y_pred (np.array): Array of predicted values
    epsilon (float): A small value to avoid division by zero
    
    Returns:
    float: SMAPE value
    """
    # Ensure the epsilon is used to avoid division by zero
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    smape_value = np.mean(np.abs(y_true - y_pred) / denominator)
    return smape_value

# Function to calculate various metrics
def calculate_metrics(y_true, y_pred, custom_metrics=None):
    """
    Calculate various performance metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        custom_metrics: Dictionary of custom metric functions
        
    Returns:
        dict: Dictionary of metric names to values
    """
    # Basic metrics
    metrics = {}
    
    # Mean Squared Error
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    
    # Mean Absolute Error
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
    
    # Symmetric Mean Absolute Percentage Error
    metrics['SMAPE'] = smape_calculator(y_true, y_pred)
    
    # R-squared (Coefficient of Determination)
    metrics['R²'] = r2_score(y_true, y_pred)

    
    # Add custom metrics if provided
    if custom_metrics:
        for metric_name, metric_func in custom_metrics.items():
            try:
                metrics[metric_name] = metric_func(y_true, y_pred)
            except Exception as e:
                st.warning(f"Error calculating metric {metric_name}: {str(e)}")
                metrics[metric_name] = float('nan')
    
    return metrics

def load_available_models():
    """
    Load the list of available trained models.
    
    Returns:
        list: List of model file paths
    """
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob("*.pkl"))
    return sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)

def load_model_from_file(model_file):
    """
    Load a trained model from a file.
    
    Args:
        model_file: Path to the model file
        
    Returns:
        dict: Model data dictionary
    """
    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def export_predictions(predictions, ground_truth, target_vars):
    """
    Export predictions and ground truth to a CSV or Excel file.
    
    Args:
        predictions: Denormalized predictions
        ground_truth: Denormalized ground truth
        target_vars: List of target variable names
        
    Returns:
        pandas.DataFrame: DataFrame with predictions and ground truth
    """
    # Create a dictionary to store the data
    export_data = {}
    
    # For each target variable and time step
    for var_idx, var_name in enumerate(target_vars):
        if predictions.ndim == 2:
            pred_vals = predictions[:, var_idx]
            true_vals = ground_truth[:, var_idx]
        else:
            pred_vals = predictions
            true_vals = ground_truth
        
        # Use timestep in column name if we have multiple timesteps
        export_data[f"{var_name}_true"] = true_vals
        export_data[f"{var_name}_pred"] = pred_vals
    
    # Create DataFrame
    export_df = pd.DataFrame(export_data)
    
    return export_df

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
    Main function to run the model evaluation component.
    This function will be called from the main application.
    """
    add_scroll_to_top()
    # Title and introduction
    st.title("Time Series Model Evaluation")
    st.write("Evaluate model performance and export predictions.")
    
    # Load custom metrics
    custom_metrics = load_custom_metrics()
    
    # Check if we have a trained model in the session state
    has_current_model = (
        st.session_state.trained_model is not None and 
        st.session_state.predictions is not None and 
        st.session_state.ground_truth is not None
    )
    
    # Load available saved models
    available_models = load_available_models()

    st.subheader("Current Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Data Source:** {st.session_state.file_name}")
        st.write(f"**Number of Rows:** {st.session_state.data_Rows}")
        st.write(f"**Number of Columns:** {st.session_state.data_Columns}")

    with col2:
        st.write(f"**Input Variables:** {', '.join(st.session_state.input_vars)}")
        st.write(f"**Target Variables:** {', '.join(st.session_state.target_vars)}")

    # Model selection section
    st.header("Select Model for Evaluation")
    
    model_source = st.radio(
        "Model Source",
        ["Current Session Model", "Saved Model"],
        index=0 if has_current_model else 1
    )
    
    if model_source == "Current Session Model" and not has_current_model:
        st.warning("No model is available in the current session. Please select a saved model or train a new model.")
        model_source = "Saved Model"
    
    # Initialize model data
    model_data = None
    
    if model_source == "Current Session Model":
        # Use the model from the current session
        model_data = {
            "model_name": st.session_state.model_config.get("model_name", "Current Model"),
            "input_vars": st.session_state.input_vars,
            "target_vars": st.session_state.target_vars,
            "norm_params": st.session_state.norm_params,
            "predictions": st.session_state.predictions,
            "ground_truth": st.session_state.ground_truth
        }
        
        st.success(f"Using model from current session: {model_data['model_name']}")
        
    else:
        # Select a saved model
        if not available_models:
            st.error("No saved models found. Please train a model first.")
            return
        
        selected_model_file = st.selectbox(
            "Select a saved model",
            available_models,
            format_func=lambda x: x.name
        )
        
        if st.button("Load Selected Model"):
            model_data = load_model_from_file(selected_model_file)
            
            if model_data:
                st.success(f"Model loaded: {selected_model_file.name}")
                
                # Store model info in session state
                st.session_state.loaded_model_data = model_data
                
                # If we have predictions and ground truth stored
                if "predictions" in model_data and "ground_truth" in model_data:
                    st.session_state.predictions = model_data["predictions"]
                    st.session_state.ground_truth = model_data["ground_truth"]
            else:
                st.error("Failed to load model data. The model file may be corrupted.")
                return
        elif hasattr(st.session_state, 'loaded_model_data'):
            # Use previously loaded model data
            model_data = st.session_state.loaded_model_data
            st.info(f"Using previously loaded model: {selected_model_file.name}")
        else:
            st.info("Click 'Load Selected Model' to load the model data.")
            return
    
    # If we have model data, proceed with evaluation
    if model_data:
        st.header("Model Information")
        
        # Show model details
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Model Name:** {model_data['model_name']}")
            st.write(f"**Input Variables:** {', '.join(model_data['input_vars'])}")
            
        with col2:
            st.write(f"**Target Variables:** {', '.join(model_data['target_vars'])}")
            if "timestamp" in model_data:
                st.write(f"**Training Date:** {model_data['timestamp']}")
        
        # Evaluation options
        st.header("Evaluation Options")
        
        # Get predictions and ground truth
        if hasattr(st.session_state, 'predictions') and hasattr(st.session_state, 'ground_truth'):
            predictions = st.session_state.predictions
            ground_truth = st.session_state.ground_truth
        elif "predictions" in model_data and "ground_truth" in model_data:
            predictions = model_data["predictions"]
            ground_truth = model_data["ground_truth"]
        else:
            st.error("No predictions available for evaluation.")
            return
        # st.write(f"Shape of Predictions: {predictions.shape}")
        # st.write(f"Shape of Ground Truth: {ground_truth.shape}")
        # Determine the shape of predictions
        output_length = predictions.shape[1] if predictions.ndim == 3 else 1
        target_vars = model_data['target_vars']
        # st.write(f"target_vars: {target_vars}")
        
        # Normalization parameters
        norm_params = model_data['norm_params']
        y_mean = norm_params['y_mean']
        y_std = norm_params['y_std']
        # st.write(f"y_mean: {y_mean}")
        # st.write(f"y_std: {y_std}")
        
        # Select which target variable to visualize
        if len(target_vars) > 1:
            target_var_idx = st.selectbox(
                "Select Target Variable to Visualize",
                range(len(target_vars)),
                format_func=lambda x: target_vars[x]
            )
        
        # Prepare data for metrics calculation across ALL time steps
        if predictions.ndim == 3:  # Multi-step prediction
            # Flatten predictions and ground truth for the selected variable
            # This combines all time steps for comprehensive evaluation
            all_pred_values = predictions[:, :, target_var_idx].flatten()
            all_true_values = ground_truth[:, :, target_var_idx].flatten()
            
            # Denormalize
            all_pred_values = denormalize_data(all_pred_values, y_mean[target_var_idx], y_std[target_var_idx])
            all_true_values = denormalize_data(all_true_values, y_mean[target_var_idx], y_std[target_var_idx])
            
        else:  # Single-step prediction
            # For single-step, there's no difference
            all_pred_values = predictions[:, target_var_idx]
            all_true_values = ground_truth[:, target_var_idx]
            
            # Denormalize
            all_pred_values = denormalize_data(all_pred_values, y_mean[target_var_idx], y_std[target_var_idx])
            all_true_values = denormalize_data(all_true_values, y_mean[target_var_idx], y_std[target_var_idx])
        
        # Select metrics to calculate
        st.subheader("Select Metrics")
        
        # Built-in metrics
        builtin_metrics = ["R²", "SMAPE", "MAPE", "RMSE", "MSE", "MAE"]
        selected_builtin_metrics = st.multiselect(
            "Select Built-in Metrics",
            builtin_metrics,
            default=builtin_metrics
        )
        
        # Custom metrics
        selected_custom_metrics = {}
        if custom_metrics:
            custom_metric_keys = st.multiselect(
                "Select Custom Metrics",
                list(custom_metrics.keys()),
                default=[],
                format_func=lambda x: x.split(".")[-1]
            )
            
            for key in custom_metric_keys:
                selected_custom_metrics[key.split(".")[-1]] = custom_metrics[key]
        
        # Create combined metrics dict
        metrics_dict = {}
        for metric in selected_builtin_metrics:
            if metric == "MSE":
                metrics_dict[metric] = lambda y_true, y_pred: mean_squared_error(y_true, y_pred)
            elif metric == "RMSE":
                metrics_dict[metric] = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
            elif metric == "MAE":
                metrics_dict[metric] = lambda y_true, y_pred: mean_absolute_error(y_true, y_pred)
            elif metric == "MAPE":
                metrics_dict[metric] = lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred)
            elif metric == "SMAPE":
                metrics_dict[metric] = lambda y_true, y_pred: smape_calculator(y_true, y_pred)
            elif metric == "R²":
                metrics_dict[metric] = lambda y_true, y_pred: r2_score(y_true, y_pred)
        
        # Add custom metrics
        metrics_dict.update(selected_custom_metrics)

        # st.write(f"Shape of all true values: {all_true_values.shape}")
        # st.write(f"Shape of all predicted values: {all_pred_values.shape}")

        # Calculate metrics
        if metrics_dict and st.button("Calculate Metrics"):
            # Calculate metrics across ALL time steps
            metrics_results = calculate_metrics(all_true_values, all_pred_values, metrics_dict)
            
            # Display metrics in a table
            st.subheader("Evaluation Metrics (Across All Time Steps)")
            
            metrics_df = pd.DataFrame({
                'Metric': list(metrics_results.keys()),
                'Value': list(metrics_results.values())
            })
            
            st.table(metrics_df)
            
            # Store metrics in session state
            st.session_state.current_metrics = metrics_results
            
            if output_length > 1:
                st.info("Metrics are calculated across all time steps for comprehensive evaluation.")
        
        # Visualization section
        st.header("Visualization")
        
        # Plot options
        plot_type = st.selectbox(
            "Select Plot Type",
            ["Time Series Plot", "Scatter Plot", "Residual Plot", "Error Distribution"]
        )
        
        # Additional options based on plot type
        if plot_type == "Time Series Plot":
            plot_window = st.slider(
                "Window Size (0 for all data)",
                0, len(all_pred_values), 0
            )
        
        elif plot_type == "Error Distribution":
            num_bins = st.slider("Number of Bins", 10, 100, 30)
        
        # Initialize session state for the figure if it doesn't exist
        if 'fig' not in st.session_state:
            st.session_state.fig = None

        # Generate the plot
        if st.button("Generate Plot"):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if plot_type == "Time Series Plot":
                if plot_window > 0:
                    # Plot a window of the data
                    plot_start = max(0, len(all_pred_values) - plot_window)
                    plot_end = len(all_pred_values)

                    ax.plot(range(plot_start, plot_end), all_true_values[plot_start:plot_end], label='Actual')
                    ax.plot(range(plot_start, plot_end), all_pred_values[plot_start:plot_end], label='Predicted', alpha=0.7)
                    ax.set_xlabel('Sample')
                else:
                    # Plot all data
                    ax.plot(all_true_values, label='Actual')
                    ax.plot(all_pred_values, label='Predicted', alpha=0.7)
                    ax.set_xlabel('Sample')
                
                ax.set_ylabel(target_vars[target_var_idx])
                ax.set_title(f'Time Series Comparison: {target_vars[target_var_idx]}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            elif plot_type == "Scatter Plot":
                ax.scatter(all_true_values, all_pred_values, alpha=0.7)
                
                # Add perfect prediction line
                min_val = min(np.min(all_true_values), np.min(all_pred_values))
                max_val = max(np.max(all_true_values), np.max(all_pred_values))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                
                
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'Predicted vs Actual: {target_vars[target_var_idx]}')
                ax.grid(True, alpha=0.3)
            
            elif plot_type == "Residual Plot":
                residuals = all_pred_values - all_true_values

                ax.scatter(all_true_values, residuals, alpha=0.7)
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Residuals (Predicted - Actual)')
                ax.set_title(f'Residual Plot: {target_vars[target_var_idx]}')
                ax.grid(True, alpha=0.3)
            
            elif plot_type == "Error Distribution":
                residuals = all_pred_values - all_true_values

                ax.hist(residuals, bins=num_bins, alpha=0.7)
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            
                
                ax.set_xlabel('Prediction Error (Predicted - Actual)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Error Distribution: {target_vars[target_var_idx]}')
                ax.grid(True, alpha=0.3)
            
            # Store the figure in session state
            st.session_state.fig = fig
            
            # Display the plot
            st.pyplot(fig)

        # Save Plot button (outside the Generate Plot button block)
        if st.button("Save Plot"):
            if st.session_state.fig is not None:
                # Create directory if it doesn't exist
                os.makedirs("plots", exist_ok=True)
                
                # Save the plot
                plot_file = f"plots/{plot_type.replace(' ', '_').lower()}_{target_vars[target_var_idx]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                st.session_state.fig.savefig(plot_file, dpi=300, bbox_inches="tight")
                st.success(f"Plot saved as {plot_file}")
            else:
                st.warning("Please generate a plot first.")
        
        # Export predictions section
        st.header("Export Predictions")
        
        # Denormalize all predictions and ground truth
        
        # Prepare data for metrics calculation across ALL time steps
        if predictions.ndim == 3:  # Multi-step prediction
            # Flatten predictions and ground truth for the selected variable
            # This combines all time steps for comprehensive evaluation
            all_pred_values = predictions.reshape(-1, predictions.shape[2])
            all_true_values = ground_truth.reshape(-1, ground_truth.shape[2])
            
            # Denormalize
            all_pred_values = denormalize_data(all_pred_values, y_mean, y_std)
            all_true_values = denormalize_data(all_true_values, y_mean, y_std)
            
        else:  # Single-step prediction
            # For single-step, there's no difference
            all_pred_values = predictions[:, target_var_idx]
            all_true_values = ground_truth[:, target_var_idx]

            # Denormalize
            all_pred_values = denormalize_data(all_pred_values, y_mean, y_std)
            all_true_values = denormalize_data(all_true_values, y_mean, y_std)

        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            file_format = st.selectbox(
                "Export Format",
                ["CSV", "Excel"],
                index=0
            )
        
            export_separate = st.checkbox("Export All Predictions Separately", value=False)

        with col2:
            if export_separate:
                default_folder = f"predictions_{model_data['model_name']}_{datetime.now().strftime('%Y%m%d')}"
                export_filename = st.text_input("Foldername", default_folder)
            else:
                default_filename = f"predictions_{model_data['model_name']}_{datetime.now().strftime('%Y%m%d')}"
                export_filename = st.text_input("Filename (without extension)", default_filename)
        
        if st.button("Export Predictions"):
            # Create directory for exports
            os.makedirs("exports", exist_ok=True)
            if export_separate:
                # Export each target variable separately
                for var_idx, var_name in enumerate(target_vars):
                    export_df = export_predictions(
                        all_pred_values[:, var_idx],
                        all_true_values[:, var_idx],
                        [var_name]
                    )

                    # Export path
                    if file_format == "CSV":
                        export_path = f"exports/{export_filename}/{var_name}.csv"
                        os.makedirs(os.path.dirname(export_path), exist_ok=True)
                        export_df.to_csv(export_path, index=False)
                    else:
                        export_path = f"exports/{export_filename}/{var_name}.xlsx"
                        os.makedirs(os.path.dirname(export_path), exist_ok=True)
                        export_df.to_excel(export_path, index=False)
                    
                    st.success(f"Predictions for {var_name} exported to {export_path}")
            else:
                # Create export dataframe
                export_df = export_predictions(
                    all_pred_values,
                    all_true_values,
                    target_vars
                )

                # Export path
                if file_format == "CSV":
                    export_path = f"exports/{export_filename}.csv"
                    export_df.to_csv(export_path, index=False)
                else:
                    export_path = f"exports/{export_filename}.xlsx"
                    export_df.to_excel(export_path, index=False)
            
                st.success(f"Predictions exported to {export_path}")
            
                # Show preview of the exported data
                st.subheader("Preview of Exported Data")
                st.dataframe(export_df.head(10))

# If run directly outside the main application
if __name__ == "__main__":
    run()