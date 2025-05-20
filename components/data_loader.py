import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

# Import the custom correlation loader
try:
    from custom_corr_loader import get_custom_corr_loader, get_correlation_method
    CUSTOM_CORR_AVAILABLE = True
except ImportError:
    CUSTOM_CORR_AVAILABLE = False
    st.warning("Custom correlation loader not found. Only built-in correlation methods will be available.")

def prepare_dataframe_for_streamlit(df):
    """Prepare DataFrame for Streamlit to prevent Arrow conversion errors"""
    # Create a deep copy to avoid modifying the original
    df_copy = df.copy()
    
    # Process all columns to ensure Arrow compatibility
    for col in df_copy.columns:
        # Handle numeric columns
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            # Replace infinities and coerce to float64 for consistency
            df_copy[col] = df_copy[col].replace([np.inf, -np.inf], np.nan)
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Handle datetime columns
        elif pd.api.types.is_datetime64_dtype(df_copy[col]):
            try:
                # Convert to datetime with no timezone
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce').dt.tz_localize(None)
            except:
                # If conversion fails, convert to string as fallback
                df_copy[col] = df_copy[col].astype(str)
        
        # Handle all other columns by converting to string
        else:
            df_copy[col] = df_copy[col].astype(str)
    
    return df_copy

def calculate_correlation_matrix(data, method_name, corr_loader=None):
    """
    Calculate correlation matrix using built-in or custom correlation methods
    
    Args:
        data: DataFrame with numeric data
        method_name: Name of the correlation method
        corr_loader: CustomCorrLoader instance (optional)
    
    Returns:
        Correlation matrix DataFrame
    """
    try:
        if method_name in ["Pearson", "Spearman", "Kendall"]:
            # Use built-in pandas correlation methods
            return data.corr(method=method_name.lower())
        elif CUSTOM_CORR_AVAILABLE and corr_loader:
            # Use custom correlation function
            custom_func = corr_loader.get_correlation_function(method_name)
            if custom_func:
                return data.corr(method=custom_func)
            else:
                st.error(f"Custom correlation function '{method_name}' not found")
                return None
        else:
            st.error(f"Correlation method '{method_name}' not supported")
            return None
    except Exception as e:
        st.error(f"Error calculating correlation matrix: {e}")
        return None

def run():
    """
    Main function to run the data loader component.
    This function will be called from the main application.
    """
    # Initialize custom correlation loader
    corr_loader = None
    if CUSTOM_CORR_AVAILABLE:
        corr_loader = get_custom_corr_loader()
    
    # Title and introduction
    st.title("Time Series Data Selection")
    st.write("Upload your time series data and select input variables and targets.")

    # Create a sidebar for options
    with st.sidebar:
        st.header("Data Options")
        
        # Add refresh button for custom correlation functions
        if CUSTOM_CORR_AVAILABLE:
            if st.button("🔄 Refresh Custom Correlations", help="Reload custom correlation functions"):
                corr_loader.refresh()
                st.success("Custom correlation functions refreshed!")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                # Detect file type and read accordingly
                file_extension = uploaded_file.name.split(".")[-1].lower()
                
                if file_extension == "csv":
                    # Provide additional CSV options
                    csv_separator = st.selectbox("CSV Separator", [",", ";", "\t", "|", " "], index=0)
                    
                    # Handle different date formats
                    date_format = st.selectbox(
                        "Date Format (if applicable)",
                        ["Auto-detect", "yyyy-mm-dd", "mm/dd/yyyy", "dd/mm/yyyy"],
                        index=0
                    )
                    
                    # Read the CSV file with the selected separator
                    data = pd.read_csv(uploaded_file, sep=csv_separator)
                else:
                    # For Excel files, show sheet selection
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_name = st.selectbox("Select Sheet", excel_file.sheet_names)
                    data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                
                data = prepare_dataframe_for_streamlit(data)
                
                # Process potential date columns
                for col in data.columns:
                    if data[col].dtype == 'object':
                        # Try to convert to datetime
                        try:
                            data[col] = pd.to_datetime(data[col])
                            st.sidebar.info(f"Column '{col}' converted to datetime format.")
                        except:
                            pass
                
                # Update session state
                st.session_state.data = data
                st.session_state.file_name = uploaded_file.name
                
                # Show file info
                st.sidebar.success(f"File loaded: {uploaded_file.name}")
                st.sidebar.info(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
                st.session_state.target_vars = []
                
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                st.session_state.data = None
                st.session_state.file_name = None

        # Initialize import_mode in session state if it doesn't exist
        if 'import_mode' not in st.session_state:
            st.session_state.import_mode = False
        
        # Toggle button to enter/exit import mode
        if st.button("Import Saved Configuration"):
            st.session_state.import_mode = not st.session_state.import_mode
        
        # If in import mode, show the configuration selection interface
        if st.session_state.import_mode:
            # List available data configurations
            data_configs_dir = Path("data_configs")
            if data_configs_dir.exists():
                # Find all config.json files within subdirectories
                config_files = list(data_configs_dir.glob("*/config.json"))
                if config_files:
                    # Sort by modification time (newest first)
                    config_files = sorted(config_files, key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    # Format directory names nicely for selection
                    config_labels = [f"{x.parent.name} ({datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})" 
                                    for x in config_files]
                    
                    # Let user select a configuration
                    config_index = st.selectbox(
                        "Select a saved configuration",
                        range(len(config_files)),
                        format_func=lambda i: config_labels[i]
                    )
                    
                    if st.button("Load Selected Configuration"):
                        selected_config = config_files[config_index]
                        # Load the configuration
                        try:
                            with open(selected_config, "r") as f:
                                config = json.load(f)
                            
                            # Look for the data file in the same directory
                            config_dir = selected_config.parent
                            data_files = list(config_dir.glob("data.*"))
                            
                            if data_files:
                                imported_file = data_files[0]
                                if imported_file.suffix == ".csv":
                                    data = pd.read_csv(imported_file)
                                else:
                                    data = pd.read_excel(imported_file)
                                
                                # Try to convert string columns to datetime if they were datetime in original
                                for col in data:
                                    if col in data and data[col].dtype == 'object':
                                        try:
                                            data[col] = pd.to_datetime(data[col])
                                        except:
                                            pass
                                
                                # Update session state
                                st.session_state.data = data
                                st.session_state.file_name = config.get("file_name")
                                st.session_state.input_vars_default = config.get("input_variables", [])
                                st.session_state.target_vars_default = config.get("target_variables", [])
                                
                                # Exit import mode
                                st.session_state.import_mode = False
                                
                                st.success("Configuration and data loaded successfully!")
                                st.rerun()
                            else:
                                st.warning("Could not find associated data file in the configuration directory.")
                        except Exception as e:
                            st.error(f"Error loading configuration: {e}")
                else:
                    st.info("No saved configurations found.")
                    if st.button("Exit Import Mode"):
                        st.session_state.import_mode = False
                        st.rerun()
            else:
                st.info("No data_configs directory found.")
                if st.button("Exit Import Mode"):
                    st.session_state.import_mode = False
                    st.rerun()
    
    # Main panel for data preview and variable selection
    if st.session_state.data is not None:

        data = st.session_state.data
        
        # Data preview section
        st.subheader("Data Preview")
        
        # Show number of rows to preview
        num_rows = st.slider("Number of rows to preview", 5, 100, 10)
        st.dataframe(data.head(num_rows))

        numeric_data = data.select_dtypes(include=['number'])
        
        # Data statistics toggle
        if st.checkbox("Show Data Statistics"):
            st.subheader("Data Statistics")
            st.write(data.describe())
            if not numeric_data.empty:
                # Get available correlation methods
                if CUSTOM_CORR_AVAILABLE and corr_loader:
                    correlation_options = corr_loader.get_all_correlation_options()
                else:
                    correlation_options = ["Pearson", "Spearman", "Kendall"]
                
                # Display info about custom correlations
                if CUSTOM_CORR_AVAILABLE and corr_loader:
                    custom_funcs = corr_loader.get_custom_correlation_functions()
                    if custom_funcs:
                        with st.expander(f"📊 Custom Correlation Functions ({len(custom_funcs)} available)"):
                            for name in custom_funcs.keys():
                                st.write(f"• {name}")
                            st.info("Add more custom correlation functions by placing them in the `custom_corr/` directory!")
                
                # Correlation method selection
                correlation_method = st.selectbox(
                    "Select Correlation Method", 
                    correlation_options, 
                    index=0,
                    help="Select built-in correlation methods or custom functions from the custom_corr directory"
                )
                
                # Calculate correlation matrix
                correlation_matrix = calculate_correlation_matrix(numeric_data, correlation_method, corr_loader)
                
                if correlation_matrix is not None:
                    # Calculate appropriate figure size and font size based on number of columns
                    num_cols = len(correlation_matrix.columns)
                    
                    # Dynamically adjust figure size based on column count
                    fig_size = num_cols
                    
                    # Dynamically adjust font size based on column count
                    font_size = fig_size + 2
                    
                    # Annotation size slightly smaller than font size
                    annot_size = max(6, font_size - 2)

                    # Create correlation heatmap with seaborn
                    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
                    heatmap = sns.heatmap(correlation_matrix, xticklabels=correlation_matrix.columns, 
                                          yticklabels=correlation_matrix.columns, annot=True, fmt=".2f", 
                                          cmap='PRGn', vmin=-1, vmax=1, annot_kws={"size": annot_size})

                    # Apply font size to tick labels
                    plt.xticks(rotation=45, ha='right', fontsize=font_size)
                    plt.yticks(rotation=0, fontsize=font_size)
                    
                    # Adjust colorbar font size
                    cbar = heatmap.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=font_size)

                    plt.title(f"{correlation_method} Correlation Heatmap", fontsize=font_size + 3, loc='center')
                    plt.tight_layout()

                    # Display the plot
                    st.pyplot(plt)
                    import io
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download Correlation Matrix",
                            data=correlation_matrix.to_csv().encode('utf-8'),
                            file_name=f'{correlation_method.lower()}_correlation_matrix.csv',
                            mime='text/csv'
                        )
                    with col2:
                        os.makedirs("plots", exist_ok=True)
                        st.download_button(
                            label="Download Correlation Heatmap",
                            data=buffer,
                            file_name=f'{correlation_method.lower()}_correlation_heatmap.png',
                            mime='image/png'
                        )
        
        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': data.columns,
            'Type': [str(dtype) for dtype in data.dtypes],
            'Non-Null Count': data.count(),
            'Null Count': data.isna().sum(),
            'Unique Values': [data[col].nunique() for col in data.columns]
        })
        st.dataframe(col_info)
        
        # Column selection section
        st.subheader("Variable Selection")

        # Get numeric columns for input selection
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Force Arrow compatibility for all columns
        for col in data.columns:
            # Handle datetime columns that might cause issues
            if col in numeric_cols:
                # Ensure numeric format is compatible
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                data[col] = pd.to_numeric(data[col], errors='coerce')

            if col in datetime_cols:
                # Ensure datetime format is compatible (strip timezone if present)
                data[col] = pd.to_datetime(data[col], errors='coerce').dt.tz_localize(None)
            
            # Handle the 'Type' column specifically if it exists
            if col in categorical_cols:
                data[col] = data[col].astype(str)
        
        # Create two columns for input and target selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Select Input Variables:")
            
            # Organize columns by type
            numeric_defaults = [col for col in st.session_state.input_vars_default if col in numeric_cols]
            st.write("Numeric Variables:")
            # Filter default values to only include variables that exist in the options
            if len(numeric_defaults)>0:
                numeric_inputs = st.multiselect(
                    "Select numeric input variables",
                    numeric_cols,
                    default=numeric_defaults
                )
            else:
                numeric_inputs = st.multiselect(
                    "Select numeric input variables",
                    numeric_cols
                )
            
            if datetime_cols:
                st.write("Date/Time Variables:")
                # Filter default values for datetime columns
                datetime_defaults = [col for col in st.session_state.input_vars_default if col in datetime_cols]
                if len(datetime_defaults)>0:
                    datetime_inputs = st.multiselect(
                        "Select datetime input variables",
                        datetime_cols,
                        default=datetime_defaults
                    )
                else:
                    datetime_inputs = st.multiselect(
                        "Select datetime input variables",
                        datetime_cols
                    )
            else:
                datetime_inputs = []
                
            if categorical_cols:
                st.write("Categorical Variables:")
                # Filter default values for categorical columns
                categorical_defaults = [col for col in st.session_state.input_vars_default if col in categorical_cols]
                if len(categorical_defaults)>0:
                    categorical_inputs = st.multiselect(
                        "Select categorical input variables",
                        categorical_cols,
                        default=categorical_defaults
                    )
                else:
                    categorical_inputs = st.multiselect(
                        "Select categorical input variables",
                        categorical_cols
                    )
            else:
                categorical_inputs = []
            
            # Combine all selected inputs
            if len(datetime_inputs) == 0:
                st.session_state.datetime_column = False
            else:
                st.session_state.datetime_column = True
            selected_inputs = numeric_inputs + datetime_inputs + categorical_inputs
            
            
            # Show number of selected inputs
            st.info(f"Selected {len(selected_inputs)} input variables")
        
        with col2:
            st.write("Select Target Variables:")
            
            # Usually targets are numeric, but allow any type
            target_options = data.columns.tolist()
            # Remove already selected inputs
            available_targets = [col for col in target_options if col not in datetime_cols and col not in categorical_cols]
            if len(st.session_state.target_vars_default)>0:
                selected_targets = st.multiselect(
                    "Select target variables",
                    available_targets,
                    default=st.session_state.target_vars_default
                )
            else:
                selected_targets = st.multiselect(
                    "Select target variables",
                    available_targets
                )
            
            # Show number of selected targets
            st.info(f"Selected {len(selected_targets)} target variables")
        
        # Plotting section
        st.subheader("Data Visualization")
        
        # Options for plotting
        plot_options = st.expander("Plot Options", expanded=False)
        
        with plot_options:

            col0, col1 = st.columns(2)

            with col0:
                plot_inputs = st.checkbox("Plot Input Variables", value=True)
                plot_targets = st.checkbox("Plot Target Variables", value=True)
            with col1:
                turn_to_log = st.checkbox("Take the logarithm of the data", value=False)
                if turn_to_log:
                    log_base = st.selectbox("Log base", ["Natural log (ln)", "Log base 10", "Log base 2"], index=1)

            # Variables to plot
            vars_to_plot = []
            if plot_inputs and selected_inputs:
                vars_to_plot.extend(selected_inputs)
            if plot_targets and selected_targets:
                vars_to_plot.extend(selected_targets)
            
            vars_to_plot = list(dict.fromkeys(vars_to_plot))

            not_to_generate = False

            # Plot type selection
            plot_type = st.selectbox(
                "Select Plot Type",
                ["Line Plot", "Scatter Plot", "Box Plot", "Histogram", "Heatmap (Correlation)"]
            )
            
            # Additional options based on plot type
            if plot_type == "Line Plot":
                use_index_as_x = st.checkbox("Use DataFrame Index as X-axis", value=False)
                x_axis_col = None
                if not use_index_as_x:
                    # Let user select x-axis column
                    x_axis_options = data.columns.tolist()
                    x_axis_col = st.selectbox("Select X-axis column", x_axis_options)
            
            elif plot_type == "Scatter Plot":
                # Add option for distribution view
                use_constant_x = st.checkbox("Use constant X-axis (for distribution view)", 
                                            help="Plot all variables at x=0 to see their distributions")
                
                if not use_constant_x:
                    # For scatter plots, user needs to select x and y variables
                    scatter_x = st.selectbox("Select X-axis variable", 
                                        data.select_dtypes(include=['number']).columns.tolist())
                    
                    # Filter out X variable from Y variables and only keep numeric ones
                    y_vars = [var for var in vars_to_plot 
                            if var != scatter_x and np.issubdtype(data[var].dtype, np.number)]
                    
                    # Check for non-numeric variables that were filtered out
                    non_numeric_y_vars = [var for var in vars_to_plot 
                                        if var != scatter_x and not np.issubdtype(data[var].dtype, np.number)]
                    
                    if non_numeric_y_vars:
                        st.warning(f"⚠️ Non-numeric Y variables will be skipped: {', '.join(non_numeric_y_vars)}")
                    
                    if not y_vars:
                        not_to_generate = True
                        if vars_to_plot:
                            st.warning(f"⚠️ No valid numeric Y variables available after filtering.")
                            st.info("Please select additional numeric variables or choose a different X-axis variable.")
                        else:
                            st.info("Please select some input/target variables to plot on the Y-axis.")
                    else:
                        not_to_generate = False
                        if len(vars_to_plot) != len(y_vars):
                            removed_count = len(vars_to_plot) - len(y_vars)
                            st.info(f"ℹ️ Filtered out {removed_count} variable(s) (X-axis or non-numeric) from Y-axis.")
                else:
                    # For distribution view, use all numeric variables
                    y_vars = [var for var in vars_to_plot if np.issubdtype(data[var].dtype, np.number)]
                    non_numeric_vars = [var for var in vars_to_plot if not np.issubdtype(data[var].dtype, np.number)]
                    
                    if non_numeric_vars:
                        st.warning(f"⚠️ Non-numeric variables will be skipped: {', '.join(non_numeric_vars)}")
                    
                    if not y_vars:
                        not_to_generate = True
                        st.warning("⚠️ No numeric variables available for distribution view.")
                    else:
                        not_to_generate = False
                        st.info(f"ℹ️ Distribution view will show {len(y_vars)} variable(s) at constant X position.")
                        
                        # Add jitter option for better visualization
                        add_jitter = st.checkbox("Add horizontal jitter", value=False, 
                                            help="Adds small random horizontal offset to avoid overlapping points")

                
            elif plot_type == "Histogram":
                hist_bins = st.slider("Number of bins", 5, 100, 20)
                
            # Plot size
            if not not_to_generate:
                col1, col2 = st.columns(2)
                with col1:
                    plot_width = st.slider("Plot Width", 6, 20, 12)
                with col2:
                    plot_height = st.slider("Plot Height", 4, 15, 6)
            
            if plot_type == "Heatmap (Correlation)":
                # Get correlation options for plotting
                if CUSTOM_CORR_AVAILABLE and corr_loader:
                    plot_correlation_options = corr_loader.get_all_correlation_options()
                else:
                    plot_correlation_options = ["Pearson", "Spearman", "Kendall"]
                
                heatmap_type = st.selectbox(
                    "Select Heatmap Type", 
                    plot_correlation_options, 
                    index=0,
                    help="Choose correlation method for the heatmap"
                )
            
            if not not_to_generate:
                generate_plot = st.button("Generate Plot")
            else:
                generate_plot = False
            
            if generate_plot:
                    
                if vars_to_plot:
                    try:
                        plt.figure(figsize=(plot_width, plot_height))
                        
                        if plot_type == "Line Plot":
                            if use_index_as_x:
                                # Plot using DataFrame index
                                for var in vars_to_plot:
                                    if turn_to_log:
                                        # Check if variable is numeric
                                        if not np.issubdtype(data[var].dtype, np.number):
                                            st.write(f"Variable '{var}' is not numeric and will be skipped for log transformation.")
                                            continue

                                        # Check for non-positive values
                                        if (data[var] <= 0).any():
                                            non_positive_count = (data[var] <= 0).sum()
                                            st.warning(f"⚠️ Variable '{var}' contains {non_positive_count} non-positive values. Adding small constant for log transformation.")
                                            plot_data = data[var] + abs(data[var].min()) + 1e-10
                                        else:
                                            plot_data = data[var].copy()

                                        # Apply selected log transformation
                                        if log_base == "Natural log (ln)":
                                            plot_data = np.log(plot_data)
                                            label_suffix = " (ln)"
                                        elif log_base == "Log base 10":
                                            plot_data = np.log10(plot_data)
                                            label_suffix = " (log10)"
                                        else:  # Log base 2
                                            plot_data = np.log2(plot_data)
                                            label_suffix = " (log2)"

                                        plt.plot(data.index, plot_data, label=f"{var}{label_suffix}")
                                    else:
                                        plt.plot(data.index, data[var], label=var)
                            else:
                                # Plot using selected x-axis
                                for var in vars_to_plot:
                                    if var != x_axis_col:  # Don't plot x against itself
                                        if turn_to_log:
                                            # Check for non-positive values in both x and y
                                            x_data = data[x_axis_col]
                                            y_data = data[var]
                                            
                                            # if (x_data <= 0).any():
                                            #     st.warning(f"X-axis variable '{x_axis_col}' contains non-positive values. Adding small constant for log transformation.")
                                            #     x_data = x_data + abs(x_data.min()) + 1e-10
                                            
                                            if (y_data <= 0).any():
                                                st.warning(f"Variable '{var}' contains non-positive values. Adding small constant for log transformation.")
                                                y_data = y_data + abs(y_data.min()) + 1e-10
                                            
                                            # Apply selected log transformation
                                            if log_base == "Natural log (ln)":
                                                # x_data = np.log(x_data)
                                                y_data = np.log(y_data)
                                            elif log_base == "Log base 10":
                                                # x_data = np.log10(x_data)
                                                y_data = np.log10(y_data)
                                            else:  # Log base 2
                                                # x_data = np.log2(x_data)
                                                y_data = np.log2(y_data)
                                            
                                            plt.plot(x_data, y_data, label=var)
                                        else:
                                            plt.plot(data[x_axis_col], data[var], label=var)
                                plt.xlabel(x_axis_col)

                        elif plot_type == "Scatter Plot":
                            if use_constant_x:
                                # Distribution view - plot all y_vars at x=0 with optional jitter
                                import random
                                
                                for i, var in enumerate(y_vars):
                                    if turn_to_log:
                                        # Check if variable is numeric
                                        if not np.issubdtype(data[var].dtype, np.number):
                                            st.warning(f"Variable '{var}' is not numeric and will be skipped.")
                                            continue
                                        
                                        y_data = data[var].copy()
                                        
                                        # Handle non-positive values
                                        if (y_data <= 0).any():
                                            non_positive_count = (y_data <= 0).sum()
                                            st.warning(f"⚠️ Variable '{var}' contains {non_positive_count} non-positive values. Adding small constant for log transformation.")
                                            y_data = y_data + abs(y_data.min()) + 1e-10
                                        
                                        # Apply selected log transformation
                                        if log_base == "Natural log (ln)":
                                            y_data = np.log(y_data)
                                            label_suffix = " (ln)"
                                        elif log_base == "Log base 10":
                                            y_data = np.log10(y_data)
                                            label_suffix = " (log10)"
                                        else:  # Log base 2
                                            y_data = np.log2(y_data)
                                            label_suffix = " (log2)"
                                        
                                        label = f"{var}{label_suffix}"
                                    else:
                                        y_data = data[var]
                                        label = var
                                    
                                    # Create x values (constant with optional jitter)
                                    if add_jitter:
                                        # Add small random horizontal offset for each variable
                                        x_values = np.full(len(y_data), i) + np.random.normal(0, 0.1, len(y_data))
                                    else:
                                        x_values = np.full(len(y_data), i)
                                    
                                    plt.scatter(x_values, y_data, label=label, alpha=0.7)
                                
                                plt.xlabel("Variable Index" + (" (with jitter)" if add_jitter else ""))
                                plt.ylabel("Values")
                                
                                # Set x-axis labels to variable names - FIXED VERSION
                                if y_vars:
                                    plt.xticks(range(len(y_vars)), y_vars, rotation=45)
                            
                            else:
                                for var in y_vars:  # Use y_vars instead of vars_to_plot
                                    if turn_to_log:
                                        # Check for non-positive values in both x and y
                                        x_data = data[scatter_x].copy()
                                        y_data = data[var].copy()
                                        
                                        if (x_data <= 0).any():
                                            st.warning(f"X-axis variable '{scatter_x}' contains non-positive values. Adding small constant for log transformation.")
                                            x_data = x_data + abs(x_data.min()) + 1e-10
                                        
                                        if (y_data <= 0).any():
                                            st.warning(f"Variable '{var}' contains non-positive values. Adding small constant for log transformation.")
                                            y_data = y_data + abs(y_data.min()) + 1e-10
                                        
                                        # Apply selected log transformation
                                        if log_base == "Natural log (ln)":
                                            x_data = np.log(x_data)
                                            y_data = np.log(y_data)
                                            label_suffix = " (ln)"
                                        elif log_base == "Log base 10":
                                            x_data = np.log10(x_data)
                                            y_data = np.log10(y_data)
                                            label_suffix = " (log10)"
                                        else:  # Log base 2
                                            x_data = np.log2(x_data)
                                            y_data = np.log2(y_data)
                                            label_suffix = " (log2)"
                                        
                                        plt.scatter(x_data, y_data, label=f"{var}{label_suffix}", alpha=0.7)
                                    else: 
                                        plt.scatter(data[scatter_x], data[var], label=var, alpha=0.7)
                                
                                plt.xlabel(scatter_x)
                        
                        elif plot_type == "Box Plot":
                            # Filter to numeric variables only for box plot
                            numeric_vars = [var for var in vars_to_plot if np.issubdtype(data[var].dtype, np.number)]
                            if numeric_vars:
                                if turn_to_log:
                                    # Check for non-positive values before log transformation
                                    plot_data = data[numeric_vars].copy()
                                    if (plot_data <= 0).any().any():
                                        st.warning("Log transformation requires positive values. Some non-positive values were found and will be handled.")
                                        # Add small constant to make all values positive
                                        plot_data = plot_data + abs(plot_data.min().min()) + 1e-10
                                    
                                    # Apply selected log transformation
                                    if log_base == "Natural log (ln)":
                                        plot_data = np.log(plot_data)
                                        title_suffix = " (ln scale)"
                                    elif log_base == "Log base 10":
                                        plot_data = np.log10(plot_data)
                                        title_suffix = " (log10 scale)"
                                    else:  # Log base 2
                                        plot_data = np.log2(plot_data)
                                        title_suffix = " (log2 scale)"
                                    
                                    plot_data.boxplot(figsize=(plot_width, plot_height))
                                    plt.suptitle(f"Box Plot{title_suffix}")
                                else:
                                    data[numeric_vars].boxplot(figsize=(plot_width, plot_height))
                                    plt.suptitle("Box Plot")
                            else:
                                st.warning("Box plots require numeric variables")

                        elif plot_type == "Histogram":
                            # Filter to numeric variables only for histogram
                            numeric_vars = [var for var in vars_to_plot if np.issubdtype(data[var].dtype, np.number)]
                            if numeric_vars:
                                if turn_to_log:
                                    # Check for non-positive values before log transformation
                                    plot_data = data[numeric_vars].copy()
                                    if (plot_data <= 0).any().any():
                                        st.warning("Log transformation requires positive values. Some non-positive values were found and will be handled.")
                                        # Add small constant to make all values positive
                                        plot_data = plot_data + abs(plot_data.min().min()) + 1e-10
                                    
                                    # Apply selected log transformation
                                    if log_base == "Natural log (ln)":
                                        plot_data = np.log(plot_data)
                                        title_suffix = " (ln scale)"
                                    elif log_base == "Log base 10":
                                        plot_data = np.log10(plot_data)
                                        title_suffix = " (log10 scale)"
                                    else:  # Log base 2
                                        plot_data = np.log2(plot_data)
                                        title_suffix = " (log2 scale)"
                                    
                                    plot_data.hist(bins=hist_bins, figsize=(plot_width, plot_height), alpha=0.7)
                                    plt.suptitle(f"Histogram{title_suffix}")
                                else:
                                    data[numeric_vars].hist(bins=hist_bins, figsize=(plot_width, plot_height), alpha=0.7)
                                    plt.suptitle("Histogram")
                            else:
                                st.warning("Histograms require numeric variables")
                        
                        elif plot_type == "Heatmap (Correlation)":
                            # Filter to numeric variables only for correlation
                            numeric_vars = [var for var in vars_to_plot if np.issubdtype(data[var].dtype, np.number)]
                            if len(numeric_vars) > 1:
                                # Calculate correlation using selected method
                                correlation = calculate_correlation_matrix(
                                    data[numeric_vars], 
                                    heatmap_type, 
                                    corr_loader
                                )
                                
                                if correlation is not None:
                                    fig_size = max(6, len(numeric_vars))
                        
                                    # Dynamically adjust font size based on column count
                                    font_size = fig_size + 2
                                    
                                    # Annotation size slightly smaller than font size
                                    annot_size = max(6, font_size - 2)

                                    # Create correlation heatmap with seaborn
                                    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
                                    heatmap = sns.heatmap(correlation, xticklabels=numeric_vars, 
                                                        yticklabels=numeric_vars, annot=True, fmt=".2f", 
                                                        cmap='PRGn', vmin=-1, vmax=1, annot_kws={"size": annot_size})

                                    # Apply font size to tick labels
                                    plt.xticks(rotation=45, ha='right', fontsize=font_size)
                                    plt.yticks(rotation=0, fontsize=font_size)
                                    
                                    # Adjust colorbar font size
                                    cbar = heatmap.collections[0].colorbar
                                    cbar.ax.tick_params(labelsize=font_size)

                                    plt.title(f"{heatmap_type} Correlation Heatmap", fontsize=font_size + 3, loc='center')
                                    plt.tight_layout()
                                else:
                                    st.error("Could not calculate correlation matrix")
                            else:
                                st.warning("Correlation heatmap requires at least 2 numeric variables")
                        
                        # Add plot decorations (except for heatmap and box plot)
                        if plot_type not in ["Heatmap (Correlation)", "Box Plot", "Histogram"]:
                            plt.ylabel("Value")
                            plt.title(f"{plot_type} of Selected Variables")
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            
                        # Display the plot
                        st.pyplot(plt)
                        
                        # Option to save the plot
                        save_plot = st.button("Save Plot")
                        if save_plot:
                            # Create a directory if it doesn't exist
                            os.makedirs("plots", exist_ok=True)
                            
                            # Save the plot
                            plot_file = f"plots/plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                            st.success(f"Plot saved as {plot_file}")
                            
                    except Exception as e:
                        st.error(f"Error generating plot: {e}")
                else:
                    st.warning("No variables selected for plotting")
        
        # Save configuration section
        st.subheader("Save Configuration")
        
        # Add this in the "Save Configuration" section after the existing "Save Current Selection" button
        if st.button("Export Data and Selections"):
            if selected_inputs and selected_targets:
                # Create directories
                export_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                export_dir = f"data_configs/data_export_{export_timestamp}"
                os.makedirs(export_dir, exist_ok=True)
                
                # Save configuration
                config = {
                    "file_name": st.session_state.file_name,
                    "input_variables": st.session_state.input_vars,
                    "target_variables": st.session_state.target_vars
                }
                with open(f"{export_dir}/config.json", "w") as f:
                    json.dump(config, f, indent=4)
                
                # Export data file
                if st.session_state.file_name.endswith('.csv'):
                    data.to_csv(f"{export_dir}/data.csv", index=False)
                else:
                    data.to_excel(f"{export_dir}/data.xlsx", index=False)
                
                st.success(f"Data and selection exported to {export_dir}")
            else:
                st.error("Please select at least one input and one target variable")
        
        st.session_state.input_vars = selected_inputs
        st.session_state.target_vars = selected_targets
        
        # Display current configuration summary
        if st.session_state.input_vars or st.session_state.target_vars:
            st.subheader("Current Selection Summary")
            
            summary = {
                "File": st.session_state.file_name,
                "Number of Input Variables": len(st.session_state.input_vars),
                "Input Variables": ", ".join(st.session_state.input_vars) if st.session_state.input_vars else "None",
                "Number of Target Variables": len(st.session_state.target_vars),
                "Target Variables": ", ".join(st.session_state.target_vars) if st.session_state.target_vars else "None"
            }
            
            for key, value in summary.items():
                st.write(f"**{key}:** {value}")
            
            # Option to proceed to model training
            if st.session_state.input_vars and st.session_state.target_vars:
                if st.button("Proceed to Model Training"):
                    # Update session state for main app
                    st.session_state.page = "model_training"
                    st.rerun()

    else:
        # Show instructions when no data is loaded
        st.info("Please upload a CSV or Excel file using the sidebar to get started.")
        
        # Add information about custom correlation functions
        if CUSTOM_CORR_AVAILABLE:
            with st.expander("📚 About Custom Correlation Functions"):
                st.write("""
                ### Adding Custom Correlation Functions
                
                You can add custom correlation functions by creating Python files in the `custom_corr/` directory.
                
                **Requirements for custom correlation functions:**
                - Must take exactly two parameters (e.g., `x`, `y`)
                - Must return a single numeric value between -1 and 1 (or NaN for invalid data)
                - Should handle NaN values appropriately
                
                **Example:**
                ```python
                def my_custom_correlation(x, y):
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() < 2:
                        return np.nan
                    # Your correlation calculation here
                    return correlation_value
                ```
                
                The functions will be automatically detected and made available in the correlation method dropdown.
                """)
        
        # Example data section
        with st.expander("Don't have data? Use example data"):
            example_option = st.selectbox(
                "Select example data",
                ["Synthetic Time Series", "Stock Price Data", "Temperature Data"]
            )
            
            if st.button("Load Example Data"):
                # Generate example data based on selection
                if example_option == "Synthetic Time Series":
                    # Create a synthetic time series dataset with realistic correlations
                    np.random.seed(42)
                    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
                    
                    # Base temperature with seasonal pattern
                    temp_base = np.linspace(0, 4*np.pi, 1000)
                    temperature = np.random.normal(25, 3, 1000) + 10 * np.sin(temp_base)
                    
                    # Humidity negatively correlated with temperature
                    humidity = 100 - temperature*1.5 + np.random.normal(0, 5, 1000)
                    humidity = np.clip(humidity, 30, 95)
                    
                    # Pressure with its own cycle, but affected by temperature
                    pressure_base = 1013 + 3 * np.sin(np.linspace(0, 8*np.pi, 1000))
                    pressure = pressure_base - 0.2 * (temperature - temperature.mean()) + np.random.normal(0, 2, 1000)
                    
                    # Precipitation related to humidity
                    precip_prob = (humidity - 40) / 60  # Higher humidity = higher chance of rain
                    precipitation = np.zeros(1000)
                    rain_indices = np.random.random(1000) < precip_prob
                    precipitation[rain_indices] = np.random.exponential(1.5, size=sum(rain_indices))
                    
                    # Wind speed related to pressure gradients
                    pressure_diff = np.abs(np.diff(pressure, prepend=[pressure[0]]))
                    wind_speed = 5 + 2 * pressure_diff + np.random.normal(0, 2, 1000)
                    wind_speed = np.clip(wind_speed, 0, 30)
                    
                    data = pd.DataFrame({
                        'Date': dates,
                        'Temperature': temperature,
                        'Humidity': humidity,
                        'Pressure': pressure,
                        'WindSpeed': wind_speed,
                        'Precipitation': precipitation
                    })
                    
                elif example_option == "Stock Price Data":
                    # Create a synthetic stock price dataset with realistic correlations
                    np.random.seed(42)
                    dates = pd.date_range(start='2019-01-01', periods=252 * 5, freq='B')
                    
                    # Initialize arrays
                    n = 252 * 5
                    returns = np.random.normal(0.0005, 0.01, n)  # Daily returns
                    
                    # Add some autocorrelation to returns (momentum effect)
                    for i in range(5, n):
                        returns[i] += 0.2 * np.mean(returns[i-5:i])
                    
                    # Create price series
                    prices = 100 * np.cumprod(1 + returns)
                    
                    # Volatility clustering (GARCH-like effect)
                    vol = np.abs(returns) * 10
                    for i in range(5, n):
                        vol[i] = 0.7 * vol[i] + 0.3 * np.mean(vol[i-5:i])
                    
                    # Volume related to volatility
                    volume_base = 1000000 + 5000000 * vol
                    
                    # Higher volume on trend days (price moves in same direction for multiple days)
                    trend = np.zeros(n)
                    for i in range(3, n):
                        if np.all(np.diff(prices[i-3:i+1]) > 0) or np.all(np.diff(prices[i-3:i+1]) < 0):
                            trend[i] = 1
                    
                    volume = volume_base * (1 + 0.5 * trend) * np.random.lognormal(0, 0.3, n)
                    
                    # Calculate OHLC with more realistic relationships
                    daily_range = 0.015 * prices * (1 + vol)  # Range is bigger when volatility is higher
                    high = prices + daily_range/2
                    low = prices - daily_range/2
                    
                    # Gap based on previous day's close and overnight sentiment
                    overnight_sentiment = np.random.normal(0, 0.005, n)
                    open_prices = np.zeros(n)
                    open_prices[0] = prices[0] * 0.999
                    
                    for i in range(1, n):
                        open_prices[i] = prices[i-1] * (1 + overnight_sentiment[i])
                    
                    data = pd.DataFrame({
                        'Date': dates,
                        'Close': prices,
                        'Open': open_prices,
                        'High': high,
                        'Low': low,
                        'Volume': volume
                    })
                    
                elif example_option == "Temperature Data":
                    # Create a synthetic temperature dataset with realistic correlations
                    np.random.seed(42)
                    dates = pd.date_range(start='2019-01-01', periods=365 * 5, freq='D')
                    n = 365 * 5
                    
                    # Base temperature with yearly and weekly cycles
                    yearly_cycle = np.sin(np.linspace(0, 2*np.pi*5, n))  # 5 years
                    base_temp = 15 + 10 * yearly_cycle
                    weekly = 2 * np.sin(np.linspace(0, 2*np.pi*5*52, n))  # Weekly fluctuations
                    temperature = base_temp + weekly + np.random.normal(0, 2, n)
                    
                    # Cloud cover correlated (negatively) with temperature
                    cloud_cover_base = 50 - 30 * yearly_cycle  # Opposite phase of temperature
                    cloud_cover = np.clip(cloud_cover_base + 15 * np.random.normal(0, 1, n), 0, 100)
                    
                    # Humidity related to temperature and cloud cover
                    humidity_base = 80 - temperature*0.8 + cloud_cover*0.3
                    humidity = np.clip(humidity_base + np.random.normal(0, 5, n), 20, 100)
                    
                    # Precipitation strongly related to cloud cover and humidity
                    precip_prob = (cloud_cover/100) * (humidity/100)
                    precipitation = np.zeros(n)
                    rain_indices = np.random.random(n) < precip_prob
                    precipitation[rain_indices] = np.random.exponential(1.5, size=sum(rain_indices))
                    
                    # Wind speed related to temperature gradients
                    temp_diff = np.abs(np.diff(temperature, prepend=[temperature[0]]))
                    wind_base = 5 + 8 * temp_diff/np.max(temp_diff)
                    
                    # Also add some seasonal patterns to wind
                    wind_seasonal = 2 * np.sin(np.linspace(0, 2*np.pi*5, n) + np.pi)  # Windier in winter
                    wind_speed = wind_base + wind_seasonal + np.random.normal(0, 1, n)
                    wind_speed = np.clip(wind_speed, 0, 25)
                    
                    data = pd.DataFrame({
                        'Date': dates,
                        'Temperature': temperature,
                        'Precipitation': precipitation,
                        'Humidity': humidity,
                        'WindSpeed': wind_speed,
                        'CloudCover': cloud_cover
                    })
                
                # Update session state with the example data
                st.session_state.data = data
                st.session_state.file_name = f"example_{example_option.lower().replace(' ', '_')}.csv"
                
                # Force refresh to show the data
                st.rerun()

# If run directly outside the main application
if __name__ == "__main__":
    run()