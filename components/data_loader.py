import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

def run():
    """
    Main function to run the data loader component.
    This function will be called from the main application.
    """
    # Title and introduction
    st.title("Time Series Data Selection")
    st.write("Upload your time series data and select input variables and targets.")

    # Create a sidebar for options
    with st.sidebar:
        st.header("Data Options")
        
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
                
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                st.session_state.data = None
                st.session_state.file_name = None

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
                if not numeric_data.empty:
                    correlation_method = st.selectbox("Select Correlation Method", ["Pearson", "Spearman", "Kendall"], index=0)
                    correlation_matrix = numeric_data.corr(method=correlation_method.lower())
                    # st.write("Correlation Matrix:")
                    # st.dataframe(correlation_matrix)
                    # Calculate appropriate figure size and font size based on number of columns
                    num_cols = len(correlation_matrix.columns)
                    
                    # Dynamically adjust figure size based on column count
                    # Base size of 8x8, increasing by 0.5 per column, capped at 20x20
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

                    co1, col2 = st.columns(2)
                    with co1:
                        st.download_button(
                            label="Download Correlation Matrix",
                            data=correlation_matrix.to_csv().encode('utf-8'),
                            file_name='correlation_matrix.csv',
                            mime='text/csv'
                        )
                    with col2:
                        os.makedirs("plots", exist_ok=True)
                        st.download_button(
                            label="Download Correlation Heatmap",
                            data=buffer,
                            file_name='correlation_heatmap.png',
                            mime='image/png'
                        )
                    
        
        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Null Count': data.isna().sum(),
            'Unique Values': [data[col].nunique() for col in data.columns]
        })
        st.dataframe(col_info)
        
        # Column selection section
        st.subheader("Variable Selection")
        
        # Create two columns for input and target selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Select Input Variables:")
            
            # Get numeric columns for input selection
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            datetime_cols = data.select_dtypes(include=['datetime']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Organize columns by type
            st.write("Numeric Variables:")
            # Filter default values to only include variables that exist in the options
            numeric_defaults = [var for var in st.session_state.input_vars if var in numeric_cols]
            numeric_inputs = st.multiselect(
                "Select numeric input variables",
                numeric_cols,
                default=numeric_defaults
            )
            
            if datetime_cols:
                st.write("Date/Time Variables:")
                # Filter default values for datetime columns
                datetime_defaults = [var for var in st.session_state.input_vars if var in datetime_cols]
                datetime_inputs = st.multiselect(
                    "Select datetime input variables",
                    datetime_cols,
                    default=datetime_defaults
                )
            else:
                datetime_inputs = []
                
            if categorical_cols:
                st.write("Categorical Variables:")
                # Filter default values for categorical columns
                categorical_defaults = [var for var in st.session_state.input_vars if var in categorical_cols]
                categorical_inputs = st.multiselect(
                    "Select categorical input variables",
                    categorical_cols,
                    default=categorical_defaults
                )
            else:
                categorical_inputs = []
            
            # Combine all selected inputs
            if len(datetime_inputs) == 0:
                st.session_state.datetime_column = False
            else:
                st.session_state.datetime_column = True
            selected_inputs = numeric_inputs + datetime_inputs + categorical_inputs
            st.session_state.input_vars = selected_inputs
            
            # Show number of selected inputs
            st.info(f"Selected {len(selected_inputs)} input variables")
        
        with col2:
            st.write("Select Target Variables:")
            
            # Usually targets are numeric, but allow any type
            target_options = data.columns.tolist()
            # Remove already selected inputs
            available_targets = [col for col in target_options if col not in datetime_cols]
            
            # Filter default target values to only include those in available options
            target_defaults = [var for var in st.session_state.target_vars if var in available_targets]
            
            selected_targets = st.multiselect(
                "Select target variables",
                available_targets,
                default=target_defaults
            )
            
            st.session_state.target_vars = selected_targets
            
            # Show number of selected targets
            st.info(f"Selected {len(selected_targets)} target variables")
        
        # Plotting section
        st.subheader("Data Visualization")
        
        # Options for plotting
        plot_options = st.expander("Plot Options", expanded=False)
        
        with plot_options:
            plot_inputs = st.checkbox("Plot Input Variables", value=True)
            plot_targets = st.checkbox("Plot Target Variables", value=True)
            
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
                # For scatter plots, user needs to select x and y variables
                scatter_x = st.selectbox("Select X-axis variable", data.columns.tolist())
                
            elif plot_type == "Histogram":
                hist_bins = st.slider("Number of bins", 5, 100, 20)
                
            # Plot size
            col1, col2 = st.columns(2)
            with col1:
                plot_width = st.slider("Plot Width", 6, 20, 12)
            with col2:
                plot_height = st.slider("Plot Height", 4, 15, 6)
            
            if plot_type == "Heatmap (Correlation)":
                heatmap_type = st.selectbox("Select Heatmap Type", ["Pearson", "Spearman", "Kendall"], index=0)

            generate_plot = st.button("Generate Plot")
            
            
            if generate_plot:
                # Variables to plot
                vars_to_plot = []
                if plot_inputs and selected_inputs:
                    vars_to_plot.extend(selected_inputs)
                if plot_targets and selected_targets:
                    vars_to_plot.extend(selected_targets)
                    
                if vars_to_plot:
                    try:
                        plt.figure(figsize=(plot_width, plot_height))
                        
                        if plot_type == "Line Plot":
                            if use_index_as_x:
                                # Plot using DataFrame index
                                for var in vars_to_plot:
                                    plt.plot(data.index, data[var], label=var)
                            else:
                                # Plot using selected x-axis
                                for var in vars_to_plot:
                                    if var != x_axis_col:  # Don't plot x against itself
                                        plt.plot(data[x_axis_col], data[var], label=var)
                                plt.xlabel(x_axis_col)
                        
                        elif plot_type == "Scatter Plot":
                            for var in vars_to_plot:
                                if var != scatter_x:  # Don't plot x against itself
                                    plt.scatter(data[scatter_x], data[var], label=var, alpha=0.7)
                            plt.xlabel(scatter_x)
                        
                        elif plot_type == "Box Plot":
                            # Filter to numeric variables only for box plot
                            numeric_vars = [var for var in vars_to_plot if np.issubdtype(data[var].dtype, np.number)]
                            if numeric_vars:
                                data[numeric_vars].boxplot(figsize=(plot_width, plot_height))
                            else:
                                st.warning("Box plots require numeric variables")
                        
                        elif plot_type == "Histogram":
                            # Filter to numeric variables only for histogram
                            numeric_vars = [var for var in vars_to_plot if np.issubdtype(data[var].dtype, np.number)]
                            if numeric_vars:
                                data[numeric_vars].hist(bins=hist_bins, figsize=(plot_width, plot_height), alpha=0.7)
                            else:
                                st.warning("Histograms require numeric variables")
                        
                        elif plot_type == "Heatmap (Correlation)":
                            # Filter to numeric variables only for correlation
                            numeric_vars = [var for var in vars_to_plot if np.issubdtype(data[var].dtype, np.number)]
                            if len(numeric_vars) > 1:
                                correlation = data[numeric_vars].corr(method=heatmap_type.lower())
                                fig_size = num_cols
                    
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
                                st.warning("Correlation heatmap requires at least 2 numeric variables")
                        
                        # Add plot decorations (except for heatmap and box plot)
                        if plot_type not in ["Heatmap (Correlation)", "Box Plot"]:
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
        
        if st.button("Save Current Selection"):
            if selected_inputs and selected_targets:
                # Update session state
                st.session_state.config = {
                    "file_name": st.session_state.file_name,
                    "input_variables": st.session_state.input_vars,
                    "target_variables": st.session_state.target_vars
                }
                
                # Create a directory if it doesn't exist
                os.makedirs("configs", exist_ok=True)
                
                # Save configuration to a JSON file
                config_file = f"configs/config_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(config_file, "w") as f:
                    json.dump(st.session_state.config, f, indent=4)
                
                st.success(f"Configuration saved as {config_file}")
                st.info("You can now proceed to the Model Training page")
            else:
                st.error("Please select at least one input and one target variable")
        
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
        
        # Example data section
        with st.expander("Don't have data? Use example data"):
            example_option = st.selectbox(
                "Select example data",
                ["Synthetic Time Series", "Stock Price Data", "Temperature Data"]
            )
            
            if st.button("Load Example Data"):
                # Generate example data based on selection
                if example_option == "Synthetic Time Series":
                    # Create a synthetic time series dataset
                    np.random.seed(42)
                    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
                    data = pd.DataFrame({
                        'Date': dates,
                        'Temperature': np.random.normal(25, 5, 100) + 10 * np.sin(np.linspace(0, 4*np.pi, 100)),
                        'Humidity': np.random.normal(60, 10, 100),
                        'Pressure': np.random.normal(1013, 5, 100),
                        'WindSpeed': np.abs(np.random.normal(0, 10, 100)),
                        'Precipitation': np.abs(np.random.normal(0, 2, 100))
                    })
                    
                elif example_option == "Stock Price Data":
                    # Create a synthetic stock price dataset
                    np.random.seed(42)
                    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
                    price = 100
                    prices = [price]
                    for _ in range(251):
                        change = np.random.normal(0, 1) / 100
                        price = price * (1 + change)
                        prices.append(price)
                    
                    data = pd.DataFrame({
                        'Date': dates,
                        'Close': prices,
                        'Open': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
                        'High': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
                        'Low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
                        'Volume': np.random.normal(1000000, 200000, 252)
                    })
                    
                elif example_option == "Temperature Data":
                    # Create a synthetic temperature dataset
                    np.random.seed(42)
                    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
                    base_temp = 15
                    
                    # Seasonal component (yearly cycle)
                    seasonal = 10 * np.sin(np.linspace(0, 2*np.pi, 365))
                    
                    # Weekly component
                    weekly = 2 * np.sin(np.linspace(0, 2*np.pi*52, 365))
                    
                    # Random noise
                    noise = np.random.normal(0, 2, 365)
                    
                    # Combine components
                    temperature = base_temp + seasonal + weekly + noise
                    
                    data = pd.DataFrame({
                        'Date': dates,
                        'Temperature': temperature,
                        'Precipitation': np.abs(np.random.normal(0, 2, 365)),
                        'Humidity': 60 + 20 * np.sin(np.linspace(0, 4*np.pi, 365)) + np.random.normal(0, 5, 365),
                        'WindSpeed': np.abs(np.random.normal(5, 3, 365)),
                        'CloudCover': np.clip(50 + 30 * np.sin(np.linspace(0, 6*np.pi, 365)) + np.random.normal(0, 10, 365), 0, 100)
                    })
                
                # Update session state with the example data
                st.session_state.data = data
                st.session_state.file_name = f"example_{example_option.lower().replace(' ', '_')}.csv"
                
                # Force refresh to show the data
                st.rerun()

# If run directly outside the main application
if __name__ == "__main__":
    run()
