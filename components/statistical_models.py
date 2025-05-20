import torch
import torch.nn as nn
import numpy as np
import warnings
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

class StatisticalSMA(nn.Module):
    """
    Standard Statistical Simple Moving Average (SMA) model.
    No training required - uses standard moving average calculation.
    
    Performs true autoregressive forecasting by using the SMA approach
    to generate each prediction step individually.
    """
    def __init__(self, input_dim, output_dim, input_length, output_length, window_size=5):
        super(StatisticalSMA, self).__init__()
        self.output_dim = output_dim
        self.output_length = output_length
        self.window_size = min(window_size, input_length)
        self.input_dim = input_dim
        
        # If input_dim != output_dim, we'll need a mapping approach
        # But we'll do this statistically rather than with nn.Linear
        self.input_to_output_map = None
        if input_dim != output_dim:
            # Initialize with identity mapping where possible
            # This will be replaced with correlation-based mapping during forward pass
            self.register_buffer('feature_correlation', 
                               torch.eye(max(input_dim, output_dim), dtype=torch.float32)[:output_dim, :input_dim])
    
    def forward(self, x):
        # x shape: [batch_size, input_length, input_dim]
        batch_size = x.size(0)
        device = x.device
        
        # Initialize predictions container with explicit dtype
        predictions = torch.zeros(batch_size, self.output_length, self.output_dim, 
                                device=device, dtype=torch.float32)
        
        # Convert to numpy for statistical calculations
        x_np = x.detach().cpu().numpy()
        
        for batch_idx in range(batch_size):
            # Start with historical data
            history = x_np[batch_idx, :, :]
            
            # Update feature mapping if needed
            if self.input_dim != self.output_dim and self.input_to_output_map is None:
                # Calculate correlation between features
                try:
                    corr_matrix = np.corrcoef(history.T)
                    # Create a mapping based on highest correlations
                    if self.output_dim < self.input_dim:
                        self.feature_correlation = torch.tensor(
                            np.abs(corr_matrix[:self.output_dim, :]).astype(np.float32),
                            device=device
                        )
                    else:
                        self.feature_correlation = torch.tensor(
                            np.abs(corr_matrix).astype(np.float32),
                            device=device
                        )
                except Exception as e:
                    # If correlation fails, keep the default mapping
                    warnings.warn(f"Correlation calculation failed: {str(e)}. Using default mapping.")
            
            # Create a mapping from output space back to input space
            if self.input_dim != self.output_dim:
                # This will store our input-space representation of forecasts
                input_space_forecast = np.zeros(self.input_dim, dtype=np.float32)
            
            # Generate forecasts for each step in the output horizon
            for step in range(self.output_length):
                # For the first step, use original history
                if step == 0:
                    # Calculate SMA for each feature
                    sma_forecast = np.mean(history[-self.window_size:, :], axis=0)
                else:
                    # Update history with our previous prediction for autoregressive forecasting
                    # But we need to map the forecast back to input space first
                    if self.input_dim != self.output_dim:
                        # Only update input dimensions that correspond to output dimensions
                        history = np.vstack([
                            history, 
                            input_space_forecast.reshape(1, -1)
                        ])
                    else:
                        # When dimensions match, we can directly use the forecast
                        history = np.vstack([history, step_forecast.reshape(1, -1)])
                    
                    # Recalculate SMA with updated history
                    sma_forecast = np.mean(history[-self.window_size:, :], axis=0)
                
                # Map input features to output dimensions
                if self.input_dim != self.output_dim:
                    if self.output_dim < self.input_dim:
                        # Fewer outputs than inputs: select top correlated features
                        step_forecast = sma_forecast[:self.output_dim]
                        
                        # For autoregression, store forecast in input space
                        # Just copy the forecast values to the corresponding input dimensions
                        input_space_forecast = sma_forecast.copy()
                    else:
                        # More outputs than inputs: repeat most important features
                        expanded_forecast = np.zeros(self.output_dim, dtype=np.float32)
                        for out_idx in range(self.output_dim):
                            # Find most correlated input feature for this output
                            in_idx = np.argmax(self.feature_correlation[out_idx, :].cpu().numpy())
                            expanded_forecast[out_idx] = sma_forecast[in_idx]
                        step_forecast = expanded_forecast
                        
                        # For autoregression, create an input-space version
                        # Keep the original input forecast
                        input_space_forecast = sma_forecast.copy()
                else:
                    step_forecast = sma_forecast
                
                # Store prediction with explicit float32 conversion
                predictions[batch_idx, step, :] = torch.tensor(
                    step_forecast.astype(np.float32), 
                    device=device
                )
        
        return predictions


class StatisticalExponentialSmoothing(nn.Module):
    """
    Standard Statistical Exponential Smoothing model using statsmodels.
    No training required - uses statsmodels implementation.
    
    Parameters:
        input_dim: Number of input features
        output_dim: Number of output features
        input_length: Length of input sequence
        output_length: Length of output sequence
        alpha: Smoothing parameter (0-1) - can be None for auto-optimization
        seasonal: Whether to use seasonal decomposition
        trend: Whether to use trend component (Holt's method)
    """
    def __init__(self, input_dim, output_dim, input_length, output_length, 
                 alpha=None, seasonal=False, trend=False):
        super(StatisticalExponentialSmoothing, self).__init__()
        self.output_dim = output_dim
        self.output_length = output_length
        self.input_dim = input_dim
        self.alpha = alpha
        self.seasonal = seasonal
        self.trend = trend
        
        # Store best alpha values (will be determined during forward pass) with explicit dtype
        self.register_buffer('best_alphas', torch.ones(input_dim, dtype=torch.float32) * 0.3)
        
        # If using trend, store beta values with explicit dtype
        if self.trend:
            self.register_buffer('best_betas', torch.ones(input_dim, dtype=torch.float32) * 0.1)
    
    def forward(self, x):
        # x shape: [batch_size, input_length, input_dim]
        batch_size = x.size(0)
        device = x.device
        
        # Initialize predictions container with explicit dtype
        predictions = torch.zeros(batch_size, self.output_length, self.output_dim, 
                                device=device, dtype=torch.float32)
        
        # Process each batch separately
        for batch_idx in range(batch_size):
            # Extract time series for this batch
            batch_series = x[batch_idx].detach().cpu().numpy()
            
            # Process each feature dimension
            feature_forecasts = []
            
            for feature_idx in range(self.input_dim):
                # Extract this feature's time series
                series = batch_series[:, feature_idx]
                
                try:
                    if self.trend:
                        # Use Holt's method for trending data
                        if self.alpha is None:
                            # Auto-optimize parameters
                            model = Holt(series).fit()
                            # Store optimized parameters with explicit float32
                            self.best_alphas[feature_idx] = torch.tensor(
                                float(model.params['smoothing_level']), 
                                device=device, 
                                dtype=torch.float32
                            )
                            self.best_betas[feature_idx] = torch.tensor(
                                float(model.params['smoothing_trend']), 
                                device=device, 
                                dtype=torch.float32
                            )
                        else:
                            # Use specified parameters
                            model = Holt(series).fit(
                                smoothing_level=float(self.best_alphas[feature_idx].cpu().numpy()),
                                smoothing_trend=float(self.best_betas[feature_idx].cpu().numpy())
                            )
                    else:
                        # Simple exponential smoothing
                        if self.alpha is None:
                            # Auto-optimize alpha
                            model = SimpleExpSmoothing(series).fit()
                            # Store optimized alpha with explicit float32
                            self.best_alphas[feature_idx] = torch.tensor(
                                float(model.params['smoothing_level']), 
                                device=device, 
                                dtype=torch.float32
                            )
                        else:
                            # Use specified alpha
                            model = SimpleExpSmoothing(series).fit(smoothing_level=self.alpha)
                    
                    # Generate forecast
                    forecast = model.forecast(self.output_length)
                    feature_forecasts.append(forecast)
                    
                except Exception as e:
                    # Fallback to simple moving average if exponential smoothing fails
                    warnings.warn(f"Exponential smoothing failed for feature {feature_idx}. Using SMA instead. Error: {str(e)}")
                    # Simple moving average fallback
                    window = min(5, len(series))
                    sma = np.mean(series[-window:])
                    forecast = np.repeat(sma, self.output_length)
                    feature_forecasts.append(forecast)
            
            # Create feature mapping if needed
            if self.input_dim != self.output_dim:
                try:
                    # Calculate correlation between features
                    corr_matrix = np.corrcoef(batch_series.T)
                    
                    # Map forecasts based on correlation
                    mapped_forecasts = []
                    
                    if self.output_dim < self.input_dim:
                        # Fewer outputs than inputs: find clusters or most important features
                        # Simple approach: just take the first output_dim features
                        for i in range(self.output_dim):
                            mapped_forecasts.append(feature_forecasts[i])
                    else:
                        # More outputs than inputs: need to duplicate some features
                        # Simple approach: use correlation to map
                        for out_idx in range(self.output_dim):
                            # Find most correlated input feature (or use modulo for simple duplication)
                            in_idx = out_idx % self.input_dim
                            mapped_forecasts.append(feature_forecasts[in_idx])
                    
                    # Convert to array and transpose
                    forecast_array = np.column_stack(mapped_forecasts)
                except Exception as e:
                    # Fallback if correlation fails
                    warnings.warn(f"Feature mapping failed: {str(e)}. Using simple mapping.")
                    mapped_forecasts = []
                    for out_idx in range(self.output_dim):
                        in_idx = out_idx % self.input_dim
                        mapped_forecasts.append(feature_forecasts[in_idx])
                    forecast_array = np.column_stack(mapped_forecasts)
            else:
                # No mapping needed
                forecast_array = np.column_stack(feature_forecasts)
            
            # Store predictions with explicit float32 conversion
            predictions[batch_idx] = torch.tensor(
                forecast_array.astype(np.float32), 
                device=device
            )
        
        return predictions


class StatisticalLinearRegression(nn.Module):
    """
    Standard Statistical Linear Regression using statsmodels OLS.
    No PyTorch training required - uses statsmodels for coefficient estimation.
    
    Features:
    - Proper OLS regression for each feature
    - Trend forecasting based on fitted parameters
    - Handles multivariate input/output
    """
    def __init__(self, input_dim, output_dim, input_length, output_length):
        super(StatisticalLinearRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.output_length = output_length
        
        # Register buffers to store regression coefficients with explicit dtype
        self.register_buffer('slopes', torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer('intercepts', torch.zeros(input_dim, dtype=torch.float32))
        
    def forward(self, x):
        # x shape: [batch_size, input_length, input_dim]
        batch_size = x.size(0)
        device = x.device
        
        # Initialize predictions with explicit dtype
        predictions = torch.zeros(batch_size, self.output_length, self.output_dim, 
                                device=device, dtype=torch.float32)
        
        # Process each batch
        for batch_idx in range(batch_size):
            # Get this batch's time series
            batch_series = x[batch_idx].detach().cpu().numpy()
            
            # Create time indices
            time_idx = np.arange(self.input_length, dtype=np.float32).reshape(-1, 1)
            
            # Process each feature dimension
            feature_forecasts = []
            
            for feature_idx in range(self.input_dim):
                # Get this feature's values
                y = batch_series[:, feature_idx]
                
                # Add constant for intercept
                X = sm.add_constant(time_idx)
                
                try:
                    # Fit OLS model
                    model = sm.OLS(y, X).fit()
                    
                    # Extract coefficients
                    intercept = model.params[0]
                    slope = model.params[1]
                    
                    # Store coefficients with explicit float32
                    self.intercepts[feature_idx] = torch.tensor(
                        float(intercept), 
                        device=device, 
                        dtype=torch.float32
                    )
                    self.slopes[feature_idx] = torch.tensor(
                        float(slope), 
                        device=device, 
                        dtype=torch.float32
                    )
                    
                    # Generate forecast for future time steps
                    future_time = np.arange(
                        self.input_length, 
                        self.input_length + self.output_length,
                        dtype=np.float32
                    ).reshape(-1, 1)
                    
                    future_X = sm.add_constant(future_time)
                    forecast = model.predict(future_X)
                    feature_forecasts.append(forecast)
                    
                except Exception as e:
                    # Fallback to naive forecasting if regression fails
                    warnings.warn(f"Linear regression failed for feature {feature_idx}. Using naive forecast. Error: {str(e)}")
                    last_value = y[-1]
                    forecast = np.repeat(last_value, self.output_length)
                    feature_forecasts.append(forecast)
            
            # Create feature mapping if needed
            if self.input_dim != self.output_dim:
                # Map features based on correlation or other criteria
                mapped_forecasts = []
                
                if self.output_dim < self.input_dim:
                    # Fewer outputs than inputs: select most important features
                    # Simple approach: take first output_dim features
                    for i in range(self.output_dim):
                        mapped_forecasts.append(feature_forecasts[i])
                else:
                    # More outputs than inputs: duplicate some features
                    for out_idx in range(self.output_dim):
                        # Use modulo to cycle through available features
                        in_idx = out_idx % self.input_dim
                        mapped_forecasts.append(feature_forecasts[in_idx])
                
                # Convert to array and transpose
                forecast_array = np.column_stack(mapped_forecasts)
            else:
                # No mapping needed
                forecast_array = np.column_stack(feature_forecasts)
            
            # Store predictions with explicit float32 conversion
            predictions[batch_idx] = torch.tensor(
                forecast_array.astype(np.float32), 
                device=device
            )
        
        return predictions


class ARIMAModel(nn.Module):
    """
    ARIMA model implementation using statsmodels.
    Uses Auto ARIMA by default to select optimal parameters.
    
    Parameters:
        input_dim: Number of input features
        output_dim: Number of output features
        input_length: Length of input sequence
        output_length: Length of output sequence
        order: ARIMA order as tuple (p,d,q)
        seasonal_order: Seasonal ARIMA component as tuple (P,D,Q,s) or None
    """
    def __init__(self, input_dim, output_dim, input_length, output_length, 
                 order=(1, 1, 1), seasonal_order=None):
        super(ARIMAModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.output_length = output_length
        self.order = order
        self.seasonal_order = seasonal_order
        
    def forward(self, x):
        # Lazy import to avoid dependency issues
        from statsmodels.tsa.arima.model import ARIMA
        
        # Try to import auto_arima if available
        has_auto_arima = False
        try:
            from pmdarima import auto_arima
            has_auto_arima = True
        except ImportError:
            warnings.warn("pmdarima not installed, using fixed ARIMA order")
        
        # x shape: [batch_size, input_length, input_dim]
        batch_size = x.size(0)
        device = x.device
        
        # Initialize predictions with explicit dtype
        predictions = torch.zeros(batch_size, self.output_length, self.output_dim, 
                                device=device, dtype=torch.float32)
        
        # Process each batch
        for batch_idx in range(batch_size):
            # Get this batch's time series
            batch_series = x[batch_idx].detach().cpu().numpy()
            
            # Process each feature dimension
            feature_forecasts = []
            
            for feature_idx in range(self.input_dim):
                # Get this feature's values
                series = batch_series[:, feature_idx]
                
                try:
                    if has_auto_arima:
                        # Use auto_arima to determine best parameters
                        auto_model = auto_arima(
                            series,
                            start_p=0, start_q=0, max_p=5, max_q=5, max_d=2,
                            seasonal=self.seasonal_order is not None,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True
                        )
                        # Extract best order
                        best_order = auto_model.order
                        
                        # Fit ARIMA with best parameters
                        model = ARIMA(series, order=best_order, 
                                    seasonal_order=self.seasonal_order)
                    else:
                        # Use specified order
                        model = ARIMA(series, order=self.order, 
                                     seasonal_order=self.seasonal_order)
                    
                    # Fit model
                    fitted_model = model.fit()
                    
                    # Generate forecast
                    forecast = fitted_model.forecast(steps=self.output_length)
                    feature_forecasts.append(forecast)
                    
                except Exception as e:
                    # Fallback to naive forecasting if ARIMA fails
                    warnings.warn(f"ARIMA failed for feature {feature_idx}. Using naive forecast. Error: {str(e)}")
                    last_value = series[-1]
                    forecast = np.repeat(last_value, self.output_length)
                    feature_forecasts.append(forecast)
            
            # Map features if needed (similar to other models)
            if self.input_dim != self.output_dim:
                mapped_forecasts = []
                
                if self.output_dim < self.input_dim:
                    for i in range(self.output_dim):
                        mapped_forecasts.append(feature_forecasts[i])
                else:
                    for out_idx in range(self.output_dim):
                        in_idx = out_idx % self.input_dim
                        mapped_forecasts.append(feature_forecasts[in_idx])
                
                forecast_array = np.column_stack(mapped_forecasts)
            else:
                forecast_array = np.column_stack(feature_forecasts)
            
            # Store predictions with explicit float32 conversion
            predictions[batch_idx] = torch.tensor(
                forecast_array.astype(np.float32), 
                device=device
            )
        
        return predictions