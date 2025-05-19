import os
import importlib.util
import inspect
import streamlit as st
from pathlib import Path

class CustomCorrLoader:
    def __init__(self, custom_corr_dir="custom_corr"):
        self.custom_corr_dir = custom_corr_dir
        self.custom_corr_functions = {}
        self.load_custom_corr_functions()
    
    def load_custom_corr_functions(self):
        """Load all custom correlation functions from the custom_corr directory"""
        if not os.path.exists(self.custom_corr_dir):
            os.makedirs(self.custom_corr_dir, exist_ok=True)
            return
        
        # Get all Python files in the custom_corr directory
        python_files = [f for f in os.listdir(self.custom_corr_dir) 
                       if f.endswith('.py') and f != '__init__.py']
        
        for file_name in python_files:
            try:
                file_path = os.path.join(self.custom_corr_dir, file_name)
                
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    f"custom_corr.{file_name[:-3]}", file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find all functions in the module
                functions = inspect.getmembers(module, inspect.isfunction)
                
                for func_name, func in functions:
                    # Validate that it's a suitable correlation function
                    if self.is_valid_correlation_function(func):
                        # Create a display name (remove underscores, capitalize)
                        display_name = func_name.replace('_', ' ').title()
                        self.custom_corr_functions[display_name] = func
                        
            except Exception as e:
                st.warning(f"Could not load correlation functions from {file_name}: {e}")
    
    def is_valid_correlation_function(self, func):
        """Check if a function is a valid correlation function"""
        try:
            # Get function signature
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Should have exactly 2 parameters (x, y)
            if len(params) != 2:
                return False
            
            # Check parameter names (should be something like x, y or arr1, arr2, etc.)
            # This is a basic check, you could make it more sophisticated
            return True
            
        except Exception:
            return False
    
    def get_custom_correlation_functions(self):
        """Return dictionary of custom correlation functions"""
        return self.custom_corr_functions
    
    def get_correlation_function(self, name):
        """Get a specific correlation function by name"""
        return self.custom_corr_functions.get(name)
    
    def get_all_correlation_options(self):
        """Get all correlation options (built-in + custom)"""
        built_in = ["Pearson", "Spearman", "Kendall"]
        custom_names = list(self.custom_corr_functions.keys())
        return built_in + custom_names
    
    def refresh(self):
        """Reload all custom correlation functions"""
        self.custom_corr_functions = {}
        self.load_custom_corr_functions()

# Global instance for use across the application
@st.cache_resource
def get_custom_corr_loader():
    """Get or create a cached instance of the CustomCorrLoader"""
    return CustomCorrLoader()

def get_correlation_method(method_name, corr_loader):
    """
    Get the actual correlation method (function or string) to use with pandas.corr()
    
    Args:
        method_name: Name of the correlation method
        corr_loader: Instance of CustomCorrLoader
    
    Returns:
        Either a string (for built-in methods) or a function (for custom methods)
    """
    if method_name in ["Pearson", "Spearman", "Kendall"]:
        return method_name.lower()
    else:
        # It's a custom correlation function
        return corr_loader.get_correlation_function(method_name)

# Example usage and testing
if __name__ == "__main__":
    loader = CustomCorrLoader()
    print("Available correlation functions:")
    for name in loader.get_all_correlation_options():
        print(f"- {name}")