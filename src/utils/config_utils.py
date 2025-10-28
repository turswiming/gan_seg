import os
import shutil
from omegaconf import OmegaConf

def load_config_with_inheritance(config_path):
    """
    Loads and merges configuration files with support for inheritance.
    
    This function implements a configuration inheritance system where each config file
    can inherit from a base config using the '__base__' field. The inheritance can be
    chained, allowing for multiple levels of configuration inheritance.
    
    Args:
        config_path (str): Path to the primary configuration file
        
    Returns:
        OmegaConf: Merged configuration object containing all inherited settings
        
    Example:
        If config.yaml contains:
            __base__: base_config.yaml
            learning_rate: 0.001
        And base_config.yaml contains:
            batch_size: 32
        The result will be a config with both batch_size and learning_rate.
    """
    config = OmegaConf.load(config_path)
    base_config_path = config_path
    while '__base__' in config:
        base_config_path = os.path.join(os.path.dirname(base_config_path), config.__base__)
        print(f"Loading base config from: {base_config_path}")
        base_config = OmegaConf.load(base_config_path)
        # Remove __base__ field to avoid interference
        config.pop('__base__')
        config = OmegaConf.merge(base_config, config)
    return config

def save_config_and_code(config, log_dir):
    """
    Saves configuration and all code files to the specified directory.
    
    This function performs two main tasks:
    1. Saves the current configuration as a YAML file
    2. Copies all relevant code files (Python, CUDA, C++, headers) to a code directory
    
    Args:
        config (OmegaConf): Configuration object to save
        log_dir (str): Directory where files should be saved
        
    Note:
        The function creates the following structure:
        log_dir/
        ├── config.yaml
        └── code/
            └── [all code files with original directory structure]
    """
    # Save config file
    config_save_path = f"{log_dir}/config.yaml"
    with open(config_save_path, "w") as config_file:
        OmegaConf.save(config=config, f=config_file)
    print(f"Config file saved to {config_save_path}")

    # Save all code files
    code_save_path = os.path.join(log_dir, "code")
    os.makedirs(code_save_path, exist_ok=True)
    current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get src directory
    code_extensions = (".py", ".cu", ".cpp", ".h", ".yaml")
    
    for root, dirs, files in os.walk(current_folder):
        for file_name in files:
            if file_name.endswith(code_extensions):
                full_file_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(full_file_path, current_folder)
                destination_path = os.path.join(code_save_path, relative_path)
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.copy(full_file_path, destination_path)

    print(f"All code files including subdirectories saved to {code_save_path}") 