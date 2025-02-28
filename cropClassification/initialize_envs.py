import yaml
import os
import sys
import torch
import random
import numpy as np
import pandas as pd

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Add device configuration based on available hardware
    if torch.cuda.is_available():
        config['device'] = 'cuda'
        print('---------- GPU (CUDA) available ----------')
    elif torch.backends.mps.is_available():
        config['device'] = 'mps'
        print('---------- MPS available ----------')
        print('Warning: MPS may have performance issues.')
    else:
        config['device'] = 'cpu'
        print('---------- Using CPU ----------')
    
    return config

def set_seed(seed=42, cudnn=True):
    """Make all the randomization processes start from a shared seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cudnn:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    print(f"Random seed {seed} has been set.")

def initialize_envs(config_path):

    
    config = load_config(config_path)
    img_df = pd.read_csv(config["inference"]["path"])
    print("Configuration loaded successfully!")
    set_seed()
    # Print some config sections as a test
    print(f"Model type: {config['model']['name']}")
    print(f"Using device: {config['device']}")
    return config, img_df