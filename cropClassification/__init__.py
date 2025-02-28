"""Crop Classification package for roadside crop type identification."""

__version__ = '0.1.0'

# Import specific components from submodules
# from .model_train.compiler import ModelCompiler  # Direct import of ModelCompiler
# from .model_train import train  # Import other functions from model_train
from .initialize_envs import initialize_envs
# from .load_data import RoadsideCropImageDataset
from .pipeline import PipelineManager

# Import other modules as needed
# from .model_eval import *
from .utils import *
# from .model.unets import UNetWithAttentionDeep
# from .model.losses import *
# from .model_inference import *

# Explicitly define what this package exports
# __all__ = [
#     'ModelCompiler',  # Make ModelCompiler available at the top level
#     'train',
#     'initialize_envs',
#     'RoadsideCropImageDataset',
#     'UNetWithAttentionDeep',
#     # Add other components you want to expose
# ]
__all__ = [
   'initialize_envs',
    'PipelineManager'    # Add other components you want to expose
]