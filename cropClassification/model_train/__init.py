# Explicitly import and expose the ModelCompiler class
from .compiler import ModelCompiler
from .train import train

# Explicitly define what this module exports
__all__ = ['ModelCompiler', 'train']