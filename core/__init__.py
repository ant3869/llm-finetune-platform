"""
Core module for LLM Fine-Tuning Platform.

This package contains the core functionality for:
- Model loading (GGUF and HuggingFace formats)
- Dataset handling (JSON, TXT, PDF, Confluence exports)
- QLoRA training with memory optimization
- Model export and quantization
"""

from .model_loader import ModelLoader
from .dataset_handler import DatasetHandler
from .trainer import Trainer

__all__ = [
    "ModelLoader",
    "DatasetHandler", 
    "Trainer",
]

__version__ = "0.1.0"
