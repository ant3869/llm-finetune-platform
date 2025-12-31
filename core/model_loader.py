"""
Model Loader for Local GGUF Models.

Handles:
- Scanning directories for .gguf files
- Loading GGUF models for inference (llama-cpp-python)
- Loading models for training (HuggingFace format)
- VRAM estimation before loading
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a discovered model."""
    name: str
    path: Path
    size_bytes: int
    size_gb: float
    format: str
    estimated_vram_gb: float


class ModelLoader:
    """
    Handles loading of local GGUF models for both inference and training.
    
    For inference: Uses llama-cpp-python to load GGUF directly
    For training: Loads corresponding HuggingFace model with 4-bit quantization
    """
    
    # Mapping of GGUF model patterns to HuggingFace model IDs for training
    # Users can extend this mapping for their specific models
    GGUF_TO_HF_MAPPING = {
        "granite": "ibm-granite/granite-3.0-8b-instruct",
        "llama": "meta-llama/Llama-2-7b-hf",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "phi": "microsoft/phi-2",
    }
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the model loader with configuration."""
        self.config = self._load_config(config_path)
        self.models_base_path = Path(self.config["paths"]["models_base"])
        self.supported_formats = self.config["model"]["supported_formats"]
        
        # Ensure models directory exists
        self.models_base_path.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "paths": {
                "models_base": "./models/base",
            },
            "model": {
                "supported_formats": [".gguf"],
                "default_context_length": 2048,
            }
        }
    
    def scan_models(self, directory: Optional[Path] = None) -> List[ModelInfo]:
        """
        Scan directory for GGUF model files.
        
        Args:
            directory: Directory to scan (defaults to models_base_path)
            
        Returns:
            List of ModelInfo objects for discovered models
        """
        scan_dir = directory or self.models_base_path
        models = []
        
        if not scan_dir.exists():
            logger.warning(f"Models directory does not exist: {scan_dir}")
            return models
        
        for ext in self.supported_formats:
            for model_path in scan_dir.rglob(f"*{ext}"):
                try:
                    info = self._get_model_info(model_path)
                    models.append(info)
                    logger.info(f"Found model: {info.name} ({info.size_gb:.2f} GB)")
                except Exception as e:
                    logger.error(f"Error reading model {model_path}: {e}")
        
        return sorted(models, key=lambda x: x.name)
    
    def _get_model_info(self, model_path: Path) -> ModelInfo:
        """Extract information about a model file."""
        size_bytes = model_path.stat().st_size
        size_gb = size_bytes / (1024 ** 3)
        
        # Estimate VRAM needed (rough heuristic)
        # GGUF models are already quantized, so VRAM ≈ file size + overhead
        estimated_vram_gb = size_gb * 1.2  # 20% overhead for context
        
        return ModelInfo(
            name=model_path.stem,
            path=model_path,
            size_bytes=size_bytes,
            size_gb=size_gb,
            format=model_path.suffix,
            estimated_vram_gb=estimated_vram_gb
        )
    
    def estimate_training_vram(self, model_size_gb: float, 
                                batch_size: int = 1,
                                seq_length: int = 512) -> Dict[str, float]:
        """
        Estimate VRAM requirements for training with QLoRA.
        
        Args:
            model_size_gb: Size of the model in GB
            batch_size: Training batch size
            seq_length: Maximum sequence length
            
        Returns:
            Dictionary with VRAM estimates for different components
        """
        # 4-bit quantized model size
        model_vram = model_size_gb * 0.5  # 4-bit ≈ 50% of FP16
        
        # LoRA adapters (typically 50-200MB)
        lora_vram = 0.2
        
        # Optimizer states (Adam: 2x parameters)
        optimizer_vram = lora_vram * 2
        
        # Activations (depends on batch size and seq length)
        activation_vram = (batch_size * seq_length * 4096 * 4) / (1024 ** 3)  # Rough estimate
        
        # Gradient checkpointing reduces activation memory by ~60%
        activation_vram *= 0.4
        
        total = model_vram + lora_vram + optimizer_vram + activation_vram
        
        return {
            "model": model_vram,
            "lora_adapters": lora_vram,
            "optimizer": optimizer_vram,
            "activations": activation_vram,
            "total": total,
            "fits_8gb": total < 7.5,  # Leave some headroom
        }
    
    def load_for_inference(self, model_path: Path, 
                           n_ctx: int = 2048,
                           n_gpu_layers: int = -1) -> Any:
        """
        Load a GGUF model for inference using llama-cpp-python.
        
        Args:
            model_path: Path to the GGUF file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            
        Returns:
            Llama model instance
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF inference. "
                "Install with: pip install llama-cpp-python"
            )
        
        logger.info(f"Loading model for inference: {model_path}")
        
        model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        
        logger.info("Model loaded successfully for inference")
        return model
    
    def load_for_training(self, model_name_or_path: str,
                          quantization_config: Optional[Dict] = None) -> Tuple[Any, Any]:
        """
        Load a model for training with QLoRA quantization.
        
        For training, we need to use HuggingFace format models, not GGUF.
        This method loads the corresponding HF model with 4-bit quantization.
        
        Args:
            model_name_or_path: HuggingFace model ID or local path
            quantization_config: Optional quantization settings override
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "transformers and bitsandbytes are required for training. "
                "Install with: pip install transformers bitsandbytes"
            )
        
        logger.info(f"Loading model for training: {model_name_or_path}")
        
        # Configure 4-bit quantization for 8GB VRAM
        quant_config = quantization_config or self.config["training"]["quantization"]
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        
        logger.info("Model loaded successfully for training")
        return model, tokenizer
    
    def get_hf_model_id(self, gguf_name: str) -> Optional[str]:
        """
        Get the HuggingFace model ID corresponding to a GGUF model name.
        
        This is a heuristic mapping - users may need to specify the correct
        HF model ID manually for their specific GGUF files.
        
        Args:
            gguf_name: Name of the GGUF model file
            
        Returns:
            HuggingFace model ID if found, None otherwise
        """
        gguf_lower = gguf_name.lower()
        
        for pattern, hf_id in self.GGUF_TO_HF_MAPPING.items():
            if pattern in gguf_lower:
                logger.info(f"Mapped GGUF '{gguf_name}' to HF model '{hf_id}'")
                return hf_id
        
        logger.warning(
            f"Could not find HF mapping for GGUF model '{gguf_name}'. "
            "Please specify the HuggingFace model ID manually."
        )
        return None
    
    def check_gpu_availability(self) -> Dict[str, Any]:
        """
        Check GPU availability and VRAM status.
        
        Returns:
            Dictionary with GPU information
        """
        info = {
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
            "error": None,
        }
        
        try:
            # Check if CUDA is available
            cuda_available = torch.cuda.is_available()
            info["cuda_available"] = cuda_available
            
            if not cuda_available:
                # Try to diagnose why
                info["error"] = "CUDA not available. Check PyTorch installation with: python -c \"import torch; print(torch.cuda.is_available())\""
                logger.warning("CUDA not available - check PyTorch CUDA installation")
                return info
            
            info["device_count"] = torch.cuda.device_count()
            
            if info["device_count"] == 0:
                info["error"] = "CUDA available but no devices found"
                return info
            
            for i in range(info["device_count"]):
                try:
                    device_info = {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                        "allocated_memory_gb": torch.cuda.memory_allocated(i) / (1024**3),
                        "cached_memory_gb": torch.cuda.memory_reserved(i) / (1024**3),
                    }
                    device_info["free_memory_gb"] = (
                        device_info["total_memory_gb"] - device_info["allocated_memory_gb"]
                    )
                    info["devices"].append(device_info)
                except Exception as device_err:
                    logger.error(f"Error getting info for GPU {i}: {device_err}")
                    
        except Exception as e:
            info["error"] = f"Error checking GPU: {str(e)}"
            logger.error(f"GPU check error: {e}")
        
        return info


# Convenience function for quick model scanning
def list_local_models(directory: str = "./models/base") -> List[ModelInfo]:
    """Scan a directory for local GGUF models."""
    loader = ModelLoader()
    return loader.scan_models(Path(directory))


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    loader = ModelLoader()
    
    print("\n=== GPU Information ===")
    gpu_info = loader.check_gpu_availability()
    print(f"CUDA Available: {gpu_info['cuda_available']}")
    if gpu_info['devices']:
        for device in gpu_info['devices']:
            print(f"  {device['name']}: {device['total_memory_gb']:.2f} GB total, "
                  f"{device['free_memory_gb']:.2f} GB free")
    
    print("\n=== Scanning for Models ===")
    models = loader.scan_models()
    if models:
        for model in models:
            print(f"  {model.name}: {model.size_gb:.2f} GB "
                  f"(~{model.estimated_vram_gb:.2f} GB VRAM for inference)")
    else:
        print("  No models found. Place .gguf files in ./models/base/")
    
    print("\n=== VRAM Estimation for 7B Model ===")
    estimate = loader.estimate_training_vram(model_size_gb=4.0)  # 7B Q4 ≈ 4GB
    print(f"  Model: {estimate['model']:.2f} GB")
    print(f"  LoRA: {estimate['lora_adapters']:.2f} GB")
    print(f"  Optimizer: {estimate['optimizer']:.2f} GB")
    print(f"  Activations: {estimate['activations']:.2f} GB")
    print(f"  Total: {estimate['total']:.2f} GB")
    print(f"  Fits 8GB VRAM: {'Yes' if estimate['fits_8gb'] else 'No'}")
