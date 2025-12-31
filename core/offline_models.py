"""
Offline Model Manager for Air-Gapped/Restricted Networks.

Provides multiple ways to obtain models when HuggingFace Hub is inaccessible:
1. Manual file download instructions (USB transfer, etc.)
2. Direct URL downloads from mirrors
3. Local model cache management
4. GGUF-first workflow (no HF needed for inference)
5. Export pre-downloaded models

This module is designed for enterprise environments where:
- Corporate firewalls block HuggingFace
- Proxy configurations don't work
- Air-gapped networks require manual transfers
"""

import os
import sys
import json
import shutil
import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelSource:
    """Information about a model source."""
    name: str
    model_id: str
    download_url: Optional[str]  # Direct download URL if available
    mirror_urls: List[str]  # Alternative mirror URLs
    file_size_gb: float
    file_hash: Optional[str]  # SHA256 for verification
    format: str  # "gguf", "safetensors", "pytorch"
    description: str
    recommended_for: List[str]  # ["inference", "training", "both"]


# Pre-defined model sources with direct download URLs where available
OFFLINE_MODEL_REGISTRY = {
    # GGUF Models - These can be used directly for inference without HuggingFace
    "phi-2-gguf": ModelSource(
        name="Phi-2 (GGUF Q4_K_M)",
        model_id="TheBloke/phi-2-GGUF",
        download_url="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        mirror_urls=[
            "https://models.ggml.ai/phi-2/phi-2.Q4_K_M.gguf",  # Example mirror
        ],
        file_size_gb=1.6,
        file_hash=None,
        format="gguf",
        description="Microsoft Phi-2 2.7B - Efficient, good for testing",
        recommended_for=["inference", "both"],
    ),
    "mistral-7b-gguf": ModelSource(
        name="Mistral 7B Instruct (GGUF Q4_K_M)",
        model_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        download_url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        mirror_urls=[],
        file_size_gb=4.4,
        file_hash=None,
        format="gguf",
        description="Mistral 7B Instruct - Great general purpose model",
        recommended_for=["inference", "both"],
    ),
    "llama-2-7b-gguf": ModelSource(
        name="Llama 2 7B Chat (GGUF Q4_K_M)",
        model_id="TheBloke/Llama-2-7B-Chat-GGUF",
        download_url="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        mirror_urls=[],
        file_size_gb=4.1,
        file_hash=None,
        format="gguf",
        description="Meta Llama 2 7B Chat - Widely compatible",
        recommended_for=["inference", "both"],
    ),
    "granite-3b-gguf": ModelSource(
        name="Granite 3B (GGUF Q4_K_M)",
        model_id="ibm-granite/granite-3.0-3b-a800m-instruct-GGUF",
        download_url="https://huggingface.co/ibm-granite/granite-3.0-3b-a800m-instruct-GGUF/resolve/main/granite-3.0-3b-a800m-instruct-Q4_K_M.gguf",
        mirror_urls=[],
        file_size_gb=2.0,
        file_hash=None,
        format="gguf",
        description="IBM Granite 3B - Enterprise focused, efficient",
        recommended_for=["inference", "both"],
    ),
}

# HuggingFace models for training (require HF access OR manual download)
HF_MODEL_REGISTRY = {
    "microsoft/phi-2": {
        "name": "Microsoft Phi-2",
        "size_gb": 5.5,
        "description": "2.7B parameters - Great for 8GB VRAM",
        "files_needed": [
            "config.json",
            "model.safetensors",  # or pytorch_model.bin
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    },
    "ibm-granite/granite-3.0-8b-instruct": {
        "name": "IBM Granite 3.0 8B",
        "size_gb": 16.0,
        "description": "8B parameters - Enterprise grade",
        "files_needed": [
            "config.json",
            "model*.safetensors",  # May be sharded
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    },
    "mistralai/Mistral-7B-v0.1": {
        "name": "Mistral 7B",
        "size_gb": 14.5,
        "description": "7B parameters - High quality",
        "files_needed": [
            "config.json",
            "model*.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    },
}


class OfflineModelManager:
    """
    Manages model downloads and caching for offline/restricted environments.
    """
    
    def __init__(self, 
                 models_dir: str = "./models",
                 cache_dir: str = "./models/cache"):
        """
        Initialize offline model manager.
        
        Args:
            models_dir: Base directory for models
            cache_dir: Directory for cached/downloaded files
        """
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        self.gguf_dir = self.models_dir / "base"
        self.hf_cache_dir = self.cache_dir / "huggingface"
        
        # Create directories
        for d in [self.models_dir, self.cache_dir, self.gguf_dir, self.hf_cache_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Track download progress
        self.download_progress: Dict[str, float] = {}
    
    def get_available_offline_models(self) -> List[Dict[str, Any]]:
        """Get list of models available for offline download."""
        models = []
        
        for key, source in OFFLINE_MODEL_REGISTRY.items():
            models.append({
                "key": key,
                **asdict(source),
                "local_path": self._get_local_path(source),
                "is_downloaded": self._check_if_downloaded(source),
            })
        
        return models
    
    def _get_local_path(self, source: ModelSource) -> Path:
        """Get local path for a model."""
        if source.format == "gguf":
            return self.gguf_dir / f"{source.name.replace(' ', '_').replace('/', '_')}.gguf"
        else:
            return self.hf_cache_dir / source.model_id.replace("/", "_")
    
    def _check_if_downloaded(self, source: ModelSource) -> bool:
        """Check if model is already downloaded."""
        local_path = self._get_local_path(source)
        
        if source.format == "gguf":
            return local_path.exists() or any(self.gguf_dir.glob(f"*{source.name.split()[0].lower()}*.gguf"))
        else:
            return local_path.exists() and any(local_path.glob("*.safetensors")) or any(local_path.glob("*.bin"))
    
    def generate_download_instructions(self, model_key: str) -> str:
        """
        Generate manual download instructions for a model.
        
        Returns step-by-step instructions for downloading outside the network.
        """
        if model_key not in OFFLINE_MODEL_REGISTRY:
            return f"Unknown model: {model_key}"
        
        source = OFFLINE_MODEL_REGISTRY[model_key]
        local_path = self._get_local_path(source)
        
        instructions = f"""
╔══════════════════════════════════════════════════════════════════╗
║  OFFLINE DOWNLOAD INSTRUCTIONS: {source.name}
╚══════════════════════════════════════════════════════════════════╝

Model: {source.name}
Format: {source.format.upper()}
Size: ~{source.file_size_gb:.1f} GB
Use Case: {', '.join(source.recommended_for)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION 1: Direct Download URL
─────────────────────────────
On a machine WITH internet access, download from:

  {source.download_url}

Then transfer the file to:
  {local_path}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION 2: Using wget/curl (on connected machine)
────────────────────────────────────────────────
# Windows (PowerShell):
Invoke-WebRequest -Uri "{source.download_url}" -OutFile "{source.name.replace(' ', '_')}.gguf"

# Linux/Mac:
wget "{source.download_url}" -O "{source.name.replace(' ', '_')}.gguf"
# or
curl -L "{source.download_url}" -o "{source.name.replace(' ', '_')}.gguf"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION 3: HuggingFace Hub CLI (on connected machine)
────────────────────────────────────────────────────
pip install huggingface_hub

# Download specific file:
huggingface-cli download {source.model_id} --local-dir ./downloaded_model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AFTER DOWNLOADING
─────────────────
1. Transfer the downloaded file(s) to your air-gapped machine
2. Place the .gguf file in: {self.gguf_dir}
3. Refresh the model list in the UI

"""
        
        if source.mirror_urls:
            instructions += """
ALTERNATIVE MIRRORS
───────────────────
If the main URL doesn't work, try these mirrors:
"""
            for i, mirror in enumerate(source.mirror_urls, 1):
                instructions += f"  {i}. {mirror}\n"
        
        return instructions
    
    def generate_hf_offline_instructions(self, model_id: str) -> str:
        """
        Generate instructions for downloading HuggingFace models for training.
        """
        if model_id not in HF_MODEL_REGISTRY:
            # Generic instructions
            model_info = {
                "name": model_id,
                "size_gb": "Unknown",
                "files_needed": ["config.json", "model.safetensors", "tokenizer.json"],
            }
        else:
            model_info = HF_MODEL_REGISTRY[model_id]
        
        local_dir = self.hf_cache_dir / model_id.replace("/", "_")
        
        instructions = f"""
╔══════════════════════════════════════════════════════════════════╗
║  HUGGINGFACE MODEL OFFLINE DOWNLOAD: {model_info['name']}
╚══════════════════════════════════════════════════════════════════╝

Model: {model_id}
Size: ~{model_info['size_gb']} GB
Files Needed: {', '.join(model_info['files_needed'][:3])}...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METHOD 1: HuggingFace Hub CLI (Recommended)
───────────────────────────────────────────
On a machine WITH internet access:

# Install the CLI
pip install huggingface_hub

# Download the entire model
huggingface-cli download {model_id} --local-dir ./{model_id.split('/')[-1]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METHOD 2: Python Script
───────────────────────
from huggingface_hub import snapshot_download

# This downloads all model files
snapshot_download(
    repo_id="{model_id}",
    local_dir="./{model_id.split('/')[-1]}",
    local_dir_use_symlinks=False
)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METHOD 3: Manual Web Download
─────────────────────────────
1. Go to: https://huggingface.co/{model_id}/tree/main
2. Download each required file:
   - config.json
   - model.safetensors (or model*.bin files)
   - tokenizer.json
   - tokenizer_config.json
   - special_tokens_map.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AFTER DOWNLOADING
─────────────────
1. Create folder: {local_dir}
2. Copy ALL downloaded files into that folder
3. Set environment variable (optional):
   
   Windows: set HF_HOME={self.hf_cache_dir}
   Linux:   export HF_HOME={self.hf_cache_dir}

4. Use the local path in training:
   --model {local_dir}

"""
        return instructions
    
    def download_model(self, model_key: str, 
                       progress_callback: Optional[callable] = None) -> Tuple[bool, str]:
        """
        Attempt to download a model directly.
        
        Args:
            model_key: Key from OFFLINE_MODEL_REGISTRY
            progress_callback: Optional callback(downloaded_bytes, total_bytes)
            
        Returns:
            Tuple of (success, message)
        """
        if model_key not in OFFLINE_MODEL_REGISTRY:
            return False, f"Unknown model: {model_key}"
        
        source = OFFLINE_MODEL_REGISTRY[model_key]
        
        if not source.download_url:
            return False, "No direct download URL available"
        
        local_path = self._get_local_path(source)
        
        try:
            logger.info(f"Downloading {source.name} from {source.download_url}")
            
            # Create a progress reporter
            def reporthook(count, block_size, total_size):
                downloaded = count * block_size
                if progress_callback:
                    progress_callback(downloaded, total_size)
                self.download_progress[model_key] = downloaded / max(total_size, 1)
            
            # Attempt download
            urllib.request.urlretrieve(
                source.download_url,
                str(local_path),
                reporthook=reporthook
            )
            
            # Verify file size
            actual_size = local_path.stat().st_size / (1024**3)
            if actual_size < source.file_size_gb * 0.9:  # Allow 10% tolerance
                local_path.unlink()
                return False, f"Downloaded file too small ({actual_size:.2f} GB vs expected {source.file_size_gb:.2f} GB)"
            
            return True, f"Successfully downloaded to {local_path}"
            
        except urllib.error.URLError as e:
            return False, f"Network error: {e.reason}. Use manual download instructions."
        except Exception as e:
            return False, f"Download failed: {str(e)}"
    
    def scan_local_models(self) -> Dict[str, List[Dict]]:
        """
        Scan for locally available models.
        
        Returns:
            Dict with "gguf" and "huggingface" model lists
        """
        result = {"gguf": [], "huggingface": []}
        
        # Scan GGUF models
        if self.gguf_dir.exists():
            for gguf_file in self.gguf_dir.glob("*.gguf"):
                size_gb = gguf_file.stat().st_size / (1024**3)
                result["gguf"].append({
                    "name": gguf_file.stem,
                    "path": str(gguf_file),
                    "size_gb": size_gb,
                    "format": "gguf",
                })
        
        # Scan HuggingFace cache
        if self.hf_cache_dir.exists():
            for model_dir in self.hf_cache_dir.iterdir():
                if model_dir.is_dir():
                    config_file = model_dir / "config.json"
                    if config_file.exists():
                        # Calculate total size
                        total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                        result["huggingface"].append({
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "size_gb": total_size / (1024**3),
                            "format": "huggingface",
                        })
        
        return result
    
    def verify_model_files(self, model_path: Path) -> Tuple[bool, List[str]]:
        """
        Verify that a HuggingFace model directory has all required files.
        
        Returns:
            Tuple of (is_valid, list of missing files)
        """
        required_files = ["config.json"]
        optional_model_files = ["model.safetensors", "pytorch_model.bin"]
        optional_tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
        
        missing = []
        
        # Check required files
        for f in required_files:
            if not (model_path / f).exists():
                missing.append(f)
        
        # Check for at least one model file
        has_model = any(
            (model_path / f).exists() or list(model_path.glob(f.replace(".safetensors", "*.safetensors")))
            for f in optional_model_files
        )
        if not has_model:
            missing.append("model.safetensors or pytorch_model.bin")
        
        # Check for tokenizer
        has_tokenizer = any((model_path / f).exists() for f in optional_tokenizer_files)
        if not has_tokenizer:
            missing.append("tokenizer files")
        
        return len(missing) == 0, missing
    
    def get_offline_workflow_guide(self) -> str:
        """Get a complete guide for offline model usage."""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║            COMPLETE OFFLINE MODEL WORKFLOW GUIDE                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

This guide helps you use LLM Fine-Tuning Platform without internet access
or when HuggingFace is blocked by corporate firewalls.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION A: GGUF-ONLY WORKFLOW (Simplest - Inference Only)
─────────────────────────────────────────────────────────

✓ No HuggingFace access needed
✓ Single file download
✓ Great for testing and inference
✗ Cannot fine-tune GGUF directly (need HF format for training)

Steps:
1. Download a GGUF model from huggingface.co or TheBloke on a personal device
2. Transfer the .gguf file via USB/network share
3. Place in: ./models/base/
4. Use the Evaluation page to test the model

Recommended GGUF models for 8GB VRAM:
• phi-2.Q4_K_M.gguf (~1.6 GB) - Fast, good for testing
• mistral-7b-instruct-v0.2.Q4_K_M.gguf (~4.4 GB) - High quality
• llama-2-7b-chat.Q4_K_M.gguf (~4.1 GB) - Widely compatible

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION B: FULL TRAINING WORKFLOW (Requires HF Model Files)
──────────────────────────────────────────────────────────

For fine-tuning, you need the original model in HuggingFace format.

Steps:
1. ON A MACHINE WITH INTERNET ACCESS:
   
   # Install huggingface_hub
   pip install huggingface_hub
   
   # Download the model
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id="microsoft/phi-2",
       local_dir="./phi-2",
       local_dir_use_symlinks=False
   )

2. TRANSFER THE FOLDER:
   Copy the entire ./phi-2/ folder to your air-gapped machine
   
3. PLACE IN CACHE:
   Move to: ./models/cache/huggingface/microsoft_phi-2/

4. TRAIN WITH LOCAL PATH:
   Use the local path instead of model ID:
   --model ./models/cache/huggingface/microsoft_phi-2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION C: HYBRID WORKFLOW (Recommended for Production)
──────────────────────────────────────────────────────

1. Train on a machine WITH internet (home/personal)
2. Export the trained LoRA adapter (small file, ~50-200 MB)
3. Also download a GGUF version of the base model
4. Transfer both to your work machine
5. For inference: Use GGUF + LoRA adapter
6. For deployment: Merge and convert to GGUF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ENVIRONMENT VARIABLES FOR OFFLINE USE
─────────────────────────────────────

# Disable HuggingFace Hub connections:
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

# Point to local cache:
set HF_HOME=./models/cache/huggingface

# Disable telemetry:
set HF_HUB_DISABLE_TELEMETRY=1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GGUF EXPORT (After Training)
────────────────────────────

After fine-tuning, you can export to GGUF format:

1. Merge LoRA adapter with base model (Export page)
2. Convert to GGUF using llama.cpp:
   
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   pip install -r requirements.txt
   python convert.py ../merged_model --outtype q4_k_m

"""


def get_offline_env_setup() -> str:
    """Get environment setup commands for offline mode."""
    return """
# PowerShell (Windows) - Run before starting the platform:
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HOME = ".\\models\\cache\\huggingface"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

# Bash (Linux/Mac):
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=./models/cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1

# Or add to your .env file:
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HOME=./models/cache/huggingface
HF_HUB_DISABLE_TELEMETRY=1
"""


if __name__ == "__main__":
    # Test the offline manager
    manager = OfflineModelManager()
    
    print("\n=== Available Offline Models ===")
    for model in manager.get_available_offline_models():
        status = "✓ Downloaded" if model["is_downloaded"] else "✗ Not downloaded"
        print(f"  {model['name']}: {status}")
    
    print("\n=== Local Models ===")
    local = manager.scan_local_models()
    print(f"  GGUF models: {len(local['gguf'])}")
    print(f"  HuggingFace models: {len(local['huggingface'])}")
    
    print("\n=== Download Instructions Example ===")
    print(manager.generate_download_instructions("phi-2-gguf")[:500] + "...")
