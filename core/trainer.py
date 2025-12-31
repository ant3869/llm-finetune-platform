"""
QLoRA Trainer for Fine-Tuning LLMs.

Implements memory-efficient fine-tuning using:
- 4-bit quantization (NF4)
- LoRA adapters
- Gradient checkpointing
- Gradient accumulation

Optimized for 8GB VRAM consumer GPUs.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime

import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainingProgress:
    """Real-time training progress information."""
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    samples_per_second: float = 0.0
    status: str = "idle"  # idle, training, paused, completed, error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "elapsed_time": self.elapsed_time,
            "estimated_remaining": self.estimated_remaining,
            "vram_used_gb": self.vram_used_gb,
            "vram_total_gb": self.vram_total_gb,
            "samples_per_second": self.samples_per_second,
            "status": self.status,
            "progress_percent": (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0
        }


@dataclass
class TrainingConfig:
    """Configuration for training run."""
    # Model
    model_name_or_path: str = ""
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training settings
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_seq_length: int = 512
    
    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = True
    optim: str = "paged_adamw_32bit"
    
    # Logging & saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50
    output_dir: str = "./models/adapters"
    
    # Control
    max_steps: int = -1  # -1 = use epochs
    
    @classmethod
    def from_preset(cls, preset_name: str, config_path: str = "config/settings.yaml") -> "TrainingConfig":
        """Load a training preset from configuration."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        presets = config.get("training", {}).get("presets", {})
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        preset = presets[preset_name]
        defaults = config.get("training", {}).get("defaults", {})
        lora_config = config.get("training", {}).get("lora", {})
        
        # Merge defaults with preset
        merged = {**defaults, **preset}
        
        return cls(
            lora_r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", cls.__dataclass_fields__["target_modules"].default_factory()),
            epochs=merged.get("epochs", 3),
            batch_size=merged.get("batch_size", 1),
            gradient_accumulation_steps=merged.get("gradient_accumulation_steps", 16),
            learning_rate=merged.get("learning_rate", 2e-4),
            warmup_ratio=merged.get("warmup_ratio", 0.03),
            weight_decay=merged.get("weight_decay", 0.001),
            max_seq_length=merged.get("max_seq_length", 512),
            gradient_checkpointing=merged.get("gradient_checkpointing", True),
            fp16=merged.get("fp16", True),
            optim=merged.get("optim", "paged_adamw_32bit"),
            logging_steps=merged.get("logging_steps", 10),
            save_steps=merged.get("save_steps", 100),
            eval_steps=merged.get("eval_steps", 50),
            max_steps=merged.get("max_steps", -1),
        )


class Trainer:
    """
    QLoRA Trainer for fine-tuning LLMs on consumer hardware.
    
    Optimized for 8GB VRAM GPUs using:
    - 4-bit quantization
    - LoRA adapters (low-rank adaptation)
    - Gradient checkpointing
    - Memory-efficient optimizers
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize trainer with optional configuration."""
        self.config = config or TrainingConfig()
        self.progress = TrainingProgress()
        self.progress_callbacks: List[Callable[[TrainingProgress], None]] = []
        self._stop_requested = False
        self._pause_requested = False
        
        # Training components (set during training)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def add_progress_callback(self, callback: Callable[[TrainingProgress], None]):
        """Add a callback to receive progress updates."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self):
        """Notify all callbacks of progress update."""
        for callback in self.progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def _update_vram_usage(self):
        """Update VRAM usage in progress."""
        if torch.cuda.is_available():
            self.progress.vram_used_gb = torch.cuda.memory_allocated() / (1024**3)
            self.progress.vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    def prepare_model(self, model_name_or_path: str):
        """
        Load and prepare model with QLoRA configuration.
        
        Args:
            model_name_or_path: HuggingFace model ID or local path
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as e:
            raise ImportError(
                "Required packages not installed. Run: "
                "pip install transformers peft bitsandbytes accelerate"
            ) from e
        
        logger.info(f"Preparing model: {model_name_or_path}")
        self.progress.status = "loading"
        self._notify_progress()
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        
        self._update_vram_usage()
        logger.info(f"VRAM usage after model load: {self.progress.vram_used_gb:.2f} GB")
        
        return self.model, self.tokenizer
    
    def train(self, 
              train_dataset,
              eval_dataset=None,
              output_dir: Optional[str] = None):
        """
        Run the training loop.
        
        Args:
            train_dataset: HuggingFace Dataset for training
            eval_dataset: Optional HuggingFace Dataset for evaluation
            output_dir: Where to save the adapter
            
        Returns:
            Path to the saved adapter
        """
        try:
            from transformers import TrainingArguments, DataCollatorForLanguageModeling
            from trl import SFTTrainer
        except ImportError as e:
            raise ImportError(
                "Required packages not installed. Run: "
                "pip install transformers trl"
            ) from e
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")
        
        output_dir = output_dir or self.config.output_dir
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        full_output_dir = Path(output_dir) / run_name
        full_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training run: {run_name}")
        logger.info(f"Output directory: {full_output_dir}")
        
        # Calculate total steps
        num_train_samples = len(train_dataset)
        steps_per_epoch = num_train_samples // (
            self.config.batch_size * self.config.gradient_accumulation_steps
        )
        
        if self.config.max_steps > 0:
            total_steps = self.config.max_steps
        else:
            total_steps = steps_per_epoch * self.config.epochs
        
        self.progress.total_steps = total_steps
        self.progress.total_epochs = self.config.epochs
        self.progress.status = "training"
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(full_output_dir),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=False,
            report_to=["tensorboard"],
            logging_dir=str(full_output_dir / "logs"),
            max_steps=self.config.max_steps if self.config.max_steps > 0 else -1,
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if self.config.gradient_checkpointing else None,
        )
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=False,  # Disable packing for simplicity
        )
        
        # Add our callback for progress tracking
        self.trainer.add_callback(self._create_progress_callback())
        
        # Run training
        start_time = time.time()
        self._stop_requested = False
        self._pause_requested = False
        
        try:
            train_result = self.trainer.train()
            
            self.progress.elapsed_time = time.time() - start_time
            self.progress.status = "completed"
            self._notify_progress()
            
            # Save the final adapter
            final_adapter_path = full_output_dir / "final_adapter"
            self.trainer.save_model(str(final_adapter_path))
            self.tokenizer.save_pretrained(str(final_adapter_path))
            
            logger.info(f"Training completed. Adapter saved to: {final_adapter_path}")
            
            # Save training config
            config_path = full_output_dir / "training_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump({
                    "model": self.config.model_name_or_path,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "max_seq_length": self.config.max_seq_length,
                    "train_samples": num_train_samples,
                    "final_loss": self.progress.loss,
                    "total_steps": self.progress.current_step,
                    "training_time_seconds": self.progress.elapsed_time,
                }, f)
            
            return final_adapter_path
            
        except Exception as e:
            self.progress.status = "error"
            self._notify_progress()
            logger.error(f"Training failed: {e}")
            raise
    
    def _create_progress_callback(self):
        """Create a HuggingFace callback for progress tracking."""
        from transformers import TrainerCallback
        
        trainer = self
        
        class ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    trainer.progress.loss = logs.get("loss", trainer.progress.loss)
                    trainer.progress.learning_rate = logs.get("learning_rate", trainer.progress.learning_rate)
                
                trainer.progress.current_step = state.global_step
                trainer.progress.current_epoch = state.epoch or 0
                trainer._update_vram_usage()
                
                # Calculate time estimates
                if state.global_step > 0:
                    elapsed = time.time() - state.start_time if hasattr(state, 'start_time') else 0
                    trainer.progress.elapsed_time = elapsed
                    steps_remaining = trainer.progress.total_steps - state.global_step
                    if elapsed > 0:
                        trainer.progress.samples_per_second = state.global_step / elapsed
                        trainer.progress.estimated_remaining = steps_remaining / (state.global_step / elapsed)
                
                trainer._notify_progress()
                
                # Check for stop/pause requests
                if trainer._stop_requested:
                    control.should_training_stop = True
                
            def on_epoch_end(self, args, state, control, **kwargs):
                trainer.progress.current_epoch = int(state.epoch)
                trainer._notify_progress()
        
        return ProgressCallback()
    
    def stop_training(self):
        """Request training to stop."""
        logger.info("Stop requested")
        self._stop_requested = True
        self.progress.status = "stopping"
        self._notify_progress()
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Resources cleaned up")


def quick_train(
    model_name: str,
    train_data_path: str,
    output_dir: str = "./models/adapters",
    preset: str = "balanced",
    progress_callback: Optional[Callable[[TrainingProgress], None]] = None
) -> Path:
    """
    Quick training function for CLI usage.
    
    Args:
        model_name: HuggingFace model ID
        train_data_path: Path to training data (JSON/JSONL)
        output_dir: Where to save the adapter
        preset: Training preset (quick_test, balanced, thorough)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to the saved adapter
    """
    from .dataset_handler import DatasetHandler
    
    # Load configuration
    config = TrainingConfig.from_preset(preset)
    config.model_name_or_path = model_name
    config.output_dir = output_dir
    
    # Initialize trainer
    trainer = Trainer(config)
    
    if progress_callback:
        trainer.add_progress_callback(progress_callback)
    
    # Load data
    handler = DatasetHandler()
    samples = handler.load_file(train_data_path)
    
    if not samples:
        raise ValueError(f"No training samples found in {train_data_path}")
    
    # Prepare datasets
    split_data = handler.prepare_dataset(samples)
    train_dataset = handler.to_hf_dataset(split_data["train"])
    eval_dataset = handler.to_hf_dataset(split_data["validation"]) if split_data["validation"] else None
    
    logger.info(f"Training samples: {len(split_data['train'])}")
    logger.info(f"Validation samples: {len(split_data['validation'])}")
    
    # Prepare model
    trainer.prepare_model(model_name)
    
    # Train
    adapter_path = trainer.train(train_dataset, eval_dataset, output_dir)
    
    # Cleanup
    trainer.cleanup()
    
    return adapter_path


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration loading
    print("=== Testing Training Configuration ===")
    
    try:
        config = TrainingConfig.from_preset("quick_test")
        print(f"Quick test preset loaded:")
        print(f"  Epochs: {config.epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Max seq length: {config.max_seq_length}")
        print(f"  LoRA r: {config.lora_r}")
    except FileNotFoundError:
        print("Config file not found - using defaults")
        config = TrainingConfig()
        print(f"Default config:")
        print(f"  Epochs: {config.epochs}")
        print(f"  Batch size: {config.batch_size}")
    
    print("\n=== GPU Check ===")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("CUDA not available - training will be slow on CPU")
    
    print("\n=== Trainer Ready ===")
    print("To run training, use train_cli.py or the web UI")
