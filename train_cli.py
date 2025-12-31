#!/usr/bin/env python3
"""
Command-Line Training Script for LLM Fine-Tuning Platform.

Simple CLI interface to run fine-tuning without the web UI.

Usage:
    python train_cli.py --model <hf_model_id> --data <data_path> [options]
    
Examples:
    # Quick test with small model
    python train_cli.py --model microsoft/phi-2 --data ./data/my_tickets.json --preset quick_test
    
    # Full training with Granite model
    python train_cli.py --model ibm-granite/granite-3.0-8b-instruct --data ./data/training.jsonl --preset balanced
    
    # Custom settings
    python train_cli.py --model microsoft/phi-2 --data ./data/data.json --epochs 5 --lr 1e-4 --lora-r 32
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.model_loader import ModelLoader
from core.dataset_handler import DatasetHandler
from core.trainer import Trainer, TrainingConfig, TrainingProgress


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


def print_progress(progress: TrainingProgress):
    """Print progress update to console."""
    if progress.total_steps > 0:
        pct = progress.current_step / progress.total_steps * 100
        bar_len = 30
        filled = int(bar_len * progress.current_step / progress.total_steps)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        eta_str = ""
        if progress.estimated_remaining > 0:
            eta_min = int(progress.estimated_remaining / 60)
            eta_sec = int(progress.estimated_remaining % 60)
            eta_str = f" ETA: {eta_min}m {eta_sec}s"
        
        print(
            f"\r[{bar}] {pct:5.1f}% | "
            f"Step {progress.current_step}/{progress.total_steps} | "
            f"Loss: {progress.loss:.4f} | "
            f"VRAM: {progress.vram_used_gb:.1f}/{progress.vram_total_gb:.1f}GB"
            f"{eta_str}",
            end="", flush=True
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLMs with QLoRA on consumer hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test run:
    python train_cli.py --model microsoft/phi-2 --data ./data/sample.json --preset quick_test
    
  Balanced training:
    python train_cli.py --model ibm-granite/granite-3.0-8b-instruct --data ./data/tickets.jsonl
    
  Custom configuration:
    python train_cli.py --model microsoft/phi-2 --data ./data/data.json \\
        --epochs 5 --lr 1e-4 --lora-r 32 --batch-size 2
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="HuggingFace model ID (e.g., 'microsoft/phi-2', 'ibm-granite/granite-3.0-8b-instruct')"
    )
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to training data file (JSON, JSONL, CSV, TXT, PDF, HTML)"
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        default="./models/adapters",
        help="Output directory for the trained adapter (default: ./models/adapters)"
    )
    
    # Preset selection
    parser.add_argument(
        "--preset", "-p",
        choices=["quick_test", "balanced", "thorough"],
        default="balanced",
        help="Training preset (default: balanced)"
    )
    
    # Training parameters (override preset)
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size per device")
    parser.add_argument("--lr", "--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, help="Maximum sequence length")
    parser.add_argument("--max-steps", type=int, help="Maximum training steps (-1 for epochs-based)")
    
    # LoRA parameters
    parser.add_argument("--lora-r", type=int, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, help="LoRA dropout")
    
    # Memory optimization
    parser.add_argument("--no-gradient-checkpointing", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--gradient-accumulation", type=int, help="Gradient accumulation steps")
    
    # Misc
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without training")
    parser.add_argument("--list-models", action="store_true", help="List local GGUF models and exit")
    parser.add_argument("--check-gpu", action="store_true", help="Check GPU availability and exit")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Handle info commands
    if args.list_models:
        loader = ModelLoader()
        models = loader.scan_models()
        if models:
            print("\nLocal GGUF Models:")
            print("-" * 60)
            for model in models:
                print(f"  {model.name}")
                print(f"    Size: {model.size_gb:.2f} GB")
                print(f"    VRAM (inference): ~{model.estimated_vram_gb:.2f} GB")
                print()
        else:
            print("\nNo local GGUF models found.")
            print("Place .gguf files in ./models/base/")
        return 0
    
    if args.check_gpu:
        loader = ModelLoader()
        gpu_info = loader.check_gpu_availability()
        print("\nGPU Information:")
        print("-" * 40)
        print(f"CUDA Available: {gpu_info['cuda_available']}")
        if gpu_info['devices']:
            for device in gpu_info['devices']:
                print(f"\n  Device {device['index']}: {device['name']}")
                print(f"    Total VRAM: {device['total_memory_gb']:.2f} GB")
                print(f"    Free VRAM: {device['free_memory_gb']:.2f} GB")
        else:
            print("No CUDA devices found.")
        return 0
    
    # Validate data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return 1
    
    print("\n" + "=" * 60)
    print("  LLM Fine-Tuning Platform - CLI Trainer")
    print("=" * 60)
    
    # Load dataset
    print(f"\n📄 Loading data: {data_path}")
    handler = DatasetHandler()
    
    try:
        samples = handler.load_file(data_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    if not samples:
        logger.error("No training samples found in data file")
        return 1
    
    stats = handler.get_statistics(samples)
    print(f"   Samples loaded: {stats.total_samples}")
    print(f"   Train/Val split: {stats.train_samples}/{stats.validation_samples}")
    print(f"   Avg input length: {stats.avg_input_length:.0f} words")
    print(f"   Avg output length: {stats.avg_output_length:.0f} words")
    
    if stats.warnings:
        print("\n   ⚠️  Warnings:")
        for warning in stats.warnings:
            print(f"      - {warning}")
    
    # Load training config
    print(f"\n⚙️  Loading preset: {args.preset}")
    
    try:
        config = TrainingConfig.from_preset(args.preset)
    except FileNotFoundError:
        logger.warning("Config file not found, using defaults")
        config = TrainingConfig()
    
    config.model_name_or_path = args.model
    config.output_dir = args.output
    
    # Apply CLI overrides
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.max_seq_length:
        config.max_seq_length = args.max_seq_length
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.lora_r:
        config.lora_r = args.lora_r
    if args.lora_alpha:
        config.lora_alpha = args.lora_alpha
    if args.lora_dropout:
        config.lora_dropout = args.lora_dropout
    if args.no_gradient_checkpointing:
        config.gradient_checkpointing = False
    if args.gradient_accumulation:
        config.gradient_accumulation_steps = args.gradient_accumulation
    
    print(f"\n🔧 Training Configuration:")
    print(f"   Model: {config.model_name_or_path}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"   Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Max seq length: {config.max_seq_length}")
    print(f"   LoRA rank: {config.lora_r}")
    print(f"   Output: {config.output_dir}")
    
    # Estimate VRAM
    loader = ModelLoader()
    estimate = loader.estimate_training_vram(
        model_size_gb=4.0,  # Rough estimate for 7B Q4
        batch_size=config.batch_size,
        seq_length=config.max_seq_length
    )
    
    print(f"\n💾 Estimated VRAM Usage:")
    print(f"   Model (4-bit): ~{estimate['model']:.1f} GB")
    print(f"   LoRA adapters: ~{estimate['lora_adapters']:.1f} GB")
    print(f"   Optimizer: ~{estimate['optimizer']:.1f} GB")
    print(f"   Activations: ~{estimate['activations']:.1f} GB")
    print(f"   Total: ~{estimate['total']:.1f} GB")
    print(f"   Fits 8GB VRAM: {'✅ Yes' if estimate['fits_8gb'] else '❌ No - reduce batch size or seq length'}")
    
    if args.dry_run:
        print("\n🔍 Dry run complete - no training performed")
        return 0
    
    # Prepare datasets
    split_data = handler.prepare_dataset(samples)
    train_dataset = handler.to_hf_dataset(split_data["train"])
    eval_dataset = handler.to_hf_dataset(split_data["validation"]) if split_data["validation"] else None
    
    # Initialize trainer
    trainer = Trainer(config)
    trainer.add_progress_callback(print_progress)
    
    print(f"\n🚀 Starting training...")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    try:
        # Prepare model
        trainer.prepare_model(config.model_name_or_path)
        
        # Run training
        adapter_path = trainer.train(train_dataset, eval_dataset, config.output_dir)
        
        print("\n")  # New line after progress bar
        print("-" * 60)
        print(f"\n✅ Training complete!")
        print(f"   Adapter saved to: {adapter_path}")
        print(f"   Total time: {trainer.progress.elapsed_time / 60:.1f} minutes")
        print(f"   Final loss: {trainer.progress.loss:.4f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        trainer.stop_training()
        return 130
        
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        logger.exception("Training error")
        return 1
        
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    sys.exit(main())
