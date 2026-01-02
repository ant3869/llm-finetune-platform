"""
CLI Training Test - Run a mini training session from command line

This tests the full training pipeline without the UI.
Run with: python tests/test_cli_training.py
"""

import sys
import os
from pathlib import Path
import tempfile
import json
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_test_dataset(output_path: Path, num_samples: int = 10):
    """Create a minimal training dataset."""
    samples = [
        {"instruction": "What is 1+1?", "input": "", "output": "The answer is 2."},
        {"instruction": "What color is the sky?", "input": "", "output": "The sky is blue."},
        {"instruction": "Say hello", "input": "", "output": "Hello! How can I help you?"},
        {"instruction": "What is Python?", "input": "", "output": "Python is a programming language."},
        {"instruction": "Count to 3", "input": "", "output": "1, 2, 3"},
        {"instruction": "What is AI?", "input": "", "output": "AI stands for Artificial Intelligence."},
        {"instruction": "Explain machine learning", "input": "briefly", "output": "Machine learning is teaching computers to learn from data."},
        {"instruction": "What is fine-tuning?", "input": "", "output": "Fine-tuning adapts a pre-trained model to specific tasks."},
        {"instruction": "Name a color", "input": "", "output": "Red is a color."},
        {"instruction": "What is 2+2?", "input": "", "output": "The answer is 4."},
    ][:num_samples]
    
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"✓ Created test dataset with {len(samples)} samples: {output_path}")
    return samples


def run_cli_training():
    """Run a CLI training test."""
    print("\n" + "="*70)
    print("🧪 CLI Training Test - Mini Training Pipeline")
    print("="*70)
    
    import torch
    
    # Check GPU
    print("\n📊 System Check:")
    print("-" * 40)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  ✓ GPU: {device_name}")
        print(f"  ✓ VRAM: {vram:.1f} GB")
    else:
        print("  ⚠ No GPU detected - training will be slow")
        response = input("Continue without GPU? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return False
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="llm_test_")
    data_path = Path(temp_dir) / "train_data.json"
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Working directory: {temp_dir}")
    
    try:
        # Step 1: Create dataset
        print("\n" + "-"*40)
        print("Step 1: Creating test dataset...")
        create_test_dataset(data_path, num_samples=5)
        
        # Step 2: Load and format data
        print("\nStep 2: Loading and formatting data...")
        from core.dataset_handler import DatasetHandler
        
        handler = DatasetHandler()
        samples = handler.load_file(str(data_path))
        dataset = handler.to_hf_dataset(samples)
        print(f"  ✓ Loaded {len(samples)} samples")
        print(f"  ✓ HF Dataset columns: {dataset.column_names}")
        print(f"  ✓ Sample text preview: {dataset[0]['text'][:100]}...")
        
        # Step 3: Configure training
        print("\nStep 3: Configuring trainer...")
        from core.trainer import TrainingConfig, Trainer
        
        config = TrainingConfig(
            model_name_or_path="microsoft/phi-2",  # Small model for testing
            epochs=1,
            batch_size=1,
            max_seq_length=128,  # Short for speed
            max_steps=3,  # Just 3 steps to verify pipeline
            learning_rate=2e-4,
            lora_r=8,
            lora_alpha=16,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=100,  # Don't save checkpoints during test
            gradient_checkpointing=True,
        )
        
        print(f"  ✓ Model: {config.model_name_or_path}")
        print(f"  ✓ Max steps: {config.max_steps}")
        print(f"  ✓ Sequence length: {config.max_seq_length}")
        print(f"  ✓ LoRA r={config.lora_r}, alpha={config.lora_alpha}")
        
        # Step 4: Load model
        print("\nStep 4: Loading model (this may take 1-2 minutes)...")
        trainer = Trainer(config)
        
        # Add progress callback
        def progress_callback(progress):
            if progress.status == "training":
                print(f"  📈 Step {progress.current_step}/{progress.total_steps} | "
                      f"Loss: {progress.loss:.4f} | "
                      f"VRAM: {progress.vram_used_gb:.2f}GB")
        
        trainer.add_progress_callback(progress_callback)
        trainer.prepare_model(config.model_name_or_path)
        print(f"  ✓ Model loaded")
        print(f"  ✓ VRAM used: {trainer.progress.vram_used_gb:.2f} GB")
        
        # Step 5: Run training
        print("\nStep 5: Running training...")
        print("-" * 40)
        
        adapter_path = trainer.train(
            train_dataset=dataset,
            output_dir=str(output_dir),
        )
        
        print("-" * 40)
        print(f"\n  ✓ Training complete!")
        print(f"  ✓ Adapter saved: {adapter_path}")
        
        # Step 6: Verify outputs
        print("\nStep 6: Verifying outputs...")
        if adapter_path and Path(adapter_path).exists():
            files = list(Path(adapter_path).glob("*"))
            print(f"  ✓ Adapter directory contains {len(files)} files:")
            for f in files[:5]:
                print(f"    - {f.name}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")
        else:
            print(f"  ⚠ Adapter path not found: {adapter_path}")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        trainer.cleanup()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n" + "="*70)
        print("✅ CLI Training Test PASSED!")
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


if __name__ == "__main__":
    success = run_cli_training()
    sys.exit(0 if success else 1)
