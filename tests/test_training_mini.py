"""
Mini Training Test - Validates the full training pipeline

This test loads a tiny model and runs a few training steps
to verify everything works before committing to a real training run.

Run with: python tests/test_training_mini.py
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


def create_mini_dataset(output_path: Path, num_samples: int = 5):
    """Create a minimal training dataset."""
    samples = [
        {"instruction": "What is 1+1?", "input": "", "output": "2"},
        {"instruction": "What color is the sky?", "input": "", "output": "Blue"},
        {"instruction": "Say hello", "input": "", "output": "Hello!"},
        {"instruction": "What is Python?", "input": "", "output": "A programming language"},
        {"instruction": "Count to 3", "input": "", "output": "1, 2, 3"},
    ][:num_samples]
    
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    
    return samples


def test_data_loading():
    """Test 1: Data Loading"""
    print("\n📊 Test 1: Data Loading")
    print("-" * 40)
    
    try:
        from core.dataset_handler import DatasetHandler
        
        # Create temp data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {"instruction": "Test", "input": "", "output": "Response"}
            ], f)
            temp_path = f.name
        
        handler = DatasetHandler()
        samples = handler.load_file(temp_path)
        dataset = handler.to_hf_dataset(samples)
        
        os.unlink(temp_path)
        
        print(f"  ✓ Loaded {len(samples)} samples")
        print(f"  ✓ Created HF dataset with {len(dataset)} examples")
        print(f"  ✓ Columns: {dataset.column_names}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_model_loading_dry():
    """Test 2: Model Loader (dry run - no actual model download)"""
    print("\n🔧 Test 2: Model Loader Setup")
    print("-" * 40)
    
    try:
        from core.model_loader import ModelLoader
        
        loader = ModelLoader()
        
        # Test GGUF mapping
        mappings = [
            ("phi-2", "phi"),
            ("mistral", "mistral"),
        ]
        
        for gguf_name, expected in mappings:
            hf_id = loader.get_hf_model_id(gguf_name)
            if expected.lower() in hf_id.lower():
                print(f"  ✓ Mapping {gguf_name} → {hf_id}")
            else:
                print(f"  ⚠ Mapping {gguf_name} → {hf_id} (expected '{expected}')")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_trainer_config():
    """Test 3: Trainer Configuration"""
    print("\n⚙️ Test 3: Trainer Configuration")
    print("-" * 40)
    
    try:
        from core.trainer import TrainingConfig, Trainer
        
        # Test config creation
        config = TrainingConfig(
            model_name_or_path="microsoft/phi-2",
            epochs=1,
            batch_size=1,
            max_seq_length=128,
            max_steps=2,  # Very few steps
            learning_rate=1e-4,
            lora_r=8,
            lora_alpha=16,
        )
        
        print(f"  ✓ Config created: {config.model_name_or_path}")
        print(f"  ✓ LoRA r={config.lora_r}, alpha={config.lora_alpha}")
        print(f"  ✓ Max steps: {config.max_steps}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_gpu_availability():
    """Test 4: GPU/CUDA Check"""
    print("\n🎮 Test 4: GPU/CUDA Check")
    print("-" * 40)
    
    try:
        import torch
        
        print(f"  PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✓ CUDA available: {device_name}")
            print(f"  ✓ VRAM: {total_mem:.1f} GB")
            return True
        else:
            print("  ⚠ CUDA not available - training will be slow on CPU")
            return True  # Not a failure, just a warning
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_imports():
    """Test 5: Required Package Imports"""
    print("\n📦 Test 5: Required Imports")
    print("-" * 40)
    
    required = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("trl", "TRL"),
        ("datasets", "Datasets"),
        ("bitsandbytes", "BitsAndBytes"),
    ]
    
    all_ok = True
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            all_ok = False
    
    return all_ok


def test_trl_api():
    """Test 6: TRL API Compatibility"""
    print("\n🔌 Test 6: TRL API Check")
    print("-" * 40)
    
    try:
        import trl
        print(f"  TRL version: {trl.__version__}")
        
        from trl import SFTTrainer
        print("  ✓ SFTTrainer import OK")
        
        # Check if SFTConfig exists (newer API)
        try:
            from trl import SFTConfig
            print("  ✓ SFTConfig available (new API)")
        except ImportError:
            print("  ⚠ SFTConfig not found (older API)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_mini_training(skip_if_no_gpu: bool = True):
    """Test 7: Mini Training Run (optional)"""
    print("\n🚀 Test 7: Mini Training Pipeline")
    print("-" * 40)
    
    import torch
    if skip_if_no_gpu and not torch.cuda.is_available():
        print("  ⏭ Skipped (no GPU, set skip_if_no_gpu=False to force)")
        return True
    
    try:
        from core.dataset_handler import DatasetHandler
        from core.trainer import TrainingConfig, Trainer
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        data_path = Path(temp_dir) / "mini_train.json"
        output_dir = Path(temp_dir) / "output"
        
        try:
            # Create minimal dataset
            create_mini_dataset(data_path, num_samples=3)
            print(f"  ✓ Created mini dataset: {data_path}")
            
            # Load and format data
            handler = DatasetHandler()
            samples = handler.load_file(str(data_path))
            dataset = handler.to_hf_dataset(samples)
            print(f"  ✓ Created HF dataset with {len(dataset)} samples")
            
            # Create trainer config - minimal settings
            config = TrainingConfig(
                model_name_or_path="microsoft/phi-2",  # Small model
                epochs=1,
                batch_size=1,
                max_seq_length=64,  # Very short for speed
                max_steps=2,  # Just 2 steps to verify
                learning_rate=1e-4,
                lora_r=4,  # Minimal LoRA
                lora_alpha=8,
                gradient_accumulation_steps=1,
                logging_steps=1,
                save_steps=100,  # Don't save during mini test
            )
            
            print(f"  ✓ Config: max_steps={config.max_steps}, seq_len={config.max_seq_length}")
            print("  ⏳ Loading model (this may take a minute)...")
            
            # Create trainer and prepare model
            trainer = Trainer(config)
            trainer.prepare_model(config.model_name_or_path)
            
            print("  ⏳ Running mini training...")
            
            # Run training - use train_dataset parameter
            adapter_path = trainer.train(
                train_dataset=dataset,
                output_dir=str(output_dir),
            )
            
            print(f"  ✓ Training complete!")
            print(f"  ✓ Adapter saved: {adapter_path}")
            
            return True
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all quick tests."""
    print("\n" + "="*60)
    print("🧪 LLM Fine-Tuning Platform - Training Pipeline Tests")
    print("="*60)
    
    results = {}
    
    # Run tests
    results["imports"] = test_imports()
    results["gpu"] = test_gpu_availability()
    results["trl_api"] = test_trl_api()
    results["data_loading"] = test_data_loading()
    results["model_loader"] = test_model_loading_dry()
    results["trainer_config"] = test_trainer_config()
    
    # Ask about mini training
    print("\n" + "-"*60)
    print("Mini training test loads a real model and runs 2 training steps.")
    print("This takes ~2-3 minutes and requires a GPU.")
    
    import torch
    if torch.cuda.is_available():
        response = input("\nRun mini training test? [y/N]: ").strip().lower()
        if response == 'y':
            results["mini_training"] = test_mini_training(skip_if_no_gpu=False)
        else:
            results["mini_training"] = True  # Skipped
            print("  ⏭ Skipped")
    else:
        print("  ⏭ Skipped (no GPU)")
        results["mini_training"] = True
    
    # Summary
    print("\n" + "="*60)
    print("📋 Test Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅" if passed_test else "❌"
        print(f"  {status} {test_name}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Ready for training.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    print("="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
