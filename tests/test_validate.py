"""
Platform Validation Test - Quick non-interactive check

This runs all the basic checks without requiring user input.
Run with: python tests/test_validate.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Run validation tests."""
    print("\n" + "="*60)
    print("🧪 LLM Fine-Tuning Platform - Validation")
    print("="*60)
    
    errors = []
    
    # Test 1: Required imports
    print("\n📦 Checking required imports...")
    required = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("trl", "TRL"),
        ("datasets", "Datasets"),
        ("bitsandbytes", "BitsAndBytes"),
        ("streamlit", "Streamlit"),
    ]
    
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            errors.append(f"Import {name} failed")
    
    # Test 2: Core module imports
    print("\n📁 Checking core modules...")
    core_modules = [
        ("core.dataset_handler", "DatasetHandler"),
        ("core.trainer", "Trainer, TrainingConfig"),
        ("core.model_loader", "ModelLoader"),
        ("core.data_cleaner", "DataCleaningPipeline"),
        ("core.offline_models", "OfflineModelManager"),
    ]
    
    for module, components in core_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            errors.append(f"Core module {module} failed")
    
    # Test 3: GPU Check
    print("\n🎮 Checking GPU...")
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  ✓ CUDA available: {device_name}")
        print(f"  ✓ VRAM: {total_mem:.1f} GB")
    else:
        print("  ⚠ No GPU detected - training will be slow")
    
    # Test 4: TRL API Check
    print("\n🔌 Checking TRL API...")
    try:
        import trl
        print(f"  ✓ TRL version: {trl.__version__}")
        
        from trl import SFTTrainer
        print("  ✓ SFTTrainer available")
        
        try:
            from trl import SFTConfig
            print("  ✓ SFTConfig available (new API)")
        except ImportError:
            print("  ⚠ SFTConfig not found - using older API")
    except Exception as e:
        print(f"  ✗ TRL error: {e}")
        errors.append("TRL API check failed")
    
    # Test 5: Data Pipeline
    print("\n📊 Testing data pipeline...")
    try:
        import tempfile
        import json
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
        
        import os
        os.unlink(temp_path)
        
        print(f"  ✓ Data loading works")
        print(f"  ✓ HF dataset creation works ({len(dataset)} samples)")
    except Exception as e:
        print(f"  ✗ Data pipeline error: {e}")
        errors.append(f"Data pipeline: {e}")
    
    # Test 6: Trainer Config
    print("\n⚙️ Testing trainer config...")
    try:
        from core.trainer import TrainingConfig, Trainer
        
        config = TrainingConfig(
            model_name_or_path="microsoft/phi-2",
            epochs=1,
            batch_size=1,
            max_seq_length=128,
        )
        
        print(f"  ✓ TrainingConfig creation works")
        print(f"  ✓ Trainer class available")
    except Exception as e:
        print(f"  ✗ Trainer config error: {e}")
        errors.append(f"Trainer config: {e}")
    
    # Summary
    print("\n" + "="*60)
    if not errors:
        print("✅ All validation checks passed!")
        print("\nPlatform is ready for use. You can:")
        print("  • Run: streamlit run ui/app.py")
        print("  • Or run mini training test: python tests/test_training_mini.py")
    else:
        print(f"❌ {len(errors)} issues found:")
        for err in errors:
            print(f"  • {err}")
    print("="*60 + "\n")
    
    return len(errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
