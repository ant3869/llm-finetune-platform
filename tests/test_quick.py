"""
Quick Tests for LLM Fine-Tuning Platform

Run with: python -m pytest tests/test_quick.py -v
Or directly: python tests/test_quick.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import tempfile
import json
import shutil


class TestDataHandler(unittest.TestCase):
    """Test data loading and processing."""
    
    def setUp(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample training data
        self.sample_data = [
            {
                "instruction": "What is Python?",
                "input": "",
                "output": "Python is a programming language."
            },
            {
                "instruction": "Explain machine learning",
                "input": "in simple terms",
                "output": "Machine learning is teaching computers to learn from data."
            },
            {
                "instruction": "What is fine-tuning?",
                "input": "",
                "output": "Fine-tuning is adapting a pre-trained model to a specific task."
            }
        ]
        
        # Save as JSON
        self.json_path = Path(self.temp_dir) / "test_data.json"
        with open(self.json_path, "w") as f:
            json.dump(self.sample_data, f)
        
        # Save as JSONL
        self.jsonl_path = Path(self.temp_dir) / "test_data.jsonl"
        with open(self.jsonl_path, "w") as f:
            for item in self.sample_data:
                f.write(json.dumps(item) + "\n")
    
    def tearDown(self):
        """Clean up temp files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_handler_import(self):
        """Test that DatasetHandler can be imported."""
        from core.dataset_handler import DatasetHandler
        self.assertTrue(True)
    
    def test_load_json(self):
        """Test loading JSON training data."""
        from core.dataset_handler import DatasetHandler
        
        handler = DatasetHandler()
        samples = handler.load_file(str(self.json_path))
        
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].instruction, "What is Python?")
    
    def test_load_jsonl(self):
        """Test loading JSONL training data."""
        from core.dataset_handler import DatasetHandler
        
        handler = DatasetHandler()
        samples = handler.load_file(str(self.jsonl_path))
        
        self.assertEqual(len(samples), 3)
    
    def test_to_hf_dataset(self):
        """Test converting samples to HuggingFace dataset."""
        from core.dataset_handler import DatasetHandler, TrainingSample
        
        handler = DatasetHandler()
        sample = TrainingSample(
            instruction="Test instruction",
            input="Test input",
            output="Test output"
        )
        
        dataset = handler.to_hf_dataset([sample])
        self.assertEqual(len(dataset), 1)
        self.assertIn("text", dataset.column_names)


class TestDataCleaner(unittest.TestCase):
    """Test data cleaning functionality."""
    
    def test_import(self):
        """Test that data cleaner can be imported."""
        from core.data_cleaner import DataCleaningPipeline, CleaningConfig, TextCleaner
        self.assertTrue(True)
    
    def test_basic_cleaning(self):
        """Test basic text cleaning."""
        from core.data_cleaner import TextCleaner, CleaningConfig
        
        config = CleaningConfig(normalize_whitespace=True)
        cleaner = TextCleaner(config)
        
        text = "Hello    world  \n\n  test"
        cleaned = cleaner.clean_text(text)
        
        self.assertNotIn("    ", cleaned)
    
    def test_delimiter_removal(self):
        """Test prefix delimiter removal."""
        from core.data_cleaner import TextCleaner, CleaningConfig
        
        config = CleaningConfig(prefix_delimiter="XX |")
        cleaner = TextCleaner(config)
        
        text = "XX | This is the actual content"
        cleaned = cleaner.clean_text(text)
        
        self.assertEqual(cleaned.strip(), "This is the actual content")
    
    def test_pipeline_presets(self):
        """Test cleaning pipeline with preset."""
        from core.data_cleaner import DataCleaningPipeline
        
        # Test all presets work
        for preset in ["minimal", "standard", "aggressive", "it_support"]:
            pipeline = DataCleaningPipeline(preset=preset)
            self.assertIsNotNone(pipeline)


class TestModelLoader(unittest.TestCase):
    """Test model loading utilities."""
    
    def test_import(self):
        """Test that ModelLoader can be imported."""
        from core.model_loader import ModelLoader
        self.assertTrue(True)
    
    def test_loader_init(self):
        """Test ModelLoader initialization."""
        from core.model_loader import ModelLoader
        
        loader = ModelLoader()
        self.assertIsNotNone(loader)
    
    def test_gguf_mapping(self):
        """Test GGUF to HuggingFace model mapping."""
        from core.model_loader import ModelLoader
        
        loader = ModelLoader()
        
        # Test known mapping
        hf_id = loader.get_hf_model_id("phi-2")
        self.assertIn("phi", hf_id.lower())


class TestTrainerConfig(unittest.TestCase):
    """Test trainer configuration."""
    
    def test_import(self):
        """Test that Trainer can be imported."""
        from core.trainer import Trainer, TrainingConfig
        self.assertTrue(True)
    
    def test_config_defaults(self):
        """Test TrainingConfig default values."""
        from core.trainer import TrainingConfig
        
        config = TrainingConfig(model_name_or_path="test/model")
        
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.lora_r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertTrue(config.gradient_checkpointing)
    
    def test_config_custom(self):
        """Test TrainingConfig with custom values."""
        from core.trainer import TrainingConfig
        
        config = TrainingConfig(
            model_name_or_path="test/model",
            epochs=5,
            learning_rate=1e-5,
            lora_r=32
        )
        
        self.assertEqual(config.epochs, 5)
        self.assertEqual(config.learning_rate, 1e-5)
        self.assertEqual(config.lora_r, 32)


class TestOfflineModels(unittest.TestCase):
    """Test offline model support."""
    
    def test_import(self):
        """Test that offline models module can be imported."""
        from core.offline_models import OfflineModelManager
        self.assertTrue(True)
    
    def test_manager_init(self):
        """Test OfflineModelManager initialization."""
        from core.offline_models import OfflineModelManager
        
        manager = OfflineModelManager()
        self.assertIsNotNone(manager)
    
    def test_available_models(self):
        """Test getting available offline models."""
        from core.offline_models import OfflineModelManager
        
        manager = OfflineModelManager()
        models = manager.get_available_offline_models()
        
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
    
    def test_download_instructions(self):
        """Test generating download instructions."""
        from core.offline_models import OfflineModelManager
        
        manager = OfflineModelManager()
        models = manager.get_available_offline_models()
        
        if models:
            instructions = manager.generate_download_instructions(models[0]["key"])
            self.assertIsInstance(instructions, str)
            self.assertGreater(len(instructions), 0)


class TestGPU(unittest.TestCase):
    """Test GPU detection."""
    
    def test_torch_import(self):
        """Test PyTorch import."""
        import torch
        self.assertTrue(True)
    
    def test_cuda_check(self):
        """Test CUDA availability check (doesn't require CUDA)."""
        import torch
        
        # Just check we can query CUDA status
        cuda_available = torch.cuda.is_available()
        self.assertIsInstance(cuda_available, bool)
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            self.assertGreaterEqual(device_count, 1)
            print(f"\n  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("\n  ⚠ CUDA not available (CPU only)")


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests (minimal)."""
    
    def setUp(self):
        """Create minimal test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Minimal training data
        self.sample_data = [
            {"instruction": "Say hello", "input": "", "output": "Hello!"},
            {"instruction": "Say goodbye", "input": "", "output": "Goodbye!"},
        ]
        
        self.data_path = Path(self.temp_dir) / "mini_data.json"
        with open(self.data_path, "w") as f:
            json.dump(self.sample_data, f)
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_data_pipeline(self):
        """Test full data loading and formatting pipeline."""
        from core.dataset_handler import DatasetHandler
        
        handler = DatasetHandler()
        
        # Load
        samples = handler.load_file(str(self.data_path))
        self.assertEqual(len(samples), 2)
        
        # Convert to HF dataset
        dataset = handler.to_hf_dataset(samples)
        self.assertEqual(len(dataset), 2)
        
        # Each should have 'text' field
        self.assertIn("text", dataset.column_names)


def run_quick_tests():
    """Run quick validation tests."""
    print("\n" + "="*60)
    print("🧪 LLM Fine-Tuning Platform - Quick Tests")
    print("="*60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestDataCleaner))
    suite.addTests(loader.loadTestsFromTestCase(TestModelLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainerConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestOfflineModels))
    suite.addTests(loader.loadTestsFromTestCase(TestGPU))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
    print("="*60 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_quick_tests()
    sys.exit(0 if success else 1)
