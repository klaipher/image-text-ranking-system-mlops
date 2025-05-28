#!/usr/bin/env python3
"""
Test script to verify the model setup and basic functionality.
"""

import sys
import traceback
from pathlib import Path
import torch

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src.config import model_config, data_config, training_config
        from src.models import create_model, DualEncoder
        from src.data import TextProcessor, Flickr8kDataLoader
        from src.training.trainer import Trainer
        from src.inference import create_predictor
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    try:
        from src.config import model_config, data_config, training_config
        
        assert model_config.embedding_dim == 256
        assert model_config.vocab_size == 10000
        assert model_config.image_size == (224, 224)
        assert data_config.train_split == 0.6
        assert training_config.random_seed == 42
        
        print(f"✅ Configuration loaded successfully!")
        print(f"   Device: {model_config.device}")
        print(f"   Embedding dim: {model_config.embedding_dim}")
        print(f"   Batch size: {model_config.batch_size}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_model_creation():
    """Test model creation and basic forward pass."""
    print("Testing model creation...")
    try:
        from src.models import create_model
        from src.config import model_config
        
        # Create model
        model = create_model()
        print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass with dummy data
        batch_size = 2
        dummy_images = torch.randn(batch_size, 3, 224, 224)
        dummy_text = torch.randint(0, 1000, (batch_size, 64))
        
        # Forward pass
        image_emb, text_emb = model(dummy_images, dummy_text)
        
        assert image_emb.shape == (batch_size, model_config.embedding_dim)
        assert text_emb.shape == (batch_size, model_config.embedding_dim)
        
        # Test similarity computation
        similarity = model.compute_similarity(image_emb, text_emb)
        assert similarity.shape == (batch_size, batch_size)
        
        # Test loss computation
        loss = model.contrastive_loss(image_emb, text_emb)
        assert loss.dim() == 0  # Scalar loss
        
        print("✅ Model creation and forward pass successful!")
        print(f"   Image embeddings shape: {image_emb.shape}")
        print(f"   Text embeddings shape: {text_emb.shape}")
        print(f"   Loss value: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False


def test_text_processor():
    """Test text processing functionality."""
    print("Testing text processor...")
    try:
        from src.data import TextProcessor
        
        # Create processor
        processor = TextProcessor(vocab_size=1000, max_length=32)
        
        # Test vocabulary building
        sample_texts = [
            "a dog playing in the park",
            "beautiful sunset over mountains",
            "people walking on the street",
            "children playing football"
        ]
        
        processor.build_vocabulary(sample_texts)
        assert processor.vocab_built
        assert len(processor.word_to_idx) > 4  # At least reserved tokens
        
        # Test encoding
        encoded = processor.encode("a dog playing")
        assert len(encoded) == 32  # max_length
        assert encoded[0] == processor.word_to_idx['<start>']
        
        # Test decoding
        decoded = processor.decode(encoded)
        assert isinstance(decoded, str)
        
        print("✅ Text processor working correctly!")
        print(f"   Vocabulary size: {len(processor.word_to_idx)}")
        print(f"   Sample encoding length: {len(encoded)}")
        print(f"   Sample decoded: '{decoded}'")
        return True
    except Exception as e:
        print(f"❌ Text processor error: {e}")
        return False


def test_data_loader():
    """Test data loader creation (without actual data)."""
    print("Testing data loader setup...")
    try:
        from src.data import Flickr8kDataLoader
        from src.config import data_config
        
        # Create loader (without downloading data)
        loader = Flickr8kDataLoader()
        
        # Check paths
        assert loader.data_dir == data_config.processed_data_path
        assert loader.text_processor is not None
        assert loader.train_transform is not None
        assert loader.eval_transform is not None
        
        print("✅ Data loader setup successful!")
        print(f"   Data directory: {loader.data_dir}")
        print(f"   Text processor vocab size: {loader.text_processor.vocab_size}")
        print("   Note: Dataset download test skipped (will download automatically when needed)")
        return True
    except Exception as e:
        print(f"❌ Data loader error: {e}")
        return False


def test_device_compatibility():
    """Test device compatibility (MPS/CUDA/CPU)."""
    print("Testing device compatibility...")
    try:
        from src.config import model_config
        
        device = model_config.device
        print(f"   Selected device: {device}")
        
        # Test tensor operations on device
        x = torch.randn(2, 3).to(device)
        y = torch.randn(2, 3).to(device)
        z = x + y
        
        assert z.device.type == device or device == 'cpu'
        
        print("✅ Device compatibility confirmed!")
        return True
    except Exception as e:
        print(f"❌ Device compatibility error: {e}")
        return False


def test_directory_creation():
    """Test that required directories are created."""
    print("Testing directory creation...")
    try:
        from src.config import data_config
        
        # Check if directories exist (should be created by config)
        required_dirs = [
            data_config.data_root,
            data_config.raw_data_path,
            data_config.processed_data_path,
            data_config.models_path,
            data_config.logs_path
        ]
        
        for dir_path in required_dirs:
            assert dir_path.exists(), f"Directory {dir_path} not created"
        
        print("✅ All required directories exist!")
        for dir_path in required_dirs:
            print(f"   {dir_path}")
        return True
    except Exception as e:
        print(f"❌ Directory creation error: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("BASELINE MODEL SETUP VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Model Creation Test", test_model_creation),
        ("Text Processor Test", test_text_processor),
        ("Data Loader Test", test_data_loader),
        ("Device Compatibility Test", test_device_compatibility),
        ("Directory Creation Test", test_directory_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            print(traceback.format_exc())
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The baseline model setup is ready.")
        print("\nNext steps:")
        print("1. Run: python train.py")
        print("2. After training, run: python inference_demo.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before proceeding.")
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 