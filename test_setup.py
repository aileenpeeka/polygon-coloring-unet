#!/usr/bin/env python3
"""
Test script to verify the Ayna ML Assignment setup.
This script tests basic functionality without requiring the full dataset.
"""

import torch
import numpy as np
from PIL import Image
import os
import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    try:
        from model.unet import UNet
        from utils import PolygonDataset, COLOR_MAP
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model():
    """Test UNet model creation and forward pass."""
    print("\nTesting UNet model...")
    try:
        from model.unet import UNet
        
        # Create model
        model = UNet()
        print(f"✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 2
        input_tensor = torch.randn(batch_size, 4, 256, 256)  # 4 channels (RGB + color condition)
        output = model(input_tensor)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {input_tensor.shape}")
        print(f"   Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_dataset():
    """Test dataset creation and loading."""
    print("\nTesting dataset...")
    try:
        from utils import PolygonDataset
        from torchvision import transforms
        
        # Create dummy dataset
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Test with non-existent directory (should handle gracefully)
        dataset = PolygonDataset("non_existent_dir", transform=transform)
        print(f"✅ Dataset created (empty: {len(dataset) == 0})")
        
        return True
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def test_color_mapping():
    """Test color mapping functionality."""
    print("\nTesting color mapping...")
    try:
        from utils import COLOR_MAP
        
        print("Color mappings:")
        for color, rgb in COLOR_MAP.items():
            print(f"   {color}: RGB{rgb}")
        
        print("✅ Color mapping test successful")
        return True
    except Exception as e:
        print(f"❌ Color mapping test failed: {e}")
        return False

def test_device():
    """Test device availability."""
    print("\nTesting device availability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("✅ Device test successful")
    return True

def create_dummy_data():
    """Create dummy data for testing."""
    print("\nCreating dummy data...")
    try:
        # Create directories
        os.makedirs("dataset/training/inputs", exist_ok=True)
        os.makedirs("dataset/training/outputs", exist_ok=True)
        os.makedirs("dataset/validation/inputs", exist_ok=True)
        os.makedirs("dataset/validation/outputs", exist_ok=True)
        
        # Create dummy images
        dummy_img = Image.new('RGB', (256, 256), color='white')
        dummy_img.save("dataset/training/inputs/dummy.png")
        dummy_img.save("dataset/training/outputs/dummy.png")
        dummy_img.save("dataset/validation/inputs/dummy.png")
        dummy_img.save("dataset/validation/outputs/dummy.png")
        
        # Create dummy data.json
        dummy_data = [
            {
                "input": "dummy.png",
                "output": "dummy.png",
                "color": "red"
            }
        ]
        
        import json
        with open("dataset/training/data.json", "w") as f:
            json.dump(dummy_data, f, indent=2)
        
        with open("dataset/validation/data.json", "w") as f:
            json.dump(dummy_data, f, indent=2)
        
        print("✅ Dummy data created successfully")
        return True
    except Exception as e:
        print(f"❌ Dummy data creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Ayna ML Assignment - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model,
        test_dataset,
        test_color_mapping,
        test_device,
        create_dummy_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Add your dataset to the dataset/ directory")
        print("2. Run: python train.py")
        print("3. Open inference.ipynb for testing")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 