#!/usr/bin/env python3
"""
Quick Start Test Script
=======================

This script tests all major components to verify the installation works correctly.
Run this after installing requirements to ensure everything is set up properly.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so local packages import correctly
# (when running this script from the Documentation folder or elsewhere)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from PIL import Image
from CoreScripts.model_deit import create_deit_model, count_parameters
from CoreScripts.model_resnet_teacher import create_teacher_model


def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)
    
    try:
        import torch
        import torchvision
        import timm
        from facenet_pytorch import MTCNN
        import cv2
        import pandas as pd
        import sklearn
        import matplotlib
        import seaborn
        
        print("✓ All required packages imported successfully")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")  # type: ignore
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        return False


def test_models():
    """Test that models can be created."""
    print("\n" + "=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    try:        
        # Test DeiT student
        print("\nCreating DeiT student model...")
        student = create_deit_model(pretrained=False, num_classes=2)
        num_params = count_parameters(student)
        print(f"✓ DeiT model created ({num_params:,} parameters)")
        
        # Test ResNet50 teacher
        print("\nCreating ResNet50 teacher model...")
        teacher = create_teacher_model(pretrained=False, num_classes=2)
        print(f"✓ ResNet50 teacher created")
        
        # Test forward pass (use model's expected input size)
        print("\nTesting forward pass...")
        dummy_input = torch.randn(2, 3, 224, 224)
        
        student.eval()
        with torch.no_grad():
            output = student(dummy_input)
        
        print(f"✓ Forward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_dataset():
    """Test dataset utilities."""
    print("\n" + "=" * 60)
    print("Testing Dataset Utilities")
    print("=" * 60)
    
    try:
        from Utils.transforms import get_train_transforms, get_val_transforms
        
        # Test transforms
        print("\nTesting transforms...")
        train_transform = get_train_transforms(256)
        val_transform = get_val_transforms(256)
        
        # Create dummy image
        dummy_img = Image.new('RGB', (256, 256), color='red')
        
        # Apply transforms
        transformed = train_transform(dummy_img)
        print(f"✓ Transforms working (output shape: {transformed})")
        # print(f"✓ Transforms working (output shape: {transformed.shape})")
        
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False


def test_loss():
    """Test distillation loss."""
    print("\n" + "=" * 60)
    print("Testing Distillation Loss")
    print("=" * 60)
    
    try:
        from CoreScripts.distillation_loss import create_distillation_loss
        
        # Create loss function
        print("\nCreating distillation loss...")
        criterion = create_distillation_loss()
        
        # Test loss computation
        batch_size = 4
        class_logits = torch.randn(batch_size, 2)
        distill_logits = torch.randn(batch_size, 2)
        teacher_labels = torch.randint(0, 2, (batch_size,))
        ground_truth = torch.randint(0, 2, (batch_size,))
        
        total_loss, class_loss, distill_loss = criterion(
            (class_logits, distill_logits),
            teacher_labels,
            ground_truth
        )
        
        print(f"✓ Loss computation successful")
        print(f"  Total loss: {total_loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Loss test failed: {e}")
        return False


def test_metrics():
    """Test evaluation metrics."""
    print("\n" + "=" * 60)
    print("Testing Metrics")
    print("=" * 60)
    
    try:
        from Utils.metrics import compute_all_metrics, print_metrics
        
        # Dummy predictions
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 0])
        y_scores = np.random.rand(10)
        
        # Compute metrics
        metrics = compute_all_metrics(y_true, y_pred, y_scores)
        
        print(f"✓ Metrics computed successfully")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False


def test_visualization():
    """Test visualization utilities."""
    print("\n" + "=" * 60)
    print("Testing Visualization")
    print("=" * 60)
    
    try:
        from Utils.visualization import plot_training_curves
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Dummy training data
        train_losses = [0.8, 0.6, 0.5, 0.4]
        val_losses = [0.7, 0.65, 0.6, 0.55]
        train_accs = [60, 70, 75, 80]
        val_accs = [65, 68, 72, 75]
        
        # Create plot (don't show, just test creation)
        plot_training_curves(
            train_losses, val_losses, train_accs, val_accs,
            save_path=None, show=False
        )
        
        print(f"✓ Visualization working")
        
        return True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "PAIN DETECTION PROJECT - QUICK START TEST" + " " * 7 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Models", test_models()))
    results.append(("Dataset", test_dataset()))
    results.append(("Loss Function", test_loss()))
    results.append(("Metrics", test_metrics()))
    results.append(("Visualization", test_visualization()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:<20} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Installation verified.")
        print("\nYou're ready to:")
        print("  1. Prepare your dataset with prepare_dataset.py")
        print("  2. Preprocess faces with preprocess_faces.py")
        print("  3. Train your model with train.py")
        print("\nSee README.md for detailed usage instructions.")
    else:
        print("✗ Some tests failed. Please check error messages above.")
        print("\nTroubleshooting:")
        print("  1. Ensure all packages are installed: pip install -r requirements.txt")
        print("  2. Check Python version (requires 3.8+)")
        print("  3. For GPU support, ensure CUDA is properly installed")
    print("=" * 60)
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
