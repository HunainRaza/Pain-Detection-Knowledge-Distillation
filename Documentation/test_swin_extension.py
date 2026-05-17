#!/usr/bin/env python3
"""
Swin Transformer Extension Test Script
=======================================

Tests the Swin Transformer teacher implementation and verifies:
1. Swin Transformer models are available in timm
2. Teacher model can be created and frozen
3. Forward pass works correctly
4. Hard label generation works
5. Integration with existing training pipeline
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Ensure project imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CoreScripts.teacher_swin.model_swin_teacher import create_swin_teacher, count_parameters
from CoreScripts.model_deit import create_deit_model
from CoreScripts.distillation_loss import create_distillation_loss
from CoreScripts.model_resnet_teacher import create_teacher_model as create_resnet_teacher


def test_timm_swin_availability():
    """Test that Swin Transformer models are available in timm."""
    print("=" * 60)
    print("Test 1: Swin Transformer Availability")
    print("=" * 60)
    
    try:
        import timm
        
        # List available Swin models
        swin_models = [m for m in timm.list_models() if 'swin' in m.lower()]
        
        print(f"\n✓ timm version: {timm.__version__}")
        print(f"✓ Found {len(swin_models)} Swin Transformer variants")
        
        print("\nAvailable Swin models:")
        for model in swin_models[:10]:  # Show first 10
            print(f"  - {model}")
        
        if len(swin_models) > 10:
            print(f"  ... and {len(swin_models) - 10} more")
        
        # Check for our default model
        default_model = 'swin_base_patch4_window7_224'
        if default_model in swin_models:
            print(f"\n✓ Default model '{default_model}' is available")
            return True
        else:
            print(f"\n✗ Default model '{default_model}' NOT found")
            print("  Please update timm: pip install --upgrade timm")
            return False
            
    except ImportError as e:
        print(f"\n✗ Error importing timm: {e}")
        print("  Please install: pip install timm")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def test_swin_teacher_creation():
    """Test Swin Transformer teacher model creation."""
    print("\n" + "=" * 60)
    print("Test 2: Swin Teacher Model Creation")
    print("=" * 60)
    
    try:       
        # Create teacher model (without pretrained weights for speed)
        print("\nCreating Swin Transformer teacher...")
        teacher = create_swin_teacher(
            pretrained=False,
            num_classes=2,
            model_variant='swin_base_patch4_window7_224',
            freeze=True
        )
        
        # Count parameters
        total_params, trainable_params = count_parameters(teacher)
        
        print(f"✓ Swin teacher created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Verify it's frozen
        if trainable_params == 0:
            print(f"✓ Model is properly frozen (0 trainable parameters)")
            return True
        else:
            print(f"✗ Model is not frozen ({trainable_params:,} trainable parameters)")
            return False
            
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("  Make sure model_swin_teacher.py is in the current directory")
        return False
    except Exception as e:
        print(f"\n✗ Error creating teacher: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass through Swin teacher."""
    print("\n" + "=" * 60)
    print("Test 3: Forward Pass")
    print("=" * 60)
    
    try:        
        teacher = create_swin_teacher(pretrained=False, num_classes=2, freeze=True)
        teacher.eval()
        
        # Test with different input sizes
        test_cases = [
            (4, 3, 224, 224, "Standard 224x224"),
            (4, 3, 256, 256, "256x256 (auto-resized)"),
        ]
        
        for batch, channels, height, width, description in test_cases:
            print(f"\nTesting {description}:")
            
            # Create dummy input
            dummy_input = torch.randn(batch, channels, height, width)
            
            # Forward pass
            with torch.no_grad():
                output = teacher(dummy_input)
            
            # Check output shape
            expected_shape = (batch, 2)
            if output.shape == expected_shape:
                print(f"  ✓ Output shape: {output.shape} (correct)")
            else:
                print(f"  ✗ Output shape: {output.shape} (expected {expected_shape})")
                return False
        
        print("\n✓ All forward pass tests passed")
        return True
        
    except Exception as e:
        print(f"\n✗ Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hard_label_generation():
    """Test hard label generation from teacher."""
    print("\n" + "=" * 60)
    print("Test 4: Hard Label Generation")
    print("=" * 60)
    
    try:       
        teacher = create_swin_teacher(pretrained=False, num_classes=2, freeze=True)
        teacher.eval()
        
        # Create dummy input
        batch_size = 8
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        # Get hard labels
        hard_labels = teacher.get_hard_labels(dummy_input)
        
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Hard labels shape: {hard_labels.shape}")
        print(f"Hard labels: {hard_labels}")
        
        # Verify labels are in correct range
        if hard_labels.shape == (batch_size,):
            print(f"✓ Hard labels shape is correct")
        else:
            print(f"✗ Hard labels shape incorrect (expected ({batch_size},))")
            return False
        
        if torch.all((hard_labels >= 0) & (hard_labels <= 1)):
            print(f"✓ All labels are in range [0, 1]")
        else:
            print(f"✗ Some labels are outside range [0, 1]")
            return False
        
        print("\n✓ Hard label generation works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ Hard label generation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_with_resnet():
    """Compare Swin teacher with ResNet teacher for consistency."""
    print("\n" + "=" * 60)
    print("Test 5: Comparison with ResNet Teacher")
    print("=" * 60)
    
    try:
        # Create both teachers
        swin_teacher = create_swin_teacher(pretrained=False, num_classes=2, freeze=True)
        resnet_teacher = create_resnet_teacher(pretrained=False, num_classes=2, freeze=True)
        
        swin_teacher.eval()
        resnet_teacher.eval()
        
        # Create same input
        dummy_input = torch.randn(4, 3, 224, 224)
        
        # Get outputs
        with torch.no_grad():
            swin_output = swin_teacher(dummy_input)
            resnet_output = resnet_teacher(dummy_input)
            
            swin_labels = swin_teacher.get_hard_labels(dummy_input)
            resnet_labels = resnet_teacher.get_hard_labels(dummy_input)
        
        print(f"\nSwin output shape: {swin_output.shape}")
        print(f"ResNet output shape: {resnet_output.shape}")
        print(f"\nSwin hard labels: {swin_labels}")
        print(f"ResNet hard labels: {resnet_labels}")
        
        # Verify shapes match
        if swin_output.shape == resnet_output.shape:
            print(f"\n✓ Both teachers produce same output shape")
        else:
            print(f"\n✗ Output shapes don't match")
            return False
        
        if swin_labels.shape == resnet_labels.shape:
            print(f"✓ Both teachers produce same label shape")
        else:
            print(f"✗ Label shapes don't match")
            return False
        
        print("\n✓ Swin and ResNet teachers are compatible")
        return True
        
    except Exception as e:
        print(f"\n✗ Comparison error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration with distillation loss."""
    print("\n" + "=" * 60)
    print("Test 6: Integration with Distillation Loss")
    print("=" * 60)
    
    try:   
        # Create models
        teacher = create_swin_teacher(pretrained=False, num_classes=2, freeze=True)
        student = create_deit_model(pretrained=False, num_classes=2)
        criterion = create_distillation_loss()
        
        teacher.eval()
        student.train()
        
        # Create dummy data
        images = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 2, (4,))
        
        # Forward pass
        with torch.no_grad():
            teacher_labels = teacher.get_hard_labels(images)
        
        student_outputs = student(images)
        
        # Compute loss
        total_loss, class_loss, distill_loss = criterion(
            student_outputs, teacher_labels, labels
        )
        
        print(f"\nTeacher labels: {teacher_labels}")
        print(f"Ground truth labels: {labels}")
        print(f"\nTotal loss: {total_loss.item():.4f}")
        print(f"Class loss: {class_loss.item():.4f}")
        print(f"Distillation loss: {distill_loss.item():.4f}")
        
        # Test backward pass
        total_loss.backward()
        
        print(f"\n✓ Backward pass successful")
        print(f"✓ Integration with training pipeline works")
        return True
        
    except Exception as e:
        print(f"\n✗ Integration error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "SWIN TRANSFORMER EXTENSION TEST SUITE" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Run tests
    results.append(("Swin Availability", test_timm_swin_availability()))
    results.append(("Model Creation", test_swin_teacher_creation()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Hard Labels", test_hard_label_generation()))
    results.append(("ResNet Comparison", test_comparison_with_resnet()))
    results.append(("Integration", test_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:<25} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Swin Transformer extension is ready.")
        print("\nYou can now:")
        print("  1. Train with Swin teacher:")
        print("     python train_multi_teacher.py --teacher swin --data ...")
        print("  2. Compare with ResNet baseline:")
        print("     python compare_teachers.py --output-dir ...")
        print("\nSee SWIN_EXTENSION_README.md for detailed instructions.")
    else:
        print("✗ Some tests failed. Please fix errors before proceeding.")
        print("\nTroubleshooting:")
        print("  1. Ensure timm is up to date: pip install --upgrade timm")
        print("  2. Check all files are in place:")
        print("     - model_swin_teacher.py")
        print("     - train_multi_teacher.py")
        print("     - compare_teachers.py")
        print("  3. Verify CUDA is available if using GPU")
    print("=" * 60)
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())