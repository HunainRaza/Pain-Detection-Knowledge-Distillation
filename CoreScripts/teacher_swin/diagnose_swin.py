#!/usr/bin/env python3
"""
Swin Transformer Training Diagnostic
=====================================

Diagnoses issues with Swin Transformer teacher training by checking:
1. Teacher model predictions on actual data
2. Input/output shapes and data flow
3. Loss values and gradients
4. Comparison with ResNet50 teacher predictions
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_swin_teacher import create_swin_teacher
from model_resnet_teacher import create_teacher_model as create_resnet_teacher
from model_deit import create_deit_model
from dataset import create_dataloaders
from Utils.transforms import get_val_transforms

def diagnose_teacher_predictions(data_dir, manifest_path):
    """
    Check what predictions the Swin teacher actually makes on real data.
    """
    print("=" * 80)
    print("DIAGNOSIS 1: Teacher Predictions on Real Data")
    print("=" * 80)
    
    # Load a small batch of real data
    _, val_loader, _, _ = create_dataloaders(
        data_dir=data_dir,
        manifest_path=manifest_path,
        train_transform=get_val_transforms(224),
        val_transform=get_val_transforms(224),
        batch_size=32,
        num_workers=0,
        random_seed=42
    )
    
    # Get one batch
    images, labels = next(iter(val_loader))
    
    print(f"\nReal data batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels distribution: Pain={labels.sum().item()}, NoPain={len(labels)-labels.sum().item()}")
    
    # Create both teachers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    swin_teacher = create_swin_teacher(pretrained=True, num_classes=2, freeze=True)
    resnet_teacher = create_resnet_teacher(pretrained=True, num_classes=2, freeze=True)
    
    swin_teacher = swin_teacher.to(device)
    resnet_teacher = resnet_teacher.to(device)
    
    images = images.to(device)
    
    # Get predictions from both teachers
    with torch.no_grad():
        swin_logits = swin_teacher(images)
        resnet_logits = resnet_teacher(images)
        
        swin_labels = swin_teacher.get_hard_labels(images)
        resnet_labels = resnet_teacher.get_hard_labels(images)
        
        swin_probs = torch.softmax(swin_logits, dim=1)
        resnet_probs = torch.softmax(resnet_logits, dim=1)
    
    print("\n" + "-" * 80)
    print("SWIN Transformer Teacher Analysis:")
    print("-" * 80)
    print(f"Logits shape: {swin_logits.shape}")
    print(f"Logits stats: min={swin_logits.min():.4f}, max={swin_logits.max():.4f}, mean={swin_logits.mean():.4f}")
    print(f"Hard labels: {swin_labels.cpu().numpy()}")
    print(f"Label distribution: Pain={swin_labels.sum().item()}, NoPain={len(swin_labels)-swin_labels.sum().item()}")
    print(f"Probabilities (Pain class): {swin_probs[:, 1].cpu().numpy()}")
    print(f"Confidence: mean={swin_probs.max(dim=1)[0].mean():.4f}, min={swin_probs.max(dim=1)[0].min():.4f}")
    
    print("\n" + "-" * 80)
    print("ResNet50 Teacher Analysis:")
    print("-" * 80)
    print(f"Logits shape: {resnet_logits.shape}")
    print(f"Logits stats: min={resnet_logits.min():.4f}, max={resnet_logits.max():.4f}, mean={resnet_logits.mean():.4f}")
    print(f"Hard labels: {resnet_labels.cpu().numpy()}")
    print(f"Label distribution: Pain={resnet_labels.sum().item()}, NoPain={len(resnet_labels)-resnet_labels.sum().item()}")
    print(f"Probabilities (Pain class): {resnet_probs[:, 1].cpu().numpy()}")
    print(f"Confidence: mean={resnet_probs.max(dim=1)[0].mean():.4f}, min={resnet_probs.max(dim=1)[0].min():.4f}")
    
    # Check agreement
    agreement = (swin_labels == resnet_labels).float().mean()
    print(f"\n" + "=" * 80)
    print(f"Teacher Agreement: {agreement.item()*100:.2f}%")
    print("=" * 80)
    
    # Check if Swin is predicting all one class
    swin_unique = torch.unique(swin_labels)
    resnet_unique = torch.unique(resnet_labels)
    
    if len(swin_unique) == 1:
        print(f"\n⚠️  WARNING: Swin teacher predicting only class {swin_unique[0].item()}")
        print("   This is a MAJOR PROBLEM - teacher is collapsed!")
    
    if len(resnet_unique) == 1:
        print(f"\n⚠️  WARNING: ResNet teacher predicting only class {resnet_unique[0].item()}")
    
    return swin_teacher, resnet_teacher, images, labels


def diagnose_gradient_flow(swin_teacher, student, images, labels):
    """
    Check if gradients are flowing properly during training.
    """
    print("\n" + "=" * 80)
    print("DIAGNOSIS 2: Gradient Flow Check")
    print("=" * 80)
    
    from distillation_loss import create_distillation_loss
    
    device = images.device
    criterion = create_distillation_loss()
    
    student = student.to(device)
    student.train()
    
    # Forward pass
    with torch.no_grad():
        teacher_labels = swin_teacher.get_hard_labels(images)
    
    student_outputs = student(images)
    
    # Compute loss
    total_loss, class_loss, distill_loss = criterion(
        student_outputs, teacher_labels, labels
    )
    
    print(f"\nLoss values:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Classification loss: {class_loss.item():.4f}")
    print(f"  Distillation loss: {distill_loss.item():.4f}")
    
    # Check for NaN or Inf
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("\n⚠️  ERROR: Loss is NaN or Inf!")
        return False
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in student.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm > 100:
                print(f"\n⚠️  WARNING: Large gradient in {name}: {grad_norm:.4f}")
    
    if grad_norms:
        print(f"\nGradient statistics:")
        print(f"  Mean: {np.mean(grad_norms):.4f}")
        print(f"  Max: {np.max(grad_norms):.4f}")
        print(f"  Min: {np.min(grad_norms):.4f}")
        
        if np.max(grad_norms) > 100:
            print("\n⚠️  WARNING: Exploding gradients detected!")
            return False
        
        if np.mean(grad_norms) < 1e-7:
            print("\n⚠️  WARNING: Vanishing gradients detected!")
            return False
    
    print("\n✓ Gradient flow appears normal")
    return True


def diagnose_input_preprocessing(data_dir, manifest_path):
    """
    Check if input preprocessing is correct for Swin.
    """
    print("\n" + "=" * 80)
    print("DIAGNOSIS 3: Input Preprocessing Check")
    print("=" * 80)
    
    from PIL import Image
    import pandas as pd
    
    # Load manifest
    manifest_df = pd.read_csv(manifest_path)
    
    # Get first image path
    first_image_path = os.path.join(data_dir, manifest_df.iloc[0]['filepath'])
    
    print(f"\nChecking image: {first_image_path}")
    
    # Load raw image
    img = Image.open(first_image_path)
    print(f"Original image size: {img.size}")
    
    # Apply transforms
    transform = get_val_transforms(224)
    img_tensor = transform(img)
    
    print(f"Transformed tensor shape: {img_tensor.shape}")
    print(f"Tensor dtype: {img_tensor.dtype}")
    print(f"Tensor range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
    
    # Check normalization
    expected_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    expected_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    print(f"\nExpected normalization:")
    print(f"  Mean: {expected_mean.view(-1).numpy()}")
    print(f"  Std: {expected_std.view(-1).numpy()}")
    
    if img_tensor.shape != (3, 224, 224):
        print("\n⚠️  WARNING: Image shape is not (3, 224, 224)!")
        return False
    
    print("\n✓ Input preprocessing appears correct")
    return True


def check_swin_model_initialization():
    """
    Check if Swin model is properly initialized.
    """
    print("\n" + "=" * 80)
    print("DIAGNOSIS 4: Swin Model Initialization")
    print("=" * 80)
    
    import timm
    
    # Check timm version
    print(f"\ntimm version: {timm.__version__}")
    
    # Check if Swin models are available
    swin_models = [m for m in timm.list_models() if 'swin_base' in m]
    print(f"Available Swin-Base models: {swin_models[:5]}")
    
    # Try to create model
    try:
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=2)
        print(f"\n✓ Swin model created successfully")
        print(f"  Model type: {type(model)}")
        
        # Check if head is correct
        if hasattr(model, 'head'):
            print(f"  Head: {model.head}")
        
        return True
    except Exception as e:
        print(f"\n⚠️  ERROR creating Swin model: {e}")
        return False


def main():
    """
    Run all diagnostics.
    """
    print("\n╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "SWIN TRANSFORMER TRAINING DIAGNOSTIC" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝\n")
    
    # Configuration
    data_dir = "./SynPainProcessed/Cropped"
    manifest_path = os.path.join(data_dir, "manifest.csv")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please update the path in this script.")
        return
    
    # Run diagnostics
    results = []
    
    # 1. Check Swin model initialization
    results.append(("Swin Model Init", check_swin_model_initialization()))
    
    # 2. Check input preprocessing
    results.append(("Input Preprocessing", diagnose_input_preprocessing(data_dir, manifest_path)))
    
    # 3. Check teacher predictions
    try:
        swin_teacher, resnet_teacher, images, labels = diagnose_teacher_predictions(data_dir, manifest_path)
        results.append(("Teacher Predictions", True))
    except Exception as e:
        print(f"\n⚠️  Error in teacher predictions: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Teacher Predictions", False))
        swin_teacher, resnet_teacher, images, labels = None, None, None, None
    
    # 4. Check gradient flow (if we got this far)
    if swin_teacher is not None:
        try:
            student = create_deit_model(pretrained=False, num_classes=2)
            grad_ok = diagnose_gradient_flow(swin_teacher, student, images, labels)
            results.append(("Gradient Flow", grad_ok))
        except Exception as e:
            print(f"\n⚠️  Error checking gradients: {e}")
            import traceback
            traceback.print_exc()
            results.append(("Gradient Flow", False))
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:<25} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("DIAGNOSIS: No obvious issues detected")
        print("\nPossible causes of poor performance:")
        print("  1. Swin teacher may not generalize well to synthetic faces")
        print("  2. Learning rate may need tuning for Swin teacher")
        print("  3. Swin may need different hyperparameters than ResNet50")
        print("  4. Dataset-specific issue with synthetic data")
    else:
        print("DIAGNOSIS: Issues detected - see above for details")
        print("\nPlease fix the identified issues before training.")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()