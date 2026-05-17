#!/usr/bin/env python3
"""
Training Script for Pain Detection with Configurable Teacher Architecture
===========================================================================

Extended training script that supports multiple teacher architectures:
- ResNet50 (baseline from original paper)
- Swin Transformer (research extension)

This enables systematic comparison of how different teacher architectures
affect knowledge distillation performance with DeiT student.

Research Question: Does architectural alignment between teacher and student
improve knowledge transfer efficiency?
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add parent directory to path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Project imports
from dataset import create_dataloaders, print_dataset_info
from model_deit import create_deit_model, count_parameters
from model_resnet_teacher import create_teacher_model as create_resnet_teacher
from model_swin_teacher import create_swin_teacher
from distillation_loss import create_distillation_loss
from Utils.transforms import get_train_transforms, get_val_transforms
from Utils.metrics import compute_accuracy
from Utils.visualization import plot_training_curves


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed (default: 42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Deterministic CUDA operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")
    print("Note: Deterministic mode may reduce training speed")


def create_teacher(teacher_type, num_classes=2, device='cuda'):
    """
    Factory function to create teacher model based on type.
    
    Args:
        teacher_type: Type of teacher ('resnet50' or 'swin')
        num_classes: Number of output classes (default: 2)
        device: Device to place model on
        
    Returns:
        Teacher model
    """
    if teacher_type.lower() == 'resnet50':
        print("\n" + "=" * 60)
        print("Creating ResNet50 Teacher (Baseline)")
        print("=" * 60)
        teacher = create_resnet_teacher(pretrained=True, num_classes=num_classes, freeze=True)
    
    elif teacher_type.lower() in ['swin', 'swin_transformer']:
        print("\n" + "=" * 60)
        print("Creating Swin Transformer Teacher (Research Extension)")
        print("=" * 60)
        teacher = create_swin_teacher(
            pretrained=True, 
            num_classes=num_classes,
            model_variant='swin_base_patch4_window7_224',
            freeze=True
        )
    
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}. Use 'resnet50' or 'swin'")
    
    teacher = teacher.to(device)
    teacher.eval()  # Ensure eval mode
    
    return teacher


def train_epoch(student, teacher, train_loader, criterion, optimizer, 
               device, scaler=None, use_amp=False):
    """
    Train for one epoch.
    
    Args:
        student: DeiT student model
        teacher: Teacher model (ResNet50 or Swin Transformer)
        train_loader: Training data loader
        criterion: Distillation loss function
        optimizer: Optimizer
        device: Device to use
        scaler: GradScaler for mixed precision (optional)
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        tuple: (average_loss, accuracy, class_loss, distill_loss)
    """
    student.train()
    teacher.eval()  # Teacher always in eval mode
    
    running_loss = 0.0
    running_class_loss = 0.0
    running_distill_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Get teacher hard labels (no gradient)
        with torch.no_grad():
            teacher_labels = teacher.get_hard_labels(images)
        
        # Forward pass with mixed precision if enabled
        if use_amp and scaler is not None:
            with autocast():
                # Student forward pass
                student_outputs = student(images)
                
                # Compute distillation loss
                total_loss, class_loss, distill_loss = criterion(
                    student_outputs, teacher_labels, labels
                )
            
            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Student forward pass
            student_outputs = student(images)
            
            # Compute distillation loss
            total_loss, class_loss, distill_loss = criterion(
                student_outputs, teacher_labels, labels
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
        
        # Get predictions from class token
        if isinstance(student_outputs, tuple):
            logits = student_outputs[0]
        else:
            logits = student_outputs
        
        _, predicted = torch.max(logits, 1)
        
        # Statistics
        running_loss += total_loss.item() * images.size(0)
        running_class_loss += class_loss.item() * images.size(0)
        running_distill_loss += distill_loss.item() * images.size(0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_class_loss = running_class_loss / total
    epoch_distill_loss = running_distill_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc, epoch_class_loss, epoch_distill_loss


def validate_epoch(student, val_loader, criterion, device):
    """
    Validate for one epoch.
    
    Args:
        student: DeiT student model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    student.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            student_outputs = student(images)
            
            # Get class logits
            if isinstance(student_outputs, tuple):
                logits = student_outputs[0]
            else:
                logits = student_outputs
            
            # Simple classification loss for validation
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            # Predictions
            _, predicted = torch.max(logits, 1)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def train_model(args):
    """
    Main training function.
    
    Args:
        args: Parsed command line arguments
    """
    print("=" * 80)
    print("Pain Detection Training with Configurable Teacher Architecture")
    print("=" * 80)
    print(f"\nTraining configuration:")
    print(f"  Teacher architecture: {args.teacher}")
    print(f"  Data directory: {args.data}")
    print(f"  Output directory: {args.output}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  Mixed precision: {args.use_amp}")
    print(f"  Random seed: {args.seed}")
    print()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory with teacher name
    output_dir = os.path.join(args.output, f"teacher_{args.teacher}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}\n")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create data loaders
    print("Preparing data loaders...")
    manifest_path = os.path.join(args.data, "manifest.csv")
    
    train_loader, val_loader, test_loader, class_counts = create_dataloaders(
        data_dir=args.data,
        manifest_path=manifest_path,
        train_transform=get_train_transforms(224),  # DeiT expects 224x224
        val_transform=get_val_transforms(224),
        batch_size=args.batch,
        num_workers=args.num_workers,
        random_seed=args.seed
    )
    
    print_dataset_info(train_loader, val_loader, test_loader, class_counts)
    
    # Create models
    print("\nInitializing models...")
    
    # Student: DeiT
    student = create_deit_model(pretrained=True, num_classes=2)
    student = student.to(device)
    print(f"Student (DeiT) parameters: {count_parameters(student):,}")
    
    # Teacher: ResNet50 or Swin Transformer
    teacher = create_teacher(args.teacher, num_classes=2, device=device)
    from model_resnet_teacher import count_parameters as count_teacher_params
    total_params, trainable_params = count_teacher_params(teacher)
    print(f"Teacher ({args.teacher}) parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Loss function
    criterion = create_distillation_loss(class_weight=1.0, distill_weight=1.0)
    
    # Optimizer
    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp else None
    
    # Training tracking
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    # Training loop
    print("\n" + "=" * 80)
    print(f"Starting Training with {args.teacher} Teacher")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc, class_loss, distill_loss = train_epoch(
            student, teacher, train_loader, criterion, optimizer,
            device, scaler, args.use_amp
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(student, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (Class: {class_loss:.4f}, Distill: {distill_loss:.4f})")
        print(f"  Train Acc:  {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'teacher_type': args.teacher,
            }, best_checkpoint_path)
            print(f"  ✓ New best model saved (Val Acc: {val_acc:.2f}%)")
    
    # Save final model
    final_checkpoint_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_accs[-1],
        'train_acc': train_accs[-1],
        'teacher_type': args.teacher,
    }, final_checkpoint_path)
    
    # Training time
    elapsed_time = time.time() - start_time
    print(f"\n" + "=" * 80)
    print(f"Training Complete!")
    print(f"=" * 80)
    print(f"Teacher architecture: {args.teacher}")
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"\nCheckpoints saved:")
    print(f"  Best model: {best_checkpoint_path}")
    print(f"  Final model: {final_checkpoint_path}")
    
    # Save training history
    history = {
        'teacher_type': args.teacher,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'training_time_minutes': elapsed_time / 60,
        'config': config
    }
    
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Training history: {history_path}")
    
    # Plot training curves
    curves_path = os.path.join(output_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                        save_path=curves_path, show=False)
    print(f"  Training curves: {curves_path}")
    
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DeiT for pain detection with configurable teacher architecture"
    )
    
    # Data
    parser.add_argument('--data', type=str, required=True,
                       help='Path to processed dataset directory')
    parser.add_argument('--output', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    
    # Teacher selection (NEW!)
    parser.add_argument('--teacher', type=str, default='resnet50',
                       choices=['resnet50', 'swin'],
                       help='Teacher architecture: resnet50 (baseline) or swin (research extension)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--batch', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5)')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--use-amp', action='store_true',
                       help='Use automatic mixed precision training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    train_model(args)