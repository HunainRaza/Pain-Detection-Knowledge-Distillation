#!/usr/bin/env python3
"""
Fine-tune Teachers on SynPain
==============================

Fine-tunes either ResNet50 or Swin Transformer on SynPain dataset
BEFORE using them for knowledge distillation.

This solves the domain shift problem where ImageNet-pretrained teachers
cannot recognize synthetic pain expressions well.

Usage:
    # Fine-tune ResNet50
    python finetune_teacher.py --teacher resnet50 --data ./SynPainProcessed/Cropped
    
    # Fine-tune Swin Transformer  
    python finetune_teacher.py --teacher swin --data ./SynPainProcessed/Cropped
"""

import os
import sys
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# Project imports
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_swin.model_swin_teacher import SwinTransformer_Teacher
from model_resnet_teacher import ResNet50_Teacher
from dataset import create_dataloaders, print_dataset_info
from Utils.transforms import get_train_transforms, get_val_transforms
from Utils.visualization import plot_training_curves


def create_teacher_for_finetuning(teacher_type, num_classes=2):
    """
    Create teacher model for fine-tuning (NOT frozen).
    
    Args:
        teacher_type: 'resnet50' or 'swin'
        num_classes: Number of output classes
        
    Returns:
        Teacher model with trainable parameters
    """
    if teacher_type == 'resnet50':
        print("\nCreating ResNet50 teacher for fine-tuning...")
        model = ResNet50_Teacher(pretrained=True, num_classes=num_classes)
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        print("ResNet50 loaded with all parameters trainable")
        
    elif teacher_type == 'swin':
        print("\nCreating Swin Transformer teacher for fine-tuning...")
        model = SwinTransformer_Teacher(
            pretrained=True,
            num_classes=num_classes,
            model_variant='swin_base_patch4_window7_224'
        )
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        print("Swin Transformer loaded with all parameters trainable")
        
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")
    
    return model


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, use_amp=False):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Track prediction distribution
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    # Check prediction diversity
    unique_preds = len(set(all_preds))
    pain_preds = sum(all_preds)
    nopain_preds = len(all_preds) - pain_preds
    
    pred_dist = {
        'unique_classes': unique_preds,
        'pain_predictions': pain_preds,
        'nopain_predictions': nopain_preds
    }
    
    return epoch_loss, epoch_acc, pred_dist


def main(args):
    """Main fine-tuning function."""
    print("=" * 80)
    print(f"Fine-tuning {args.teacher.upper()} Teacher on SynPain")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Teacher: {args.teacher}")
    print(f"  Data directory: {args.data}")
    print(f"  Output directory: {args.output}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print()
    
    # Create output directory
    output_dir = os.path.join(args.output, f"finetuned_{args.teacher}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(os.path.join(output_dir, 'finetune_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create data loaders
    print("Preparing data loaders...")
    manifest_path = os.path.join(args.data, "manifest.csv")
    
    train_loader, val_loader, test_loader, class_counts = create_dataloaders(
        data_dir=args.data,
        manifest_path=manifest_path,
        train_transform=get_train_transforms(224),
        val_transform=get_val_transforms(224),
        batch_size=args.batch,
        num_workers=args.num_workers,
        random_seed=42
    )
    
    print_dataset_info(train_loader, val_loader, test_loader, class_counts)
    
    # Create model
    model = create_teacher_for_finetuning(args.teacher, num_classes=2)
    model = model.to(device)
    
    trainable_params = count_parameters(model)
    print(f"\nTrainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.teacher == 'swin':
        # Swin benefits from AdamW with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        # ResNet50 works well with standard Adam
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Mixed precision
    scaler = GradScaler() if args.use_amp else None
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Fine-tuning")
    print("=" * 80)
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, args.use_amp
        )
        
        val_loss, val_acc, pred_dist = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"  Learning rate: {current_lr:.2e}")
        print(f"  Predictions: Pain={pred_dist['pain_predictions']}, NoPain={pred_dist['nopain_predictions']}")
        
        # Warn if model is predicting only one class
        if pred_dist['unique_classes'] == 1:
            print(f"  ⚠️  WARNING: Model predicting only one class!")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(output_dir, f'best_{args.teacher}_teacher.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'teacher_type': args.teacher,
                'prediction_distribution': pred_dist,
            }, checkpoint_path)
            print(f"  ✓ New best model saved (Val Acc: {val_acc:.2f}%)")
    
    elapsed_time = time.time() - start_time
    
    # Save final model
    final_path = os.path.join(output_dir, f'final_{args.teacher}_teacher.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_accs[-1],
        'teacher_type': args.teacher,
    }, final_path)
    
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
    
    history_path = os.path.join(output_dir, 'finetuning_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    curves_path = os.path.join(output_dir, 'finetuning_curves.png')
    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                        save_path=curves_path, show=False)
    
    print(f"\n" + "=" * 80)
    print("Fine-tuning Complete!")
    print("=" * 80)
    print(f"Teacher: {args.teacher.upper()}")
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"\nCheckpoints saved:")
    print(f"  Best model: {checkpoint_path}")
    print(f"  Final model: {final_path}")
    print(f"  Training history: {history_path}")
    print(f"  Training curves: {curves_path}")
    
    # Final validation
    print("\n" + "=" * 80)
    print("Final Validation Check")
    print("=" * 80)
    
    if best_val_acc < 90:
        print(f"⚠️  WARNING: Validation accuracy ({best_val_acc:.2f}%) is below 90%")
        print("   Consider training for more epochs or adjusting hyperparameters")
    elif best_val_acc < 95:
        print(f"✓ Acceptable performance: {best_val_acc:.2f}%")
        print("  Model can be used as teacher, but may benefit from more training")
    else:
        print(f"✓ Excellent performance: {best_val_acc:.2f}%")
        print("  Model is ready to use as teacher for knowledge distillation!")
    
    print("\nTo use this fine-tuned teacher, you'll need to:")
    print("  1. Modify train_multi_teacher.py to load this checkpoint")
    print("  2. Freeze the model parameters after loading")
    print("  3. Proceed with student training as normal")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune teacher on SynPain before using for distillation"
    )
    
    # Required
    parser.add_argument('--teacher', type=str, required=True,
                       choices=['resnet50', 'swin'],
                       help='Teacher architecture to fine-tune')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to processed dataset directory')
    
    # Optional
    parser.add_argument('--output', type=str, default='./finetuned_teachers',
                       help='Output directory for fine-tuned models')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of fine-tuning epochs (default: 15)')
    parser.add_argument('--batch', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-amp', action='store_true',
                       help='Use automatic mixed precision')
    
    args = parser.parse_args()
    
    main(args)