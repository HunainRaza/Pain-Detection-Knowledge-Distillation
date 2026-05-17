#!/usr/bin/env python3
"""
Evaluation Script for Fine-tuned Swin Transformer Teacher
==========================================================

Evaluates the fine-tuned Swin Transformer teacher model on test set with comprehensive metrics:
- Accuracy
- Confusion matrix
- Per-class precision, recall, F1
- ROC/AUC

This helps understand the upper bound of what the teacher knows before distillation.

Usage:
    python evaluate_swin_teacher.py \
        --checkpoint ./finetuned_teachers/finetuned_swin/best_swin_teacher.pth \
        --data ./SynPainProcessed/Cropped \
        --output ./evaluation_swin_teacher

Maps to research: Evaluating teacher performance establishes baseline for distillation comparison.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import time

# Add parent directories to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEACHER_SWIN_DIR = Path(__file__).resolve().parent
if str(TEACHER_SWIN_DIR) not in sys.path:
    sys.path.insert(0, str(TEACHER_SWIN_DIR))

# Project imports
from model_swin_teacher import SwinTransformer_Teacher, count_parameters
from dataset import create_dataloaders
from Utils.transforms import get_val_transforms
from Utils.metrics import compute_all_metrics, print_metrics
from Utils.visualization import plot_confusion_matrix, plot_roc_curve, visualize_predictions


def evaluate_teacher(model, test_loader, device):
    """
    Evaluate fine-tuned Swin teacher on test set.
    
    Args:
        model: Fine-tuned Swin teacher model
        test_loader: Test data loader
        device: Device to use
        
    Returns:
        dict: Evaluation results with predictions and metrics
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_images = []
    
    print("\nEvaluating fine-tuned Swin teacher on test set...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass through teacher
            outputs = model(images)
            
            # Get predictions and probabilities
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Store some images for visualization (limit memory usage)
            if len(all_images) < 32:
                all_images.append(images.cpu())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Get probabilities for positive class (Pain = class 1)
    pain_probs = all_probs[:, 1]
    
    # Compute metrics
    metrics = compute_all_metrics(all_labels, all_preds, pain_probs)
    
    results = {
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs,
        'pain_probabilities': pain_probs,
        'metrics': metrics,
        'sample_images': torch.cat(all_images[:4]) if all_images else None
    }
    
    return results


def main(args):
    """
    Main evaluation function for Swin teacher.
    
    Args:
        args: Parsed command line arguments
    """
    print("=" * 70)
    print("Fine-tuned Swin Transformer Teacher Evaluation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data directory: {args.data}")
    print(f"  Output directory: {args.output}")
    print(f"  Model variant: {args.swin_variant}")
    print()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading test data...")
    manifest_path = os.path.join(args.data, "manifest.csv")
    
    _, _, test_loader, class_counts = create_dataloaders(
        data_dir=args.data,
        manifest_path=manifest_path,
        train_transform=get_val_transforms(224),  # Not used for test
        val_transform=get_val_transforms(224),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Test set size: {len(test_loader.dataset)} images")
    print(f"Test batches: {len(test_loader)}\n")
    
    # Create model architecture
    print("Loading Swin Transformer teacher...")
    model = SwinTransformer_Teacher(
        pretrained=False,  # Will load weights from checkpoint
        num_classes=2,
        model_variant=args.swin_variant
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model_state_dict from checkpoint")
        if 'epoch' in checkpoint:
            print(f"  Checkpoint from epoch: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint:
            print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")
        if 'model_variant' in checkpoint:
            print(f"  Model variant: {checkpoint['model_variant']}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print("Loaded state dict directly")
    
    model = model.to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")
    
    # Set to evaluation mode
    model.eval()
    
    # Evaluate
    start_time = time.time()
    results = evaluate_teacher(model, test_loader, device)
    eval_time = time.time() - start_time
    
    print(f"\nEvaluation completed in {eval_time:.2f} seconds")
    
    # Print metrics
    print_metrics(results['metrics'], title="Swin Teacher Test Set Evaluation Results")
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.output, 'swin_teacher_test_metrics.json')
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    metrics_to_save = {
        'model': 'Swin Transformer Teacher (Fine-tuned)',
        'model_variant': args.swin_variant,
        'checkpoint': args.checkpoint,
        'evaluation_time_seconds': eval_time,
        **convert_to_native(results['metrics'])
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Plot confusion matrix
    cm = np.array(results['metrics']['confusion_matrix'])
    cm_path = os.path.join(args.output, 'swin_teacher_confusion_matrix.png')
    plot_confusion_matrix(cm, class_names=['NoPain', 'Pain'], 
                         save_path=cm_path, show=False)
    
    # Plot normalized confusion matrix
    cm_norm_path = os.path.join(args.output, 'swin_teacher_confusion_matrix_normalized.png')
    plot_confusion_matrix(cm, class_names=['NoPain', 'Pain'], 
                         save_path=cm_norm_path, show=False, normalize=True)
    
    print(f"✓ Confusion matrices saved to: {args.output}/")
    
    # Plot ROC curve
    if 'roc_auc' in results['metrics']:
        roc_path = os.path.join(args.output, 'swin_teacher_roc_curve.png')
        plot_roc_curve(results['labels'], results['pain_probabilities'],
                      save_path=roc_path, show=False)
        print(f"✓ ROC curve saved to: {roc_path}")
    
    # Visualize sample predictions
    if results['sample_images'] is not None and len(results['sample_images']) > 0:
        sample_path = os.path.join(args.output, 'swin_teacher_sample_predictions.png')
        
        # Get corresponding labels and predictions for sample images
        num_samples = len(results['sample_images'])
        sample_labels = results['labels'][:num_samples]
        sample_preds = results['predictions'][:num_samples]
        sample_probs = results['probabilities'][:num_samples]
        
        visualize_predictions(
            results['sample_images'],
            sample_labels,
            sample_preds,
            sample_probs,
            class_names=['NoPain', 'Pain'],
            num_samples=min(num_samples, 16),
            save_path=sample_path,
            show=False
        )
        print(f"✓ Sample predictions saved to: {sample_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Swin Teacher Evaluation Complete!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Accuracy: {results['metrics']['accuracy']*100:.2f}%")
    print(f"  F1 Score: {results['metrics']['f1']*100:.2f}%")
    print(f"  ROC-AUC:  {results['metrics'].get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in results['metrics'] else "")
    print(f"\nAll results saved to: {args.output}/")
    print()
    
    # Return metrics for programmatic use
    return results['metrics']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Swin Transformer teacher model on test set"
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to fine-tuned Swin teacher checkpoint (.pth file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to processed dataset directory (with manifest.csv)')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default='./evaluation_swin_teacher',
                       help='Output directory for evaluation results')
    parser.add_argument('--swin-variant', type=str, default='swin_base_patch4_window7_224',
                       help='Swin Transformer variant (default: swin_base_patch4_window7_224)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    main(args)
