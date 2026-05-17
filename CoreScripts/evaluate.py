#!/usr/bin/env python3
"""
Evaluation Script for Pain Detection Model
===========================================

Evaluates trained model on test set with comprehensive metrics:
- Accuracy
- Confusion matrix
- Per-class precision, recall, F1
- ROC/AUC

Maps to paper: Section IV "Experimental Results" - evaluation methodology
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add parent directory to path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Project imports
from dataset import create_dataloaders
from model_deit import create_deit_model
from Utils.transforms import get_val_transforms
from Utils.metrics import compute_all_metrics, print_metrics
from Utils.visualization import plot_confusion_matrix, plot_roc_curve, visualize_predictions


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
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
    
    print("\nEvaluating on test set...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get class logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Get predictions and probabilities
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Store some images for visualization
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
    Main evaluation function.
    
    Args:
        args: Parsed command line arguments
    """
    print("=" * 60)
    print("Pain Detection Model Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data directory: {args.data}")
    print(f"  Output directory: {args.output}")
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
        train_transform=get_val_transforms(224),  # DeiT expects 224x224
        val_transform=get_val_transforms(224),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Test set size: {len(test_loader.dataset)} images")  # type: ignore
    print(f"Test batches: {len(test_loader)}\n")
    
    # Load model
    print("Loading model...")
    model = create_deit_model(pretrained=False, num_classes=2)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print_metrics(results['metrics'], title="Test Set Evaluation Results")
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.output, 'test_metrics.json')
    
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
    
    metrics_to_save = convert_to_native(results['metrics'])
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Plot confusion matrix
    cm = np.array(results['metrics']['confusion_matrix'])
    cm_path = os.path.join(args.output, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names=['NoPain', 'Pain'], 
                         save_path=cm_path, show=False)
    
    # Plot normalized confusion matrix
    cm_norm_path = os.path.join(args.output, 'confusion_matrix_normalized.png')
    plot_confusion_matrix(cm, class_names=['NoPain', 'Pain'], 
                         save_path=cm_norm_path, show=False, normalize=True)
    
    print(f"✓ Confusion matrices saved to: {args.output}/")
    
    # Plot ROC curve
    if 'roc_auc' in results['metrics']:
        roc_path = os.path.join(args.output, 'roc_curve.png')
        plot_roc_curve(results['labels'], results['pain_probabilities'],
                      save_path=roc_path, show=False)
        print(f"✓ ROC curve saved to: {roc_path}")
    
    # Visualize sample predictions
    if results['sample_images'] is not None and len(results['sample_images']) > 0:
        sample_path = os.path.join(args.output, 'sample_predictions.png')
        
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
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nAll results saved to: {args.output}/")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained pain detection model on test set"
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to processed dataset directory')
    parser.add_argument('--output', type=str, default='./evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    main(args)