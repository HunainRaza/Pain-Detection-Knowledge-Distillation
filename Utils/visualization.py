#!/usr/bin/env python3
"""
Visualization Utilities
========================

Functions for visualizing:
- Training curves (loss and accuracy)
- Confusion matrix heatmaps
- Sample predictions with overlays
- ROC curves

Maps to paper: Visualization of results and training progress
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import torch
from PIL import Image


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                         save_path=None, show=True):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(cm, class_names=None, save_path=None, show=True, 
                         normalize=False):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix (2, 2) numpy array
        class_names: List of class names (default: ['NoPain', 'Pain'])
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        normalize: Whether to normalize by row (default: False)
    """
    if class_names is None:
        class_names = ['NoPain', 'Pain']
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                linewidths=0.5, linecolor='gray')
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(y_true, y_scores, save_path=None, show=True):
    """
    Plot ROC curve.
    
    Args:
        y_true: Ground truth labels (N,)
        y_scores: Predicted probabilities for positive class (N,)
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_predictions(images, true_labels, pred_labels, pred_probs,
                         class_names=None, num_samples=8, save_path=None, show=True):
    """
    Visualize sample predictions with ground truth and predictions.
    
    Args:
        images: Batch of images (N, 3, H, W) tensor
        true_labels: Ground truth labels (N,)
        pred_labels: Predicted labels (N,)
        pred_probs: Prediction probabilities (N, num_classes)
        class_names: List of class names (default: ['NoPain', 'Pain'])
        num_samples: Number of samples to display
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    if class_names is None:
        class_names = ['NoPain', 'Pain']
    
    num_samples = min(num_samples, len(images))
    
    # Create grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        # Denormalize image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Get labels
        true_label = class_names[true_labels[i]]
        pred_label = class_names[pred_labels[i]]
        confidence = pred_probs[i][pred_labels[i]] * 100
        
        # Determine color (green if correct, red if incorrect)
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        
        # Plot
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)",
            color=color, fontweight='bold', fontsize=10
        )
    
    # Hide remaining subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    """
    Test visualization functions with dummy data.
    """
    print("=" * 60)
    print("Testing Visualization Functions")
    print("=" * 60)
    
    # Test training curves
    print("\nTesting training curves...")
    train_losses = [0.8, 0.6, 0.5, 0.4, 0.35]
    val_losses = [0.7, 0.65, 0.6, 0.55, 0.5]
    train_accs = [60, 70, 75, 80, 82]
    val_accs = [65, 68, 72, 75, 78]
    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                        save_path='test_training_curves.png', show=False)
    
    # Test confusion matrix
    print("\nTesting confusion matrix...")
    cm = np.array([[850, 150], [200, 800]])
    plot_confusion_matrix(cm, save_path='test_confusion_matrix.png', show=False)
    
    # Test ROC curve
    print("\nTesting ROC curve...")
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)
    plot_roc_curve(y_true, y_scores, save_path='test_roc_curve.png', show=False)
    
    print("\n✓ Visualization test successful!")
    print("\nTest images saved:")
    print("  - test_training_curves.png")
    print("  - test_confusion_matrix.png")
    print("  - test_roc_curve.png")
