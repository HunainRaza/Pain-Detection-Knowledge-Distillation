#!/usr/bin/env python3
"""
Evaluation Metrics
==================

Helper functions for computing evaluation metrics:
- Accuracy
- Precision, Recall, F1-score (per-class and overall)
- Confusion matrix
- ROC/AUC

Maps to paper: Section IV "Experimental Results" - evaluation metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.
    
    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        
    Returns:
        float: Accuracy score
    """
    return accuracy_score(y_true, y_pred)


def compute_precision_recall_f1(y_true, y_pred, average='binary'):
    """
    Compute precision, recall, and F1-score.
    
    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        average: Averaging method ('binary', 'macro', 'weighted')
        
    Returns:
        tuple: (precision, recall, f1)
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return precision, recall, f1


def compute_per_class_metrics(y_true, y_pred, class_names=None):
    """
    Compute per-class precision, recall, and F1-score.
    
    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        class_names: List of class names (default: ['NoPain', 'Pain'])
        
    Returns:
        dict: Per-class metrics
    """
    if class_names is None:
        class_names = ['NoPain', 'Pain']
    
    # Compute per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Organize into dictionary
    metrics = {}
    for i, class_name in enumerate(class_names):
        if i < len(precision):
            metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i])
            }
    
    return metrics


def compute_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        
    Returns:
        numpy.ndarray: Confusion matrix (2, 2)
    """
    return confusion_matrix(y_true, y_pred)


def compute_roc_auc(y_true, y_scores):
    """
    Compute ROC AUC score.
    
    Args:
        y_true: Ground truth labels (N,)
        y_scores: Predicted probabilities for positive class (N,)
        
    Returns:
        float: ROC AUC score
    """
    try:
        auc = roc_auc_score(y_true, y_scores)
        return auc
    except ValueError:
        # If only one class present in y_true
        return 0.0


def compute_roc_curve(y_true, y_scores):
    """
    Compute ROC curve points.
    
    Args:
        y_true: Ground truth labels (N,)
        y_scores: Predicted probabilities for positive class (N,)
        
    Returns:
        tuple: (fpr, tpr, thresholds)
    """
    return roc_curve(y_true, y_scores)


def compute_all_metrics(y_true, y_pred, y_scores=None):
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_scores: Predicted probabilities (N,) - optional
        
    Returns:
        dict: All computed metrics
    """
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = compute_accuracy(y_true, y_pred)
    
    # Overall precision, recall, F1
    precision, recall, f1 = compute_precision_recall_f1(y_true, y_pred, average='binary')
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # Per-class metrics
    metrics['per_class'] = compute_per_class_metrics(y_true, y_pred)
    
    # Confusion matrix
    metrics['confusion_matrix'] = compute_confusion_matrix(y_true, y_pred).tolist()
    
    # ROC AUC (if scores provided)
    if y_scores is not None:
        metrics['roc_auc'] = compute_roc_auc(y_true, y_scores)
    
    return metrics


def print_metrics(metrics, title="Evaluation Metrics"):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics from compute_all_metrics
        title: Title for the metrics display
    """
    print("=" * 60)
    print(title)
    print("=" * 60)
    
    # Overall metrics
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    # ROC AUC if available
    if 'roc_auc' in metrics:
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1-Score:  {class_metrics['f1']:.4f}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"                Predicted")
    print(f"               NoPain  Pain")
    print(f"  Actual NoPain  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"         Pain    {cm[1,0]:5d}  {cm[1,1]:5d}")
    print()


if __name__ == "__main__":
    """
    Test metrics computation.
    """
    print("=" * 60)
    print("Testing Metrics Functions")
    print("=" * 60)
    
    # Dummy data
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 0])
    y_scores = np.random.rand(10)
    
    print(f"\nTest data:")
    print(f"  y_true: {y_true}")
    print(f"  y_pred: {y_pred}")
    
    # Compute all metrics
    metrics = compute_all_metrics(y_true, y_pred, y_scores)
    
    # Print metrics
    print_metrics(metrics, title="Test Metrics")
    
    print("✓ Metrics computation test successful!")
