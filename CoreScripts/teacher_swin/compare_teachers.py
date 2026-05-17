#!/usr/bin/env python3
"""
Teacher Architecture Comparison Analysis
=========================================

Compares training results and performance metrics between different teacher
architectures (ResNet50 vs Swin Transformer) to evaluate the research hypothesis:

Hypothesis: Architectural alignment between teacher and student networks
(both attention-based) improves knowledge distillation efficiency.

This script:
1. Loads training histories from both teacher experiments
2. Compares convergence speed, final accuracy, and training stability
3. Generates comparative visualizations
4. Produces statistical analysis for thesis
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_training_history(output_dir, teacher_name):
    """
    Load training history for a specific teacher.
    
    Args:
        output_dir: Base output directory
        teacher_name: Teacher architecture name (e.g., 'resnet50', 'swin')
        
    Returns:
        dict: Training history data
    """
    history_path = os.path.join(output_dir, f'teacher_{teacher_name}', 'training_history.json')
    
    if not os.path.exists(history_path):
        print(f"Warning: Training history not found at {history_path}")
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history


def plot_comparative_training_curves(histories, save_path=None):
    """
    Plot side-by-side comparison of training curves for different teachers.
    
    Args:
        histories: Dictionary of {teacher_name: history_data}
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'resnet50': '#1f77b4', 'swin': '#ff7f0e'}
    
    # Plot 1: Training Loss Comparison
    ax1 = axes[0, 0]
    for teacher_name, history in histories.items():
        epochs = range(1, len(history['train_losses']) + 1)
        ax1.plot(epochs, history['train_losses'], 
                label=f"{teacher_name.upper()} Teacher",
                color=colors.get(teacher_name, 'gray'),
                linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss Comparison
    ax2 = axes[0, 1]
    for teacher_name, history in histories.items():
        epochs = range(1, len(history['val_losses']) + 1)
        ax2.plot(epochs, history['val_losses'],
                label=f"{teacher_name.upper()} Teacher",
                color=colors.get(teacher_name, 'gray'),
                linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy Comparison
    ax3 = axes[1, 0]
    for teacher_name, history in histories.items():
        epochs = range(1, len(history['train_accs']) + 1)
        ax3.plot(epochs, history['train_accs'],
                label=f"{teacher_name.upper()} Teacher",
                color=colors.get(teacher_name, 'gray'),
                linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax3.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy Comparison
    ax4 = axes[1, 1]
    for teacher_name, history in histories.items():
        epochs = range(1, len(history['val_accs']) + 1)
        ax4.plot(epochs, history['val_accs'],
                label=f"{teacher_name.upper()} Teacher",
                color=colors.get(teacher_name, 'gray'),
                linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax4.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Comparative training curves saved to: {save_path}")
    
    plt.close()


def plot_performance_summary(histories, save_path=None):
    """
    Create bar chart comparing final performance metrics.
    
    Args:
        histories: Dictionary of {teacher_name: history_data}
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    teacher_names = list(histories.keys())
    colors = {'resnet50': '#1f77b4', 'swin': '#ff7f0e'}
    
    # Metrics to compare
    best_val_accs = [histories[t]['best_val_acc'] for t in teacher_names]
    final_train_accs = [histories[t]['train_accs'][-1] for t in teacher_names]
    training_times = [histories[t]['training_time_minutes'] for t in teacher_names]
    
    # Plot 1: Best Validation Accuracy
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(teacher_names)), best_val_accs,
                    color=[colors.get(t, 'gray') for t in teacher_names])
    ax1.set_xticks(range(len(teacher_names)))
    ax1.set_xticklabels([t.upper() for t in teacher_names], fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Best Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim([min(best_val_accs) - 2, max(best_val_accs) + 2])
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, best_val_accs)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Final Training Accuracy
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(teacher_names)), final_train_accs,
                    color=[colors.get(t, 'gray') for t in teacher_names])
    ax2.set_xticks(range(len(teacher_names)))
    ax2.set_xticklabels([t.upper() for t in teacher_names], fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Final Training Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylim([min(final_train_accs) - 2, max(final_train_accs) + 2])
    
    for i, (bar, val) in enumerate(zip(bars2, final_train_accs)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Training Time
    ax3 = axes[2]
    bars3 = ax3.bar(range(len(teacher_names)), training_times,
                    color=[colors.get(t, 'gray') for t in teacher_names])
    ax3.set_xticks(range(len(teacher_names)))
    ax3.set_xticklabels([t.upper() for t in teacher_names], fontsize=12)
    ax3.set_ylabel('Time (minutes)', fontsize=12)
    ax3.set_title('Training Time', fontsize=14, fontweight='bold')
    
    for i, (bar, val) in enumerate(zip(bars3, training_times)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} min',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Performance summary saved to: {save_path}")
    
    plt.close()


def compute_convergence_metrics(histories):
    """
    Compute convergence speed and stability metrics.
    
    Args:
        histories: Dictionary of {teacher_name: history_data}
        
    Returns:
        dict: Convergence metrics for each teacher
    """
    metrics = {}
    
    for teacher_name, history in histories.items():
        val_accs = np.array(history['val_accs'])
        
        # Find epoch where 90% of final accuracy is reached
        final_acc = val_accs[-1]
        threshold_90 = 0.9 * final_acc
        epochs_to_90 = np.argmax(val_accs >= threshold_90) + 1 if np.any(val_accs >= threshold_90) else len(val_accs)
        
        # Calculate stability (variance in last 5 epochs)
        stability = np.std(val_accs[-5:])
        
        # Calculate improvement rate (average gain per epoch in first 10 epochs)
        improvement_rate = np.mean(np.diff(val_accs[:10])) if len(val_accs) >= 10 else 0
        
        metrics[teacher_name] = {
            'epochs_to_90_percent': epochs_to_90,
            'final_stability_std': stability,
            'early_improvement_rate': improvement_rate,
            'peak_validation_acc': np.max(val_accs),
            'final_validation_acc': val_accs[-1]
        }
    
    return metrics


def generate_comparison_report(histories, convergence_metrics, save_path):
    """
    Generate text report comparing teacher architectures.
    
    Args:
        histories: Dictionary of {teacher_name: history_data}
        convergence_metrics: Dictionary of convergence metrics
        save_path: Path to save report
    """
    report = []
    report.append("=" * 80)
    report.append("TEACHER ARCHITECTURE COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    report.append("Research Question: Does architectural alignment between teacher and student")
    report.append("improve knowledge distillation efficiency?")
    report.append("")
    
    # Compare each teacher
    for teacher_name in histories.keys():
        report.append("-" * 80)
        report.append(f"{teacher_name.upper()} Teacher")
        report.append("-" * 80)
        
        history = histories[teacher_name]
        metrics = convergence_metrics[teacher_name]
        
        report.append(f"\nFinal Performance:")
        report.append(f"  Best Validation Accuracy: {history['best_val_acc']:.2f}%")
        report.append(f"  Final Training Accuracy: {history['train_accs'][-1]:.2f}%")
        report.append(f"  Final Validation Accuracy: {metrics['final_validation_acc']:.2f}%")
        report.append(f"  Peak Validation Accuracy: {metrics['peak_validation_acc']:.2f}%")
        
        report.append(f"\nConvergence Characteristics:")
        report.append(f"  Epochs to 90% of final accuracy: {metrics['epochs_to_90_percent']}")
        report.append(f"  Early improvement rate: {metrics['early_improvement_rate']:.4f}% per epoch")
        report.append(f"  Stability (std of last 5 epochs): {metrics['final_stability_std']:.4f}%")
        
        report.append(f"\nTraining Efficiency:")
        report.append(f"  Total training time: {history['training_time_minutes']:.2f} minutes")
        report.append(f"  Average time per epoch: {history['training_time_minutes']/len(history['train_losses']):.2f} minutes")
        report.append("")
    
    # Comparative Analysis
    report.append("=" * 80)
    report.append("COMPARATIVE ANALYSIS")
    report.append("=" * 80)
    report.append("")
    
    teachers = list(histories.keys())
    if len(teachers) == 2:
        t1, t2 = teachers
        h1, h2 = histories[t1], histories[t2]
        m1, m2 = convergence_metrics[t1], convergence_metrics[t2]
        
        acc_diff = h1['best_val_acc'] - h2['best_val_acc']
        report.append(f"Accuracy Difference ({t1.upper()} - {t2.upper()}):")
        report.append(f"  {acc_diff:+.2f}% ({abs(acc_diff)/max(h1['best_val_acc'], h2['best_val_acc'])*100:.2f}% relative)")
        report.append("")
        
        conv_diff = m1['epochs_to_90_percent'] - m2['epochs_to_90_percent']
        report.append(f"Convergence Speed Difference:")
        report.append(f"  {t1.upper()} reaches 90% in {m1['epochs_to_90_percent']} epochs")
        report.append(f"  {t2.upper()} reaches 90% in {m2['epochs_to_90_percent']} epochs")
        report.append(f"  Difference: {conv_diff:+d} epochs")
        report.append("")
        
        time_diff = h1['training_time_minutes'] - h2['training_time_minutes']
        report.append(f"Training Time Difference:")
        report.append(f"  {t1.upper()}: {h1['training_time_minutes']:.2f} minutes")
        report.append(f"  {t2.upper()}: {h2['training_time_minutes']:.2f} minutes")
        report.append(f"  Difference: {time_diff:+.2f} minutes ({abs(time_diff)/max(h1['training_time_minutes'], h2['training_time_minutes'])*100:.2f}% relative)")
        report.append("")
    
    report.append("=" * 80)
    report.append("CONCLUSION")
    report.append("=" * 80)
    report.append("")
    
    # Determine which teacher performed better
    best_teacher = max(histories.keys(), key=lambda t: histories[t]['best_val_acc'])
    report.append(f"Best Performing Teacher: {best_teacher.upper()}")
    report.append(f"  Validation Accuracy: {histories[best_teacher]['best_val_acc']:.2f}%")
    report.append("")
    
    # Research implications
    if 'swin' in histories and 'resnet50' in histories:
        swin_better = histories['swin']['best_val_acc'] > histories['resnet50']['best_val_acc']
        
        if swin_better:
            report.append("Research Findings:")
            report.append("  ✓ Attention-based teacher (Swin) outperformed CNN teacher (ResNet50)")
            report.append("  ✓ Supports hypothesis: Architectural alignment improves distillation")
            report.append("  ✓ Transformer-to-Transformer knowledge transfer is more effective")
        else:
            report.append("Research Findings:")
            report.append("  • CNN teacher (ResNet50) achieved higher accuracy than Swin Transformer")
            report.append("  • Architectural alignment hypothesis not strongly supported")
            report.append("  • Other factors may influence distillation effectiveness")
    
    report.append("")
    report.append("=" * 80)
    
    # Save report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Comparison report saved to: {save_path}")
    
    # Also print to console
    print("\n" + '\n'.join(report))


def main(args):
    """
    Main comparison analysis function.
    
    Args:
        args: Command line arguments
    """
    print("=" * 80)
    print("Teacher Architecture Comparison Analysis")
    print("=" * 80)
    print()
    
    # Load training histories
    print("Loading training histories...")
    histories = {}
    
    for teacher_name in args.teachers:
        history = load_training_history(args.output_dir, teacher_name)
        if history is not None:
            histories[teacher_name] = history
            print(f"  ✓ Loaded {teacher_name} teacher history")
        else:
            print(f"  ✗ Could not load {teacher_name} teacher history")
    
    if len(histories) < 2:
        print("\nError: Need at least 2 teacher histories for comparison")
        print("Please train models with both teacher architectures first:")
        print("  python train_multi_teacher.py --teacher resnet50 --data ... --output ...")
        print("  python train_multi_teacher.py --teacher swin --data ... --output ...")
        return
    
    print(f"\nLoaded {len(histories)} teacher histories")
    print()
    
    # Create output directory for comparison results
    comparison_dir = os.path.join(args.output_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Generate visualizations
    print("Generating comparative visualizations...")
    
    curves_path = os.path.join(comparison_dir, 'comparative_training_curves.png')
    plot_comparative_training_curves(histories, save_path=curves_path)
    
    summary_path = os.path.join(comparison_dir, 'performance_summary.png')
    plot_performance_summary(histories, save_path=summary_path)
    
    # Compute convergence metrics
    print("\nComputing convergence metrics...")
    convergence_metrics = compute_convergence_metrics(histories)
    
    # Generate comparison report
    report_path = os.path.join(comparison_dir, 'comparison_report.txt')
    generate_comparison_report(histories, convergence_metrics, report_path)
    
    # Save metrics to JSON
    metrics_path = os.path.join(comparison_dir, 'convergence_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(convergence_metrics, f, indent=2)
    print(f"\n✓ Convergence metrics saved to: {metrics_path}")
    
    print("\n" + "=" * 80)
    print("Comparison Analysis Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {comparison_dir}/")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare training results from different teacher architectures"
    )
    
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Base output directory containing teacher results')
    parser.add_argument('--teachers', nargs='+', default=['resnet50', 'swin'],
                       help='List of teacher architectures to compare')
    
    args = parser.parse_args()
    
    main(args)