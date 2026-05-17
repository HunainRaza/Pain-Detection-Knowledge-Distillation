#!/usr/bin/env python3
"""
Training Script with Teacher-Specific Hyperparameters
======================================================

This variant allows different learning rates and settings for different teachers.
Use this if the standard training doesn't work well for Swin Transformer.
"""

import argparse

# Import the main training function
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_multi_teacher import train_model, set_seed


class TeacherConfig:
    """Configuration for different teacher architectures."""
    
    @staticmethod
    def get_config(teacher_type):
        """
        Get recommended hyperparameters for specific teacher.
        
        Args:
            teacher_type: 'resnet50' or 'swin'
            
        Returns:
            dict: Recommended hyperparameters
        """
        configs = {
            'resnet50': {
                'lr': 1e-5,
                'weight_decay': 0.0,
                'lr_step_size': 10,
                'lr_gamma': 0.5,
                'warmup_epochs': 0,
            },
            'swin': {
                'lr': 5e-6,  # Lower learning rate for Swin
                'weight_decay': 1e-4,  # Add weight decay
                'lr_step_size': 10,
                'lr_gamma': 0.5,
                'warmup_epochs': 3,  # Add warmup
            }
        }
        
        return configs.get(teacher_type, configs['resnet50'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train with teacher-specific hyperparameters"
    )
    
    # Data
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='./outputs')
    
    # Teacher
    parser.add_argument('--teacher', type=str, default='swin',
                       choices=['resnet50', 'swin'])
    
    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=64)
    
    # Hyperparameters (will be overridden by teacher config if not specified)
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (uses teacher-specific default if not set)')
    parser.add_argument('--weight-decay', type=float, default=None)
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Get teacher-specific configuration
    teacher_config = TeacherConfig.get_config(args.teacher)
    
    # Apply teacher-specific defaults if not specified
    if args.lr is None:
        args.lr = teacher_config['lr']
        print(f"\nUsing teacher-specific learning rate: {args.lr}")
    
    if args.weight_decay is None:
        args.weight_decay = teacher_config.get('weight_decay', 0.0)
        if args.weight_decay > 0:
            print(f"Using teacher-specific weight decay: {args.weight_decay}")
    
    # Train model
    train_model(args)