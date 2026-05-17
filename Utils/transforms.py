#!/usr/bin/env python3
"""
Data Augmentation Transforms
=============================

Augmentations used during training as mentioned in the paper:
- Random horizontal flip
- Small rotation (±10°)
- Color jitter
- Random crop/scale

Maps to paper: Section III-A, augmentation during training
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def get_train_transforms(image_size=224):
    """
    Get training data augmentation transforms.
    
    Augmentations per paper:
    - Random horizontal flip
    - Small rotation (±10°)
    - Color jitter
    - Random crop/scale
    - Normalization (ImageNet stats for pretrained models)
    
    Args:
        image_size: Target image size (default: 224 for DeiT)
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        # Resize to 256 first (our images are already 256, but this ensures it)
        transforms.Resize(256),
        # Random resized crop to 224 (scale 0.8-1.0)
        transforms.RandomResizedCrop(
            image_size, 
            scale=(0.8, 1.0),
            interpolation=InterpolationMode.BILINEAR
        ),
        # Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
        # Small random rotation (±10 degrees)
        transforms.RandomRotation(
            degrees=10,
            interpolation=InterpolationMode.BILINEAR
        ),
        # Color jitter
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        # Convert to tensor
        transforms.ToTensor(),
        # Normalize with ImageNet statistics
        # (required for pretrained models)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(image_size=224):
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size (default: 224 for DeiT)
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        # Resize and center crop to 224 to match DeiT
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        # Convert to tensor
        transforms.ToTensor(),
        # Normalize with ImageNet statistics
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_inference_transforms(image_size=224):
    """
    Get inference transforms (same as validation).
    
    Args:
        image_size: Target image size (default: 224 for DeiT)
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return get_val_transforms(image_size)


def denormalize_image(tensor):
    """
    Denormalize an image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        
    Returns:
        torch.Tensor: Denormalized tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    return tensor * std + mean