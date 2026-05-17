#!/usr/bin/env python3
"""
PyTorch Dataset for SynPain
============================

Implements PyTorch Dataset class with stratified train/val/test splits.
Split ratios: 70% train / 15% val / 15% test (stratified by label)

Maps to paper: Data preparation and loading for training/evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add parent directory to path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class SynPainDataset(Dataset):
    """
    PyTorch Dataset for SynPain pain detection.
    
    Returns (image, label) where:
    - image: Tensor of shape (3, 256, 256)
    - label: 0 for NoPain, 1 for Pain
    """
    
    def __init__(self, data_dir, manifest_df, transform=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Base directory containing images
            manifest_df: DataFrame with columns [filepath, label, ...]
            transform: torchvision transforms to apply
        """
        self.data_dir = data_dir
        self.manifest = manifest_df.reset_index(drop=True)
        self.transform = transform
        
        # Map labels to integers
        self.label_map = {'NoPain': 0, 'Pain': 1}
        
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            tuple: (image, label) where label is 0 (NoPain) or 1 (Pain)
        """
        row = self.manifest.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, row['filepath'])
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.label_map[row['label']]
        
        return image, label
    
    def get_label_distribution(self):
        """Get distribution of labels in this dataset."""
        labels = self.manifest['label'].value_counts()
        return labels.to_dict()


def create_data_splits(manifest_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                      random_seed=42):
    """
    Create stratified train/val/test splits from manifest.
    
    Per paper: 70% train / 15% val / 15% test, stratified by label
    
    Args:
        manifest_path: Path to manifest.csv
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for testing (default: 0.15)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Load manifest
    manifest_df = pd.read_csv(manifest_path)
    
    # Extract labels for stratification
    labels = manifest_df['label'].values
    
    # First split: separate train from (val + test)
    train_df, temp_df = train_test_split(
        manifest_df,
        train_size=train_ratio,
        stratify=labels,  # type: ignore
        random_state=random_seed
    )
    
    # Second split: separate val from test
    val_size = val_ratio / (val_ratio + test_ratio)
    temp_labels = temp_df['label'].values
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        stratify=temp_labels,
        random_state=random_seed
    )
    
    return train_df, val_df, test_df


def _seed_worker(worker_id, base_seed=42):
    """
    Seed worker for reproducible DataLoader (module-level for Windows pickle compatibility).
    
    Args:
        worker_id: Worker ID
        base_seed: Base random seed
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_dataloaders(data_dir, manifest_path, train_transform, val_transform,
                      batch_size=64, num_workers=4, random_seed=42):
    """
    Create train/val/test DataLoaders with stratified splits.
    
    Args:
        data_dir: Base directory containing images
        manifest_path: Path to manifest.csv
        train_transform: Transforms for training data
        val_transform: Transforms for val/test data
        batch_size: Batch size (default: 64 per paper)
        num_workers: Number of worker processes
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_counts)
    """
    # Create splits
    train_df, val_df, test_df = create_data_splits(
        manifest_path, 
        random_seed=random_seed
    )
    
    # Create datasets
    train_dataset = SynPainDataset(data_dir, train_df, train_transform)
    val_dataset = SynPainDataset(data_dir, val_df, val_transform)
    test_dataset = SynPainDataset(data_dir, test_df, val_transform)
    
    # Setup for reproducible DataLoader
    g = torch.Generator()
    g.manual_seed(random_seed)
    
    # Only use worker_init_fn if num_workers > 0 (Windows compatibility)
    if num_workers > 0:
        # Use functools.partial to bind random_seed
        from functools import partial
        worker_init_fn = partial(_seed_worker, base_seed=random_seed)
    else:
        worker_init_fn = None
    
    # Create DataLoaders with reproducible shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Calculate class counts for reference
    class_counts = {
        'train': train_dataset.get_label_distribution(),
        'val': val_dataset.get_label_distribution(),
        'test': test_dataset.get_label_distribution()
    }
    
    return train_loader, val_loader, test_loader, class_counts


def print_dataset_info(train_loader, val_loader, test_loader, class_counts):
    """
    Print dataset statistics.
    
    Args:
        train_loader, val_loader, test_loader: DataLoaders
        class_counts: Dictionary of class distributions
    """
    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    print("\nSplit sizes:")
    print(f"  Training:   {len(train_loader.dataset):6d} images")
    print(f"  Validation: {len(val_loader.dataset):6d} images")
    print(f"  Test:       {len(test_loader.dataset):6d} images")
    print(f"  Total:      {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset):6d} images")
    
    print("\nClass distribution:")
    for split_name, counts in class_counts.items():
        print(f"\n  {split_name.capitalize()}:")
        for label, count in counts.items():
            total = sum(counts.values())
            print(f"    {label}: {count:6d} ({count/total*100:5.2f}%)")
    
    print("\nBatch configuration:")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print()


if __name__ == "__main__":
    """
    Test dataset loading and display statistics.
    """
    import argparse
    from Utils.transforms import get_train_transforms, get_val_transforms
    
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument('--data', type=str, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    args = parser.parse_args()
    
    # Paths
    manifest_path = os.path.join(args.data, "manifest.csv")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_counts = create_dataloaders(
        data_dir=args.data,
        manifest_path=manifest_path,
        train_transform=get_train_transforms(),
        val_transform=get_val_transforms(),
        batch_size=args.batch_size
    )
    
    # Print info
    print_dataset_info(train_loader, val_loader, test_loader, class_counts)
    
    # Test loading a batch
    print("Testing batch loading...")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print("✓ Dataset loading successful!")