#!/usr/bin/env python3
"""
Manifest Verification and Repair Tool
======================================

Checks if manifest.csv contains all images in Pain/NoPain folders
and offers to rebuild it if incomplete.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def count_images_in_folder(folder_path):
    """Count image files in a folder."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Use a set to avoid double-counting on case-insensitive filesystems (Windows)
    image_files = set()
    
    for ext in image_extensions:
        for img_path in Path(folder_path).glob(f'*{ext}'):
            # Store normalized path to avoid duplicates
            image_files.add(img_path.resolve())
    
    return len(image_files)


def get_all_images_in_folder(folder_path, label):
    """Get all image files with their labels."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Use dict with resolved path as key to avoid duplicates on Windows
    images_dict = {}
    
    for ext in image_extensions:
        for img_path in Path(folder_path).glob(f'*{ext}'):
            resolved_path = img_path.resolve()
            if resolved_path not in images_dict:
                images_dict[resolved_path] = {
                    'filepath': str(Path(label) / img_path.name),
                    'label': label,
                    'full_path': str(img_path)
                }
    
    return list(images_dict.values())


def check_manifest(data_dir):
    """Check manifest completeness."""
    print("=" * 70)
    print("Manifest Verification Tool")
    print("=" * 70)
    
    data_dir = os.path.abspath(data_dir)
    print(f"\nChecking: {data_dir}\n")
    
    # Paths
    manifest_path = os.path.join(data_dir, "manifest.csv")
    pain_folder = os.path.join(data_dir, "Pain")
    nopain_folder = os.path.join(data_dir, "NoPain")
    
    # Check folders exist
    if not os.path.exists(pain_folder) or not os.path.exists(nopain_folder):
        print("❌ Error: Pain or NoPain folder not found!")
        return
    
    # Count images in folders
    print("Counting images in folders...")
    pain_count = count_images_in_folder(pain_folder)
    nopain_count = count_images_in_folder(nopain_folder)
    total_files = pain_count + nopain_count
    
    print(f"  Pain folder:   {pain_count:,} images")
    print(f"  NoPain folder: {nopain_count:,} images")
    print(f"  Total:         {total_files:,} images")
    
    # Check manifest
    if not os.path.exists(manifest_path):
        print(f"\n❌ Manifest not found at: {manifest_path}")
        print("\nWould you like to create a new manifest? (This will list all images)")
        return create_new_manifest(data_dir, pain_folder, nopain_folder)
    
    # Read manifest
    print(f"\nReading manifest...")
    try:
        manifest_df = pd.read_csv(manifest_path)
        manifest_count = len(manifest_df)
        
        pain_in_manifest = len(manifest_df[manifest_df['label'] == 'Pain'])
        nopain_in_manifest = len(manifest_df[manifest_df['label'] == 'NoPain'])
        
        print(f"  Pain in manifest:   {pain_in_manifest:,} entries")
        print(f"  NoPain in manifest: {nopain_in_manifest:,} entries")
        print(f"  Total in manifest:  {manifest_count:,} entries")
        
        # Compare
        print("\n" + "=" * 70)
        print("Analysis:")
        print("=" * 70)
        
        issues_found = False
        
        if manifest_count < total_files:
            print(f"\n❌ CRITICAL: Manifest is incomplete!")
            print(f"   Missing {total_files - manifest_count:,} images ({(total_files - manifest_count)/total_files*100:.1f}%)")
            issues_found = True
        
        if pain_in_manifest < pain_count:
            print(f"\n❌ CRITICAL: Missing Pain images in manifest!")
            print(f"   Folder has {pain_count:,} but manifest has {pain_in_manifest:,}")
            print(f"   Missing: {pain_count - pain_in_manifest:,} Pain images")
            issues_found = True
        
        if nopain_in_manifest < nopain_count:
            print(f"\n⚠️  WARNING: Missing NoPain images in manifest!")
            print(f"   Folder has {nopain_count:,} but manifest has {nopain_in_manifest:,}")
            print(f"   Missing: {nopain_count - nopain_in_manifest:,} NoPain images")
            issues_found = True
        
        if not issues_found:
            print(f"\n✓ Manifest is complete!")
            print(f"  All {total_files:,} images are accounted for.")
            return True
        
        # Offer to rebuild
        print("\n" + "=" * 70)
        print("RECOMMENDATION: Rebuild manifest to include all images")
        print("=" * 70)
        
        response = input("\nRebuild manifest now? This will backup the old one. (yes/no): ")
        if response.lower() in ['yes', 'y']:
            return create_new_manifest(data_dir, pain_folder, nopain_folder, backup=True)
        else:
            print("\nManifest not rebuilt. Please fix manually or re-run prepare_dataset.py")
            return False
            
    except Exception as e:
        print(f"\n❌ Error reading manifest: {e}")
        return False


def create_new_manifest(data_dir, pain_folder, nopain_folder, backup=False):
    """Create a complete manifest from scratch."""
    manifest_path = os.path.join(data_dir, "manifest.csv")
    
    # Backup old manifest if it exists
    if backup and os.path.exists(manifest_path):
        backup_path = manifest_path.replace('.csv', '_backup.csv')
        os.rename(manifest_path, backup_path)
        print(f"\n✓ Old manifest backed up to: {backup_path}")
    
    print("\nCreating new manifest...")
    print("This may take a few minutes...\n")
    
    all_entries = []
    
    # Process Pain images
    print("Processing Pain folder...")
    pain_images = get_all_images_in_folder(pain_folder, 'Pain')
    for img_info in tqdm(pain_images, desc="Pain images"):
        # Try to extract original filename from the split filename
        filename = os.path.basename(img_info['full_path'])
        
        # Check if it's a split image (has _L_ or _R_)
        if '_L_' in filename or '_R_' in filename:
            # Extract original ID and info
            parts = filename.split('_')
            orig_id = parts[0]
            split_id = 'L' if '_L_' in filename else 'R'
            
            all_entries.append({
                'filepath': img_info['filepath'],
                'label': img_info['label'],
                'original_filename': 'unknown',  # We don't have this info
                'split_id': split_id
            })
        else:
            # Not a split image, just add as-is
            all_entries.append({
                'filepath': img_info['filepath'],
                'label': img_info['label'],
                'original_filename': filename,
                'split_id': 'unknown'
            })
    
    # Process NoPain images
    print("Processing NoPain folder...")
    nopain_images = get_all_images_in_folder(nopain_folder, 'NoPain')
    for img_info in tqdm(nopain_images, desc="NoPain images"):
        filename = os.path.basename(img_info['full_path'])
        
        if '_L_' in filename or '_R_' in filename:
            parts = filename.split('_')
            orig_id = parts[0]
            split_id = 'L' if '_L_' in filename else 'R'
            
            all_entries.append({
                'filepath': img_info['filepath'],
                'label': img_info['label'],
                'original_filename': 'unknown',
                'split_id': split_id
            })
        else:
            all_entries.append({
                'filepath': img_info['filepath'],
                'label': img_info['label'],
                'original_filename': filename,
                'split_id': 'unknown'
            })
    
    # Create DataFrame and save
    manifest_df = pd.DataFrame(all_entries)
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"\n✓ New manifest created: {manifest_path}")
    print(f"  Total entries: {len(manifest_df):,}")
    print(f"  Pain: {len(manifest_df[manifest_df['label'] == 'Pain']):,}")
    print(f"  NoPain: {len(manifest_df[manifest_df['label'] == 'NoPain']):,}")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python check_manifest.py <path_to_SynPainProcessed>")
        print("\nExample:")
        print("  python check_manifest.py ./SynPainProcessed")
        sys.exit(1)
    
    check_manifest(sys.argv[1])