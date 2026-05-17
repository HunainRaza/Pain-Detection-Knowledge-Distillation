#!/usr/bin/env python3
"""
Dataset Preparation for SynPain Dataset
========================================

This script processes the SynPain dataset which contains side-by-side facial images.
Each image is split into left and right halves and labeled according to the following rule:

LABELING RULE (as corrected by user):
- If filename contains "Pain": left half -> NoPain, right half -> Pain
- If filename contains "NoPain": left half -> NoPain, right half -> NoPain

Maps to paper: Section III-A "Databases and Pre-Processing"
"""

import os
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def parse_synpain_filename(filename):
    """
    Parse SynPain filename to extract metadata.
    
    Filename format: [ID]_[expression]_[gender]_[age].jpg
    Example: 1010300012_Pain_Woman_Young.jpg
    
    Args:
        filename: SynPain image filename
        
    Returns:
        dict: Parsed metadata including expression, gender, age, ID
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) >= 4:
        return {
            'id': parts[0],
            'expression': parts[1],
            'gender': parts[2],
            'age': parts[3]
        }
    else:
        return None


def split_and_label_image(image_path, output_dir, manifest_data):
    """
    Split a side-by-side image into left and right halves and apply labeling rule.
    
    CRITICAL LABELING RULE:
    - If "Pain" in filename: left -> NoPain, right -> Pain
    - If "NoPain" in filename: left -> NoPain, right -> NoPain
    
    Args:
        image_path: Path to side-by-side image
        output_dir: Base output directory
        manifest_data: List to append manifest entries
        
    Returns:
        tuple: (num_pain, num_nopain) images created
    """
    filename = os.path.basename(image_path)
    metadata = parse_synpain_filename(filename)
    
    if metadata is None:
        print(f"Warning: Could not parse filename {filename}, skipping")
        return 0, 0
    
    try:
        # Load image
        img = Image.open(image_path)
        width, height = img.size
        
        # Handle odd width by using integer division
        mid_point = width // 2
        
        # Split into left and right halves (pixel-accurate)
        left_half = img.crop((0, 0, mid_point, height))
        right_half = img.crop((mid_point, 0, width, height))
        
        # Apply labeling rule based on filename
        expression = metadata['expression']
        orig_id = metadata['id']
        
        num_pain = 0
        num_nopain = 0
        
        if "Pain" in expression and "NoPain" not in expression:
            # Filename contains "Pain" -> left=NoPain, right=Pain
            left_label = "NoPain"
            right_label = "Pain"
        elif "NoPain" in expression:
            # Filename contains "NoPain" -> both=NoPain
            left_label = "NoPain"
            right_label = "NoPain"
        else:
            print(f"Warning: Unexpected expression '{expression}' in {filename}")
            return 0, 0
        
        # Save left half
        left_output_dir = os.path.join(output_dir, left_label)
        os.makedirs(left_output_dir, exist_ok=True)
        left_filename = f"{orig_id}_L_{left_label}.jpg"
        left_path = os.path.join(left_output_dir, left_filename)
        left_half.save(left_path, quality=95)
        
        manifest_data.append({
            'filepath': os.path.relpath(left_path, output_dir),
            'label': left_label,
            'original_filename': filename,
            'split_id': 'L'
        })
        
        if left_label == "Pain":
            num_pain += 1
        else:
            num_nopain += 1
        
        # Save right half
        right_output_dir = os.path.join(output_dir, right_label)
        os.makedirs(right_output_dir, exist_ok=True)
        right_filename = f"{orig_id}_R_{right_label}.jpg"
        right_path = os.path.join(right_output_dir, right_filename)
        right_half.save(right_path, quality=95)
        
        manifest_data.append({
            'filepath': os.path.relpath(right_path, output_dir),
            'label': right_label,
            'original_filename': filename,
            'split_id': 'R'
        })
        
        if right_label == "Pain":
            num_pain += 1
        else:
            num_nopain += 1
        
        return num_pain, num_nopain
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return 0, 0


def process_dataset(raw_dir, output_dir):
    """
    Process entire SynPain dataset.
    
    Args:
        raw_dir: Directory containing raw SynPain images
        output_dir: Output directory for processed images
    """
    print("=" * 60)
    print("SynPain Dataset Preparation")
    print("=" * 60)
    print(f"\nInput directory: {raw_dir}")
    print(f"Output directory: {output_dir}")
    print("\nApplying labeling rule:")
    print("  - If 'Pain' in filename: left=NoPain, right=Pain")
    print("  - If 'NoPain' in filename: left=NoPain, right=NoPain")
    print()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Pain"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "NoPain"), exist_ok=True)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    video_files = []
    
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(root, file))
            elif ext == '.mp4':
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images")
    
    if video_files:
        print(f"Found {len(video_files)} video files (will be logged but not processed)")
        video_log_path = os.path.join(output_dir, "video_files_detected.txt")
        with open(video_log_path, 'w') as f:
            for vf in video_files:
                f.write(f"{vf}\n")
        print(f"Video file list saved to: {video_log_path}")
    
    # Process all images
    manifest_data = []
    total_pain = 0
    total_nopain = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        num_pain, num_nopain = split_and_label_image(img_path, output_dir, manifest_data)
        total_pain += num_pain
        total_nopain += num_nopain
    
    # Create manifest CSV
    manifest_df = pd.DataFrame(manifest_data)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nTotal images created: {len(manifest_data)}")
    print(f"  - Pain: {total_pain}")
    print(f"  - NoPain: {total_nopain}")
    print(f"\nManifest saved to: {manifest_path}")
    print(f"\nOutput directory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── Pain/")
    print(f"    ├── NoPain/")
    print(f"    └── manifest.csv")
    

def run_sanity_check():
    """
    Sanity check: Verify labeling logic on example filenames.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK: Labeling Logic Verification")
    print("=" * 60)
    
    test_cases = [
        ("1010300012_Pain_Woman_Young.jpg", "Pain", "NoPain", "Pain"),
        ("1000100045_NoPain_Man_Old.jpg", "NoPain", "NoPain", "NoPain"),
        ("1111400089_Pain_Man_Young.jpg", "Pain", "NoPain", "Pain"),
        ("1001200123_NoPain_Woman_Old.jpg", "NoPain", "NoPain", "NoPain"),
    ]
    
    print("\nTest cases:")
    print(f"{'Filename':<40} {'Expression':<15} {'Left Label':<15} {'Right Label':<15}")
    print("-" * 85)
    
    all_passed = True
    for filename, expected_expr, expected_left, expected_right in test_cases:
        metadata = parse_synpain_filename(filename)

        # parse_synpain_filename may return None for malformed filenames;
        # guard against that to satisfy static checkers and avoid runtime errors.
        if metadata is None:
            expression = ""
            left_label = "ERROR"
            right_label = "ERROR"
            print(f"Warning: Could not parse filename {filename} during sanity check")
        else:
            expression = metadata['expression']

            # Apply labeling rule
            if "Pain" in expression and "NoPain" not in expression:
                left_label = "NoPain"
                right_label = "Pain"
            elif "NoPain" in expression:
                left_label = "NoPain"
                right_label = "NoPain"
            else:
                left_label = "ERROR"
                right_label = "ERROR"
        
        status = "✓" if (left_label == expected_left and right_label == expected_right) else "✗"
        print(f"{filename:<40} {expression:<15} {left_label:<15} {right_label:<15} {status}")
        
        if left_label != expected_left or right_label != expected_right:
            all_passed = False
    
    print("-" * 85)
    if all_passed:
        print("✓ All sanity checks passed!")
    else:
        print("✗ Some sanity checks failed!")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare SynPain dataset: split side-by-side images and apply labeling rule"
    )
    parser.add_argument(
        '--raw', 
        type=str, 
        required=True,
        help='Path to raw SynPain images directory'
    )
    parser.add_argument(
        '--out', 
        type=str, 
        default='./SynPainProcessed',
        help='Output directory for processed images (default: ./SynPainProcessed)'
    )
    parser.add_argument(
        '--sanity-check',
        action='store_true',
        help='Run sanity check on labeling logic'
    )
    
    args = parser.parse_args()
    
    if args.sanity_check:
        run_sanity_check()
    else:
        process_dataset(args.raw, args.out)
