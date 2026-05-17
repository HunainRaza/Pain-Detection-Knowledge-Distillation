#!/usr/bin/env python3
"""
Simple Dataset Status Checker
Shows the ACTUAL current state of your dataset.
"""

import os
from pathlib import Path


def count_images(folder):
    """Count images, avoiding duplicates on Windows."""
    if not os.path.exists(folder):
        return 0
    
    image_files = set()
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    for ext in extensions:
        for f in Path(folder).glob(f'*{ext}'):
            image_files.add(f.resolve())
    
    return len(image_files)


def main():
    base = "./SynPainProcessed"
    
    print("=" * 70)
    print("CURRENT DATASET STATUS")
    print("=" * 70)
    
    # Original split images
    print("\n📁 Original Split Images (from prepare_dataset.py):")
    pain_orig = count_images(f"{base}/Pain")
    nopain_orig = count_images(f"{base}/NoPain")
    print(f"   Pain:   {pain_orig:,} images")
    print(f"   NoPain: {nopain_orig:,} images")
    print(f"   Total:  {pain_orig + nopain_orig:,} images")
    
    # Cropped images
    print("\n📁 Cropped/Preprocessed Images (from preprocess_faces.py):")
    if os.path.exists(f"{base}/Cropped"):
        pain_crop = count_images(f"{base}/Cropped/Pain")
        nopain_crop = count_images(f"{base}/Cropped/NoPain")
        print(f"   Pain:   {pain_crop:,} images")
        print(f"   NoPain: {nopain_crop:,} images")
        print(f"   Total:  {pain_crop + nopain_crop:,} images")
    else:
        print("   (Not created yet)")
        pain_crop = 0
        nopain_crop = 0
    
    # Manifest
    print("\n📄 Manifest.csv:")
    manifest_path = f"{base}/manifest.csv"
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        total_manifest = len(lines) - 1  # subtract header
        
        # Count Pain/NoPain in manifest
        pain_manifest = sum(1 for line in lines if ',Pain,' in line)
        nopain_manifest = sum(1 for line in lines if ',NoPain,' in line)
        
        print(f"   Pain entries:   {pain_manifest:,}")
        print(f"   NoPain entries: {nopain_manifest:,}")
        print(f"   Total entries:  {total_manifest:,}")
    else:
        print("   (Not found)")
        pain_manifest = 0
        nopain_manifest = 0
        total_manifest = 0
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Check if manifest matches original images
    if total_manifest == pain_orig + nopain_orig:
        print("\n✅ Manifest is COMPLETE - matches all original split images")
    elif total_manifest < pain_orig + nopain_orig:
        missing = (pain_orig + nopain_orig) - total_manifest
        print(f"\n❌ Manifest is INCOMPLETE - missing {missing:,} images!")
        print(f"   Original images: {pain_orig + nopain_orig:,}")
        print(f"   Manifest entries: {total_manifest:,}")
        print(f"   Missing: {missing:,} ({missing/(pain_orig + nopain_orig)*100:.1f}%)")
    
    # Check if cropping is complete
    if pain_crop + nopain_crop > 0:
        if pain_crop + nopain_crop == total_manifest:
            print(f"\n✅ Cropping is COMPLETE - all manifest entries were processed")
        elif pain_crop + nopain_crop < total_manifest:
            missing_crop = total_manifest - (pain_crop + nopain_crop)
            print(f"\n⚠️  Cropping is INCOMPLETE - only {pain_crop + nopain_crop:,}/{total_manifest:,} processed")
            print(f"   Missing: {missing_crop:,} images")
    
    # Check Pain class imbalance
    if pain_crop > 0 and nopain_crop > 0:
        pain_ratio = pain_crop / (pain_crop + nopain_crop) * 100
        print(f"\n📊 Class Balance in Cropped Dataset:")
        print(f"   Pain:   {pain_crop:,} ({pain_ratio:.1f}%)")
        print(f"   NoPain: {nopain_crop:,} ({100-pain_ratio:.1f}%)")
        
        if pain_ratio < 10:
            print(f"\n   ❌ SEVERE IMBALANCE! Pain class is only {pain_ratio:.1f}%")
            print(f"      This will cause training problems!")
        elif pain_ratio < 20:
            print(f"\n   ⚠️  Moderate imbalance. Consider using class weights in training.")
        else:
            print(f"\n   ✅ Acceptable balance for training.")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if total_manifest < pain_orig + nopain_orig:
        print("\n🔧 ACTION REQUIRED:")
        print("   1. Re-run prepare_dataset.py on your ORIGINAL raw SynPain images")
        print("   2. This should create manifest with all images")
        print("   3. Then re-run preprocess_faces.py")
    elif pain_crop == 0 or nopain_crop == 0:
        print("\n🔧 NEXT STEP:")
        print("   Run preprocess_faces.py to create cropped images")
    elif pain_crop + nopain_crop < total_manifest:
        print("\n🔧 ACTION REQUIRED:")
        print("   Re-run preprocess_faces.py to process all images")
    elif pain_ratio < 10 if pain_crop > 0 else False:
        print("\n🔧 CRITICAL ISSUE:")
        print("   Your Pain class is severely underrepresented!")
        print("   Go back to Step 1 and check your original dataset.")
    else:
        print("\n✅ Dataset looks good! Ready for training.")
        print("   Next: python train.py --data ./SynPainProcessed/Cropped ...")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()