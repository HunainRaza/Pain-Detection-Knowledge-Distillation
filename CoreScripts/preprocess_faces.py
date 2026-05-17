#!/usr/bin/env python3
"""
Face Preprocessing with MTCNN
==============================

This script detects, aligns, and crops faces to 256x256 using MTCNN.
Implements fallback center-crop for failed detections.

Maps to paper: Section III-A "Databases and Pre-Processing"
- MTCNN for face detection
- Alignment and cropping to 256x256
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch


class FacePreprocessor:
    """
    Face detection and preprocessing using MTCNN.
    """
    
    def __init__(self, target_size=256, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize face preprocessor.
        
        Args:
            target_size: Output image size (default: 256x256)
            device: Device for MTCNN ('cuda' or 'cpu')
        """
        self.target_size = target_size
        self.device = device
        
        # Initialize MTCNN for face detection
        # keep_all=False returns only the most prominent face
        self.mtcnn = MTCNN(
            keep_all=False,
            device=self.device,
            post_process=False
        )
        
        print(f"MTCNN initialized on device: {self.device}")
    
    def detect_and_align(self, image):
        """
        Detect face using MTCNN and return aligned/cropped face.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            numpy array: Aligned face (256x256) or None if detection fails
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Detect face and landmarks. facenet-pytorch's `detect` may return
        # either (boxes, probs, landmarks) or (boxes, landmarks) depending
        # on version/options. To avoid static typing/unpacking issues, handle
        # the return as a single variable and unpack by length.
        det = self.mtcnn.detect(image, landmarks=True)

        boxes = probs = landmarks = None
        if det is None:
            boxes = None
        else:
            # det can be a tuple of length 3 (boxes, probs, landmarks)
            # or length 2 (boxes, landmarks). Use indexing to avoid
            # direct tuple unpacking (which Pylance flags).
            if isinstance(det, tuple) and len(det) == 3:
                boxes, probs, landmarks = det
            elif isinstance(det, tuple) and len(det) == 2:
                boxes, landmarks = det
                probs = None
            else:
                boxes = None

        if boxes is None or (hasattr(boxes, '__len__') and len(boxes) == 0):
            return None

        # Choose the most confident detection when confidences are available
        if probs is not None:
            try:
                idx = int(np.argmax(probs))
            except Exception:
                idx = 0
        else:
            idx = 0

        box = boxes[idx]
        landmark = landmarks[idx] if landmarks is not None else None
        
        # Convert to numpy for processing
        img_np = np.array(image)
        
        # Align face if landmarks available
        if landmark is not None:
            aligned_face = self._align_face(img_np, landmark)
        else:
            # Just crop using bounding box
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)
            aligned_face = img_np[y1:y2, x1:x2]
        
        # Resize to target size
        if aligned_face is not None and aligned_face.size > 0:
            aligned_face = cv2.resize(aligned_face, (self.target_size, self.target_size))
            return aligned_face
        
        return None
    
    def _align_face(self, image, landmarks):
        """
        Align face based on eye landmarks.
        
        Args:
            image: numpy array image
            landmarks: facial landmarks from MTCNN
            
        Returns:
            numpy array: Aligned face region
        """
        # Extract eye positions (landmarks: left_eye, right_eye, nose, mouth_left, mouth_right)
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Calculate angle for alignment
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Calculate center point between eyes
        eye_center = ((left_eye[0] + right_eye[0]) // 2, 
                      (left_eye[1] + right_eye[1]) // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        
        # Apply rotation
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                flags=cv2.INTER_CUBIC)
        
        # Calculate face bounding box with some margin
        eye_distance = np.sqrt((dX ** 2) + (dY ** 2))
        margin = int(eye_distance * 1.5)
        
        x1 = max(0, int(eye_center[0] - margin))
        y1 = max(0, int(eye_center[1] - margin))
        x2 = min(aligned.shape[1], int(eye_center[0] + margin))
        y2 = min(aligned.shape[0], int(eye_center[1] + margin))
        
        return aligned[y1:y2, x1:x2]
    
    def center_crop_fallback(self, image):
        """
        Fallback: Center crop and resize if MTCNN fails.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            numpy array: Center-cropped face (256x256)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w = image.shape[:2]
        
        # Take center square crop
        size = min(h, w)
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        
        cropped = image[y1:y1+size, x1:x1+size]
        
        # Resize to target
        resized = cv2.resize(cropped, (self.target_size, self.target_size))
        
        return resized


def process_images(input_dir, output_dir, log_failures=True):
    """
    Process all images in the dataset with face detection and preprocessing.
    
    Args:
        input_dir: Directory with split images (Pain/ and NoPain/ subdirs)
        output_dir: Output directory for cropped faces
        log_failures: Whether to log MTCNN detection failures
    """
    print("=" * 60)
    print("Face Preprocessing with MTCNN")
    print("=" * 60)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Initialize preprocessor
    preprocessor = FacePreprocessor(target_size=256)
    
    # Load manifest
    manifest_path = os.path.join(input_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        print("Please run prepare_dataset.py first.")
        return
    
    manifest_df = pd.read_csv(manifest_path)
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "Pain"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "NoPain"), exist_ok=True)
    
    # Track statistics
    stats = {
        'total': 0,
        'mtcnn_success': 0,
        'mtcnn_failure': 0,
        'fallback_used': 0,
        'failed': 0
    }
    
    failures = []
    new_manifest_data = []
    
    # Process each image
    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Processing faces"):
        stats['total'] += 1
        
        input_path = os.path.join(input_dir, row['filepath'])
        
        if not os.path.exists(input_path):
            print(f"Warning: File not found: {input_path}")
            stats['failed'] += 1
            continue
        
        try:
            # Load image
            image = Image.open(input_path).convert('RGB')
            
            # Try MTCNN detection
            processed = preprocessor.detect_and_align(image)
            
            if processed is not None:
                stats['mtcnn_success'] += 1
                used_fallback = False
            else:
                # Fallback to center crop
                stats['mtcnn_failure'] += 1
                stats['fallback_used'] += 1
                processed = preprocessor.center_crop_fallback(image)
                used_fallback = True
                failures.append(row['filepath'])
            
            # Save processed image
            output_path = os.path.join(output_dir, row['filepath'])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert BGR to RGB if needed
            if len(processed.shape) == 3 and processed.shape[2] == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            cv2.imwrite(output_path, processed)
            
            # Update manifest
            new_manifest_data.append({
                'filepath': row['filepath'],
                'label': row['label'],
                'original_filename': row['original_filename'],
                'split_id': row['split_id'],
                'mtcnn_detected': not used_fallback
            })
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            stats['failed'] += 1
    
    # Save updated manifest
    new_manifest_df = pd.DataFrame(new_manifest_data)
    new_manifest_path = os.path.join(output_dir, "manifest.csv")
    new_manifest_df.to_csv(new_manifest_path, index=False)
    
    # Save failure log
    if log_failures and failures:
        failure_log_path = os.path.join(output_dir, "mtcnn_failures.txt")
        with open(failure_log_path, 'w') as f:
            for fail in failures:
                f.write(f"{fail}\n")
        print(f"\nMTCNN failures logged to: {failure_log_path}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nTotal images processed: {stats['total']}")
    print(f"MTCNN detections successful: {stats['mtcnn_success']} ({stats['mtcnn_success']/stats['total']*100:.1f}%)")
    print(f"MTCNN detections failed: {stats['mtcnn_failure']} ({stats['mtcnn_failure']/stats['total']*100:.1f}%)")
    print(f"Fallback center-crop used: {stats['fallback_used']}")
    print(f"Total failures: {stats['failed']}")
    print(f"\nUpdated manifest saved to: {new_manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess faces: MTCNN detection, alignment, and crop to 256x256"
    )
    parser.add_argument(
        '--in',
        dest='input_dir',
        type=str,
        required=True,
        help='Input directory with split images (output from prepare_dataset.py)'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='./SynPainProcessed/Cropped',
        help='Output directory for cropped faces'
    )
    parser.add_argument(
        '--no-log-failures',
        action='store_true',
        help='Do not log MTCNN detection failures'
    )
    
    args = parser.parse_args()
    
    process_images(args.input_dir, args.out, log_failures=not args.no_log_failures)
