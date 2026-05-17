#!/usr/bin/env python3
"""
Inference Script for Pain Detection
====================================

Supports:
- Single image inference (side-by-side or preprocessed)
- Batch inference on a folder
- Output predictions with confidence scores

Maps to paper: Deployment and inference on new images
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Project imports
from model_deit import create_deit_model
from Utils.transforms import get_inference_transforms


class PainDetector:
    """
    Pain detection inference wrapper.
    """
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize detector with trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = ['NoPain', 'Pain']
        
        # Load model
        print(f"Loading model from {checkpoint_path}")
        self.model = create_deit_model(pretrained=False, num_classes=2)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transforms
        self.transform = get_inference_transforms(224)  # DeiT expects 224x224
        
        print(f"Model loaded successfully on {self.device}")
    
    def _is_side_by_side(self, image):
        """
        Detect if image is side-by-side format.
        Simple heuristic: if width > 1.5 * height, assume side-by-side
        
        Args:
            image: PIL Image
            
        Returns:
            bool: True if likely side-by-side format
        """
        width, height = image.size
        return width > 1.5 * height
    
    def _split_side_by_side(self, image):
        """
        Split side-by-side image into left and right halves.
        
        Args:
            image: PIL Image
            
        Returns:
            tuple: (left_half, right_half)
        """
        width, height = image.size
        mid_point = width // 2
        
        left_half = image.crop((0, 0, mid_point, height))
        right_half = image.crop((mid_point, 0, width, height))
        
        return left_half, right_half
    
    def predict_single(self, image_path, handle_side_by_side=True):
        """
        Predict on a single image.
        
        Args:
            image_path: Path to image file
            handle_side_by_side: Automatically detect and split side-by-side images
            
        Returns:
            dict: Prediction results
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        results = []
        
        # Check if side-by-side
        if handle_side_by_side and self._is_side_by_side(image):
            print(f"Detected side-by-side image, splitting into halves...")
            left_half, right_half = self._split_side_by_side(image)
            
            # Predict on both halves
            for half_name, half_img in [('left', left_half), ('right', right_half)]:
                pred = self._predict_image(half_img)
                pred['half'] = half_name
                results.append(pred)
        else:
            # Single image prediction
            pred = self._predict_image(image)
            pred['half'] = 'full'
            results.append(pred)
        
        return results
    
    def _predict_image(self, image):
        """
        Internal method to predict on a preprocessed PIL image.
        
        Args:
            image: PIL Image
            
        Returns:
            dict: Prediction result
        """
        # Transform image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
            # Get class logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Get probabilities and prediction
            probs = F.softmax(logits, dim=1)[0]
            pred_class_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_class_idx].item())
        
        return {
            'predicted_class': self.class_names[pred_class_idx],
            'predicted_class_idx': pred_class_idx,
            'confidence': confidence,
            'probabilities': {
                'NoPain': float(probs[0].item()),
                'Pain': float(probs[1].item())
            }
        }
    
    def predict_batch(self, image_dir, output_csv=None):
        """
        Predict on all images in a directory.
        
        Args:
            image_dir: Directory containing images
            output_csv: Path to save results CSV (optional)
            
        Returns:
            pandas.DataFrame: Results dataframe
        """
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images in {image_dir}")
        
        # Predict on each image
        results = []
        
        for image_path in tqdm(image_files, desc='Processing images'):
            try:
                predictions = self.predict_single(str(image_path), handle_side_by_side=False)
                
                for pred in predictions:
                    results.append({
                        'filename': image_path.name,
                        'filepath': str(image_path),
                        'predicted_class': pred['predicted_class'],
                        'confidence': pred['confidence'],
                        'prob_nopain': pred['probabilities']['NoPain'],
                        'prob_pain': pred['probabilities']['Pain']
                    })
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # Create dataframe
        df = pd.DataFrame(results)
        
        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\n✓ Results saved to: {output_csv}")
        
        return df


def main(args):
    """
    Main inference function.
    
    Args:
        args: Parsed command line arguments
    """
    print("=" * 60)
    print("Pain Detection Inference")
    print("=" * 60)
    print()
    
    # Initialize detector
    detector = PainDetector(args.checkpoint, device=args.device)
    
    if args.image:
        # Single image inference
        print(f"\nProcessing single image: {args.image}")
        print()
        
        results = detector.predict_single(args.image, handle_side_by_side=True)
        
        # Print results
        print("\nPrediction Results:")
        print("-" * 40)
        
        for i, result in enumerate(results):
            if len(results) > 1:
                print(f"\n{result['half'].capitalize()} half:")
            
            print(f"  Predicted class: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']*100:.2f}%")
            print(f"  Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"    {class_name}: {prob*100:.2f}%")
    
    elif args.folder:
        # Batch inference
        print(f"\nProcessing folder: {args.folder}")
        
        output_csv = args.output if args.output else os.path.join(args.folder, 'predictions.csv')
        
        df = detector.predict_batch(args.folder, output_csv)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Batch Inference Summary")
        print("=" * 60)
        print(f"\nTotal images processed: {len(df)}")
        print(f"\nPrediction distribution:")
        print(df['predicted_class'].value_counts())
        print(f"\nAverage confidence: {df['confidence'].mean()*100:.2f}%")
        
        if args.output:
            print(f"\n✓ Results saved to: {args.output}")
    
    else:
        print("Error: Please specify either --image or --folder")
    
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on images for pain detection"
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str,
                           help='Path to single image file')
    input_group.add_argument('--folder', type=str,
                           help='Path to folder containing images')
    
    parser.add_argument('--output', type=str,
                       help='Output CSV path for batch inference results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    main(args)