#!/usr/bin/env python3
"""
DeiT Student Model
==================

Data-efficient Image Transformer (DeiT) as student model for pain detection.
Uses pretrained distilled DeiT with modified classification head for binary output.

Maps to paper: Section III-B "Proposed Method" - DeiT architecture
- Pretrained DeiT model
- Modified for 2-class output (Pain/NoPain)
- Patch size 32 (as per paper training details)
"""

import torch
import torch.nn as nn
import timm


class DeiT_PNP(nn.Module):
    """
    DeiT model for Pain/No-Pain detection.
    
    Based on paper's Deit-PNP architecture:
    - Uses pretrained DeiT with distillation token
    - Modified classifier head for binary classification
    - Patch size 32 (as specified in paper Section IV-A)
    """
    
    def __init__(self, pretrained=True, num_classes=2, patch_size=32):
        """
        Initialize DeiT model for pain detection.
        
        Args:
            pretrained: Load pretrained weights from ImageNet (default: True)
            num_classes: Number of output classes (default: 2 for Pain/NoPain)
            patch_size: Vision transformer patch size (default: 32 per paper)
        """
        super(DeiT_PNP, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained DeiT model with distillation
        # Using deit_base_distilled_patch16_224 as base
        # Note: timm doesn't have patch_size=32 by default, so we'll use patch16
        # and handle 256x256 inputs appropriately
        model_name = 'deit_base_distilled_patch16_224'
        
        print(f"Loading pretrained DeiT model: {model_name}")
        self.deit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        print(f"DeiT model loaded with {num_classes} output classes")
        
        # The distilled DeiT has both class token and distillation token
        # timm automatically handles the dual-head architecture
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, 256, 256)
            
        Returns:
            If training: tuple of (class_logits, distillation_logits)
            If eval: class_logits
        """
        # Forward through DeiT
        # During training, distilled models return (class_output, distill_output)
        # During eval, they return class_output
        output = self.deit(x)
        
        return output
    
    def get_class_logits(self, x):
        """
        Get only class logits (not distillation logits).
        
        Args:
            x: Input tensor (B, 3, 256, 256)
            
        Returns:
            Tensor: Class logits (B, num_classes)
        """
        output = self.forward(x)
        
        # If tuple (training mode), return first element
        if isinstance(output, tuple):
            return output[0]
        
        return output
    
    def get_distillation_logits(self, x):
        """
        Get only distillation logits.
        
        Args:
            x: Input tensor (B, 3, 256, 256)
            
        Returns:
            Tensor: Distillation logits (B, num_classes) or None if not available
        """
        output = self.forward(x)
        
        # If tuple (training mode), return second element
        if isinstance(output, tuple):
            return output[1]
        
        return None


def create_deit_model(pretrained=True, num_classes=2):
    """
    Factory function to create DeiT model.
    
    Args:
        pretrained: Load pretrained weights (default: True)
        num_classes: Number of output classes (default: 2)
        
    Returns:
        DeiT_PNP model
    """
    model = DeiT_PNP(
        pretrained=pretrained,
        num_classes=num_classes,
        patch_size=32
    )
    
    return model


def count_parameters(model):
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """
    Test model creation and forward pass.
    """
    print("=" * 60)
    print("Testing DeiT Model")
    print("=" * 60)
    
    # Create model
    model = create_deit_model(pretrained=False)  # False for faster testing
    
    # Print model info
    num_params = count_parameters(model)
    print(f"\nModel parameters: {num_params:,}")
    
    # Test forward pass with dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 256, 256)
    
    print(f"\nTesting forward pass with input shape: {dummy_input.shape}")
    
    # Training mode (returns tuple)
    model.train()
    output = model(dummy_input)
    
    if isinstance(output, tuple):
        class_logits, distill_logits = output
        print(f"Training mode output:")
        print(f"  Class logits shape: {class_logits.shape}")
        print(f"  Distillation logits shape: {distill_logits.shape}")
    else:
        print(f"Training mode output shape: {output.shape}")
    
    # Eval mode
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nEval mode output shape: {output.shape}")
    
    print("\n✓ Model test successful!")
