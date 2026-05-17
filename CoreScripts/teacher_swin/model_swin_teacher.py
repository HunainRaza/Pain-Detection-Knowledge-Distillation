#!/usr/bin/env python3
"""
Swin Transformer Teacher Model
================================

Swin Transformer pretrained on ImageNet used as teacher for knowledge distillation.
Modified final layer for binary pain/no-pain classification.

Research Extension: Testing whether attention-based teachers provide superior
knowledge transfer to DeiT students compared to purely convolutional approaches.

Maps to research hypothesis: Architectural alignment between teacher and student
(both attention-based) may improve knowledge distillation efficiency.
"""

import torch
import torch.nn as nn
import timm


class SwinTransformer_Teacher(nn.Module):
    """
    Swin Transformer teacher model for knowledge distillation.
    
    Research motivation: Test whether an attention-based teacher (Swin Transformer)
    provides better knowledge transfer to DeiT student compared to CNN teacher (ResNet50).
    
    Architecture advantages:
    - Hierarchical feature representation with shifted windows
    - Self-attention mechanism (similar to student's architecture)
    - Efficient computational complexity
    - Strong ImageNet pre-training
    """
    
    def __init__(self, pretrained=True, num_classes=2, model_variant='swin_base_patch4_window7_224'):
        """
        Initialize Swin Transformer teacher.
        
        Args:
            pretrained: Load ImageNet pretrained weights (default: True)
            num_classes: Number of output classes (default: 2)
            model_variant: Swin model variant to use (default: swin_base_patch4_window7_224)
                Options:
                - 'swin_tiny_patch4_window7_224': Swin-T (28M params)
                - 'swin_small_patch4_window7_224': Swin-S (50M params)
                - 'swin_base_patch4_window7_224': Swin-B (88M params) - RECOMMENDED
                - 'swin_base_patch4_window12_384': Swin-B (88M params, larger input)
        """
        super(SwinTransformer_Teacher, self).__init__()
        
        self.model_variant = model_variant
        self.num_classes = num_classes
        
        # Load pretrained Swin Transformer with original 1000 classes first
        print(f"Loading pretrained Swin Transformer as teacher model: {model_variant}")
        
        # Load pretrained Swin - timm handles head replacement automatically
        # when we specify num_classes
        self.swin = timm.create_model(
            model_variant,
            pretrained=pretrained,
            num_classes=num_classes  # timm will replace head correctly
        )
        
        if pretrained:
            # The head was already replaced by timm, but let's reinitialize it
            # for better convergence on our specific task
            if hasattr(self.swin, 'head'):
                if hasattr(self.swin.head, 'fc'):
                    # ClassifierHead structure
                    nn.init.xavier_uniform_(self.swin.head.fc.weight)
                    nn.init.zeros_(self.swin.head.fc.bias)
                    print(f"Reinitialized head.fc: {self.swin.head.fc.in_features} -> {num_classes}")
                elif isinstance(self.swin.head, nn.Linear):
                    # Simple Linear head
                    nn.init.xavier_uniform_(self.swin.head.weight)
                    nn.init.zeros_(self.swin.head.bias)
                    print(f"Reinitialized head: {self.swin.head.in_features} -> {num_classes}")
            elif hasattr(self.swin, 'fc'):
                nn.init.xavier_uniform_(self.swin.fc.weight)
                nn.init.zeros_(self.swin.fc.bias)
                print(f"Reinitialized fc: {self.swin.fc.in_features} -> {num_classes}")
        
        print(f"Teacher model initialized with {num_classes} output classes")
        print(f"Expected input size: 224x224 (will be resized if needed)")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W) - typically (B, 3, 224, 224)
               Note: If input is 256x256, it will be resized to 224x224
            
        Returns:
            Tensor: Logits (B, num_classes)
        """
        # Swin Transformer expects 224x224 input
        # Resize if needed
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        return self.swin(x)
    
    def get_hard_labels(self, x):
        """
        Get hard labels (argmax) for distillation.
        
        Per paper: Hard distillation uses teacher's predicted class as label.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Tensor: Hard labels (B,) with values in {0, 1}
        """
        with torch.no_grad():
            logits = self.forward(x)
            hard_labels = torch.argmax(logits, dim=1)
        
        return hard_labels
    
    def freeze(self):
        """
        Freeze all parameters (teacher is not trained).
        """
        for param in self.parameters():
            param.requires_grad = False
        
        print("Teacher model frozen (all parameters set to requires_grad=False)")


def create_swin_teacher(pretrained=True, num_classes=2, model_variant='swin_base_patch4_window7_224', 
                       freeze=True):
    """
    Factory function to create Swin Transformer teacher model.
    
    Args:
        pretrained: Load pretrained weights (default: True)
        num_classes: Number of output classes (default: 2)
        model_variant: Swin model variant (default: swin_base_patch4_window7_224)
        freeze: Freeze parameters after creation (default: True)
        
    Returns:
        SwinTransformer_Teacher model
    """
    model = SwinTransformer_Teacher(
        pretrained=pretrained, 
        num_classes=num_classes,
        model_variant=model_variant
    )
    
    if freeze:
        model.freeze()
    
    # Set to eval mode (no dropout, normalization in eval mode)
    model.eval()
    
    return model


def count_parameters(model):
    """
    Count total and trainable parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable


if __name__ == "__main__":
    """
    Test Swin Transformer teacher model creation and inference.
    """
    print("=" * 60)
    print("Testing Swin Transformer Teacher Model")
    print("=" * 60)
    
    # Test different model variants
    variants = [
        'swin_tiny_patch4_window7_224',
        'swin_small_patch4_window7_224',
        'swin_base_patch4_window7_224'
    ]
    
    for variant in variants:
        print(f"\nTesting {variant}:")
        print("-" * 60)
        
        # Create teacher model
        teacher = create_swin_teacher(
            pretrained=False,  # False for faster testing
            model_variant=variant,
            freeze=True
        )
        
        # Print model info
        total_params, trainable_params = count_parameters(teacher)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        batch_size = 4
        
        # Test with 224x224 input
        print("\nTesting with 224x224 input:")
        dummy_input_224 = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            logits = teacher(dummy_input_224)
            hard_labels = teacher.get_hard_labels(dummy_input_224)
        
        print(f"  Input shape: {dummy_input_224.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Hard labels shape: {hard_labels.shape}")
        print(f"  Hard labels: {hard_labels}")
        
        # Test with 256x256 input (will be resized)
        print("\nTesting with 256x256 input (automatic resize):")
        dummy_input_256 = torch.randn(batch_size, 3, 256, 256)
        
        with torch.no_grad():
            logits = teacher(dummy_input_256)
            hard_labels = teacher.get_hard_labels(dummy_input_256)
        
        print(f"  Input shape: {dummy_input_256.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Hard labels shape: {hard_labels.shape}")
        print(f"  Hard labels: {hard_labels}")
    
    print("\n" + "=" * 60)
    print("✓ Swin Transformer teacher model test successful!")
    print("=" * 60)
    
    # Print recommendation
    print("\nRECOMMENDED CONFIGURATION:")
    print("  Model: swin_base_patch4_window7_224")
    print("  Reasons:")
    print("    - Balance between capacity and efficiency")
    print("    - Strong ImageNet pretrained performance")
    print("    - Similar size to ResNet50 for fair comparison")
    print("    - 224x224 input matches DeiT expectations")