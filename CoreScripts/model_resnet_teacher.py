#!/usr/bin/env python3
"""
ResNet50 Teacher Model
======================

ResNet50 pretrained on ImageNet used as teacher for knowledge distillation.
Modified final layer for binary pain/no-pain classification.

Maps to paper: Section III-B "Proposed Method" - Teacher model
- ResNet50 as teacher (specified in Fig. 2)
- Provides hard labels for distillation
- Not trained, used only for inference
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50_Teacher(nn.Module):
    """
    ResNet50 teacher model for knowledge distillation.
    
    Per paper: ResNet50 pretrained on ImageNet, modified for binary classification.
    Used to provide teacher labels during student training.
    """
    
    def __init__(self, pretrained=True, num_classes=2):
        """
        Initialize ResNet50 teacher.
        
        Args:
            pretrained: Load ImageNet pretrained weights (default: True)
            num_classes: Number of output classes (default: 2)
        """
        super(ResNet50_Teacher, self).__init__()
        
        # Load pretrained ResNet50
        print("Loading pretrained ResNet50 as teacher model")
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify final fully connected layer for binary classification
        # Original: 2048 -> 1000 (ImageNet classes)
        # Modified: 2048 -> 2 (Pain/NoPain)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
        print(f"Teacher model initialized with {num_classes} output classes")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, 256, 256)
            
        Returns:
            Tensor: Logits (B, num_classes)
        """
        return self.resnet(x)
    
    def get_hard_labels(self, x):
        """
        Get hard labels (argmax) for distillation.
        
        Per paper: Hard distillation uses teacher's predicted class as label.
        
        Args:
            x: Input tensor (B, 3, 256, 256)
            
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


def create_teacher_model(pretrained=True, num_classes=2, freeze=True):
    """
    Factory function to create ResNet50 teacher model.
    
    Args:
        pretrained: Load pretrained weights (default: True)
        num_classes: Number of output classes (default: 2)
        freeze: Freeze parameters after creation (default: True)
        
    Returns:
        ResNet50_Teacher model
    """
    model = ResNet50_Teacher(pretrained=pretrained, num_classes=num_classes)
    
    if freeze:
        model.freeze()
    
    # Set to eval mode (no dropout, batch norm in eval mode)
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
    Test teacher model creation and inference.
    """
    print("=" * 60)
    print("Testing ResNet50 Teacher Model")
    print("=" * 60)
    
    # Create teacher model
    teacher = create_teacher_model(pretrained=False, freeze=True)  # False for faster testing
    
    # Print model info
    total_params, trainable_params = count_parameters(teacher)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 256, 256)
    
    print(f"\nTesting forward pass with input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        logits = teacher(dummy_input)
        hard_labels = teacher.get_hard_labels(dummy_input)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Hard labels shape: {hard_labels.shape}")
    print(f"Hard labels: {hard_labels}")
    
    print("\n✓ Teacher model test successful!")
