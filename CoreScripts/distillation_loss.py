#!/usr/bin/env python3
"""
Knowledge Distillation Loss
============================

Implements hard distillation loss as described in the paper.

Maps to paper: Section III-B and Fig. 2
- Hard distillation: teacher provides hard labels (argmax, temperature=1)
- Loss = BCE(student_class, ground_truth) + BCE(student_distill, teacher_label)
- Both components use Binary Cross Entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardDistillationLoss(nn.Module):
    """
    Hard distillation loss for DeiT training with teacher.
    
    Per paper (Section III-B, Fig. 2):
    - L_total = L_BCE + L_teacher
    - L_BCE: Binary cross entropy between student class token and ground truth
    - L_teacher: Binary cross entropy between student distillation token and teacher hard label
    - Hard distillation: Use argmax of teacher logits as label (temperature = 1)
    """
    
    def __init__(self, class_weight=1.0, distill_weight=1.0):
        """
        Initialize distillation loss.
        
        Args:
            class_weight: Weight for classification loss (default: 1.0)
            distill_weight: Weight for distillation loss (default: 1.0)
        """
        super(HardDistillationLoss, self).__init__()
        
        self.class_weight = class_weight
        self.distill_weight = distill_weight
        
        # Binary cross entropy loss
        self.bce_loss = nn.CrossEntropyLoss()
        
        print(f"Hard Distillation Loss initialized:")
        print(f"  Class weight: {class_weight}")
        print(f"  Distillation weight: {distill_weight}")
    
    def forward(self, student_outputs, teacher_labels, ground_truth_labels):
        """
        Compute combined distillation loss.
        
        Args:
            student_outputs: Student model outputs
                - If tuple: (class_logits, distill_logits)
                - If tensor: class_logits only
            teacher_labels: Hard labels from teacher (B,) with values in {0, 1}
            ground_truth_labels: Ground truth labels (B,) with values in {0, 1}
            
        Returns:
            tuple: (total_loss, class_loss, distill_loss)
        """
        # Extract student outputs
        if isinstance(student_outputs, tuple):
            class_logits, distill_logits = student_outputs
        else:
            class_logits = student_outputs
            distill_logits = None
        
        # Classification loss: BCE between student class token and ground truth
        # Maps to L_BCE in Fig. 2
        class_loss = self.bce_loss(class_logits, ground_truth_labels)
        
        # Distillation loss: BCE between student distillation token and teacher label
        # Maps to L_teacher in Fig. 2
        if distill_logits is not None:
            distill_loss = self.bce_loss(distill_logits, teacher_labels)
        else:
            # If no distillation logits, use class logits for distillation too
            distill_loss = self.bce_loss(class_logits, teacher_labels)
        
        # Total loss: weighted sum
        total_loss = (self.class_weight * class_loss + 
                     self.distill_weight * distill_loss)
        
        return total_loss, class_loss, distill_loss


def create_distillation_loss(class_weight=1.0, distill_weight=1.0):
    """
    Factory function to create distillation loss.
    
    Args:
        class_weight: Weight for classification loss (default: 1.0)
        distill_weight: Weight for distillation loss (default: 1.0)
        
    Returns:
        HardDistillationLoss instance
    """
    return HardDistillationLoss(class_weight, distill_weight)


if __name__ == "__main__":
    """
    Test distillation loss computation.
    """
    print("=" * 60)
    print("Testing Hard Distillation Loss")
    print("=" * 60)
    
    # Create loss function
    loss_fn = create_distillation_loss(class_weight=1.0, distill_weight=1.0)
    
    # Dummy data
    batch_size = 4
    num_classes = 2
    
    # Student outputs (class logits, distillation logits)
    class_logits = torch.randn(batch_size, num_classes)
    distill_logits = torch.randn(batch_size, num_classes)
    student_outputs = (class_logits, distill_logits)
    
    # Teacher hard labels (0 or 1)
    teacher_labels = torch.randint(0, 2, (batch_size,))
    
    # Ground truth labels (0 or 1)
    ground_truth_labels = torch.randint(0, 2, (batch_size,))
    
    print(f"\nTest inputs:")
    print(f"  Class logits shape: {class_logits.shape}")
    print(f"  Distill logits shape: {distill_logits.shape}")
    print(f"  Teacher labels: {teacher_labels}")
    print(f"  Ground truth labels: {ground_truth_labels}")
    
    # Compute loss
    total_loss, class_loss, distill_loss = loss_fn(
        student_outputs, 
        teacher_labels, 
        ground_truth_labels
    )
    
    print(f"\nLoss values:")
    print(f"  Classification loss: {class_loss.item():.4f}")
    print(f"  Distillation loss: {distill_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    
    # Test backward pass
    total_loss.backward()
    print(f"\n✓ Backward pass successful!")
    
    print("\n✓ Distillation loss test successful!")
