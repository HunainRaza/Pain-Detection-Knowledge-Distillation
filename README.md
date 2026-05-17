# Pain Detection from Facial Expressions

Implementation of "Pain Detection From Facial Expressions Based on Transformers and Distillation" (El Morabit & Rivenq, 2022) adapted for the **SynPain** synthetic dataset.

## Overview

This codebase implements the DeiT-PNP (Data-efficient Image Transformer for Pain/No-Pain detection) architecture with knowledge distillation from a ResNet50 teacher model. The implementation follows the paper's methodology while adapting it to work with the SynPain synthetic pain expression dataset.

### Key Features

- **DeiT Student Model**: Pretrained data-efficient image transformer for pain detection
- **ResNet50 Teacher**: Provides hard labels for knowledge distillation
- **Hard Distillation Loss**: Binary cross-entropy on both class and distillation tokens
- **SynPain Dataset Support**: Handles side-by-side image format with proper labeling
- **MTCNN Face Detection**: Robust face detection and alignment preprocessing
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1, confusion matrix, ROC/AUC
- **Mixed Precision Training**: Optional AMP support for faster training
- **Reproducibility**: Seeded operations and deterministic mode

## Paper Reference

```
El Morabit, S., & Rivenq, A. (2022). 
Pain Detection From Facial Expressions Based on Transformers and Distillation. 
2022 11th International Symposium on Signal, Image, Video and Communications (ISIVC).
```

### Key Hyperparameters (from paper)

- **Input size**: 256×256
- **Patch size**: 32 (for Vision Transformer)
- **Epochs**: 30
- **Batch size**: 64
- **Learning rate**: 1e-5
- **Optimizer**: Adam
- **Loss**: Binary Cross Entropy (BCE) for both class and distillation tokens

## Dataset: SynPain

The SynPain dataset contains synthetic facial images in **side-by-side format**:
- Each image shows left and right facial portraits
- Images are generated using AI tools (Ideogram + Runway)
- Covers diverse demographics (age, gender, ethnicity)

### Labeling Rule (CRITICAL)

The labeling logic for side-by-side images:

```
If filename contains "Pain":
    Left half  -> NoPain
    Right half -> Pain

If filename contains "NoPain":
    Left half  -> NoPain
    Right half -> NoPain
```

This rule is implemented in `prepare_dataset.py`.

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone or download this repository
cd pain_detection_project

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0 (for DeiT)
- facenet-pytorch >= 2.5.0 (for MTCNN)
- opencv-python >= 4.8.0
- pandas, numpy, scikit-learn, matplotlib, seaborn

## Usage

### 1. Prepare Dataset

Split side-by-side images and apply labeling rule:

```bash
python prepare_dataset.py \
    --raw /path/to/SynPain/raw/images \
    --out ./SynPainProcessed
```

**Output**: Creates `SynPainProcessed/` with:
- `Pain/` - Images labeled as pain
- `NoPain/` - Images labeled as no pain  
- `manifest.csv` - Metadata for all processed images

**Optional**: Run sanity check on labeling logic:

```bash
python prepare_dataset.py --sanity-check
```

### 2. Preprocess Faces

Detect, align, and crop faces using MTCNN:

```bash
python preprocess_faces.py \
    --in ./SynPainProcessed \
    --out ./SynPainProcessed/Cropped
```

**Features**:
- MTCNN face detection with landmark alignment
- Fallback to center-crop if detection fails
- Resizes all faces to 256×256
- Logs detection failures to `mtcnn_failures.txt`

### 3. Train Model

Use the unified training entrypoint which supports multiple teacher architectures (ResNet50 and Swin):

```bash
# Train with configurable teacher (resnet50 or swin)
python CoreScripts/teacher_swin/train_multi_teacher.py \
    --teacher resnet50 \
    --data ./SynPainProcessed/Cropped \
    --output ./outputs \
    --epochs 30 \
    --batch 64 \
    --lr 1e-5 \
    --use-amp
```

If you only want to fine-tune a teacher model (before using it for distillation), use:

```bash
python CoreScripts/finetune_teacher.py --teacher swin --data ./SynPainProcessed/Cropped
```

**Key Arguments**:
- `--teacher`: `resnet50` or `swin` (for `train_multi_teacher.py`)
- `--data`: Path to preprocessed dataset
- `--output`: Directory for checkpoints and logs
- `--epochs`: Number of training epochs (default: 30)
- `--batch`: Batch size (default: 64)
- `--lr`: Learning rate (default: 1e-5)
- `--use-amp`: Enable mixed precision training (faster)
- `--seed`: Random seed for reproducibility (default: 42)

**Outputs**:
- `best_model.pth` - Best model by validation accuracy
- `final_model.pth` - Model after final epoch
- `training_history.json` - Loss and accuracy per epoch
- `training_curves.png` - Visualization of training progress

**Training Time**: ~2-4 hours on NVIDIA RTX 3090 (depends on dataset size)

### 4. Evaluate Model

Comprehensive evaluation on test set:

```bash
python evaluate.py \
    --checkpoint ./outputs/best_model.pth \
    --data ./SynPainProcessed/Cropped \
    --output ./evaluation
```

**Outputs**:
- `test_metrics.json` - All metrics in JSON format
- `confusion_matrix.png` - Confusion matrix heatmap
- `confusion_matrix_normalized.png` - Normalized confusion matrix
- `roc_curve.png` - ROC curve with AUC score
- `sample_predictions.png` - Visual sample of predictions

**Metrics Computed**:
- Overall accuracy
- Precision, recall, F1-score (overall and per-class)
- Confusion matrix
- ROC AUC score

### 5. Run Inference

#### Single Image

```bash
python inference.py \
    --checkpoint ./outputs/best_model.pth \
    --image /path/to/test_image.jpg
```

**Handles side-by-side images automatically**: If detected, splits and predicts both halves.

#### Batch Inference

```bash
python inference.py \
    --checkpoint ./outputs/best_model.pth \
    --folder /path/to/images/ \
    --output predictions.csv
```

**Output CSV** contains:
- `filename`: Image filename
- `predicted_class`: NoPain or Pain
- `confidence`: Prediction confidence (0-1)
- `prob_nopain`, `prob_pain`: Class probabilities

## Project Structure

```
pain_detection_project/
│
├── prepare_dataset.py          # Split side-by-side images, apply labeling
├── preprocess_faces.py          # MTCNN face detection and preprocessing
├── dataset.py                   # PyTorch Dataset and DataLoader creation
├── model_deit.py                # DeiT student model definition
├── model_resnet_teacher.py      # ResNet50 teacher model
├── distillation_loss.py         # Hard distillation loss implementation
├── train.py                     # Training script with distillation
├── evaluate.py                  # Evaluation on test set
├── inference.py                 # Single and batch inference
│
├── utils/
│   ├── __init__.py
│   ├── transforms.py            # Data augmentation transforms
│   ├── metrics.py               # Evaluation metrics
│   └── visualization.py         # Plotting and visualization
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Implementation Details

### Data Preprocessing

1. **Face Detection**: MTCNN detects face and 5 facial landmarks
2. **Alignment**: Faces aligned based on eye positions
3. **Cropping**: Face region extracted with margin
4. **Resizing**: All faces resized to 256×256
5. **Fallback**: Center-crop used if MTCNN fails

### Data Augmentation (Training Only)

- Random resized crop (scale 0.8-1.0)
- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization with ImageNet statistics

### Model Architecture

**Student (DeiT)**:
- Base: `deit_base_distilled_patch16_224` from timm
- Modified for 256×256 input
- Binary classification head (2 classes)
- Distillation token for learning from teacher

**Teacher (ResNet50)**:
- Pretrained on ImageNet
- Frozen during training (no gradient updates)
- Provides hard labels (argmax) for distillation

### Loss Function

Hard distillation loss combines two components:

```
L_total = L_BCE + L_teacher

L_BCE     = BCE(student_class_token, ground_truth)
L_teacher = BCE(student_distill_token, teacher_hard_label)
```

Both use Binary Cross Entropy (BCE).

### Training Strategy

- **70/15/15** stratified split (train/val/test)
- Adam optimizer with learning rate 1e-5
- Learning rate decay (StepLR: 0.5 every 10 epochs)
- Mixed precision training (optional, via AMP)
- Best model saved by validation accuracy

## Expected Results

Based on the paper's results on UNBC-McMaster and BioVid datasets:

| Dataset | DeiT-PNP | GoogleNet-PNP |
|---------|----------|---------------|
| UNBC-McMaster | 84.15% | 80.01% |
| BioVid Heat Pain | 72.11% | 65.75% |

Your results on SynPain may vary depending on:
- Dataset size and quality
- Distribution of Pain vs NoPain images
- Quality of synthetic face generation
- MTCNN detection success rate

## Reproducibility

For reproducible results use the unified training entrypoint (select `--teacher` as needed):

```bash
python CoreScripts/teacher_swin/train_multi_teacher.py \
    --teacher resnet50 \
    --data ./SynPainProcessed/Cropped \
    --seed 42 \
    --epochs 30 \
    --batch 64 \
    --lr 1e-5
```

**Note**: Deterministic mode is enabled by default, which may slightly reduce training speed.

## GPU Requirements

- **Minimum**: 8GB VRAM (batch size 32)
- **Recommended**: 16GB+ VRAM (batch size 64)
- **Training time**: 2-4 hours on RTX 3090 (varies with dataset size)

For limited VRAM, reduce batch size when running the training entrypoint:

```bash
python CoreScripts/teacher_swin/train_multi_teacher.py --batch 32  # or even 16
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python train.py --batch 32

# Or use CPU (very slow)
python train.py --device cpu
```

### MTCNN Detection Failures

If many faces fail detection (check `mtcnn_failures.txt`):
- Images may be low quality
- Faces may be too small or partially occluded
- Fallback center-crop is used automatically

### Low Accuracy

- Check class balance in dataset (use `python dataset.py --data ./SynPainProcessed/Cropped`)
- Ensure labeling rule was applied correctly
- Try training for more epochs
- Check if teacher model loaded correctly

## Example Workflow

Complete end-to-end example:

```bash
# 1. Prepare dataset
python prepare_dataset.py --raw /data/SynPain --out ./SynPainProcessed

# 2. Preprocess faces
python preprocess_faces.py --in ./SynPainProcessed --out ./SynPainProcessed/Cropped

# 3. Check dataset (optional)
python dataset.py --data ./SynPainProcessed/Cropped

# 4. Train model
python CoreScripts/teacher_swin/train_multi_teacher.py \
    --teacher resnet50 \
    --data ./SynPainProcessed/Cropped \
    --output ./outputs \
    --epochs 30 \
    --batch 64 \
    --use-amp

# 5. Evaluate on test set
python evaluate.py \
    --checkpoint ./outputs/best_model.pth \
    --data ./SynPainProcessed/Cropped \
    --output ./evaluation

# 6. Run inference on new images
python inference.py \
    --checkpoint ./outputs/best_model.pth \
    --image test_image.jpg
```

## Posters & Artifacts

This repository includes a `Posters/` folder containing presentation posters you can publish alongside the code. Current files (examples):

- `Posters/Pain Detection via Knowledge Distillation - Final Poster.pdf`
- `Posters/Pain-Detection-via-Knowledge-Distillation-PosterA1.pdf`

Recommendations for what to push to GitHub:
- Keep `Posters/` in the repo — it's small and provides evidence of the project deliverable.
- Avoid committing large datasets (e.g. the full SynPain RAW images). Instead include:
    - A small sample subset (e.g. 50–200 example images) in `examples/` or `data_samples/` with a clear README describing how it was sampled.
    - Processed manifest and a small processed sample (from `SynPainProcessed/`) if you want reproducible quick demos.
    - Evaluation artifacts (`evaluation/` JSON metrics, small `training_history.json`) — these are usually small and helpful for reviewers.

If you want to publish full trained models or large artifacts, consider using Git LFS or an external storage (Zenodo, Google Drive, or an institutional repository) and add URLs in this README.

## Citation

If you use this code or the SynPain dataset, please cite:

```bibtex
@inproceedings{elmorabit2022pain,
  title={Pain Detection From Facial Expressions Based on Transformers and Distillation},
  author={El Morabit, Safaa and Rivenq, Atika},
  booktitle={2022 11th International Symposium on Signal, Image, Video and Communications (ISIVC)},
  year={2022},
  organization={IEEE}
}
```

## License

This implementation is provided for research and educational purposes.

## Contact

For questions or issues, please check:
1. This README for troubleshooting
2. Code comments for implementation details
3. Original paper for methodology clarification
