# Transaction Image Classification Project

## Project Overview
This project implements deep learning models for binary classification of transaction images using ResNet50 and MobileNet architectures. The ResNet50 model achieves 87.49% accuracy, while the MobileNet implementation reaches 94.04% accuracy in classifying images into 'Yes' and 'No' categories after hyperparameter optimization.

## Features
- Dual architecture implementation (ResNet50 and MobileNet)
- Transfer learning with pretrained models
- Data augmentation for handling class imbalance
- Hyperparameter tuning using Keras Tuner
- Model conversion to ONNX format for deployment
- TPU/GPU support for training

## Model Architectures

### ResNet50 Implementation
- Base: ResNet50 (pretrained on ImageNet)
- Additional layers:
  - Global Average Pooling
  - Dense layers (128, 64 units) with ReLU activation
  - Batch Normalization
  - Dropout layers (0.5)
  - Final Dense layer with sigmoid activation

### MobileNet Implementation
- Base: MobileNet (pretrained on ImageNet)
- Additional layers:
  - Global Average Pooling
  - Dense layer with ReLU activation
  - Batch Normalization
  - Dropout layers for regularization
  - Final Dense layer with sigmoid activation

## Implementation Details

### Data Preprocessing
- Image size: 128x128 pixels
- Data augmentation techniques:
  - Rotation (up to 30 degrees)
  - Width/Height shifts (20%)
  - Shear transformation (20%)
  - Zoom range (20%)
  - Horizontal flips
  - Brightness adjustment (0.8-1.2 range)

### Training Configuration
- Batch size: 32
- Learning rate: Optimized through Keras Tuner
- Loss function: Binary Cross-entropy
- Optimizer: Adam
- Class weights: Balanced using sklearn's compute_class_weight
- Early stopping with patience of 3-10 epochs
- Learning rate reduction on plateau

### Hyperparameter Optimization
Used Keras Tuner to optimize:
- Dropout rates
- Learning rates
- Best hyperparameters selected based on validation accuracy

## Results
- ResNet50 Test Accuracy: 87.49%
- MobileNet Test Accuracy: 94.04%
- Model evaluation includes:
  - Classification reports
  - ROC-AUC scores
  - Training/validation accuracy and loss curves

## Dependencies
```
tensorflow
keras
keras-tuner
sklearn
numpy
pillow
tf2onnx
```
## Setup and Usage

1. Install dependencies:
```bash
pip install keras-tuner tensorflow sklearn pillow tf2onnx
```

2. Prepare dataset structure:
```
transaction-yes-no/
├── train/
│   ├── Yes/
│   └── No/
└── test/
    ├── Yes/
    └── No/
```

3. Train models:
```python
python train.py
```

4. Convert to ONNX:
```bash
python3 -m tf2onnx.convert --saved-model rn_model --output best_hprn_model.onnx
```

## Model Performance Optimization
- Implemented comprehensive data augmentation
- Applied dropout and batch normalization
- Used early stopping and learning rate reduction
- Utilized hyperparameter tuning
- Balanced dataset through augmentation and undersampling

## Inference Using ONNX Model
```python
import onnxruntime
import numpy as np
from PIL import Image

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Load ONNX model
session = onnxruntime.InferenceSession('best_hprn_model.onnx')

# Prepare input image
input_image = load_and_preprocess_image('path_to_image.jpg')

# Run inference
results = session.run(None, {'input': input_image})
```

## Limitations and Future Improvements
- Current ResNet50 accuracy (87.49%) is below target
- Limited hyperparameter tuning due to resource constraints
- Potential improvements:
  - Experiment with different architectures
