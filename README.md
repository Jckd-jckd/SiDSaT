# SiDSaT: Shape-integrated Dual-Spectrum-aware Transformer for Metasurface Design

A deep learning framework for predicting phase responses of metasurface structures using a novel dual-scale attention mechanism.

## Overview

SiDSaT (Shape-integrated Dual-Spectrum-aware Transformer) is a specialized neural network architecture designed for metasurface phase prediction. The model combines global and local attention mechanisms with shape-aware processing to accurately predict phase responses across different wavelengths for both circular and rectangular metasurface unit cells.

## Key Features

- **Dual-Scale Attention**: Combines global perception transformers for cross-fragment associations and local perception transformers for intra-fragment details
- **Shape-Aware Processing**: Incorporates geometric shape information (circle/rectangle) into the learning process
- **Multi-Wavelength Support**: Predicts phase responses across 81 wavelength points (800-1600nm)
- **Advanced Data Augmentation**: Includes noise injection and spectral smoothing for robust training
- **Stratified Sampling**: Ensures balanced representation of different shape types and parameter ranges

## Architecture Components

### Core Modules

1. **FCLModule**: Feature Cascade Learning module for dimension expansion and multi-branch processing
2. **GlobalPerceptionTransformer**: Captures long-range dependencies across feature fragments
3. **LocalPerceptionTransformer**: Extracts fine-grained local patterns within fragments
4. **ShapeAwareModule**: Modulates features based on geometric shape information
5. **NFFModule**: Normalized Feed-Forward module with shared components

### Loss Function

- **ShapeAwarePhaseAwareLoss**: Custom loss function that applies different weights based on shape types
- Uses MSE loss with shape-specific weighting factors
- Separate handling for sine and cosine phase components

## Requirements

```
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

## Data Format

The model expects `.mat` files containing:
- `r`: Geometric parameters (radius for circles, side length for rectangles)
- `phase_dispersion`: Phase response data across wavelengths
- Shape type information (0 for circles, 1 for rectangles)

## Usage

### Basic Training

```python
import torch
from SiDSaT import SiDSaTMetaSurfaceModel, EnhancedMetaSurfaceDataset

# Initialize model
model = SiDSaTMetaSurfaceModel(
    d_model=768,
    num_heads=8,
    num_fragments=12,
    fragment_size=64
)

# Load dataset
dataset = EnhancedMetaSurfaceDataset(['circle_data.mat', 'rect_data.mat'])

# Training loop (see main section in SiDSaT.py for complete example)
```

### Model Parameters

- `d_model`: Model dimension (default: 768)
- `num_heads`: Number of attention heads (default: 8)
- `num_fragments`: Number of fragments for global attention (default: 12)
- `fragment_size`: Size of each fragment for local attention (default: 64)

### Hyperparameters

Key hyperparameters can be adjusted in the `HYPERPARAMS` dictionary:

```python
HYPERPARAMS = {
    'batch_size_train': 64,
    'batch_size_val': 128,
    'learning_rate': 9e-4,
    'max_lr': 9e-3,
    'weight_decay': 5e-5,
    'phase_weight': 20.0,
    'epochs': 3000,
    'early_stop_patience': 200
}
```

## Training Process

1. **Data Preparation**: Stratified sampling ensures balanced representation
2. **Model Training**: Uses OneCycleLR scheduler with gradient clipping
3. **Validation**: Early stopping based on validation loss
4. **Evaluation**: Separate MSE and MAE metrics for different shape types
5. **Visualization**: Automatic plotting of training curves and sample predictions

## Output

The model predicts phase responses as sine and cosine components:
- `phase_sin`: Sine component of phase response
- `phase_cos`: Cosine component of phase response
- Final phase: `torch.atan2(phase_sin, phase_cos)`

## Model Saving

Trained models are automatically saved with comprehensive information:
- Model state dictionary
- Optimizer state
- Parameter scaler
- Training/validation loss history
- Hyperparameters

## Performance Metrics

- **MSE (Mean Squared Error)**: Primary training loss
- **MAE (Mean Absolute Error)**: Evaluation metric
- Separate metrics for circular and rectangular structures

## File Structure

```
├── SiDSaT.py          # Main model implementation
├── README.md          # This file
├── *.mat             # Data files
└── net/              # Saved models directory
```
