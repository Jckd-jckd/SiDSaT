# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.preprocessing import StandardScaler
import os

HYPERPARAMS = {
    # Data configuration
    'data_paths': ['401_81_circle.mat', '401_81_rect.mat'],
    'wavelength_n': 81,
    
    # Batch size optimization - adjusted based on model complexity and GPU memory
    'batch_size_train': 64,  # Smaller batch size for better gradient update frequency and stability
    'batch_size_val': 128,   # Larger batch size for validation
    
    # Learning rate strategy optimization - OneCycleLR parameter adjustment
    'learning_rate': 9e-4,    # Lower initial learning rate for training stability
    'max_lr': 9e-3,          # Moderate maximum learning rate to avoid training instability
    'weight_decay': 5e-5,     # Enhanced regularization to prevent overfitting
    
    # Loss weight optimization - based on metasurface physical properties
    'phase_weight': 20.0,     # Phase loss weight
    
    # Training strategy optimization
    'epochs': 3000,           # Reduced total epochs with better learning rate strategy
    'early_stop_patience': 200, # Increased early stopping patience to avoid premature stopping
    
    # OneCycleLR scheduler parameter optimization
    'pct_start': 0.15,        # Reduced warmup phase proportion
    'div_factor': 10.0,       # Adjust initial learning rate divisor
    'final_div_factor': 5e3,  # Adjust final learning rate divisor
    
    # Gradient clipping and data augmentation
    'grad_clip_norm': 2,    # Stricter gradient clipping
    'noise_std': 0.005,       # Reduced data noise
    'phase_noise': 0.005      # Reduced phase noise
}

paths = HYPERPARAMS['data_paths']
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  

class EnhancedMetaSurfaceDataset(Dataset):
    def __init__(self, mat_paths, param_key='r', phase_key='phase_dispersion'):
        if isinstance(mat_paths, str):
            mat_paths = [mat_paths] 
            
        all_params = []
        all_phases = []
        all_shape_types = [] 
        
        for i, path in enumerate(mat_paths):
            data = loadmat(path)
            
            params = data[param_key].astype(np.float32)
            phases = data[phase_key].astype(np.float32)
            
            shape_type = 1 if 'rect' in path.lower() else 0
            shape_types = np.full((params.shape[0], 1), shape_type, dtype=np.float32)
            
            all_params.append(params)
            all_phases.append(phases)
            all_shape_types.append(shape_types)
        
        self.params = np.vstack(all_params)
        self.phase = np.vstack(all_phases)
        self.shape_types = np.vstack(all_shape_types)
        
        self.param_scaler = StandardScaler()
        self.params_normalized = self.param_scaler.fit_transform(self.params)

        self.phase_sin = np.sin(self.phase)
        self.phase_cos = np.cos(self.phase)

        # Data augmentation using optimized noise parameters
        noise_std = HYPERPARAMS['noise_std']
        self.params_normalized = self.params_normalized + np.random.normal(0, noise_std, self.params_normalized.shape)
        
        phase_noise = np.random.normal(0, HYPERPARAMS['phase_noise'], self.phase.shape)
        self.phase_sin = np.sin(self.phase + phase_noise)
        self.phase_cos = np.cos(self.phase + phase_noise)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return (
            torch.tensor(np.hstack([self.params_normalized[idx], self.shape_types[idx]]), dtype=torch.float32),
            torch.tensor(self.phase_sin[idx], dtype=torch.float32),
            torch.tensor(self.phase_cos[idx], dtype=torch.float32)
        )

# SiDSaT Architecture Implementation
class FCLModule(nn.Module):
    """Fully Connected Layer Module - Implements shape-specific parameter preprocessing and dimension expansion"""
    def __init__(self, input_dim=1, target_dim=512,dropout=0.15):
        super().__init__()
        # Shape-specific parameter preprocessing branches
        # Circle parameter preprocessing (shape_type = 0)
        self.circle_preprocessor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU()
        )
        
        # Rectangle parameter preprocessing (shape_type = 1)
        self.rect_preprocessor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU()
        )
        
        # Unified subsequent dimension expansion stage
        self.stage2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.stage3 = nn.Sequential(
            nn.Linear(256, target_dim),
            nn.ReLU(),
            nn.Linear(target_dim, target_dim),
            nn.LayerNorm(target_dim)
        )
        
    def forward(self, x, shape_type):
        # x: [batch_size, input_dim], shape_type: [batch_size, 1]
        batch_size = x.size(0)
        
        # Select different preprocessing branches based on shape type
        circle_mask = (shape_type < 0.5).squeeze(-1)  # [batch_size]
        rect_mask = ~circle_mask  # [batch_size]
        
        # Initialize output tensor
        x1 = torch.zeros(batch_size, 256, device=x.device, dtype=x.dtype)
        
        # Use circle preprocessor for circular samples
        if circle_mask.any():
            circle_indices = circle_mask.nonzero(as_tuple=True)[0]
            x1[circle_indices] = self.circle_preprocessor(x[circle_indices])
        
        # Use rectangle preprocessor for rectangular samples
        if rect_mask.any():
            rect_indices = rect_mask.nonzero(as_tuple=True)[0]
            x1[rect_indices] = self.rect_preprocessor(x[rect_indices])
        
        # Unified subsequent processing
        x2 = self.stage2(x1) + x1
        x3 = self.stage3(x2)
        return x3

class GlobalPerceptionTransformer(nn.Module):
    """Global Perception Transformer - Cross-fragment multi-head attention"""
    def __init__(self, d_model=512, num_heads=8, num_fragments=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_fragments = num_fragments
        self.fragment_dim = d_model // num_fragments
        self.head_dim = self.fragment_dim // num_heads
        
        # Projection matrices - dimension correction
        self.q_proj = nn.Linear(self.fragment_dim, self.fragment_dim)
        self.k_proj = nn.Linear(self.fragment_dim, self.fragment_dim)
        self.v_proj = nn.Linear(self.fragment_dim, self.fragment_dim)
        self.out_proj = nn.Linear(self.fragment_dim, self.fragment_dim)
        
        # Normalization and feedforward network
        self.norm1 = nn.LayerNorm(self.fragment_dim)
        self.norm2 = nn.LayerNorm(self.fragment_dim)
        self.nff = NFFModule(d_model)
        
    def forward(self, x):
        # x shape: [batch_size, d_model]
        batch_size = x.size(0)
        
        # Reshape to fragment format [batch_size, num_fragments, fragment_dim]
        x_reshaped = x.view(batch_size, self.num_fragments, self.fragment_dim)
        
        # Multi-head attention
        residual = x_reshaped
        x_norm = self.norm1(x_reshaped)
        
        # Generate Q, K, V
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        
        # Reshape to multi-head format
        q = q.view(batch_size, self.num_fragments, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, self.num_fragments, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, self.num_fragments, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Cross-fragment attention computation
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # Weighted average of value matrix V using attention weights
        
        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.num_fragments, self.fragment_dim)
        attn_output = self.out_proj(attn_output)
        x_reshaped = residual + attn_output
        
        # Flatten back to original dimensions
        x_flat = x_reshaped.view(batch_size, self.d_model)
        
        # Feedforward network
        residual = x_flat
        ffn_output = self.nff(x_flat)
        
        return residual + ffn_output

class NFFModule(nn.Module):
    """Normalization and Feedforward Network Module - SiDSaT shared component"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Normalization + feedforward network + residual connection
        normalized = self.norm(x)
        ffn_output = self.ffn(normalized)
        return x + ffn_output

class LocalPerceptionTransformer(nn.Module):
    """Local Perception Transformer - Intra-fragment self-attention"""
    def __init__(self, d_model=512, num_heads=8, fragment_size=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.fragment_size = fragment_size
        self.head_dim = fragment_size // num_heads
        
        # Projection matrices
        self.q_proj = nn.Linear(fragment_size, fragment_size)
        self.k_proj = nn.Linear(fragment_size, fragment_size)
        self.v_proj = nn.Linear(fragment_size, fragment_size)
        self.out_proj = nn.Linear(fragment_size, fragment_size)
        
        # NFF modules - using d_model dimensions
        self.nff1 = NFFModule(d_model)
        self.nff2 = NFFModule(d_model)
        
    def forward(self, x):
        # x shape: [batch_size, d_model]
        batch_size = x.size(0)
        num_fragments = self.d_model // self.fragment_size
        
        # Reshape to fragment format [batch_size, num_fragments, fragment_size]
        x_reshaped = x.view(batch_size, num_fragments, self.fragment_size)
        
        # Perform local self-attention on each fragment
        output_fragments = []
        for i in range(num_fragments):
            fragment = x_reshaped[:, i:i+1, :]  # [batch_size, 1, fragment_size]
            
            # Intra-fragment self-attention
            residual = fragment
            
            # Generate Q, K, V
            q = self.q_proj(fragment)
            k = self.k_proj(fragment)
            v = self.v_proj(fragment)
            
            # Local attention computation (using sigmoid instead of softmax)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = torch.sigmoid(attn_scores)
            
            # Hadamard product
            attn_output = attn_weights * v
            attn_output = self.out_proj(attn_output)
            fragment = residual + attn_output
            
            output_fragments.append(fragment)
        
        # Concatenate all fragments
        x_reshaped = torch.cat(output_fragments, dim=1)
        output_flat = x_reshaped.view(batch_size, -1)
        
        # Apply NFF modules
        output_flat = self.nff1(output_flat)
        output_flat = self.nff2(output_flat)
        
        return output_flat

# Shape-aware module to enhance feature extraction for different shapes
class ShapeAwareModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Increase intermediate layer dimensions to improve feature representation
        self.shape_embedding = nn.Sequential(
            nn.Linear(1, 32),  # Reduce initial dimension to avoid overfitting
            nn.GELU(),
            nn.Dropout(0.2),  # Add dropout to enhance generalization
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Linear(128, dim),
            nn.LayerNorm(dim)
        )
        
        # Gating mechanism for flexible feature fusion control
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, shape_type):
        # Extract shape features
        shape_features = self.shape_embedding(shape_type)
        # Calculate gating weights
        gate_value = self.gate(shape_features)
        # Use gating mechanism for feature modulation, allowing model to learn feature importance
        return x * gate_value + shape_features * (1 - gate_value)

class SiDSaTMetaSurfaceModel(nn.Module):
    """SiDSaT-based Metasurface Model"""
    def __init__(self, d_model=512, num_heads=8, num_fragments=8, fragment_size=64):
        super().__init__()
        self.d_model = d_model
        
        # FCL module - dimension expansion
        self.fcl_module = FCLModule(input_dim=1, target_dim=d_model)
        
        # Shape-aware module
        self.shape_aware = ShapeAwareModule(d_model)
        
        # Global perception transformer
        self.global_transformer = GlobalPerceptionTransformer(
            d_model=d_model, 
            num_heads=num_heads, 
            num_fragments=num_fragments
        )
        
        # Local perception transformer
        self.local_transformer = LocalPerceptionTransformer(
            d_model=d_model, 
            num_heads=num_heads, 
            fragment_size=fragment_size
        )
        
        # Phase prediction head
        self.phase_feature = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        self.phase_cos_head = nn.Sequential(
            nn.Linear(d_model // 2, HYPERPARAMS['wavelength_n'])
        )

        self.phase_sin_head = nn.Sequential(
            nn.Linear(d_model // 2, HYPERPARAMS['wavelength_n'])
        )
        
        # 1D convolution smoothing layer
        self.spectral_smoothing = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # Separate geometric parameters and shape type
        params = x[:, 0].unsqueeze(1)  # Geometric parameters [batch_size, 1]
        shape_type = x[:, 1].unsqueeze(1)  # Shape type [batch_size, 1]
        
        # FCL module for dimension expansion
        features = self.fcl_module(params, shape_type)  # [batch_size, d_model]
        
        # Apply shape-aware modulation
        features = self.shape_aware(features, shape_type)
        
        # Global perception transformer - capture cross-fragment associations
        global_features = self.global_transformer(features)
        
        # Local perception transformer - capture intra-fragment details
        local_features = self.local_transformer(global_features)
        
        # Directly use local transformer output, skip feature fusion layer
        fused_features = local_features
        
        # Phase prediction
        phase_features = self.phase_feature(fused_features)
        
        phase_cos_raw = self.phase_cos_head(phase_features)  # [batch_size, wavelength_n * 2]
        phase_sin_raw = self.phase_sin_head(phase_features)  # [batch_size, wavelength_n * 2]
        
        # Apply spectral smoothing

        phase_sin_smoothed = self.spectral_smoothing(phase_sin_raw.unsqueeze(1)).squeeze(1)
        phase_cos_smoothed = self.spectral_smoothing(phase_cos_raw.unsqueeze(1)).squeeze(1)
        phase_sin = 0.5 * phase_sin_raw + 0.5 * phase_sin_smoothed
        phase_cos = 0.5 * phase_cos_raw + 0.5 * phase_cos_smoothed

        # Normalize phase components
        phase_norm = torch.sqrt(phase_sin**2 + phase_cos**2 + 1e-10)
        phase_sin = phase_sin / phase_norm
        phase_cos = phase_cos / phase_norm
        
        return phase_sin, phase_cos


def create_stratified_sampler(dataset, train_indices):
    """Create stratified sampler to balance different shape types"""
    # Get shape type for each sample in training set
    shape_types = []
    for idx in train_indices:
        params_with_shape, _, _ = dataset[idx]
        shape_type = int(round(params_with_shape[-1].item()))
        shape_types.append(shape_type)
    
    shape_types = np.array(shape_types)
    
    # Calculate sample count for each category
    circle_count = np.sum(shape_types == 0)
    rect_count = np.sum(shape_types == 1)
    
    print(f"Circle samples in training set: {circle_count}, Rectangle samples: {rect_count}")
    
    # Calculate weights to balance categories
    total_samples = len(shape_types)
    circle_weight = total_samples / (2 * circle_count) if circle_count > 0 else 0
    rect_weight = total_samples / (2 * rect_count) if rect_count > 0 else 0
    
    # Assign weights to each sample
    sample_weights = np.zeros(len(shape_types))
    sample_weights[shape_types == 0] = circle_weight
    sample_weights[shape_types == 1] = rect_weight
    
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)





class ShapeAwarePhaseAwareLoss(nn.Module):
    def __init__(self, phase_weight=2.0):
        super().__init__()
        self.phase_criterion = nn.MSELoss(reduction='none')  
        self.phase_weight = phase_weight
        
        # Optimized shape-specific weight adjustment factors
        self.circle_phase_factor = 1.0
        self.rect_phase_factor = 1.3  # Moderately increase rectangular phase weight to avoid excessive bias

    def forward(self, phase_sin_pred, phase_cos_pred, phase_sin_true, phase_cos_true, shape_types):
        # Calculate basic loss
        phase_sin_loss_raw = self.phase_criterion(phase_sin_pred, phase_sin_true)
        phase_cos_loss_raw = self.phase_criterion(phase_cos_pred, phase_cos_true)
        
        # Adjust weights based on shape type
        batch_size = phase_sin_pred.size(0)
        weighted_phase_loss = 0
        
        for i in range(batch_size):
            shape_type = shape_types[i].item()
            
            # Select appropriate weight factor
            phase_factor = self.circle_phase_factor if shape_type < 0.5 else self.rect_phase_factor
            
            # Accumulate weighted loss
            weighted_phase_loss += (phase_sin_loss_raw[i].mean() + phase_cos_loss_raw[i].mean()) * phase_factor
        
        # Calculate average loss
        weighted_phase_loss /= batch_size
        
        # Return total loss
        return self.phase_weight * weighted_phase_loss

def phase_reconstruct(sin_pred, cos_pred):
    return torch.atan2(sin_pred, cos_pred)

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EnhancedMetaSurfaceDataset(paths)

    # Get shape types and geometric parameters of all samples for stratified sampling
    shape_types = []
    geom_params = []
    for i in range(len(dataset)):
        params_with_shape, _, _ = dataset[i]
        shape_type = int(round(params_with_shape[-1].item()))
        geom_param = params_with_shape[0].item()  # Geometric parameter r
        shape_types.append(shape_type)
        geom_params.append(geom_param)

    shape_types = np.array(shape_types)
    geom_params = np.array(geom_params)

    # Create composite stratification labels: shape type + geometric parameter binning
    # Divide geometric parameters into 5 bins, stratify within each shape type
    kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    geom_bins = kbins.fit_transform(geom_params.reshape(-1, 1)).flatten().astype(int)

    # Create composite labels: shape_type*10 + geometric_parameter_bin
    stratify_labels = shape_types * 10 + geom_bins

    print(f"Data stratification information:")
    for shape in [0, 1]:
        shape_name = "Circle" if shape == 0 else "Rectangle"
        shape_mask = shape_types == shape
        shape_geom_bins = geom_bins[shape_mask]
        print(f"{shape_name} samples: {np.sum(shape_mask)}")
        for bin_idx in range(5):
            bin_count = np.sum(shape_geom_bins == bin_idx)
            if bin_count > 0:
                bin_params = geom_params[shape_mask][shape_geom_bins == bin_idx]
                print(f"  Parameter bin {bin_idx}: {bin_count} samples, range: [{bin_params.min():.6f}, {bin_params.max():.6f}]")

    # Use composite labels for stratified sampling to split training and validation sets
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=0.2, 
        stratify=stratify_labels, 
        random_state=42
    )

    print(f"\nTraining samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
    print(f"Circle samples in training set: {np.sum(shape_types[train_indices] == 0)}, Rectangle samples: {np.sum(shape_types[train_indices] == 1)}")
    print(f"Circle samples in validation set: {np.sum(shape_types[val_indices] == 0)}, Rectangle samples: {np.sum(shape_types[val_indices] == 1)}")

    # Validate parameter distribution within each shape type
    for shape in [0, 1]:
        shape_name = "Circle" if shape == 0 else "Rectangle"
        train_shape_mask = shape_types[train_indices] == shape
        val_shape_mask = shape_types[val_indices] == shape
        
        if np.sum(train_shape_mask) > 0 and np.sum(val_shape_mask) > 0:
            train_shape_bins = geom_bins[train_indices][train_shape_mask]
            val_shape_bins = geom_bins[val_indices][val_shape_mask]
            
            print(f"\n{shape_name} sample parameter distribution validation:")
            for bin_idx in range(5):
                train_bin_count = np.sum(train_shape_bins == bin_idx)
                val_bin_count = np.sum(val_shape_bins == bin_idx)
                print(f"  Parameter bin {bin_idx}: Training {train_bin_count}, Validation {val_bin_count}")

    # Create subsets
    from torch.utils.data import Subset
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    # Create stratified sampler for balanced sampling within training set
    stratified_sampler = create_stratified_sampler(dataset, train_indices)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=HYPERPARAMS['batch_size_train'], 
                            sampler=stratified_sampler, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=HYPERPARAMS['batch_size_val'], shuffle=False)
    # Create SiDSaT model
    model = SiDSaTMetaSurfaceModel(
        d_model=768,
        num_heads=8, 
        num_fragments=12,
        fragment_size=64
    ).to(device)

    criterion = ShapeAwarePhaseAwareLoss(phase_weight=HYPERPARAMS['phase_weight'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERPARAMS['learning_rate'], weight_decay=HYPERPARAMS['weight_decay'])  

    # Calculate total training steps
    total_steps = len(train_loader) * HYPERPARAMS['epochs']

    # Use OneCycleLR scheduler
    scheduler = OneCycleLR(optimizer, 
                          max_lr=HYPERPARAMS['max_lr'],
                          total_steps=total_steps,
                          pct_start=HYPERPARAMS['pct_start'],
                          div_factor=HYPERPARAMS['div_factor'],
                          final_div_factor=HYPERPARAMS['final_div_factor'])  

    best_val_loss = float('inf')
    early_stop_patience = HYPERPARAMS['early_stop_patience']  
    patience_counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(HYPERPARAMS['epochs']):  
        model.train()
        train_loss = 0.0
        for params_with_shape, phase_sin, phase_cos in train_loader:
            params_with_shape = params_with_shape.to(device)
            phase_sin = phase_sin.to(device)
            phase_cos = phase_cos.to(device)
            
            # Extract shape types
            shape_types = params_with_shape[:, 1]
            
            optimizer.zero_grad()
            phase_sin_pred, phase_cos_pred = model(params_with_shape)
            
            # Use shape-aware loss function
            loss = criterion(phase_sin_pred, phase_cos_pred, phase_sin, phase_cos, shape_types)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), HYPERPARAMS['grad_clip_norm'])
            optimizer.step()
            scheduler.step()  # OneCycleLR needs to be called after each batch
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for params_with_shape, phase_sin, phase_cos in val_loader:
                params_with_shape = params_with_shape.to(device)
                phase_sin = phase_sin.to(device)
                phase_cos = phase_cos.to(device)

                # Extract shape types
                shape_types = params_with_shape[:, 1]

                phase_sin_pred, phase_cos_pred = model(params_with_shape)
                # Use shape-aware loss function
                val_loss += criterion(phase_sin_pred, phase_cos_pred, phase_sin, phase_cos, shape_types).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'enhanced_model332.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    model.load_state_dict(torch.load('enhanced_model332.pth'))
    model.eval()

    with torch.no_grad():
        test_indices = np.random.choice(len(val_set), 5, replace=False)

        plt.figure(figsize=(15, 20))

        
        # Get number of test samples for dynamic subplot configuration
        num_samples = len(test_indices)
        
        for i, idx in enumerate(test_indices):
            params_with_shape, phase_sin_true, phase_cos_true = val_set[idx]
            phase_sin_pred, phase_cos_pred = model(params_with_shape.unsqueeze(0).to(device))

            shape_type = int(round(params_with_shape[-1].item()))
            shape_name = "Rectangle" if shape_type == 1 else "Circle"

            phase_true = phase_reconstruct(phase_sin_true, phase_cos_true).numpy()
            phase_pred = phase_reconstruct(phase_sin_pred, phase_cos_pred).cpu().numpy().flatten()

            phase_mse = np.mean((phase_true - phase_pred) ** 2)
            phase_mae = np.mean(np.abs(phase_true - phase_pred))

            plt.subplot(num_samples, 1, i + 1)
            plt.plot(phase_true, 'b--', label='True')
            plt.plot(phase_pred, 'r-', label='Predicted')
            plt.title(f'Phase ({shape_name}, Sample {i + 1})')
            plt.legend()

        plt.tight_layout()
        plt.show()

    circle_phase_mse = []
    rect_phase_mse = []
    # Add MAE lists
    circle_phase_mae = []
    rect_phase_mae = []

    with torch.no_grad():
        for idx in range(len(val_set)):
            params_with_shape, phase_sin_true, phase_cos_true = val_set[idx]
            phase_sin_pred, phase_cos_pred = model(params_with_shape.unsqueeze(0).to(device))
            
            shape_type = int(round(params_with_shape[-1].item()))
            
            phase_true = phase_reconstruct(phase_sin_true, phase_cos_true).numpy()
            phase_pred = phase_reconstruct(phase_sin_pred, phase_cos_pred).cpu().numpy().flatten()
            
            phase_mse = np.mean((phase_true - phase_pred) ** 2)
            # Calculate MAE
            phase_mae = np.mean(np.abs(phase_true - phase_pred))
            
            if shape_type == 0:
                circle_phase_mse.append(phase_mse)
                circle_phase_mae.append(phase_mae)
            else:
                rect_phase_mse.append(phase_mse)
                rect_phase_mae.append(phase_mae)

    if circle_phase_mae:
        print(f"Circle metasurface unit average phase MSE: {np.mean(circle_phase_mse):.4f}")
        print(f"Circle metasurface unit average phase MAE: {np.mean(circle_phase_mae):.4f}")
    if rect_phase_mae:
        print(f"Rectangle metasurface unit average phase MSE: {np.mean(rect_phase_mse):.4f}")
        print(f"Rectangle metasurface unit average phase MAE: {np.mean(rect_phase_mae):.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()
    # Calculate average MAE values
    avg_circle_phase_mae = np.mean(circle_phase_mae) if circle_phase_mae else 0
    avg_rect_phase_mae = np.mean(rect_phase_mae) if rect_phase_mae else 0

    # Build filename containing wavelength_n and mae information
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"enhanced_model_wl{HYPERPARAMS['wavelength_n']}_800-1600_phase{avg_circle_phase_mae:.4f}_{avg_rect_phase_mae:.4f}_{current_time}.pth"
    save_path = r'net\{}'.format(file_name)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'param_scaler': dataset.param_scaler,
        'train_losses': train_losses,  
        'val_losses': val_losses,      
        'hyperparams': HYPERPARAMS,    
    }, save_path)

    print(f"Model training and evaluation completed. Final model saved as: {save_path}")