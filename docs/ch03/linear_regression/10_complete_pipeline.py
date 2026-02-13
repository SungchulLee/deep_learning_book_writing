"""
==============================================================================
10_complete_pipeline.py
==============================================================================
DIFFICULTY: ⭐⭐⭐⭐⭐ (Advanced)

DESCRIPTION:
    Complete production-ready training pipeline with all best practices.
    Includes train/val/test split, early stopping, model checkpointing,
    logging, and comprehensive evaluation.

TOPICS COVERED:
    - Complete training pipeline
    - Train/validation/test splits
    - Early stopping
    - Model checkpointing
    - Learning rate scheduling
    - Comprehensive evaluation
    - Reproducibility

PREREQUISITES:
    - All previous tutorials

TIME: ~40 minutes
==============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

print("=" * 70)
print("COMPLETE PRODUCTION-READY TRAINING PIPELINE")
print("=" * 70)

# ============================================================================
# PART 1: CONFIGURATION AND REPRODUCIBILITY
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: CONFIGURATION")
print("=" * 70)

class Config:
    """Configuration class for hyperparameters"""
    # Data
    test_size = 0.2
    val_size = 0.2  # From training set
    
    # Model
    hidden_sizes = []  # Empty for linear, or [64, 32] for MLP
    
    # Training
    batch_size = 128
    n_epochs = 200
    learning_rate = 0.01
    weight_decay = 0.001  # L2 regularization
    
    # Early stopping
    patience = 15
    min_delta = 1e-4
    
    # Learning rate scheduler
    use_scheduler = True
    scheduler_patience = 5
    scheduler_factor = 0.5
    
    # Paths
    checkpoint_dir = '/home/claude/pytorch_linear_regression_tutorial/checkpoints'
    log_dir = '/home/claude/pytorch_linear_regression_tutorial/logs'
    
    # Reproducibility
    random_seed = 42

config = Config()

# Set seeds for reproducibility
def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True  # Uncomment for full reproducibility
    # torch.backends.cudnn.benchmark = False

set_seed(config.random_seed)
print(f"Random seed set to: {config.random_seed}")
print(f"Configuration loaded")

# Create directories
os.makedirs(config.checkpoint_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)
print(f"Directories created")

# ============================================================================
# PART 2: DATA PREPARATION
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: DATA PREPARATION")
print("=" * 70)

# Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target

print(f"Dataset: California Housing")
print(f"  Total samples: {len(X)}")
print(f"  Features: {X.shape[1]}")

# Split into train/val/test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=config.test_size, random_state=config.random_seed
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=config.val_size, random_state=config.random_seed
)

print(f"\nData split:")
print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Create DataLoaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled),
    torch.FloatTensor(y_train_scaled).reshape(-1, 1)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val_scaled),
    torch.FloatTensor(y_val_scaled).reshape(-1, 1)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test_scaled),
    torch.FloatTensor(y_test_scaled).reshape(-1, 1)
)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

print(f"\nDataLoaders created with batch_size={config.batch_size}")

# ============================================================================
# PART 3: MODEL DEFINITION
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: MODEL DEFINITION")
print("=" * 70)

class RegressionModel(nn.Module):
    """Flexible regression model"""
    
    def __init__(self, n_features, hidden_sizes=[]):
        super(RegressionModel, self).__init__()
        
        layers = []
        in_features = n_features
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

model = RegressionModel(X_train.shape[1], config.hidden_sizes)
print(f"Model created:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# PART 4: TRAINING UTILITIES
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: TRAINING UTILITIES")
print("=" * 70)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop

class ModelCheckpoint:
    """Save best model"""
    
    def __init__(self, filepath, mode='min'):
        self.filepath = filepath
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(self, model, val_loss):
        if self.mode == 'min':
            is_better = val_loss < self.best_score
        else:
            is_better = val_loss > self.best_score
        
        if is_better:
            self.best_score = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, self.filepath)
            return True
        return False

class Logger:
    """Log training metrics"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        self.logs = []
    
    def log(self, epoch, train_loss, val_loss, lr):
        entry = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'learning_rate': float(lr)
        }
        self.logs.append(entry)
    
    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
checkpoint = ModelCheckpoint(
    os.path.join(config.checkpoint_dir, 'best_model.pth'),
    mode='min'
)
logger = Logger(config.log_dir)

print("Training utilities initialized:")
print(f"  Early stopping: patience={config.patience}")
print(f"  Model checkpointing: enabled")
print(f"  Logging: enabled")

# ============================================================================
# PART 5: TRAINING LOOP
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: TRAINING")
print("=" * 70)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

# Learning rate scheduler
scheduler = None
if config.use_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
        verbose=True
    )

print(f"Optimizer: Adam (lr={config.learning_rate}, weight_decay={config.weight_decay})")
if scheduler:
    print(f"Scheduler: ReduceLROnPlateau")

# Training history
history = {
    'train_loss': [],
    'val_loss': []
}

print(f"\nStarting training...")
print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'LR':<10} {'Best':<6}")
print("-" * 60)

for epoch in range(config.n_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # Store history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    
    # Learning rate scheduling
    if scheduler:
        scheduler.step(val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    # Checkpoint
    is_best = checkpoint(model, val_loss)
    
    # Logging
    logger.log(epoch + 1, train_loss, val_loss, current_lr)
    
    # Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<6} {train_loss:<12.6f} {val_loss:<12.6f} {current_lr:<10.2e} {'✓' if is_best else '':<6}")
    
    # Early stopping
    if early_stopping(val_loss):
        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
        break

logger.save()
print(f"\nTraining completed!")
print(f"  Logs saved to: {logger.log_file}")

# Load best model
best_checkpoint = torch.load(os.path.join(config.checkpoint_dir, 'best_model.pth'))
model.load_state_dict(best_checkpoint['model_state_dict'])
print(f"  Best model loaded (val_loss={best_checkpoint['val_loss']:.6f})")

# ============================================================================
# PART 6: EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: COMPREHENSIVE EVALUATION")
print("=" * 70)

def evaluate_model(model, loader, scaler_y, dataset_name=""):
    """Comprehensive model evaluation"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            y_pred = model(batch_X)
            predictions.append(y_pred)
            targets.append(batch_y)
    
    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    
    # Inverse transform to original scale
    predictions_orig = scaler_y.inverse_transform(predictions)
    targets_orig = scaler_y.inverse_transform(targets)
    
    # Calculate metrics
    mse = mean_squared_error(targets_orig, predictions_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_orig, predictions_orig)
    r2 = r2_score(targets_orig, predictions_orig)
    
    print(f"\n{dataset_name} Set Metrics:")
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MSE:       {mse:.4f}")
    print(f"  RMSE:      {rmse:.4f}")
    print(f"  MAE:       {mae:.4f} (${"%.2f" % (mae*100)}k)")
    
    return predictions_orig, targets_orig, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

# Evaluate on all sets
train_pred, train_true, train_metrics = evaluate_model(model, train_loader, scaler_y, "Train")
val_pred, val_true, val_metrics = evaluate_model(model, val_loader, scaler_y, "Validation")
test_pred, test_true, test_metrics = evaluate_model(model, test_loader, scaler_y, "Test")

# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("PART 7: VISUALIZATION")
print("=" * 70)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Training history
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training History')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# 2. Predictions vs Actual (Test)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(test_true, test_pred, alpha=0.5, s=20)
ax2.plot([test_true.min(), test_true.max()], [test_true.min(), test_true.max()], 
         'r--', lw=2, label='Perfect prediction')
ax2.set_xlabel('Actual Price ($100k)')
ax2.set_ylabel('Predicted Price ($100k)')
ax2.set_title(f'Test Set: Predictions vs Actual (R²={test_metrics["r2"]:.4f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Residual plot
ax3 = fig.add_subplot(gs[0, 2])
residuals = test_true - test_pred
ax3.scatter(test_pred, residuals, alpha=0.5, s=20)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Predicted Price ($100k)')
ax3.set_ylabel('Residuals')
ax3.set_title('Residual Plot (Test Set)')
ax3.grid(True, alpha=0.3)

# 4. Error distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Residual')
ax4.set_ylabel('Frequency')
ax4.set_title('Residual Distribution')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Performance comparison
ax5 = fig.add_subplot(gs[1, 1])
datasets = ['Train', 'Val', 'Test']
r2_scores = [train_metrics['r2'], val_metrics['r2'], test_metrics['r2']]
colors = ['green', 'orange', 'blue']
bars = ax5.bar(datasets, r2_scores, color=colors, alpha=0.7)
ax5.set_ylabel('R² Score')
ax5.set_title('Model Performance Across Datasets')
ax5.set_ylim([0, 1])
ax5.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{score:.4f}', ha='center', va='bottom')

# 6. MAE comparison
ax6 = fig.add_subplot(gs[1, 2])
mae_scores = [train_metrics['mae']*100, val_metrics['mae']*100, test_metrics['mae']*100]
bars = ax6.bar(datasets, mae_scores, color=colors, alpha=0.7)
ax6.set_ylabel('MAE ($1000s)')
ax6.set_title('Mean Absolute Error')
ax6.grid(True, alpha=0.3, axis='y')

# 7. Training summary
ax7 = fig.add_subplot(gs[2, :])
summary = f"""
TRAINING SUMMARY

Configuration:
  - Model: {'Linear' if not config.hidden_sizes else f'MLP {config.hidden_sizes}'}
  - Optimizer: Adam (lr={config.learning_rate}, weight_decay={config.weight_decay})
  - Batch size: {config.batch_size}
  - Epochs: {len(history['train_loss'])}
  - Early stopping patience: {config.patience}

Final Metrics:
  Train - R²: {train_metrics['r2']:.4f}, MAE: ${train_metrics['mae']*100:.2f}k, RMSE: ${train_metrics['rmse']*100:.2f}k
  Val   - R²: {val_metrics['r2']:.4f}, MAE: ${val_metrics['mae']*100:.2f}k, RMSE: ${val_metrics['rmse']*100:.2f}k
  Test  - R²: {test_metrics['r2']:.4f}, MAE: ${test_metrics['mae']*100:.2f}k, RMSE: ${test_metrics['rmse']*100:.2f}k

Observations:
  - {"No significant overfitting" if abs(train_metrics['r2'] - test_metrics['r2']) < 0.05 else "Some overfitting detected"}
  - Best validation loss: {best_checkpoint['val_loss']:.6f}
  - Model saved to: {config.checkpoint_dir}/best_model.pth
"""
ax7.text(0.1, 0.9, summary, transform=ax7.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax7.axis('off')

plt.savefig('/home/claude/pytorch_linear_regression_tutorial/10_complete_pipeline_results.png', dpi=100, bbox_inches='tight')
print("Visualization saved")
plt.show()

print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)
print("""
CONGRATULATIONS! You've completed a production-ready ML pipeline!

This tutorial demonstrated:
✓ Configuration management
✓ Reproducibility (random seeds)
✓ Proper train/val/test split
✓ Feature scaling
✓ DataLoader for efficient batching
✓ Early stopping
✓ Model checkpointing
✓ Learning rate scheduling
✓ Comprehensive logging
✓ Multiple evaluation metrics
✓ Professional visualizations

Next Steps:
1. Try with different models (add hidden layers)
2. Experiment with hyperparameters
3. Apply to your own datasets
4. Add GPU support (.to('cuda'))
5. Implement cross-validation
6. Add data augmentation (for image tasks)
7. Deploy the model

You now have a solid foundation for PyTorch ML projects!
""")
