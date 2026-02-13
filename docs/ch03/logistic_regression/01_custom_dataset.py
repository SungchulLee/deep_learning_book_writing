"""
================================================================================
01_custom_dataset.py - Creating Custom Dataset Classes
================================================================================

LEARNING OBJECTIVES:
- Create custom Dataset classes for any data format
- Implement __len__ and __getitem__ methods
- Handle various data types (CSV, images, text)
- Apply transformations during loading
- Best practices for data loading

TIME TO COMPLETE: ~2 hours
DIFFICULTY: ⭐⭐⭐⭐☆ (Advanced)
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Callable

print("="*80)
print("CUSTOM DATASET CLASSES")
print("="*80)

# =============================================================================
# PART 1: WHY CUSTOM DATASETS?
# =============================================================================

print("\n" + "="*80)
print("PART 1: WHEN TO USE CUSTOM DATASETS")
print("="*80)

print("""
Built-in TensorDataset works for simple cases, but custom Dataset classes are
needed for:

✓ Loading data from files (don't fit in memory)
✓ Complex data types (images, text, audio)
✓ On-the-fly preprocessing/augmentation
✓ Multiple data sources
✓ Special sampling strategies
✓ Memory-efficient loading

Custom Dataset Template:
------------------------
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # Load file paths, metadata, etc.
        pass
    
    def __len__(self):
        # Return total number of samples
        return num_samples
    
    def __getitem__(self, idx):
        # Load and return one sample
        # Can apply transformations here
        return sample, label
""")

# =============================================================================
# PART 2: EXAMPLE 1 - CSV DATASET
# =============================================================================

print("\n" + "="*80)
print("PART 2: CUSTOM DATASET FOR CSV FILES")
print("="*80)

class CSVDataset(Dataset):
    """
    Custom Dataset for loading data from CSV files
    
    Args:
        csv_file (str): Path to CSV file
        feature_cols (list): List of feature column names
        target_col (str): Name of target column
        transform (callable, optional): Transformation to apply
    """
    
    def __init__(self, 
                 csv_file: str,
                 feature_cols: list,
                 target_col: str,
                 transform: Optional[Callable] = None):
        
        # Load CSV file
        self.data = pd.read_csv(csv_file)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.transform = transform
        
        # Extract features and targets
        self.X = self.data[feature_cols].values.astype(np.float32)
        self.y = self.data[target_col].values.astype(np.float32)
        
        print(f"Loaded CSV with {len(self)} samples")
        print(f"Features: {len(feature_cols)}")
    
    def __len__(self):
        """Return the total number of samples"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get one sample
        
        Args:
            idx (int): Index of sample to retrieve
            
        Returns:
            tuple: (features, target)
        """
        # Get features and target
        features = torch.FloatTensor(self.X[idx])
        target = torch.FloatTensor([self.y[idx]])
        
        # Apply transformation if provided
        if self.transform:
            features = self.transform(features)
        
        return features, target


# Create sample CSV data
print("\nCreating sample CSV data...")
sample_data = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'feature3': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})

csv_path = "/home/claude/pytorch_logistic_regression_tutorial/03_advanced/sample_data.csv"
sample_data.to_csv(csv_path, index=False)

# Use custom CSV dataset
feature_cols = ['feature1', 'feature2', 'feature3']
target_col = 'target'

csv_dataset = CSVDataset(csv_path, feature_cols, target_col)

# Create DataLoader
csv_loader = DataLoader(csv_dataset, batch_size=32, shuffle=True)

print(f"\nDataset length: {len(csv_dataset)}")
print(f"Number of batches: {len(csv_loader)}")

# Test loading one batch
for batch_X, batch_y in csv_loader:
    print(f"\nFirst batch:")
    print(f"  Features shape: {batch_X.shape}")
    print(f"  Targets shape: {batch_y.shape}")
    break

# =============================================================================
# PART 3: EXAMPLE 2 - MEMORY-EFFICIENT DATASET
# =============================================================================

print("\n" + "="*80)
print("PART 3: MEMORY-EFFICIENT DATASET")
print("="*80)

class MemoryEfficientDataset(Dataset):
    """
    Dataset that loads data only when needed (lazy loading)
    Useful for large datasets that don't fit in RAM
    """
    
    def __init__(self, data_dir: Path, file_extension: str = '.npy'):
        self.data_dir = Path(data_dir)
        self.file_extension = file_extension
        
        # Only store file paths, not actual data
        self.file_paths = sorted(list(self.data_dir.glob(f'*{file_extension}')))
        
        print(f"Found {len(self.file_paths)} files")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Load data from disk on-demand"""
        # Load file only when requested
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        
        # Assume last column is target
        features = torch.FloatTensor(data[:-1])
        target = torch.FloatTensor([data[-1]])
        
        return features, target


# =============================================================================
# PART 4: EXAMPLE 3 - DATASET WITH AUGMENTATION
# =============================================================================

print("\n" + "="*80)
print("PART 4: DATASET WITH ON-THE-FLY AUGMENTATION")
print("="*80)

class AugmentedDataset(Dataset):
    """
    Dataset with data augmentation applied during loading
    """
    
    def __init__(self, X, y, augment: bool = True, noise_std: float = 0.1):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        self.augment = augment
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        features = self.X[idx].clone()
        target = self.y[idx]
        
        # Apply augmentation during training
        if self.augment:
            # Add Gaussian noise
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise
            
            # Could add other augmentations:
            # - Random scaling
            # - Random feature dropout
            # - etc.
        
        return features, target


# Demonstrate augmentation
print("\nDemonstrating augmentation...")
X_sample = np.random.randn(100, 5)
y_sample = np.random.randint(0, 2, 100)

aug_dataset = AugmentedDataset(X_sample, y_sample, augment=True, noise_std=0.1)
no_aug_dataset = AugmentedDataset(X_sample, y_sample, augment=False)

# Get same sample with and without augmentation
orig_features, _ = no_aug_dataset[0]
aug_features, _ = aug_dataset[0]

print(f"Original sample: {orig_features[:3]}")
print(f"Augmented sample: {aug_features[:3]}")
print(f"Difference: {(aug_features - orig_features)[:3]}")

# =============================================================================
# PART 5: TRAINING WITH CUSTOM DATASET
# =============================================================================

print("\n" + "="*80)
print("PART 5: TRAINING WITH CUSTOM DATASET")
print("="*80)

# Create model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Use CSV dataset for training
model = LogisticRegression(3)  # 3 features
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 20
print(f"Training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in csv_loader:
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * len(batch_X)
        predicted_classes = (predictions >= 0.5).float()
        correct += (predicted_classes == batch_y).sum().item()
        total += len(batch_X)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"Loss: {avg_loss:.4f} "
              f"Accuracy: {accuracy:.4f}")

print("\n✓ Training completed with custom dataset!")

# =============================================================================
# KEY TAKEAWAYS
# =============================================================================

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. CUSTOM DATASET STRUCTURE
   ✓ Must inherit from torch.utils.data.Dataset
   ✓ Must implement __len__ and __getitem__
   ✓ Can add any initialization logic

2. WHEN TO USE
   ✓ Data doesn't fit in memory
   ✓ Complex data types (images, text, audio)
   ✓ On-the-fly preprocessing
   ✓ Data augmentation during training
   ✓ Multiple data sources

3. BEST PRACTICES
   ✓ Load data lazily (in __getitem__)
   ✓ Store only file paths in memory
   ✓ Cache preprocessed data if possible
   ✓ Use proper indexing
   ✓ Return consistent tensor types

4. COMMON PATTERNS
   ✓ CSV/Excel: Load file paths, read in __getitem__
   ✓ Images: Store image paths, load with PIL/OpenCV
   ✓ Text: Store file paths, tokenize on-the-fly
   ✓ Large datasets: Memory-mapped arrays

5. AUGMENTATION
   ✓ Apply in __getitem__ during training
   ✓ Use flags to enable/disable
   ✓ Random transformations
   ✓ Increases dataset size effectively
""")

print("\n" + "="*80)
print("EXERCISES")
print("="*80)
print("""
1. EASY: Add normalization to CSVDataset
   Apply StandardScaler in __init__

2. MEDIUM: Create ImageDataset class:
   - Load images from folder
   - Apply transforms (resize, normalize)
   - Handle RGB/grayscale

3. MEDIUM: Implement caching:
   - Cache loaded samples in memory
   - Clear cache when memory full

4. HARD: Create TextDataset:
   - Load text files
   - Tokenize on-the-fly
   - Create vocabulary
   - Return padded sequences

5. HARD: Implement weighted sampling:
   - Balance imbalanced classes
   - Use WeightedRandomSampler
   - Compare with simple oversampling
""")

print("\n" + "="*80)
print("NEXT: 02_multiclass_classification.py - Beyond binary classification")
print("="*80)
