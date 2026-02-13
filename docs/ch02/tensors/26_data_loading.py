"""Tutorial 26: Data Loading - Efficient data pipelines with DataLoader"""
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

class CustomDataset(Dataset):
    """Example custom dataset."""
    def __init__(self, size=100):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def main():
    header("1. Basic Dataset and DataLoader")
    dataset = CustomDataset(size=100)
    print(f"Dataset size: {len(dataset)}")
    print(f"First sample: {dataset[0]}")
    
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    print(f"\nDataLoader created with batch_size=10")
    print(f"Number of batches: {len(dataloader)}")
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: data shape={data.shape}, labels shape={labels.shape}")
        if batch_idx == 2:
            break
    
    header("2. DataLoader Parameters")
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,      # Shuffle data each epoch
        num_workers=0,     # Number of parallel workers (0 = main process)
        drop_last=False,   # Drop incomplete last batch?
        pin_memory=False   # Pin memory for faster GPU transfer
    )
    print("Key DataLoader parameters:")
    print(f"  batch_size: {dataloader.batch_size}")
    print(f"  shuffle: True")
    print(f"  num_workers: {dataloader.num_workers}")
    print(f"  drop_last: {dataloader.drop_last}")
    
    header("3. TensorDataset - Quick Dataset")
    X = torch.randn(100, 5)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=20)
    print("TensorDataset: Simple way to create dataset from tensors")
    print(f"Dataset size: {len(dataset)}")
    for data, labels in dataloader:
        print(f"Batch: {data.shape}, {labels.shape}")
        break
    
    header("4. Training Loop with DataLoader")
    import torch.nn as nn
    import torch.optim as optim
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    dataset = CustomDataset(size=200)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("Training for 2 epochs:")
    for epoch in range(2):
        total_loss = 0
        for batch_idx, (data, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
    
    header("5. Data Augmentation Example")
    class AugmentedDataset(Dataset):
        def __init__(self, size=100):
            self.data = torch.randn(size, 3, 32, 32)  # Images
            self.labels = torch.randint(0, 10, (size,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            img = self.data[idx]
            # Simple augmentation: random flip
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[2])  # Horizontal flip
            return img, self.labels[idx]
    
    aug_dataset = AugmentedDataset(size=50)
    aug_dataloader = DataLoader(aug_dataset, batch_size=10)
    print("Dataset with random horizontal flip augmentation")
    for img, label in aug_dataloader:
        print(f"Batch: images={img.shape}, labels={label.shape}")
        break
    
    header("6. Splitting Dataset")
    from torch.utils.data import random_split
    
    dataset = CustomDataset(size=100)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Total dataset size: {len(dataset)}")
    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print("\nTrain and validation loaders created!")
    
    header("7. Collate Function - Custom Batching")
    def custom_collate(batch):
        """Custom function to batch samples."""
        data = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        # Add custom processing here
        return data, labels
    
    dataloader = DataLoader(dataset, batch_size=10, collate_fn=custom_collate)
    print("Using custom collate function")
    for data, labels in dataloader:
        print(f"Batch: {data.shape}, {labels.shape}")
        break
    
    header("8. Best Practices")
    print("""
    DataLoader Best Practices:
    
    1. Use multiple workers (num_workers > 0) for faster loading
    2. Enable pin_memory=True when using GPU
    3. Shuffle training data (shuffle=True)
    4. Don't shuffle validation/test data
    5. Use appropriate batch size (powers of 2 often work well)
    6. Prefetch data with persistent_workers=True
    7. Use drop_last=True if batch size matters
    8. Implement efficient __getitem__ in custom datasets
    9. Cache preprocessed data when possible
    10. Profile data loading time vs training time
    """)

if __name__ == "__main__":
    main()
