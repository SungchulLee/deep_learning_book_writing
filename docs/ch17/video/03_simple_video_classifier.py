"""
Module 34: Video Understanding - Beginner Level
File 03: Simple Video Classifier - Building a Complete Video Classification System

This file covers:
- Building a complete 3D CNN classifier
- Creating synthetic video dataset
- Training loop implementation
- Evaluation metrics for video classification
- Model checkpointing and inference

Mathematical Foundation:
Classification Objective:
    Minimize cross-entropy loss over video-label pairs
    
    L = -Σ y_i * log(p_i)
    
    where:
    - y_i = true label (one-hot encoded)
    - p_i = predicted probability for class i
    - p = softmax(f(V)) where f is the 3D CNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


#=============================================================================
# PART 1: SIMPLE 3D CNN CLASSIFIER
#=============================================================================

class Simple3DCNN(nn.Module):
    """
    Lightweight 3D CNN for video classification.
    
    Architecture:
        - 3 convolutional blocks with increasing channels
        - Global average pooling
        - Fully connected classifier
        
    Simpler than C3D, good for learning and small datasets.
    """
    
    def __init__(self, 
                 num_classes: int = 10,
                 input_channels: int = 3,
                 dropout: float = 0.5):
        """
        Initialize simple 3D CNN.
        
        Args:
            num_classes: Number of action classes
            input_channels: Number of input channels (3 for RGB)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Convolutional Block 1: 3 → 32 channels
        # Input: (B, 3, T, H, W)
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        # Output: (B, 32, T/2, H/2, W/2)
        
        # Convolutional Block 2: 32 → 64 channels
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        # Output: (B, 64, T/4, H/4, W/4)
        
        # Convolutional Block 3: 64 → 128 channels
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        # Output: (B, 128, T/8, H/8, W/8)
        
        # Global Average Pooling
        # Reduces spatiotemporal dimensions to 1x1x1
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input video (B, C, T, H, W)
            
        Returns:
            Class logits (B, num_classes)
        """
        # Feature extraction through conv blocks
        x = self.conv_block1(x)  # (B, 32, T/2, H/2, W/2)
        x = self.conv_block2(x)  # (B, 64, T/4, H/4, W/4)
        x = self.conv_block3(x)  # (B, 128, T/8, H/8, W/8)
        
        # Global pooling: (B, 128, T/8, H/8, W/8) → (B, 128, 1, 1, 1)
        x = self.global_avg_pool(x)
        
        # Classification
        x = self.classifier(x)
        
        return x


#=============================================================================
# PART 2: SYNTHETIC VIDEO DATASET
#=============================================================================

class SyntheticVideoDataset(Dataset):
    """
    Synthetic video dataset for demonstration.
    
    Creates videos with different motion patterns for classification:
        - Class 0: Horizontal movement (left to right)
        - Class 1: Vertical movement (top to bottom)
        - Class 2: Diagonal movement
        - Class 3: Circular movement
        - Class 4: Static (no movement)
    """
    
    def __init__(self,
                 num_samples: int = 1000,
                 num_frames: int = 16,
                 height: int = 64,
                 width: int = 64,
                 num_classes: int = 5):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of video samples
            num_frames: Number of frames per video
            height: Frame height
            width: Frame width
            num_classes: Number of classes
        """
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_classes = num_classes
        
        # Generate all videos and labels
        self.videos, self.labels = self._generate_dataset()
    
    def _create_moving_square(self, 
                            motion_type: int,
                            t: int) -> np.ndarray:
        """
        Create frame with moving square based on motion type.
        
        Args:
            motion_type: Type of motion (0-4)
            t: Time step (frame index)
            
        Returns:
            Frame as numpy array (H, W, 3)
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # Square size
        square_size = 8
        
        # Calculate position based on motion type
        if motion_type == 0:  # Horizontal
            x = int((t / self.num_frames) * (self.width - square_size))
            y = self.height // 2 - square_size // 2
        
        elif motion_type == 1:  # Vertical
            x = self.width // 2 - square_size // 2
            y = int((t / self.num_frames) * (self.height - square_size))
        
        elif motion_type == 2:  # Diagonal
            progress = t / self.num_frames
            x = int(progress * (self.width - square_size))
            y = int(progress * (self.height - square_size))
        
        elif motion_type == 3:  # Circular
            angle = 2 * np.pi * (t / self.num_frames)
            radius = min(self.height, self.width) // 4
            center_x, center_y = self.width // 2, self.height // 2
            x = int(center_x + radius * np.cos(angle) - square_size // 2)
            y = int(center_y + radius * np.sin(angle) - square_size // 2)
        
        else:  # Static
            x = self.width // 2 - square_size // 2
            y = self.height // 2 - square_size // 2
        
        # Ensure within bounds
        x = max(0, min(x, self.width - square_size))
        y = max(0, min(y, self.height - square_size))
        
        # Draw square (different color per class)
        color = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
        ])[motion_type]
        
        frame[y:y+square_size, x:x+square_size] = color
        
        return frame
    
    def _generate_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate complete dataset.
        
        Returns:
            videos: Tensor of shape (N, T, H, W, 3)
            labels: Tensor of shape (N,)
        """
        videos = []
        labels = []
        
        for i in range(self.num_samples):
            # Random class
            label = np.random.randint(0, self.num_classes)
            
            # Generate video
            video_frames = []
            for t in range(self.num_frames):
                frame = self._create_moving_square(label, t)
                video_frames.append(frame)
            
            video = np.array(video_frames)  # (T, H, W, 3)
            
            # Add slight noise for robustness
            video = video + np.random.randn(*video.shape) * 0.02
            video = np.clip(video, 0, 1)
            
            videos.append(video)
            labels.append(label)
        
        # Convert to tensors
        # (N, T, H, W, 3) → (N, T, 3, H, W) → (N, 3, T, H, W)
        videos = torch.FloatTensor(np.array(videos))
        videos = videos.permute(0, 1, 4, 2, 3)  # (N, T, 3, H, W)
        videos = videos.permute(0, 2, 1, 3, 4)  # (N, 3, T, H, W)
        
        labels = torch.LongTensor(labels)
        
        return videos, labels
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get single video-label pair.
        
        Args:
            idx: Index
            
        Returns:
            video: (3, T, H, W)
            label: class index
        """
        return self.videos[idx], self.labels[idx]


#=============================================================================
# PART 3: TRAINING PIPELINE
#=============================================================================

class VideoClassifier:
    """
    Complete training and evaluation pipeline for video classification.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cpu'):
        """
        Initialize classifier.
        
        Args:
            model: 3D CNN model
            device: Device for training ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self,
                   train_loader: DataLoader,
                   optimizer: optim.Optimizer,
                   criterion: nn.Module) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for videos, labels in train_loader:
            # Move to device
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(videos)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self,
                val_loader: DataLoader,
                criterion: nn.Module) -> Tuple[float, float]:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for videos, labels in val_loader:
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(videos)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: int = 20,
             learning_rate: float = 0.001) -> Dict:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training history dictionary
        """
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        print(f"\nTraining on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*80)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  ✓ New best validation accuracy: {best_val_acc:.2f}%")
        
        print("="*80)
        print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': best_val_acc
        }


#=============================================================================
# PART 4: VISUALIZATION
#=============================================================================

def plot_training_history(history: Dict):
    """
    Plot training and validation metrics.
    
    Args:
        history: Training history dictionary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/34_video_understanding/03_training_curves.png',
                dpi=150, bbox_inches='tight')
    print(f"Training curves saved to 03_training_curves.png")
    plt.close()


#=============================================================================
# PART 5: MAIN DEMONSTRATION
#=============================================================================

def main():
    """
    Main demonstration of simple video classifier.
    """
    print(__doc__)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "="*80)
    print("SIMPLE VIDEO CLASSIFIER DEMONSTRATION")
    print("="*80)
    
    # Configuration
    num_classes = 5
    num_frames = 16
    height, width = 64, 64
    batch_size = 16
    num_epochs = 15
    
    # Create dataset
    print("\n1. Creating synthetic video dataset...")
    train_dataset = SyntheticVideoDataset(
        num_samples=800,
        num_frames=num_frames,
        height=height,
        width=width,
        num_classes=num_classes
    )
    
    val_dataset = SyntheticVideoDataset(
        num_samples=200,
        num_frames=num_frames,
        height=height,
        width=width,
        num_classes=num_classes
    )
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Video shape: (3, {num_frames}, {height}, {width})")
    print(f"   Number of classes: {num_classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for debugging
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\n2. Creating 3D CNN model...")
    model = Simple3DCNN(num_classes=num_classes)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create classifier and train
    print(f"\n3. Training model...")
    classifier = VideoClassifier(model, device=device)
    history = classifier.train(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        learning_rate=0.001
    )
    
    # Plot results
    print("\n4. Visualizing training history...")
    plot_training_history(history)
    
    # Test predictions
    print("\n5. Testing predictions on validation set...")
    classifier.model.eval()
    
    class_names = ['Horizontal', 'Vertical', 'Diagonal', 'Circular', 'Static']
    
    with torch.no_grad():
        videos, labels = next(iter(val_loader))
        videos = videos.to(device)
        outputs = classifier.model(videos)
        _, predicted = torch.max(outputs, 1)
        
        print(f"\nSample predictions:")
        for i in range(min(5, len(labels))):
            true_label = class_names[labels[i].item()]
            pred_label = class_names[predicted[i].item()]
            correct = "✓" if labels[i] == predicted[i] else "✗"
            print(f"  {correct} True: {true_label:12s} | Predicted: {pred_label:12s}")
    
    # Summary
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print(f"""
    1. Complete Pipeline:
       - Dataset creation with motion patterns
       - Model training with validation
       - Performance monitoring and visualization
    
    2. Model Performance:
       - Best validation accuracy: {history['best_val_acc']:.2f}%
       - Simple 3D CNN can learn basic motion patterns
       - Synthetic data good for prototyping
    
    3. Training Insights:
       - 3D CNNs learn spatiotemporal features end-to-end
       - Batch size limited by memory (3D tensors are large)
       - Learning rate scheduling helps convergence
    
    4. Next Steps:
       - Try real video datasets (UCF-101, Kinetics)
       - Add data augmentation for robustness
       - Experiment with deeper architectures
       - Consider two-stream networks for better performance
    """)


if __name__ == "__main__":
    main()
