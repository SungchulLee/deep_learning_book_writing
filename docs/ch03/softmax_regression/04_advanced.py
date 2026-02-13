"""
===============================================================================
LEVEL 4: Advanced Softmax Regression Techniques
===============================================================================
Difficulty: Intermediate-Advanced
Prerequisites: Level 1, 2, 3
Learning Goals:
  - Implement softmax regression from scratch (numpy)
  - Advanced regularization techniques (L2, dropout, batch normalization)
  - Learning rate scheduling
  - Early stopping
  - Gradient clipping
  - Custom loss functions and metrics

Time to complete: 60-90 minutes
===============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("LEVEL 4: ADVANCED SOFTMAX REGRESSION TECHNIQUES")
print("=" * 80)


# =============================================================================
# PART 1: Softmax Regression from Scratch (NumPy)
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: Implementing Softmax Regression from Scratch")
print("=" * 80)

class SoftmaxRegressionNumPy:
    """
    Softmax regression implemented from scratch using only NumPy.
    This helps understand what PyTorch does under the hood.
    """
    
    def __init__(self, input_dim, num_classes, lr=0.01, reg_lambda=0.01):
        """
        Initialize the softmax regression model.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            lr: Learning rate
            reg_lambda: L2 regularization parameter
        """
        # Initialize weights and bias with small random values
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.loss_history = []
        
    def softmax(self, z):
        """
        Compute softmax values for each row of z.
        
        Args:
            z: Logits of shape (batch_size, num_classes)
        
        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward pass: compute class probabilities.
        
        Args:
            X: Input features of shape (batch_size, input_dim)
        
        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        logits = np.dot(X, self.W) + self.b
        probs = self.softmax(logits)
        return probs
    
    def compute_loss(self, X, y):
        """
        Compute cross-entropy loss with L2 regularization.
        
        Args:
            X: Input features
            y: True labels (class indices)
        
        Returns:
            Average loss value
        """
        m = X.shape[0]  # Number of samples
        probs = self.forward(X)
        
        # Cross-entropy loss
        # For each sample, get the probability of the true class
        correct_logprobs = -np.log(probs[range(m), y] + 1e-10)
        data_loss = np.sum(correct_logprobs) / m
        
        # L2 regularization loss
        reg_loss = 0.5 * self.reg_lambda * np.sum(self.W * self.W)
        
        total_loss = data_loss + reg_loss
        return total_loss
    
    def backward(self, X, y):
        """
        Backward pass: compute gradients.
        
        Args:
            X: Input features
            y: True labels
        """
        m = X.shape[0]
        probs = self.forward(X)
        
        # Compute gradient of loss w.r.t. logits
        # This is a key insight: d(loss)/d(logits) = (probs - one_hot_labels) / m
        dlogits = probs.copy()
        dlogits[range(m), y] -= 1  # Subtract 1 from true class probabilities
        dlogits /= m
        
        # Compute gradients w.r.t. weights and bias
        dW = np.dot(X.T, dlogits) + self.reg_lambda * self.W  # Add L2 gradient
        db = np.sum(dlogits, axis=0, keepdims=True)
        
        return dW, db
    
    def train_step(self, X, y):
        """
        Perform one gradient descent step.
        
        Args:
            X: Training features
            y: Training labels
        """
        # Compute gradients
        dW, db = self.backward(X, y)
        
        # Update weights
        self.W -= self.lr * dW
        self.b -= self.lr * db
        
        # Compute and store loss
        loss = self.compute_loss(X, y)
        self.loss_history.append(loss)
        
        return loss
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Input features
        
        Returns:
            Predicted class indices
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def accuracy(self, X, y):
        """
        Compute accuracy.
        
        Args:
            X: Input features
            y: True labels
        
        Returns:
            Accuracy value
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Generate synthetic data for testing
print("Generating synthetic dataset...")
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, n_classes=3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {len(np.unique(y))}")

# Train the NumPy model
print("\nTraining NumPy implementation...")
model_numpy = SoftmaxRegressionNumPy(
    input_dim=X_train.shape[1],
    num_classes=3,
    lr=0.1,
    reg_lambda=0.01
)

num_epochs = 200
for epoch in range(num_epochs):
    loss = model_numpy.train_step(X_train, y_train)
    
    if (epoch + 1) % 50 == 0:
        train_acc = model_numpy.accuracy(X_train, y_train)
        test_acc = model_numpy.accuracy(X_test, y_test)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, "
              f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

final_test_acc = model_numpy.accuracy(X_test, y_test)
print(f"\n✅ NumPy implementation final test accuracy: {final_test_acc:.2%}")


# =============================================================================
# PART 2: Advanced Model with Batch Normalization
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: Batch Normalization")
print("=" * 80)

"""
Batch Normalization:
-------------------
- Normalizes the inputs to each layer
- Helps with training stability
- Allows higher learning rates
- Acts as a regularizer
- Reduces internal covariate shift
"""

class AdvancedClassifier(nn.Module):
    """
    Neural network with batch normalization.
    
    Batch normalization normalizes layer inputs, which:
    1. Speeds up training
    2. Reduces sensitivity to initialization
    3. Acts as a regularizer
    """
    
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes,
                 use_batchnorm=True, dropout_rate=0.3):
        super(AdvancedClassifier, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        
        # First layer
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)      # Batch normalization here
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)      # Batch normalization here
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


# Create datasets
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Compare models with and without batch normalization
print("\nComparing models with/without batch normalization...")

models_config = [
    ("Without BatchNorm", False),
    ("With BatchNorm", True)
]

for name, use_bn in models_config:
    print(f"\nTraining: {name}")
    print("-" * 40)
    
    model = AdvancedClassifier(
        input_size=20,
        hidden_size1=64,
        hidden_size2=32,
        num_classes=3,
        use_batchnorm=use_bn,
        dropout_rate=0.3
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_t).float().mean()
    
    print(f"Final test accuracy: {accuracy:.4f}")


# =============================================================================
# PART 3: Learning Rate Scheduling
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: Learning Rate Scheduling")
print("=" * 80)

"""
Learning Rate Scheduling:
------------------------
Instead of using a fixed learning rate, we can:
- Start with a high LR for fast initial learning
- Gradually decrease it for fine-tuning
- Use different strategies: step decay, exponential, cosine, etc.
"""

class LRSchedulerDemo:
    """Demonstration of different learning rate schedules."""
    
    @staticmethod
    def step_lr_schedule(initial_lr, epoch, step_size=30, gamma=0.1):
        """
        Step decay: Reduce LR by gamma every step_size epochs.
        
        Example: LR = 0.1 → 0.01 → 0.001 (if gamma=0.1, step_size=30)
        """
        return initial_lr * (gamma ** (epoch // step_size))
    
    @staticmethod
    def exponential_schedule(initial_lr, epoch, gamma=0.95):
        """
        Exponential decay: LR = initial_lr * gamma^epoch
        
        Smooth, continuous decay.
        """
        return initial_lr * (gamma ** epoch)
    
    @staticmethod
    def cosine_schedule(initial_lr, epoch, total_epochs):
        """
        Cosine annealing: LR follows a cosine curve.
        
        Starts high, smoothly decreases, popular for modern training.
        """
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))


# Visualize different schedules
print("Visualizing learning rate schedules...")
epochs = range(100)
initial_lr = 0.1

schedules = {
    'Constant': [initial_lr] * 100,
    'Step Decay': [LRSchedulerDemo.step_lr_schedule(initial_lr, e) for e in epochs],
    'Exponential': [LRSchedulerDemo.exponential_schedule(initial_lr, e) for e in epochs],
    'Cosine': [LRSchedulerDemo.cosine_schedule(initial_lr, e, 100) for e in epochs]
}

# Uncomment to plot
# plt.figure(figsize=(10, 6))
# for name, lrs in schedules.items():
#     plt.plot(epochs, lrs, label=name, linewidth=2)
# plt.xlabel('Epoch')
# plt.ylabel('Learning Rate')
# plt.title('Learning Rate Schedules')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()


# Train with learning rate scheduling
print("\nTraining with StepLR scheduler...")
model_scheduled = AdvancedClassifier(20, 64, 32, 3, use_batchnorm=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_scheduled.parameters(), lr=0.01)

# Create scheduler: reduce LR by 0.1 every 20 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

num_epochs = 60
lr_history = []

for epoch in range(num_epochs):
    model_scheduled.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model_scheduled(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    # Step the scheduler
    scheduler.step()
    
    # Track learning rate
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Learning Rate = {current_lr:.6f}")

print("✅ Training with LR scheduling complete")


# =============================================================================
# PART 4: Early Stopping
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: Early Stopping")
print("=" * 80)

"""
Early Stopping:
--------------
Stop training when validation loss stops improving.
This prevents overfitting and saves time.
"""

class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        """
        Args:
            patience: How many epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        """
        Check if we should stop training.
        
        Args:
            val_loss: Current validation loss
            model: Current model (to save best version)
        
        Returns:
            True if we should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"  Validation loss improved: {self.best_loss:.4f} → {val_loss:.4f}")
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop


# Demonstrate early stopping
print("Training with early stopping...")

# Split data into train and validation
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

train_loader_es = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader_es = DataLoader(val_subset, batch_size=32, shuffle=False)

model_es = AdvancedClassifier(20, 64, 32, 3, use_batchnorm=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_es.parameters(), lr=0.001)

early_stopping = EarlyStopping(patience=10, min_delta=0.001, verbose=True)

max_epochs = 100
for epoch in range(max_epochs):
    # Train
    model_es.train()
    train_loss = 0
    for X_batch, y_batch in train_loader_es:
        optimizer.zero_grad()
        outputs = model_es(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model_es.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader_es:
            outputs = model_es(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader_es)
    avg_val_loss = val_loss / len(val_loader_es)
    
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
          f"Val Loss = {avg_val_loss:.4f}")
    
    # Check early stopping
    if early_stopping(avg_val_loss, model_es):
        print(f"\n✅ Early stopping triggered at epoch {epoch+1}")
        # Load best model
        model_es.load_state_dict(early_stopping.best_model)
        break

if epoch == max_epochs - 1:
    print(f"\n✅ Training completed all {max_epochs} epochs")


# =============================================================================
# PART 5: Gradient Clipping
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: Gradient Clipping")
print("=" * 80)

"""
Gradient Clipping:
-----------------
Prevents exploding gradients by limiting their magnitude.
Useful for training stability, especially with RNNs/LSTMs.
"""

def train_with_gradient_clipping(model, train_loader, criterion, optimizer,
                                max_norm=1.0, num_epochs=10):
    """
    Train model with gradient clipping.
    
    Args:
        max_norm: Maximum norm for gradients
    """
    print(f"Training with gradient clipping (max_norm={max_norm})...")
    
    for epoch in range(num_epochs):
        model.train()
        total_grad_norm = 0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            total_grad_norm += grad_norm
            num_batches += 1
            
            optimizer.step()
        
        avg_grad_norm = total_grad_norm / num_batches
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Avg gradient norm = {avg_grad_norm:.4f}")
    
    print("✅ Training with gradient clipping complete")


model_clip = AdvancedClassifier(20, 64, 32, 3, use_batchnorm=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_clip.parameters(), lr=0.001)

train_with_gradient_clipping(model_clip, train_loader, criterion, optimizer,
                            max_norm=1.0, num_epochs=20)


# =============================================================================
# PART 6: Custom Loss Function - Label Smoothing
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: Label Smoothing")
print("=" * 80)

"""
Label Smoothing:
---------------
Instead of using hard targets (0 or 1), use soft targets.
Example: Instead of [0, 1, 0], use [0.05, 0.9, 0.05]

Benefits:
- Prevents overconfidence
- Acts as regularization
- Often improves generalization
"""

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        """
        Args:
            num_classes: Number of classes
            smoothing: Smoothing parameter (0 = no smoothing, 1 = uniform)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, predictions, targets):
        """
        Compute label smoothing loss.
        
        Args:
            predictions: Model logits (batch_size, num_classes)
            targets: True class indices (batch_size,)
        """
        # Apply log softmax
        log_probs = torch.nn.functional.log_softmax(predictions, dim=1)
        
        # Create smoothed targets
        with torch.no_grad():
            # Start with uniform distribution
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            # Set true class to higher probability
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Compute loss
        loss = torch.sum(-smooth_targets * log_probs, dim=1)
        return loss.mean()


# Compare standard CE vs label smoothing
print("Comparing standard CrossEntropy vs Label Smoothing...")

configs = [
    ("Standard CrossEntropy", nn.CrossEntropyLoss(), 0.0),
    ("Label Smoothing (0.1)", LabelSmoothingCrossEntropy(3, smoothing=0.1), 0.1)
]

for name, criterion, smoothing in configs:
    print(f"\n{name}:")
    print("-" * 40)
    
    model = AdvancedClassifier(20, 64, 32, 3, use_batchnorm=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for a few epochs
    for epoch in range(30):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_t).float().mean()
    
    print(f"Test accuracy: {accuracy:.4f}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY - What You Learned")
print("=" * 80)

print("""
✅ Implemented softmax regression from scratch in NumPy
✅ Used batch normalization for training stability
✅ Applied learning rate scheduling strategies
✅ Implemented early stopping to prevent overfitting
✅ Used gradient clipping for training stability
✅ Explored label smoothing as a regularization technique

Advanced Techniques Summary:
---------------------------
1. Batch Normalization
   - Normalizes layer inputs
   - Speeds up training
   - Acts as regularizer

2. Learning Rate Scheduling
   - Step decay: Drop LR at intervals
   - Exponential: Smooth continuous decay
   - Cosine: Following cosine curve

3. Early Stopping
   - Monitor validation loss
   - Stop when no improvement
   - Save best model

4. Gradient Clipping
   - Prevents exploding gradients
   - Limits gradient magnitude
   - Improves stability

5. Label Smoothing
   - Soft targets instead of hard
   - Prevents overconfidence
   - Better generalization

Best Practices:
--------------
• Use batch normalization for deeper networks
• Start with higher LR, schedule it down
• Always use early stopping with validation set
• Clip gradients for RNNs or unstable training
• Consider label smoothing for better generalization

Next Steps:
-----------
→ Level 5: Compare multiple datasets and architectures
→ Experiment with different combinations of techniques
→ Try on your own datasets

🎉 Congratulations! You've mastered advanced techniques!
""")
