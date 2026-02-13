# Quick Reference Guide

A cheat sheet for quick lookups. Bookmark this page!

## 📚 File Finder

### By Topic:

**Math & Foundations**
- Linear regression from scratch → `01_linear_regression_numpy.py`
- PyTorch vs NumPy → `02_linear_regression_pytorch.py`
- Manual backprop → `03_simple_nn_manual.py`

**PyTorch Basics**
- Autograd deep dive → `04_autograd_introduction.py`
- Single neuron → `05_simple_perceptron.py`
- Multi-layer network → `06_two_layer_network.py`
- nn.Module pattern → `07_nn_module_and_optimizers.py`

**Building Networks**
- MNIST basic → `08_mnist_basic.py`
- MNIST detailed → `09_mnist_classification_detailed.py`
- nn.Sequential → `10_using_sequential.py`
- Custom modules → `11_custom_module.py`
- Activation functions → `12_activation_functions.py`
- Loss functions → `13_loss_functions.py`

**Advanced Techniques**
- Dropout → `14_dropout_regularization.py`
- All regularization → `15_regularization_techniques_detailed.py`
- Batch norm basic → `16_batch_normalization.py`
- Batch norm detailed → `17_batch_normalization_detailed.py`
- Learning rate scheduling → `18_learning_rate_scheduling.py`
- Weight initialization → `19_weight_initialization.py`

**Applications**
- CIFAR-10 images → `20_cifar10_classifier.py`
- Regression problems → `21_regression_task.py`
- Multi-task learning → `22_multi_output_network.py`
- Very deep networks → `23_deep_network.py`

---

## 🔧 Common Patterns

### Standard Training Loop
```python
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
```

### nn.Module Pattern
```python
class MyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

### Data Loading
```python
from torch.utils.data import DataLoader, TensorDataset

# Create dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Create loader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use in training
for batch_X, batch_y in loader:
    # training code here
    pass
```

---

## 🎯 Problem-Specific Quick Guides

### For Classification:
```python
# Architecture
model = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, num_classes)
    # NO SOFTMAX! (CrossEntropyLoss includes it)
)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Evaluation
_, predicted = torch.max(outputs, 1)
accuracy = (predicted == labels).float().mean()
```

### For Regression:
```python
# Architecture
model = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)  # Single output
    # NO ACTIVATION at end!
)

# Loss & Optimizer
criterion = nn.MSELoss()  # or nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Evaluation
mse = torch.mean((predictions - targets)**2)
mae = torch.mean(torch.abs(predictions - targets))
```

### For Binary Classification:
```python
# Architecture
model = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()  # For binary classification
)

# Loss & Optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy
# Or use BCEWithLogitsLoss (no sigmoid needed)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Evaluation
predictions = (outputs > 0.5).float()
accuracy = (predictions == labels).float().mean()
```

---

## 📊 Hyperparameter Guidelines

### Learning Rates:
```python
Adam: 0.001 (default, good starting point)
SGD: 0.01 to 0.1 (needs higher values)
RMSprop: 0.001

# If loss doesn't decrease: try smaller (÷10)
# If learning too slow: try larger (×10)
```

### Batch Sizes:
```python
Small datasets (<1000): 16-32
Medium (1K-100K): 32-128
Large (>100K): 64-512

# Larger batch = more stable, but needs more memory
# Smaller batch = regularization effect
```

### Hidden Layer Sizes:
```python
Simple problems: 32-128
Medium problems: 128-512
Complex problems: 512-2048

# Rule of thumb: between input size and output size
```

### Dropout Rates:
```python
Light regularization: 0.1-0.2
Medium: 0.3-0.4
Heavy: 0.5
Extreme: 0.6-0.7 (rarely used)

# Too high (>0.7): underfitting
# Too low (<0.1): not effective
```

### Number of Epochs:
```python
Quick experiment: 10-20
Typical training: 50-100
Production: 100-500+

# Use early stopping based on validation loss
```

---

## 🎨 Activation Function Chooser

```
Hidden Layers:
├─ Default → ReLU
├─ Deep networks → LeakyReLU or ELU
├─ RNNs → Tanh
└─ Very deep (>50 layers) → Swish or GELU

Output Layer:
├─ Multi-class classification → None (CrossEntropyLoss handles it)
├─ Binary classification → Sigmoid
├─ Regression → None (linear)
└─ Multi-label classification → Sigmoid (per output)
```

---

## 🎯 Loss Function Chooser

```
Problem Type → Loss Function

Multi-class Classification → CrossEntropyLoss
Binary Classification → BCELoss or BCEWithLogitsLoss
Regression (general) → MSELoss
Regression (robust to outliers) → L1Loss (MAE)
Multi-label Classification → BCEWithLogitsLoss
Semantic Segmentation → CrossEntropyLoss (per pixel)
```

---

## 🔍 Debugging Checklist

### Loss is not decreasing:
- [ ] Check learning rate (try 0.001, 0.0001)
- [ ] Verify loss function is appropriate
- [ ] Ensure optimizer is updating (check `param.grad`)
- [ ] Verify `optimizer.zero_grad()` is called
- [ ] Check data is normalized
- [ ] Try simpler model first

### Loss is NaN or Inf:
- [ ] Learning rate too high → reduce by 10x
- [ ] Check for division by zero
- [ ] Ensure numerical stability in loss
- [ ] Verify data has no NaN values
- [ ] Add gradient clipping: `torch.nn.utils.clip_grad_norm_()`

### Training is slow:
- [ ] Use GPU if available: `.cuda()`
- [ ] Increase batch size (if memory allows)
- [ ] Use DataLoader with num_workers > 0
- [ ] Profile code to find bottlenecks
- [ ] Use mixed precision training (advanced)

### Overfitting (train good, test bad):
- [ ] Add dropout
- [ ] Add L2 regularization (weight decay)
- [ ] Use data augmentation
- [ ] Get more training data
- [ ] Use simpler model (fewer parameters)
- [ ] Early stopping

### Underfitting (both train and test bad):
- [ ] Increase model capacity (more/larger layers)
- [ ] Reduce regularization (lower dropout, weight decay)
- [ ] Train for more epochs
- [ ] Increase learning rate
- [ ] Check if data quality is good
- [ ] Verify features are informative

---

## 💻 Code Snippets

### Save/Load Model:
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = MyNetwork()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### GPU Usage:
```python
# Check availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to GPU
model = model.to(device)

# Move data to GPU
inputs = inputs.to(device)
labels = labels.to(device)
```

### Train/Eval Mode:
```python
# Training (enables dropout, batchnorm updating)
model.train()

# Evaluation (disables dropout, batchnorm in eval mode)
model.eval()
with torch.no_grad():  # No gradient computation
    outputs = model(inputs)
```

### Learning Rate Scheduling:
```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(num_epochs):
    train_one_epoch()
    scheduler.step()  # Update LR
```

### Gradient Clipping:
```python
# Prevent exploding gradients
max_grad_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

### Data Normalization:
```python
# Normalize to mean=0, std=1
mean = X.mean(dim=0)
std = X.std(dim=0)
X_normalized = (X - mean) / std

# For images (to [-1, 1])
transform = transforms.Normalize((0.5,), (0.5,))
```

---

## 📈 Performance Benchmarks

### MNIST (Handwritten Digits):
```
Random: 10%
Basic MLP: 95-97%
Good MLP: 98-99%
CNN: 99.5%+
```

### CIFAR-10 (Color Images):
```
Random: 10%
Basic MLP: 40-50%
Good MLP: 55-60%
Basic CNN: 70-75%
Good CNN: 90-95%
```

---

## 🚀 Performance Tips

1. **Use GPU**: 10-100x faster than CPU
2. **Batch size**: Larger = faster (if fits in memory)
3. **DataLoader workers**: Use `num_workers=4`
4. **Mixed precision**: `torch.cuda.amp` (advanced)
5. **Profile**: Find bottlenecks with `torch.profiler`

---

## 📚 Quick Links to Files

**Need to understand math?** → Level 0  
**Learning PyTorch?** → Level 1  
**Building first models?** → Level 2  
**Preventing overfitting?** → Level 3  
**Real applications?** → Level 4  

**Stuck on something?** → Check the README in that level folder!

---

**Remember**: This is a reference guide. For learning, go through the files sequentially!
