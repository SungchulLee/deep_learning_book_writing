# MNIST Neural Network with TensorBoard Visualization

A comprehensive PyTorch implementation demonstrating neural network training on the MNIST dataset with TensorBoard integration for visualization and monitoring.

## 📋 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Understanding the Code](#understanding-the-code)
- [TensorBoard Features](#tensorboard-features)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Common Issues](#common-issues)
- [Additional Resources](#additional-resources)

## 🎯 Overview

This project demonstrates:
- **Neural Network Training**: Building and training a fully connected network
- **TensorBoard Integration**: Visualizing training metrics, model architecture, and performance
- **Best Practices**: Proper PyTorch workflow with data loading, training loops, and evaluation

### What is MNIST?

MNIST (Modified National Institute of Standards and Technology) is a dataset of 70,000 handwritten digits:
- 60,000 training images
- 10,000 test images
- Each image is 28x28 pixels in grayscale
- 10 classes (digits 0-9)

### What is TensorBoard?

TensorBoard is TensorFlow's visualization toolkit that works with PyTorch. It provides:
- Real-time training metrics visualization
- Model architecture graphs
- Image and histogram logging
- Precision-Recall curves
- Hyperparameter tuning visualization

## 📦 Requirements

### Python Version
- Python 3.7 or higher

### Required Packages
```
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
tensorboard>=2.6.0
```

## 🚀 Installation

### Step 1: Clone or Download the Project
```bash
# If you have git
git clone <repository-url>
cd mnist-tensorboard

# Or download and extract the ZIP file
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install torch torchvision matplotlib tensorboard
```

### Step 4: Verify Installation
```bash
python -c "import torch; import torchvision; import tensorboard; print('All packages installed successfully!')"
```

## 📁 Project Structure

```
mnist-tensorboard/
│
├── tensorboard_mnist_commented.py    # Fully commented main script
├── config.py                          # Hyperparameter configuration
├── utils.py                           # Helper functions
├── advanced_tensorboard.py            # Advanced features
├── README.md                          # This file
│
├── data/                              # MNIST dataset (auto-downloaded)
│   └── MNIST/
│
└── runs/                              # TensorBoard logs (auto-generated)
    └── mnist1/
        └── events.out.tfevents.*
```

## 💻 Usage

### Basic Usage

1. **Run the training script:**
```bash
python tensorboard_mnist_commented.py
```

2. **Start TensorBoard in a new terminal:**
```bash
tensorboard --logdir=runs
```

3. **Open your browser and navigate to:**
```
http://localhost:6006
```

### What Happens During Execution

1. **Data Download**: First run downloads MNIST dataset (~50MB) to `./data`
2. **Training**: Model trains for 1 epoch (configurable)
3. **Logging**: Metrics are logged every 100 steps to TensorBoard
4. **Evaluation**: Model is evaluated on test set after training
5. **Results**: Final accuracy and PR curves are saved

### Expected Output

```
Using device: cuda  # or 'cpu' if no GPU

Model Architecture:
NeuralNet(
  (l1): Linear(in_features=784, out_features=500, bias=True)
  (relu): ReLU()
  (l2): Linear(in_features=500, out_features=10, bias=True)
)

============================================================
STARTING TRAINING
============================================================
Epoch [1/1], Step [100/938], Loss: 0.4521
Epoch [1/1], Step [200/938], Loss: 0.2134
...
============================================================
TRAINING COMPLETED
============================================================

============================================================
EVALUATING MODEL ON TEST SET
============================================================

Accuracy of the network on the 10000 test images: 97.50%

TensorBoard logs saved to: runs/mnist1
To view, run: tensorboard --logdir=runs
============================================================
```

## 🧠 Understanding the Code

### Key Components

#### 1. Model Architecture
```python
Input (784) → Linear Layer (500) → ReLU → Linear Layer (10) → Output
```

- **Input Layer**: 784 neurons (28×28 flattened image)
- **Hidden Layer**: 500 neurons with ReLU activation
- **Output Layer**: 10 neurons (one per digit class)

#### 2. Training Loop Structure
```python
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 1. Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # 2. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 3. Log metrics
        writer.add_scalar('loss', loss.item(), global_step)
```

#### 3. Evaluation Process
```python
with torch.no_grad():  # Disable gradient computation
    for data, target in test_loader:
        output = model(data)
        # Calculate accuracy
        # Generate predictions for PR curves
```

### Hyperparameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_size` | 784 | Flattened image size (28×28) |
| `hidden_size` | 500 | Hidden layer neurons |
| `num_classes` | 10 | Output classes (digits 0-9) |
| `num_epochs` | 1 | Training iterations over dataset |
| `batch_size` | 64 | Samples per gradient update |
| `learning_rate` | 0.001 | Step size for optimization |

## 📊 TensorBoard Features

### 1. Scalars Tab
- **Training Loss**: Monitor loss decrease over time
- **Accuracy**: Track model performance during training
- **Smooth Curves**: Adjust smoothing slider for clearer trends

### 2. Images Tab
- **Sample Images**: View MNIST digits the model is trained on
- **Grid Layout**: See multiple examples at once

### 3. Graphs Tab
- **Model Architecture**: Visual representation of network layers
- **Data Flow**: See how tensors move through the network
- **Operation Details**: Click nodes for tensor shapes and operations

### 4. PR Curves Tab
- **Per-Class Curves**: One curve for each digit (0-9)
- **Precision-Recall Trade-off**: Understand classification performance
- **Threshold Selection**: Find optimal decision boundaries

### Navigation Tips
- Use the left sidebar to switch between tabs
- Adjust time ranges with the slider at the bottom
- Download plots as SVG or PNG using the download button
- Use the settings gear icon for additional options

## 🎛️ Hyperparameter Tuning

### Experimenting with Different Configurations

1. **Modify `config.py`** or directly in the script:
```python
# Try different architectures
hidden_size = 1000  # Larger network

# Try different learning rates
learning_rate = 0.0001  # Slower learning

# Train longer
num_epochs = 5  # More iterations
```

2. **Compare Results in TensorBoard:**
```bash
# Run multiple experiments
python tensorboard_mnist_commented.py  # First run: runs/mnist1
# Edit config and run again
python tensorboard_mnist_commented.py  # Second run: runs/mnist2

# View all runs together
tensorboard --logdir=runs
```

3. **What to Look For:**
- **Loss Curves**: Should decrease smoothly
- **Accuracy**: Should increase and plateau
- **Overfitting**: Training accuracy much higher than test accuracy

### Recommended Experiments

| Experiment | Modification | Expected Result |
|------------|--------------|-----------------|
| Deeper Network | Add more layers | May improve accuracy |
| Higher Learning Rate | `lr = 0.01` | Faster training, might be unstable |
| Lower Learning Rate | `lr = 0.0001` | Slower but more stable |
| Larger Batch Size | `batch_size = 256` | Faster training, less noise |
| Smaller Batch Size | `batch_size = 16` | More frequent updates, more noise |
| More Epochs | `num_epochs = 10` | Better convergence |

## 🐛 Common Issues

### Issue 1: CUDA Out of Memory
**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce batch size: `batch_size = 32`
- Use CPU instead: `device = torch.device('cpu')`
- Close other GPU-using applications

### Issue 2: TensorBoard Not Loading
**Error:** Browser shows "TensorBoard not found"

**Solutions:**
```bash
# Check if TensorBoard is running
ps aux | grep tensorboard

# Kill existing instances and restart
pkill -f tensorboard
tensorboard --logdir=runs --port=6006
```

### Issue 3: No Plots Showing
**Problem:** TensorBoard shows no data

**Solutions:**
- Wait a few seconds for logs to sync
- Refresh the browser (F5)
- Check that `runs/` directory contains log files
- Ensure the script completed without errors

### Issue 4: Import Errors
**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solutions:**
```bash
# Verify installation
pip list | grep torch

# Reinstall if necessary
pip install --upgrade torch torchvision tensorboard
```

### Issue 5: Slow Training
**Problem:** Training takes too long

**Solutions:**
- Enable GPU if available (CUDA)
- Increase batch size
- Reduce network size
- Use fewer epochs for quick tests

## 🔧 Advanced Usage

### Custom TensorBoard Logging

```python
# Log hyperparameters
writer.add_hparams(
    {'lr': learning_rate, 'batch_size': batch_size},
    {'accuracy': final_accuracy}
)

# Log confusion matrix
writer.add_figure('confusion_matrix', fig)

# Log model weights
for name, param in model.named_parameters():
    writer.add_histogram(name, param, epoch)

# Log gradient norms
for name, param in model.named_parameters():
    writer.add_histogram(f'{name}.grad', param.grad, epoch)
```

### Multiple Runs Comparison

```python
# Create different log directories for experiments
writer = SummaryWriter(f'runs/experiment_{experiment_id}')
```

Then view all experiments together:
```bash
tensorboard --logdir=runs
```

## 📚 Additional Resources

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Torchvision Datasets](https://pytorch.org/vision/stable/datasets.html)

### Tutorials
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [TensorBoard with PyTorch](https://pytorch.org/docs/stable/tensorboard.html)
- [Deep Learning Fundamentals](https://www.deeplearningbook.org/)

### Related Topics
- Convolutional Neural Networks (CNNs) for better image recognition
- Data augmentation techniques
- Transfer learning
- Model deployment

## 📝 Notes

- First run will download MNIST dataset (~50MB)
- Training on CPU takes ~2-3 minutes per epoch
- Training on GPU takes ~10-20 seconds per epoch
- TensorBoard logs are cumulative (new runs add to existing logs)
- Clean `runs/` directory to start fresh: `rm -rf runs/*`

## 🤝 Contributing

Feel free to experiment and modify the code! Some ideas:
- Try different model architectures
- Add data augmentation
- Implement learning rate scheduling
- Add validation set monitoring
- Experiment with different optimizers (SGD, RMSprop)

## 📄 License

This code is based on [Patrick Loeber's PyTorch Tutorial](https://github.com/patrickloeber/pytorchTutorial) and is provided for educational purposes.

---

**Happy Learning! 🎓🚀**

For questions or issues, refer to the [Common Issues](#common-issues) section or check the official documentation.
