# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Basic Script
```bash
python tensorboard_mnist_commented.py
```

### Step 3: View Results in TensorBoard
Open a new terminal and run:
```bash
tensorboard --logdir=runs
```
Then open your browser and go to: http://localhost:6006

---

## 📦 What's Included

### Files:
1. **tensorboard_mnist_commented.py** - Fully commented basic implementation
2. **advanced_tensorboard.py** - Advanced features with enhanced logging
3. **config.py** - Centralized configuration management
4. **utils.py** - Helper functions for training and visualization
5. **README.md** - Comprehensive documentation
6. **requirements.txt** - Python package dependencies
7. **QUICK_START.md** - This file

---

## 🎯 Usage Examples

### Example 1: Basic Training
```bash
python tensorboard_mnist_commented.py
```
- Trains for 1 epoch
- Logs to `runs/mnist1`
- Takes ~2-3 minutes on CPU

### Example 2: Advanced Training
```bash
python advanced_tensorboard.py
```
- Includes advanced logging features
- Confusion matrix, per-class metrics
- Model checkpointing
- Learning rate scheduling

### Example 3: Custom Configuration
Edit the script or use config.py:
```python
from config import Config

config = Config()
config.num_epochs = 5
config.learning_rate = 0.01
config.hidden_size = 1000
```

### Example 4: Using Preset Experiments
```python
from config import DeepNetworkExperiment

config = DeepNetworkExperiment()
# This automatically sets optimized parameters
```

---

## 📊 TensorBoard Navigation

### Main Tabs:
- **SCALARS**: View training/test loss and accuracy
- **IMAGES**: See sample MNIST digits
- **GRAPHS**: Visualize model architecture
- **DISTRIBUTIONS**: Weight and gradient distributions
- **HISTOGRAMS**: Detailed historical distributions
- **HPARAMS**: Compare hyperparameters (advanced_tensorboard.py)
- **PR CURVES**: Precision-recall curves per class

### Tips:
- Use the smoothing slider (left sidebar) for cleaner plots
- Toggle runs on/off by clicking in the legend
- Download plots as SVG or PNG
- Use the settings icon for more options

---

## 🔧 Common Commands

### Run Training
```bash
# Basic
python tensorboard_mnist_commented.py

# Advanced
python advanced_tensorboard.py
```

### Start TensorBoard
```bash
# Basic
tensorboard --logdir=runs

# Specify port
tensorboard --logdir=runs --port=6007

# Allow remote access
tensorboard --logdir=runs --host=0.0.0.0
```

### Clean Up Logs
```bash
# Remove all TensorBoard logs
rm -rf runs/*

# Remove specific experiment
rm -rf runs/mnist1
```

### Test Configuration
```bash
# Print configuration without training
python config.py
```

### Test Utilities
```bash
# Check available functions
python utils.py
```

---

## 🧪 Experiment Ideas

### 1. Learning Rate Comparison
Run multiple times with different learning rates:
```python
config.learning_rate = 0.0001  # Run 1
config.learning_rate = 0.001   # Run 2
config.learning_rate = 0.01    # Run 3
```

### 2. Network Size Comparison
```python
config.hidden_size = 128   # Small
config.hidden_size = 500   # Medium
config.hidden_size = 1000  # Large
```

### 3. Batch Size Impact
```python
config.batch_size = 16    # Small batches
config.batch_size = 64    # Medium batches
config.batch_size = 256   # Large batches
```

### 4. Training Duration
```python
config.num_epochs = 1     # Quick test
config.num_epochs = 5     # Standard
config.num_epochs = 10    # Extended
```

---

## 🐛 Troubleshooting

### Problem: "No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision tensorboard
```

### Problem: TensorBoard not showing data
**Solution:**
1. Wait a few seconds for logs to sync
2. Refresh browser (F5)
3. Check that training completed successfully
4. Verify `runs/` directory exists and has files

### Problem: CUDA out of memory
**Solution:**
```python
# Option 1: Use CPU
config.device = torch.device('cpu')

# Option 2: Reduce batch size
config.batch_size = 32
```

### Problem: Training too slow
**Solution:**
1. Check if GPU is being used (should see "Using device: cuda")
2. Increase batch size if memory allows
3. Use fewer epochs for quick tests

---

## 📈 Expected Results

### Typical Performance:
- **Accuracy**: 96-98% after 1 epoch
- **Training Time**: 
  - CPU: ~2-3 minutes per epoch
  - GPU: ~10-20 seconds per epoch
- **Final Loss**: ~0.1-0.3

### What to Look For:
- ✅ Loss should decrease steadily
- ✅ Accuracy should increase
- ✅ Test metrics close to training metrics
- ⚠️ If test << training accuracy: overfitting

---

## 🎓 Learning Path

1. **Start Here**: Run `tensorboard_mnist_commented.py`
2. **Read Code**: Study the comments in the basic script
3. **Explore TensorBoard**: Open http://localhost:6006
4. **Try Advanced**: Run `advanced_tensorboard.py`
5. **Experiment**: Modify hyperparameters in `config.py`
6. **Compare**: Run multiple experiments and compare
7. **Customize**: Add your own features using `utils.py`

---

## 📚 Next Steps

### Improve the Model:
- Add dropout for regularization
- Use batch normalization
- Implement a CNN instead of fully connected
- Try data augmentation

### Advanced TensorBoard:
- Log embedding projections
- Add audio/video logging
- Create custom visualizations
- Use TensorBoard.dev for sharing

### Deploy the Model:
- Export to ONNX format
- Create a web API with Flask/FastAPI
- Build a simple UI with Streamlit
- Deploy to mobile with PyTorch Mobile

---

## 🤝 Contributing

Feel free to:
- Experiment with different architectures
- Add new visualization functions
- Create custom training strategies
- Share your results!

---

## 📞 Getting Help

- Check the full README.md for detailed information
- Review PyTorch documentation: https://pytorch.org/docs
- TensorBoard guide: https://pytorch.org/docs/stable/tensorboard.html
- Stack Overflow for specific issues

---

**Happy Experimenting! 🎉**
