# Quick Start Guide - Learning Rate Schedulers

## 🚀 Get Running in 2 Minutes!

### Step 1: Extract and Install
```bash
# Extract the zip file
unzip scheduler_complete.zip
cd scheduler_project

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Your First Scheduler
```bash
# Try StepLR (basic example)
python scheduler.py --scheduler step --epochs 50

# View results: lr_schedule_step.png and training_curves_step.png
```

### Step 3: Try Different Schedulers
```bash
# Cosine Annealing (smooth decay)
python scheduler.py --scheduler cosine --epochs 50

# OneCycle (fast convergence)
python scheduler.py --scheduler onecycle --epochs 20 --max_lr 0.5

# Plateau (adaptive)
python scheduler.py --scheduler plateau --patience 5 --scheduler_step_on val_loss
```

---

## 📊 What You'll See

### Console Output
```
============================================================
LEARNING RATE SCHEDULER DEMONSTRATION
============================================================

Scheduler: STEP
Epochs: 50
Initial Learning Rate: 0.1
Batch Size: 32
============================================================

Random seed set to: 42
Using device: cuda

Building data loaders...
Training samples: 800
Validation samples: 200
Steps per epoch: 25

Epoch [  1/ 50] | Train Loss: 2.2156 | Train Acc:  15.25% | ...
Epoch [  2/ 50] | Train Loss: 2.0234 | Train Acc:  28.50% | ...
...
```

### Generated Files
- `lr_schedule_step.png` - Learning rate over time
- `training_curves_step.png` - Loss, accuracy, and LR curves

---

## 🎯 Quick Examples

### Example 1: Compare 3 Schedulers
```bash
python scheduler.py --scheduler step --epochs 50
python scheduler.py --scheduler cosine --epochs 50
python scheduler.py --scheduler onecycle --epochs 20

# Compare the generated plots!
```

### Example 2: Tune Parameters
```bash
# Try different gamma values
python scheduler.py --scheduler step --gamma 0.1  # Strong decay
python scheduler.py --scheduler step --gamma 0.5  # Moderate
python scheduler.py --scheduler step --gamma 0.9  # Gentle
```

### Example 3: Custom Model Size
```bash
# Small model
python scheduler.py --hidden_dim 50

# Large model
python scheduler.py --hidden_dim 200
```

---

## 📚 File Structure

```
scheduler_project/
├── scheduler.py                               # Main entry point (run this!)
├── README.md                                  # Full documentation
├── SCHEDULER_GUIDE.md                         # Detailed scheduler guide
├── QUICK_START.md                             # This file
├── requirements.txt                           # Dependencies
└── scheduler/                                 # Package
    ├── __init__.py
    ├── config.py        # Configuration & arguments
    ├── data_loader.py   # Data generation
    ├── model.py         # Neural network
    ├── train.py         # Training logic & schedulers
    └── utils.py         # Helper functions
```

---

## 💡 Common Commands

```bash
# See all options
python scheduler.py --help

# StepLR with custom parameters
python scheduler.py --scheduler step --step_size 20 --gamma 0.5

# MultiStepLR with milestones
python scheduler.py --scheduler multistep --milestones 10,20,30

# CosineAnnealing
python scheduler.py --scheduler cosine --t_max 50 --eta_min 0

# OneCycle (fast!)
python scheduler.py --scheduler onecycle --epochs 10 --max_lr 0.5

# ExponentialLR
python scheduler.py --scheduler exponential --gamma 0.95

# CyclicLR (batch-level updates)
python scheduler.py --scheduler cyclical --base_lr 1e-3 --max_lr 1e-1 --scheduler_step_on batch

# ReduceLROnPlateau (adaptive)
python scheduler.py --scheduler plateau --patience 5 --factor 0.5 --scheduler_step_on val_loss
```

---

## 🎓 Learning Path

1. **Start Here**: Run basic StepLR example
2. **Read Output**: Understand console output and plots
3. **Try Others**: Run different schedulers
4. **Compare**: Look at generated plots side-by-side
5. **Experiment**: Modify parameters
6. **Read Code**: Check the fully-commented code
7. **Deep Dive**: Read SCHEDULER_GUIDE.md for details

---

## ❓ Troubleshooting

**Import Error:**
```bash
pip install torch matplotlib numpy
```

**CUDA Error (optional - not critical):**
```bash
# Use CPU instead
python scheduler.py --device cpu --scheduler step
```

**Slow Training:**
- Reduce epochs: `--epochs 10`
- Reduce batch size: `--batch_size 16`
- Use smaller model: `--hidden_dim 50`

---

## 🎯 Next Steps

1. **Read README.md** - Comprehensive documentation
2. **Read SCHEDULER_GUIDE.md** - Detailed scheduler explanations
3. **Explore Code** - Every line is commented!
4. **Experiment** - Try different combinations

---

## 📖 Quick Reference

| Scheduler | Command | Best For |
|-----------|---------|----------|
| StepLR | `--scheduler step` | Classical training |
| MultiStepLR | `--scheduler multistep` | Custom milestones |
| ExponentialLR | `--scheduler exponential` | Smooth decay |
| CosineAnnealing | `--scheduler cosine` | Modern architectures |
| OneCycleLR | `--scheduler onecycle` | Fast results |
| CyclicLR | `--scheduler cyclical` | Exploration |
| ReduceLROnPlateau | `--scheduler plateau` | Adaptive/unknown |

---

**Have fun exploring learning rate schedulers! 🚀**

For detailed information, see README.md and SCHEDULER_GUIDE.md
