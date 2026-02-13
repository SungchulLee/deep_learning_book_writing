# Quick Start Guide - Diffusion Models Tutorial

## Installation

1. **Install Python 3.8+** (if not already installed)

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision numpy matplotlib tqdm scikit-learn
```

## Three Ways to Learn

### 1. Understanding the Theory (10-15 minutes)
Start here if you're new to diffusion models:
```bash
# Read the theory guide
cat theory_guide.md
# or open in your favorite text editor
```

### 2. Quick Demo with 2D Data (5-10 minutes)
Run the simple 2D example to see diffusion in action:
```bash
python simple_diffusion.py
```

This will:
- Generate Swiss roll data
- Show the forward diffusion process
- Train a simple model
- Generate new samples
- Create visualization plots

**Output files:**
- `2d_forward_diffusion.png`: How noise is added
- `2d_training_loss.png`: Training progress
- `2d_comparison.png`: Original vs generated data

### 3. Real Image Generation with MNIST (30-60 minutes)

#### Training from Scratch
```bash
python example_train.py
```

**What happens:**
- Downloads MNIST dataset automatically
- Trains for 100 epochs (~30-60 minutes on CPU, ~5-10 minutes on GPU)
- Saves sample images every 10 epochs in `samples/` folder
- Saves final model as `mnist_diffusion_model.pt`

**Pro tip:** Start with fewer epochs for testing:
```python
# Edit example_train.py and change:
'epochs': 20,  # Instead of 100
```

#### Generating Samples from Trained Model
```bash
python example_sample.py
```

**What you get:**
- Multiple grids of generated digits
- Interpolation between different digits
- Various visualization formats

## Quick Experiments

### Experiment 1: Modify Noise Schedule
```python
# In mnist_diffusion.py or simple_diffusion.py, change:
from diffusion_utils import linear_beta_schedule
betas = linear_beta_schedule(timesteps)  # Instead of cosine
```

### Experiment 2: Change Model Size
```python
# In mnist_diffusion.py, modify MNISTDiffusion.__init__:
self.model = SimpleUNet(
    base_channels=32,  # Try 32, 64, or 128
    time_emb_dim=128   # Try 128, 256, or 512
)
```

### Experiment 3: Different Timesteps
```python
# In example_train.py:
'timesteps': 500,  # Try 100, 500, or 2000
```

## Project Structure

```
diffusion_tutorial/
├── README.md                 # Overview and contents
├── theory_guide.md          # Detailed theory explanation
├── requirements.txt         # Python dependencies
├── QUICK_START.md          # This file
│
├── diffusion_utils.py       # Core utilities (noise schedules, sampling)
├── unet_architecture.py     # Neural network architectures
│
├── simple_diffusion.py      # 2D toy example
├── mnist_diffusion.py       # Complete MNIST implementation
│
├── example_train.py         # Training script
└── example_sample.py        # Sampling script
```

## Troubleshooting

### "Out of Memory" Error
- Reduce batch size in training config
- Use CPU instead of GPU for small experiments
- Reduce model size (base_channels parameter)

### "No module named 'torch'"
```bash
pip install torch torchvision --upgrade
```

### Slow Training
- Use GPU if available (PyTorch will detect automatically)
- Reduce number of timesteps
- Use smaller model
- Reduce number of epochs for quick testing

### MNIST Download Issues
If MNIST download fails:
1. Check internet connection
2. Try again (sometimes server is busy)
3. Manually download from http://yann.lecun.com/exdb/mnist/

## Learning Path for Undergraduates

**Week 1: Foundations**
1. Read theory_guide.md
2. Run simple_diffusion.py
3. Understand the 2D visualizations
4. Modify noise schedules and observe effects

**Week 2: Implementation**
1. Study diffusion_utils.py functions
2. Understand forward and reverse processes
3. Trace through the training loop
4. Run MNIST training (start with 20 epochs)

**Week 3: Architecture**
1. Study unet_architecture.py
2. Understand time embeddings
3. Trace the U-Net forward pass
4. Experiment with model sizes

**Week 4: Experiments**
1. Train full MNIST model
2. Generate and analyze samples
3. Try different hyperparameters
4. Compare results with different settings

**Week 5: Extensions**
1. Implement conditional generation
2. Try different datasets
3. Implement DDIM sampling
4. Read recent papers

## Common Questions

**Q: How long does training take?**
A: On GPU: 5-10 minutes for 100 epochs. On CPU: 30-60 minutes.

**Q: Can I use my own images?**
A: Yes! Modify mnist_diffusion.py to load your dataset instead of MNIST.

**Q: What GPU do I need?**
A: Any CUDA-compatible GPU works. Even a modest GPU is 5-10x faster than CPU.

**Q: How do I use a pretrained model?**
A: Run example_sample.py with a checkpoint file present.

**Q: Can I train on color images?**
A: Yes! Change in_channels=3 in the model initialization.

## Next Steps

After completing this tutorial:
1. Read the DDPM paper (Ho et al., 2020)
2. Implement conditional generation (class labels)
3. Try latent diffusion models
4. Explore Stable Diffusion and DALL-E 2
5. Implement classifier-free guidance

## Resources

- Original DDPM paper: https://arxiv.org/abs/2006.11239
- Annotated diffusion tutorial: https://huggingface.co/blog/annotated-diffusion
- PyTorch documentation: https://pytorch.org/docs/

## Getting Help

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review the theory guide for concepts
3. Read code comments in the implementation files
4. Experiment with simpler versions first (2D before MNIST)

Happy learning! 🚀
