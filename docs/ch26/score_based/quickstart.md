# Quick Start Guide
## 04 Score-Based Methods Leading to Diffusion

### Installation

```bash
# Extract the zip file
unzip 04_Score_Based_Methods_To_Diffusion.zip
cd 04_Score_Based_Methods_To_Diffusion

# Install dependencies
pip install -r requirements.txt
```

### Running Your First Module

```bash
# Start with Module 01 - Score Functions Basics
python 01_score_functions_basics.py
```

This will:
- Explain score functions conceptually
- Generate 3 visualization plots
- Show connection to Bayesian inference
- Take ~5 minutes to complete

### Recommended Learning Sequence

**Week 1: Foundations (8-10 hours)**
```bash
python 01_score_functions_basics.py          # 2-3 hours
python 02_score_matching_theory.py           # 3-4 hours  
python 03_langevin_dynamics.py               # 2-3 hours
```

**Week 2: Implementation (10-12 hours)**
```bash
python 04_implementing_dsm_practical.py      # 3-4 hours
python 05_sliced_score_matching.py           # 2 hours
python 06_score_network_architectures.py     # 2-3 hours
```

**Week 3: Advanced (12-15 hours)**
```bash
python 07_noise_conditional_scores.py        # 3-4 hours
python 08_score_sde_framework.py             # 3-4 hours
python 09_image_generation_unet.py           # 4-5 hours
python 10_complete_diffusion_connection.py   # 2-3 hours
```

### Module Descriptions

**Beginner Level (01-03)**
- 01: What are score functions? Connection to Bayesian inference
- 02: How to learn scores from data (score matching)
- 03: How to use scores for sampling (Langevin dynamics)

**Intermediate Level (04-06)**
- 04: Practical DSM implementation with best practices
- 05: Sliced score matching for efficiency
- 06: Designing score networks with time conditioning

**Advanced Level (07-10)**
- 07: Multi-scale scores and annealed Langevin
- 08: Continuous-time SDE framework
- 09: Image generation with U-Net architecture
- 10: Complete unification with diffusion models

### What You'll Generate

Each module creates visualizations:
- `01_*.png` - Score field visualizations
- `02_*.png` - DSM training results
- `03_*.png` - Langevin sampling trajectories
- `04_*.png` - Practical implementation results
- `07_*.png` - Annealed sampling evolution

### Troubleshooting

**Import Error:**
```bash
# Make sure PyTorch is installed
pip install torch torchvision
```

**CUDA Error:**
```python
# Run on CPU (add to top of any module)
import torch
torch.set_default_device('cpu')
```

**Slow Performance:**
- Modules 01-07 run fine on CPU
- Modules 08-09 benefit from GPU
- Reduce `n_epochs` or `n_steps` for faster testing

### Next Steps

After completing all modules:
1. Read README.md for comprehensive theory
2. Try the exercises at the end of each module
3. Implement your own variations
4. Move on to full diffusion model implementation

### Getting Help

- Check inline comments (50% of each file)
- Review mathematical derivations in docstrings
- Consult README.md for connections between modules
- Each module has EXERCISES section for practice

### Prerequisites Review

Before starting, you should be comfortable with:
- Bayesian inference (Prior × Likelihood → Posterior)
- Basic calculus (gradients, chain rule)
- Python and NumPy
- PyTorch fundamentals (tensors, autograd, nn.Module)

Review `01_Bayesian_Inference` package if needed!

---

**Ready to begin? Start with `python 01_score_functions_basics.py`**

Good luck! 🚀
