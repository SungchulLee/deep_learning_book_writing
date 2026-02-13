# Getting Started with Neural ODEs

## Quick Start Guide

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt --break-system-packages

# Key package: torchdiffeq (Neural ODE solver for PyTorch)
pip install torchdiffeq --break-system-packages
```

### 2. Your First Neural ODE (5 minutes)

```python
import torch
from torchdiffeq import odeint

# Step 1: Define ODE function (neural network)
class ODEFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 2)
        )
    
    def forward(self, t, y):
        return self.net(y)

# Step 2: Create model and solve
func = ODEFunc()
y0 = torch.randn(1, 2)  # Initial condition
t = torch.linspace(0, 1, 10)  # Time points

# Step 3: Solve ODE!
trajectory = odeint(func, y0, t)
print(f"Trajectory shape: {trajectory.shape}")  # (10, 1, 2)
```

### 3. Learning Path

**Week 1: Foundations (Level 1)**
- Day 1-2: `01_ode_basics.py` - ODE fundamentals
- Day 3-4: `02_euler_method.py` - Numerical integration  
- Day 5: `03_rk4_integration.py` - Better methods
- Day 6-7: `04_simple_neural_ode.py` - First Neural ODE!

**Week 2: Core Concepts (Level 2)**
- Day 1-2: `05_adjoint_method.py` - Memory efficiency
- Day 3-4: `06_ode_blocks.py` - Architecture building blocks
- Day 5: `07_classification_neural_ode.py` - Applications
- Day 6-7: `08_time_series_neural_ode.py` - Sequential data

**Week 3: Advanced Topics (Level 3)**
- Day 1-3: `09_continuous_normalizing_flows.py` - Generative models
- Day 4-5: `10_augmented_neural_ode.py` - Expressivity
- Day 6-7: `11_latent_ode.py` + `12_stochastic_differential_equations.py`

### 4. Running Examples

```bash
# Navigate to package directory
cd neural_ode_package

# Run a module
python level1_beginner/01_ode_basics.py

# Run with GPU (if available)
python level2_intermediate/07_classification_neural_ode.py
```

### 5. Common Patterns

#### Pattern 1: Training a Neural ODE

```python
import torch
from torchdiffeq import odeint_adjoint  # Memory-efficient!

# Define model
class NeuralODE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = ODEFunc()  # Your ODE function
    
    def forward(self, y0, t):
        return odeint_adjoint(self.func, y0, t, method='dopri5')

# Training loop
model = NeuralODE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(y0, t)
    loss = loss_fn(pred, target)
    loss.backward()  # Adjoint method handles gradients!
    optimizer.step()
```

#### Pattern 2: Using Different Solvers

```python
# Fast but less accurate
trajectory = odeint(func, y0, t, method='euler')

# Standard choice (adaptive Runge-Kutta)
trajectory = odeint(func, y0, t, method='dopri5')

# For stiff equations
trajectory = odeint(func, y0, t, method='adams')
```

#### Pattern 3: Controlling Solver Tolerance

```python
# Higher accuracy (slower)
trajectory = odeint(func, y0, t, method='dopri5', 
                   rtol=1e-7, atol=1e-9)

# Faster (less accurate)
trajectory = odeint(func, y0, t, method='dopri5',
                   rtol=1e-3, atol=1e-5)
```

### 6. Troubleshooting

**Problem: "Memory error during training"**
```python
# Solution: Use adjoint method
from torchdiffeq import odeint_adjoint  # instead of odeint
```

**Problem: "Training is very slow"**
```python
# Solutions:
# 1. Relax tolerance
odeint(func, y0, t, rtol=1e-3, atol=1e-4)  

# 2. Use simpler solver
odeint(func, y0, t, method='euler')

# 3. Reduce hidden dimensions in network
```

**Problem: "Numerical instability / NaN values"**
```python
# Solutions:
# 1. Smaller learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 2. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Better initialization
for m in model.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.1)
```

### 7. Key Concepts Checklist

After completing this package, you should understand:

- [ ] What ODEs are and why they matter for deep learning
- [ ] Numerical integration methods (Euler, RK4)
- [ ] How Neural ODEs connect to ResNets
- [ ] Adjoint sensitivity method for memory efficiency
- [ ] Continuous Normalizing Flows for generative modeling
- [ ] When to use Neural ODEs vs standard networks
- [ ] Trade-offs: flexibility vs computational cost

### 8. Next Steps

**Deepen Understanding:**
- Read original paper: Chen et al. "Neural ODEs" (NeurIPS 2018)
- Explore FFJORD: Grathwohl et al. (2019)
- Study Latent ODEs: Rubanova et al. (2019)

**Apply to Projects:**
- Time series forecasting with irregular sampling
- Generative modeling with CNFs
- Physics-informed neural networks
- Continuous control in reinforcement learning

**Explore Extensions:**
- Neural CDEs (Controlled Differential Equations)
- Neural SDEs (Stochastic Differential Equations)
- Second-order Neural ODEs
- Hamiltonian Neural Networks

### 9. Resources

**Documentation:**
- torchdiffeq: https://github.com/rtqichen/torchdiffeq
- PyTorch: https://pytorch.org/docs/

**Papers:**
- Neural ODEs (2018): https://arxiv.org/abs/1806.07366
- FFJORD (2019): https://arxiv.org/abs/1810.01367
- Augmented ODEs (2019): https://arxiv.org/abs/1904.01681

**Community:**
- GitHub discussions
- PyTorch forums
- ML research conferences (NeurIPS, ICML, ICLR)

### 10. Contact & Support

This is an educational package designed for self-paced learning.

For issues with the code:
- Check the module comments (heavily documented!)
- Review error messages and traceback
- Verify environment setup (Python 3.8+, PyTorch 2.0+)

For conceptual questions:
- Re-read the mathematical derivations in comments
- Visualize the results (all modules generate plots!)
- Review the KEY TAKEAWAYS sections

---

**Happy Learning! 🚀**

*"The future of AI is continuous, not discrete."*
