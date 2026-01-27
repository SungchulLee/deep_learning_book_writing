# Neural ODE Applications

## Learning Objectives

By the end of this section, you will:

- Apply Neural ODEs to time series forecasting and interpolation
- Implement Neural ODEs for irregular time series with missing data
- Use Neural ODEs for physical system modeling
- Build latent ODE models for sequential data
- Understand domain-specific considerations and best practices

## Prerequisites

- Neural ODE fundamentals and training
- Adjoint sensitivity method
- Time series basics
- Sequence modeling (RNN/LSTM concepts)

---

## 1. Time Series Modeling

### 1.1 Why Neural ODEs for Time Series?

Traditional sequence models (RNNs, LSTMs) operate on discrete time steps:

$$h_{t+1} = f(h_t, x_t)$$

**Limitations:**
- Fixed time discretization
- Struggle with irregular sampling
- No natural handling of missing data
- Cannot interpolate between observations

**Neural ODE advantage:** Model the underlying continuous dynamics:

$$\frac{dh}{dt} = f_\theta(h(t), t)$$

This naturally handles:
- Arbitrary observation times
- Variable-length gaps
- Missing data
- Continuous interpolation

### 1.2 Basic Time Series Neural ODE

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class TimeSeriesNeuralODE(nn.Module):
    """
    Neural ODE for regularly-sampled time series.
    
    Learns continuous dynamics from sequential observations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        # Encoder: observations -> initial hidden state
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Dynamics: dh/dt = f(h, t)
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim * 2),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Decoder: hidden state -> predictions
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.hidden_dim = hidden_dim
    
    def ode_func(self, t, h):
        """ODE dynamics."""
        batch_size = h.shape[0]
        t_vec = t.expand(batch_size, 1)
        th = torch.cat([h, t_vec], dim=-1)
        return self.dynamics(th)
    
    def forward(self, x0, t_eval):
        """
        Forward pass for time series prediction.
        
        Args:
            x0: Initial observation (batch, input_dim)
            t_eval: Times at which to predict (n_times,)
        
        Returns:
            predictions: (n_times, batch, output_dim)
        """
        # Encode initial observation
        h0 = self.encoder(x0)
        
        # Evolve hidden state through time
        h_trajectory = odeint(self.ode_func, h0, t_eval, method='dopri5')
        
        # Decode to predictions
        predictions = self.decoder(h_trajectory)
        
        return predictions


def train_time_series_node():
    """
    Example: Learn sine wave dynamics.
    """
    # Generate synthetic data
    t = torch.linspace(0, 4 * np.pi, 100)
    true_series = torch.sin(t)
    
    # Add noise
    noisy_series = true_series + 0.1 * torch.randn_like(true_series)
    
    # Create model
    model = TimeSeriesNeuralODE(input_dim=1, hidden_dim=32, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # Training: predict future from initial condition
    n_epochs = 500
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Use first point as initial condition
        x0 = noisy_series[0:1].unsqueeze(0)  # (1, 1)
        
        # Predict all time points
        predictions = model(x0, t)  # (100, 1, 1)
        predictions = predictions.squeeze()
        
        # Loss: MSE on observed points
        loss = nn.functional.mse_loss(predictions, noisy_series)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    return model, t, true_series, noisy_series
```

---

## 2. Irregular Time Series

### 2.1 The Irregular Sampling Challenge

Real-world time series often have:
- Non-uniform sampling intervals
- Missing observations
- Event-driven measurements

Standard approaches (padding, interpolation) lose information or introduce artifacts.

### 2.2 ODE-RNN: Handling Irregular Data

**ODE-RNN** (Rubanova et al., 2019) combines:
- Neural ODE for continuous evolution between observations
- RNN updates when new data arrives

```python
class ODERNN(nn.Module):
    """
    ODE-RNN for irregularly-sampled time series.
    
    Between observations: evolve state via ODE
    At observations: update state via GRU
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # ODE dynamics between observations
        self.ode_func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # GRU cell for incorporating observations
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        
        # Decoder
        self.decoder = nn.Linear(hidden_dim, output_dim)
    
    def ode_dynamics(self, t, h):
        return self.ode_func(h)
    
    def forward(self, observations, times, masks=None):
        """
        Process irregularly-sampled time series.
        
        Args:
            observations: (batch, seq_len, input_dim)
            times: (batch, seq_len) - observation times
            masks: (batch, seq_len) - 1 if observed, 0 if missing
        
        Returns:
            hidden_states: (batch, seq_len, hidden_dim)
            predictions: (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = observations.shape
        device = observations.device
        
        if masks is None:
            masks = torch.ones(batch_size, seq_len, device=device)
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        hidden_states = []
        predictions = []
        
        for i in range(seq_len):
            if i > 0:
                # Evolve hidden state from previous time to current time
                dt = times[:, i] - times[:, i-1]
                
                # For each sample, integrate ODE for appropriate duration
                # Simplified: use average dt (proper implementation handles per-sample)
                t_span = torch.tensor([0., dt.mean().item()])
                h = odeint(self.ode_dynamics, h, t_span, method='euler',
                          options={'step_size': 0.1})[-1]
            
            # Update with observation (if present)
            obs_i = observations[:, i]
            mask_i = masks[:, i].unsqueeze(-1)
            
            # GRU update where we have observations
            h_updated = self.gru_cell(obs_i, h)
            h = mask_i * h_updated + (1 - mask_i) * h
            
            hidden_states.append(h)
            predictions.append(self.decoder(h))
        
        hidden_states = torch.stack(hidden_states, dim=1)
        predictions = torch.stack(predictions, dim=1)
        
        return hidden_states, predictions


def demo_irregular_time_series():
    """
    Demonstrate ODE-RNN on irregular data.
    """
    batch_size = 32
    seq_len = 50
    input_dim = 2
    hidden_dim = 64
    output_dim = 2
    
    # Generate irregular time series
    # Random time gaps
    dt = torch.rand(batch_size, seq_len) * 0.5 + 0.1  # gaps between 0.1 and 0.6
    times = torch.cumsum(dt, dim=1)
    
    # Observations: noisy spiral
    true_traj = torch.stack([
        torch.sin(times),
        torch.cos(times)
    ], dim=-1)
    observations = true_traj + 0.1 * torch.randn_like(true_traj)
    
    # Random missing data (30% missing)
    masks = (torch.rand(batch_size, seq_len) > 0.3).float()
    
    # Model
    model = ODERNN(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    for epoch in range(100):
        optimizer.zero_grad()
        
        _, predictions = model(observations, times, masks)
        
        # Loss only on observed points
        loss = (masks.unsqueeze(-1) * (predictions - observations)**2).sum()
        loss = loss / masks.sum()
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    return model
```

---

## 3. Latent ODE Models

### 3.1 Motivation

Sometimes we don't observe the state directly—only noisy measurements. **Latent ODEs** model:

- Latent state evolves continuously: $\frac{dz}{dt} = f_\theta(z)$
- Observations are generated from latent state: $x_t \sim p(x|z(t))$

This is a continuous-time version of state-space models.

### 3.2 Latent ODE Architecture

```python
class LatentODE(nn.Module):
    """
    Latent ODE for sequential data with latent dynamics.
    
    Architecture:
    1. Encoder: observations -> latent initial state distribution
    2. ODE: evolve latent state continuously
    3. Decoder: latent state -> observation distribution
    """
    
    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Recognition network (encoder)
        # Process observations backwards to get q(z_0 | x_{1:T})
        self.encoder_rnn = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        
        # Latent initial state distribution parameters
        self.z0_mean = nn.Linear(hidden_dim, latent_dim)
        self.z0_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Latent ODE dynamics
        self.ode_func = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder: z -> x distribution
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def encode(self, x, t):
        """
        Encode observations to latent initial state distribution.
        
        Args:
            x: (batch, seq_len, obs_dim)
            t: (batch, seq_len) observation times
        
        Returns:
            z0_mean, z0_logvar: (batch, latent_dim)
        """
        # Reverse sequence for backward processing
        x_reversed = torch.flip(x, dims=[1])
        
        # RNN encoding
        _, h_n = self.encoder_rnn(x_reversed)
        h_n = h_n.squeeze(0)  # (batch, hidden_dim)
        
        # Latent distribution parameters
        z0_mean = self.z0_mean(h_n)
        z0_logvar = self.z0_logvar(h_n)
        
        return z0_mean, z0_logvar
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        """Decode latent state to observation."""
        return self.decoder(z)
    
    def latent_ode(self, t, z):
        """Latent space dynamics."""
        return self.ode_func(z)
    
    def forward(self, x, t):
        """
        Full forward pass.
        
        Args:
            x: observations (batch, seq_len, obs_dim)
            t: times (seq_len,)
        
        Returns:
            x_recon: reconstructed observations
            z0_mean, z0_logvar: for KL divergence
        """
        batch_size = x.shape[0]
        
        # Encode
        z0_mean, z0_logvar = self.encode(x, t)
        
        # Sample initial latent state
        z0 = self.reparameterize(z0_mean, z0_logvar)
        
        # Solve latent ODE
        z_trajectory = odeint(self.latent_ode, z0, t, method='dopri5')
        # Shape: (seq_len, batch, latent_dim)
        
        # Decode
        x_recon = self.decode(z_trajectory)
        # Shape: (seq_len, batch, obs_dim)
        
        # Transpose to (batch, seq_len, obs_dim)
        x_recon = x_recon.transpose(0, 1)
        
        return x_recon, z0_mean, z0_logvar
    
    def loss(self, x, t, beta=1.0):
        """
        ELBO loss = reconstruction + KL divergence.
        """
        x_recon, z0_mean, z0_logvar = self.forward(x, t)
        
        # Reconstruction loss (Gaussian likelihood)
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        recon_loss = recon_loss / x.shape[0]  # Average over batch
        
        # KL divergence: KL(q(z0|x) || p(z0))
        # p(z0) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + z0_logvar - z0_mean**2 - z0_logvar.exp())
        kl_loss = kl_loss / x.shape[0]
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def sample(self, n_samples, t):
        """
        Generate new sequences by sampling from prior.
        """
        device = next(self.parameters()).device
        
        # Sample z0 from prior
        z0 = torch.randn(n_samples, self.latent_dim, device=device)
        
        # Solve ODE
        z_trajectory = odeint(self.latent_ode, z0, t, method='dopri5')
        
        # Decode
        x_samples = self.decode(z_trajectory).transpose(0, 1)
        
        return x_samples
```

---

## 4. Physical System Modeling

### 4.1 Physics-Informed Neural ODEs

For physical systems, we can incorporate domain knowledge:

$$\frac{dq}{dt} = v$$
$$\frac{dv}{dt} = f_{NN}(q, v)$$

The neural network only learns the forces, while kinematics are encoded.

```python
class PhysicsInformedODE(nn.Module):
    """
    Neural ODE with physics-based structure.
    
    Models second-order dynamics: d²q/dt² = f(q, v)
    where f is learned by a neural network.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.state_dim = state_dim
        
        # Neural network learns forces/accelerations
        self.force_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, t, state):
        """
        State: [position, velocity]
        Dynamics: dq/dt = v, dv/dt = f(q, v)
        """
        q = state[..., :self.state_dim]  # Position
        v = state[..., self.state_dim:]  # Velocity
        
        # Position changes according to velocity (physics)
        dqdt = v
        
        # Acceleration learned from network
        qv = torch.cat([q, v], dim=-1)
        dvdt = self.force_net(qv)
        
        return torch.cat([dqdt, dvdt], dim=-1)


class HamiltonianNeuralODE(nn.Module):
    """
    Hamiltonian Neural ODE: learns energy-conserving dynamics.
    
    Hamilton's equations:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q
    
    The network learns the Hamiltonian H(q, p), and dynamics
    are derived from it.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.dim = dim
        
        # Network outputs scalar Hamiltonian
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1)
        )
    
    def hamiltonian(self, q, p):
        """Compute Hamiltonian H(q, p)."""
        qp = torch.cat([q, p], dim=-1)
        return self.hamiltonian_net(qp)
    
    def forward(self, t, state):
        """
        Compute Hamiltonian dynamics.
        """
        q = state[..., :self.dim]
        p = state[..., self.dim:]
        
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        
        H = self.hamiltonian(q, p)
        
        # Hamilton's equations via autograd
        dHdq = torch.autograd.grad(H.sum(), q, create_graph=True)[0]
        dHdp = torch.autograd.grad(H.sum(), p, create_graph=True)[0]
        
        dqdt = dHdp   # ∂H/∂p
        dpdt = -dHdq  # -∂H/∂q
        
        return torch.cat([dqdt, dpdt], dim=-1)


def demo_pendulum():
    """
    Learn pendulum dynamics with Hamiltonian Neural ODE.
    """
    import numpy as np
    
    # True pendulum dynamics
    def true_pendulum(t, state, g=9.81, L=1.0):
        theta, omega = state[..., 0], state[..., 1]
        dtheta = omega
        domega = -(g/L) * torch.sin(theta)
        return torch.stack([dtheta, domega], dim=-1)
    
    # Generate training data
    n_trajectories = 100
    t = torch.linspace(0, 2, 50)
    
    # Random initial conditions
    theta0 = torch.randn(n_trajectories, 1) * 0.5
    omega0 = torch.randn(n_trajectories, 1) * 0.5
    state0 = torch.cat([theta0, omega0], dim=-1)
    
    # Generate trajectories
    with torch.no_grad():
        true_trajectories = odeint(true_pendulum, state0, t)
    # Add noise
    noisy_trajectories = true_trajectories + 0.02 * torch.randn_like(true_trajectories)
    
    # Train Hamiltonian Neural ODE
    model = HamiltonianNeuralODE(dim=1, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(500):
        optimizer.zero_grad()
        
        # Predict from initial conditions
        pred_trajectories = odeint(model, state0, t)
        
        loss = nn.functional.mse_loss(pred_trajectories, noisy_trajectories)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    # Test energy conservation
    with torch.no_grad():
        test_state0 = torch.tensor([[0.5, 0.0]])
        test_traj = odeint(model, test_state0, torch.linspace(0, 10, 200))
        
        q = test_traj[..., 0]
        p = test_traj[..., 1]
        
        energies = []
        for i in range(len(test_traj)):
            H = model.hamiltonian(q[i:i+1], p[i:i+1])
            energies.append(H.item())
    
    energy_variation = max(energies) - min(energies)
    print(f"Energy variation: {energy_variation:.6f}")
    print("(Should be close to 0 for good energy conservation)")
    
    return model
```

---

## 5. Best Practices and Tips

### 5.1 Architecture Design

```python
# Good: Smooth activations for stable ODE dynamics
dynamics = nn.Sequential(
    nn.Linear(dim, hidden),
    nn.Tanh(),  # or Softplus, GELU
    nn.Linear(hidden, dim)
)

# Avoid: Discontinuous activations
dynamics = nn.Sequential(
    nn.Linear(dim, hidden),
    nn.ReLU(),  # Can cause issues
    nn.Linear(hidden, dim)
)
```

### 5.2 Training Stability

```python
# Gradient clipping is important
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Use learning rate warmup
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=100
)

# Consider regularization
kinetic_reg = 0.01 * (dynamics_output ** 2).mean()
loss = task_loss + kinetic_reg
```

### 5.3 Evaluation

```python
# Use tighter tolerances for evaluation
def evaluate(model, x, t):
    model.eval()
    with torch.no_grad():
        # Tighter tolerances than training
        pred = odeint(model.dynamics, x, t, 
                     rtol=1e-7, atol=1e-9,
                     method='dopri5')
    return pred
```

---

## 6. Key Takeaways

1. **Neural ODEs naturally handle irregular time series** by modeling continuous-time dynamics.

2. **ODE-RNN combines continuous evolution with discrete updates** for sequential data with observations.

3. **Latent ODEs model hidden dynamics** when observations are noisy or partial.

4. **Physics-informed Neural ODEs** incorporate domain knowledge (kinematics, energy conservation).

5. **Hamiltonian Neural ODEs** automatically conserve energy, suitable for physical systems.

---

## 7. Exercises

### Exercise 1: Weather Forecasting

Implement a Neural ODE for weather prediction using irregularly sampled sensor data. Handle missing measurements gracefully.

### Exercise 2: Bouncing Ball

Extend the physics-informed ODE to handle discontinuous dynamics (collisions) by combining ODE integration with event detection.

### Exercise 3: Latent Video Dynamics

Build a Latent ODE for video prediction where the latent space captures motion dynamics and a decoder (CNN) generates frames.

---

## References

1. Rubanova, Y., Chen, R. T. Q., & Duvenaud, D. (2019). Latent ODEs for Irregularly-Sampled Time Series. *NeurIPS*.

2. Greydanus, S., Dzamba, M., & Yosinski, J. (2019). Hamiltonian Neural Networks. *NeurIPS*.

3. Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020). Neural Controlled Differential Equations for Irregular Time Series. *NeurIPS*.

4. Chen, R. T. Q., Amos, B., & Nickel, M. (2021). Learning Neural Event Functions for Ordinary Differential Equations. *ICLR*.
