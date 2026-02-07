# Continuous Dynamics in Finance

## Learning Objectives

By the end of this section, you will:

- Apply Neural ODEs to continuous-time financial modeling
- Implement models for irregular time series from tick data and event-driven markets
- Build physics-informed Neural ODEs that encode financial structure
- Use latent ODE models for hidden state estimation in financial systems
- Understand Hamiltonian Neural ODEs for energy-conserving dynamics

## Prerequisites

- Neural ODE fundamentals, adjoint method, and `torchdiffeq` (Section 27.1)
- Continuous normalizing flows (Section 27.2)
- Time series basics and sequence modeling concepts
- Familiarity with financial instruments (helpful but not required)

---

## 1. Why Neural ODEs for Finance?

### 1.1 Finance as Continuous Dynamics

Financial systems are inherently continuous-time. Classical quantitative finance models this explicitly:

- **Geometric Brownian Motion:** $dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$
- **Vasicek interest rates:** $dr_t = \kappa(\theta - r_t) \, dt + \sigma \, dW_t$
- **Heston stochastic volatility:** $dv_t = \kappa(\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW_t$

Neural ODEs provide a natural framework for learning these dynamics from data, replacing hand-specified drift and diffusion with learned functions while preserving the continuous-time structure.

### 1.2 Advantages Over Discrete Models

Traditional sequence models (RNNs, LSTMs, Transformers) operate on fixed discrete time steps. Financial data violates this assumption:

- **Tick data** arrives at irregular intervals (milliseconds to minutes)
- **Corporate events** (earnings, dividends) occur on scheduled but non-uniform dates
- **Market microstructure** effects operate on multiple time scales simultaneously
- **Missing data** from holidays, trading halts, and thin markets

Neural ODEs handle all of these naturally because the underlying dynamics are continuous—observations can occur at any time, and the model interpolates consistently between them.

---

## 2. Time Series Modeling

### 2.1 Basic Financial Time Series Neural ODE

```python
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint

class FinancialTimeSeriesODE(nn.Module):
    """
    Neural ODE for continuous-time financial time series.
    
    Learns the drift function of the underlying dynamics from
    sequential observations of prices or returns.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        # Encoder: observations → initial hidden state
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
        
        # Decoder: hidden state → predictions
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
        h0 = self.encoder(x0)
        h_trajectory = odeint(self.ode_func, h0, t_eval, method='dopri5')
        predictions = self.decoder(h_trajectory)
        return predictions
```

### 2.2 Training on Financial Data

```python
def train_financial_node(model, prices, timestamps, n_epochs=500):
    """
    Train Neural ODE on financial time series.
    
    Args:
        model: FinancialTimeSeriesODE instance
        prices: Price series tensor (seq_len, n_features)
        timestamps: Observation times (seq_len,)
        n_epochs: Training epochs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Normalize timestamps to [0, 1]
    t_norm = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
    
    # Use log-returns as features
    log_returns = torch.log(prices[1:] / prices[:-1])
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Initial condition from first observation
        x0 = log_returns[0:1].unsqueeze(0)  # (1, input_dim)
        
        # Predict all subsequent returns
        predictions = model(x0, t_norm[1:])  # (seq_len-1, 1, output_dim)
        predictions = predictions.squeeze(1)
        
        # MSE loss on observed returns
        loss = nn.functional.mse_loss(predictions, log_returns)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    return model
```

---

## 3. Irregular Time Series: ODE-RNN for Tick Data

### 3.1 The Irregular Sampling Challenge

Real financial data exhibits severe irregularity:

- Tick-by-tick data arrives at random intervals
- Trading activity varies dramatically between market open and close
- Different instruments have different observation frequencies
- Market closures, holidays, and trading halts create large gaps

### 3.2 ODE-RNN for Financial Data

The **ODE-RNN** (Rubanova et al., 2019) addresses this by combining continuous evolution between observations with discrete updates at observation times:

```python
class FinancialODERNN(nn.Module):
    """
    ODE-RNN for irregularly-sampled financial time series.
    
    Between observations: evolve hidden state via Neural ODE
    At observations: update hidden state via GRU cell
    
    This handles tick data, event-driven observations, and
    arbitrary time gaps naturally.
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
        
        # GRU cell for incorporating new observations
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        
        # Decoder for predictions
        self.decoder = nn.Linear(hidden_dim, output_dim)
    
    def ode_dynamics(self, t, h):
        return self.ode_func(h)
    
    def forward(self, observations, times, masks=None):
        """
        Process irregularly-sampled financial time series.
        
        Args:
            observations: (batch, seq_len, input_dim)
            times: (batch, seq_len) - observation timestamps
            masks: (batch, seq_len) - 1 if observed, 0 if missing
        
        Returns:
            hidden_states: (batch, seq_len, hidden_dim)
            predictions: (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = observations.shape
        device = observations.device
        
        if masks is None:
            masks = torch.ones(batch_size, seq_len, device=device)
        
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        hidden_states = []
        predictions = []
        
        for i in range(seq_len):
            if i > 0:
                # Evolve hidden state from previous time to current time
                dt = times[:, i] - times[:, i-1]
                t_span = torch.tensor([0., dt.mean().item()])
                h = odeint(self.ode_dynamics, h, t_span, 
                          method='euler',
                          options={'step_size': 0.1})[-1]
            
            # Update with observation (if present)
            obs_i = observations[:, i]
            mask_i = masks[:, i].unsqueeze(-1)
            
            h_updated = self.gru_cell(obs_i, h)
            h = mask_i * h_updated + (1 - mask_i) * h
            
            hidden_states.append(h)
            predictions.append(self.decoder(h))
        
        hidden_states = torch.stack(hidden_states, dim=1)
        predictions = torch.stack(predictions, dim=1)
        
        return hidden_states, predictions


def demo_tick_data():
    """
    Demonstrate ODE-RNN on simulated tick data.
    """
    batch_size = 32
    seq_len = 50
    input_dim = 3   # price, volume, spread
    hidden_dim = 64
    output_dim = 1   # next return prediction
    
    # Simulate irregular tick arrivals
    dt = torch.rand(batch_size, seq_len) * 0.5 + 0.1
    times = torch.cumsum(dt, dim=1)
    
    # Simulated tick features
    observations = torch.randn(batch_size, seq_len, input_dim) * 0.01
    
    # Random missing data (30% of ticks have missing features)
    masks = (torch.rand(batch_size, seq_len) > 0.3).float()
    
    model = FinancialODERNN(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(100):
        optimizer.zero_grad()
        _, predictions = model(observations, times, masks)
        
        # Loss only on observed points
        loss = (masks.unsqueeze(-1) * (predictions - observations[..., :1])**2).sum()
        loss = loss / masks.sum()
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    return model
```

---

## 4. Latent ODE Models for Financial Systems

### 4.1 Motivation

In many financial settings, we observe noisy measurements of an underlying latent state:

- **Factor models:** Observed returns are driven by unobserved factors
- **Credit risk:** Default intensity is a latent process observed through spread movements
- **Market microstructure:** True price is latent, observed prices include microstructure noise

**Latent ODEs** model this by learning continuous dynamics in latent space with a separate observation model.

### 4.2 Financial Latent ODE

```python
class FinancialLatentODE(nn.Module):
    """
    Latent ODE for financial systems with hidden dynamics.
    
    Architecture:
    1. Encoder: observations → latent initial state distribution
    2. Latent ODE: evolve latent factors continuously
    3. Decoder: latent factors → observable quantities
    
    This is a continuous-time analog of state-space models,
    suitable for factor modeling and hidden state estimation.
    """
    
    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Recognition network (encoder)
        self.encoder_rnn = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        
        # Latent initial state distribution
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
        
        # Decoder: latent → observables
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def encode(self, x, t):
        """Encode observations to latent initial state distribution."""
        x_reversed = torch.flip(x, dims=[1])
        _, h_n = self.encoder_rnn(x_reversed)
        h_n = h_n.squeeze(0)
        
        z0_mean = self.z0_mean(h_n)
        z0_logvar = self.z0_logvar(h_n)
        
        return z0_mean, z0_logvar
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        """Decode latent state to observables."""
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
            x_recon, z0_mean, z0_logvar
        """
        z0_mean, z0_logvar = self.encode(x, t)
        z0 = self.reparameterize(z0_mean, z0_logvar)
        
        # Solve latent ODE
        z_trajectory = odeint(self.latent_ode, z0, t, method='dopri5')
        
        # Decode
        x_recon = self.decode(z_trajectory).transpose(0, 1)
        
        return x_recon, z0_mean, z0_logvar
    
    def loss(self, x, t, beta=1.0):
        """ELBO loss = reconstruction + KL divergence."""
        x_recon, z0_mean, z0_logvar = self.forward(x, t)
        
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        recon_loss = recon_loss / x.shape[0]
        
        # KL divergence: KL(q(z0|x) || p(z0)), p(z0) = N(0, I)
        kl_loss = -0.5 * torch.sum(
            1 + z0_logvar - z0_mean**2 - z0_logvar.exp()
        )
        kl_loss = kl_loss / x.shape[0]
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def sample(self, n_samples, t):
        """Generate new trajectories from the prior."""
        device = next(self.parameters()).device
        z0 = torch.randn(n_samples, self.latent_dim, device=device)
        
        z_trajectory = odeint(self.latent_ode, z0, t, method='dopri5')
        x_samples = self.decode(z_trajectory).transpose(0, 1)
        
        return x_samples
```

---

## 5. Physics-Informed Financial Neural ODEs

### 5.1 Encoding Financial Structure

For financial systems, we can incorporate domain knowledge into the Neural ODE architecture. Classical finance provides strong structural priors:

- **Mean reversion:** Rates and spreads revert to long-run levels
- **No-arbitrage:** Risk-neutral drift is constrained by the risk-free rate
- **Volatility clustering:** Squared returns are autocorrelated
- **Second-order dynamics:** Position (price) and velocity (momentum) are coupled

```python
class MeanRevertingODE(nn.Module):
    """
    Neural ODE with mean-reverting structure.
    
    Encodes the Ornstein-Uhlenbeck structure:
        dx/dt = κ(θ - x) + f_NN(x, t)
    
    where κ, θ are learned mean-reversion parameters
    and f_NN captures deviations from simple mean-reversion.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.dim = dim
        
        # Learnable mean-reversion parameters
        self.kappa = nn.Parameter(torch.ones(dim) * 0.5)   # Speed
        self.theta = nn.Parameter(torch.zeros(dim))          # Level
        
        # Neural network for residual dynamics
        self.residual_net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Initialize residual to be small
        for m in self.residual_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, x):
        """
        dx/dt = κ(θ - x) + f_NN(x, t)
        
        The mean-reversion term provides the structural backbone,
        while the neural network captures nonlinear deviations.
        """
        # Structural mean-reversion
        mean_reversion = torch.relu(self.kappa) * (self.theta - x)
        
        # Neural network residual
        batch_size = x.shape[0]
        t_vec = t.expand(batch_size, 1)
        xt = torch.cat([x, t_vec], dim=-1)
        residual = self.residual_net(xt)
        
        return mean_reversion + residual
```

### 5.2 Hamiltonian Neural ODE for Market Dynamics

Financial markets exhibit approximate conservation laws (e.g., total market value tends to be preserved in the short term through substitution effects). Hamiltonian Neural ODEs can capture this:

```python
class FinancialHamiltonianODE(nn.Module):
    """
    Hamiltonian Neural ODE for market dynamics.
    
    Models the market as a Hamiltonian system where:
    - q: "position" (log-prices or factor levels)
    - p: "momentum" (order flow, sentiment, or momentum factors)
    
    Hamilton's equations:
        dq/dt = ∂H/∂p   (prices respond to momentum)
        dp/dt = -∂H/∂q  (momentum responds to price levels)
    
    The Hamiltonian H(q, p) is learned, automatically
    preserving the symplectic structure of the dynamics.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.dim = dim
        
        # Network outputs scalar Hamiltonian H(q, p)
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
        """Compute Hamiltonian dynamics via autograd."""
        q = state[..., :self.dim]
        p = state[..., self.dim:]
        
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        
        H = self.hamiltonian(q, p)
        
        dHdq = torch.autograd.grad(H.sum(), q, create_graph=True)[0]
        dHdp = torch.autograd.grad(H.sum(), p, create_graph=True)[0]
        
        dqdt = dHdp    # ∂H/∂p
        dpdt = -dHdq   # -∂H/∂q
        
        return torch.cat([dqdt, dpdt], dim=-1)
```

### 5.3 Second-Order Financial Dynamics

Many financial quantities have natural second-order structure (price and momentum):

```python
class SecondOrderFinancialODE(nn.Module):
    """
    Neural ODE with second-order structure for financial dynamics.
    
    Models: d²x/dt² = f_NN(x, dx/dt)
    As system: dx/dt = v, dv/dt = f_NN(x, v)
    
    The network only learns the "acceleration" (forces acting on prices),
    while kinematics (velocity → position) are encoded structurally.
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
        q = state[..., :self.state_dim]   # Price levels
        v = state[..., self.state_dim:]   # Momentum
        
        dqdt = v  # Position changes according to velocity (structural)
        
        qv = torch.cat([q, v], dim=-1)
        dvdt = self.force_net(qv)  # Acceleration learned
        
        return torch.cat([dqdt, dvdt], dim=-1)
```

---

## 6. Practical Considerations

### 6.1 Architecture Design for Finance

```python
# Good: Smooth activations for stable dynamics
dynamics = nn.Sequential(
    nn.Linear(dim, hidden),
    nn.Tanh(),       # Bounded, Lipschitz
    nn.Linear(hidden, dim)
)

# Also good: Softplus for non-negative quantities (volatility, intensity)
vol_dynamics = nn.Sequential(
    nn.Linear(dim, hidden),
    nn.Softplus(),   # Smooth, non-negative
    nn.Linear(hidden, dim)
)

# Avoid: ReLU in ODE dynamics (non-smooth, can violate Lipschitz)
bad_dynamics = nn.Sequential(
    nn.Linear(dim, hidden),
    nn.ReLU(),       # Problematic for ODE theory
    nn.Linear(hidden, dim)
)
```

### 6.2 Training Stability

```python
# Gradient clipping is essential for financial data
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Learning rate warmup prevents early instability
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=100
)

# Kinetic energy regularization encourages smooth dynamics
kinetic_reg = 0.01 * (dynamics_output ** 2).mean()
loss = task_loss + kinetic_reg
```

### 6.3 Evaluation with Tight Tolerances

```python
def evaluate_financial_model(model, x, t):
    """Use tighter tolerances for financial predictions."""
    model.eval()
    with torch.no_grad():
        pred = odeint(model.dynamics, x, t, 
                     rtol=1e-7, atol=1e-9,
                     method='dopri5')
    return pred
```

---

## 7. Key Takeaways

1. **Neural ODEs naturally model continuous-time financial dynamics**, replacing hand-specified drift functions with learned networks while preserving continuous-time structure.

2. **ODE-RNN handles irregular financial data** (tick data, event-driven observations, missing values) by combining continuous state evolution with discrete observation updates.

3. **Latent ODEs estimate hidden financial states** (latent factors, unobserved risk processes) from noisy observations, providing a continuous-time analog of state-space models.

4. **Physics-informed architectures encode financial structure** such as mean reversion, momentum dynamics, and energy conservation, improving both sample efficiency and interpretability.

5. **Hamiltonian Neural ODEs** automatically preserve symplectic structure, making them suitable for modeling conservative aspects of market dynamics.

---

## 8. Exercises

### Exercise 1: Interest Rate Modeling

Implement a mean-reverting Neural ODE to model the Fed Funds rate. Compare the learned $\kappa$ and $\theta$ parameters with estimates from a standard Vasicek model calibration. Does the neural network residual capture nonlinearities that Vasicek misses?

### Exercise 2: Tick Data Interpolation

Build an ODE-RNN for tick-by-tick equity data. Train on irregularly sampled mid-prices and use the model to interpolate prices at regular 1-second intervals. Evaluate against linear interpolation and previous-tick methods.

### Exercise 3: Latent Factor Discovery

Train a Latent ODE on a panel of equity returns with `latent_dim=3`. Interpret the learned latent factors by correlating them with known factors (market, size, value). Does the continuous-time formulation discover different factor dynamics than a discrete VAR model?

### Exercise 4: Energy-Conserving Portfolio Dynamics

Model the dynamics of a portfolio's sector allocations using a Hamiltonian Neural ODE. Verify that the learned Hamiltonian approximately conserves a quantity interpretable as "total portfolio energy." How does this constraint affect out-of-sample prediction accuracy?

---

## References

1. Rubanova, Y., Chen, R. T. Q., & Duvenaud, D. (2019). Latent ODEs for Irregularly-Sampled Time Series. *NeurIPS*.
2. Greydanus, S., Dzamba, M., & Yosinski, J. (2019). Hamiltonian Neural Networks. *NeurIPS*.
3. Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020). Neural Controlled Differential Equations for Irregular Time Series. *NeurIPS*.
4. Chen, R. T. Q., Amos, B., & Nickel, M. (2021). Learning Neural Event Functions for Ordinary Differential Equations. *ICLR*.
5. Gierjatowicz, P., Sabate-Vidales, M., Šiska, D., Szpruch, L., & Žurič, Ž. (2020). Robust Pricing and Hedging via Neural SDEs. *SSRN*.
6. Cuchiero, C., Khosrawi, M., & Teichmann, J. (2020). A Generative Adversarial Network Approach to Calibration of Local Stochastic Volatility Models. *Risks*.
