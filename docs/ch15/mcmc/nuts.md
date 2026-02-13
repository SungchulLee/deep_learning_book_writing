# No-U-Turn Sampler (NUTS)

The No-U-Turn Sampler (NUTS) is an adaptive extension of HMC that automatically tunes the trajectory length $L$. By detecting when the trajectory begins to "turn around," NUTS eliminates one of the most difficult tuning parameters while maintaining—and often improving—sampling efficiency.

---

## Motivation

### The Trajectory Length Problem

Standard HMC requires specifying the number of leapfrog steps $L$. This choice is critical:

| $L$ too small | $L$ too large |
|---------------|---------------|
| Random-walk behavior | Wasted computation |
| High autocorrelation | Trajectory doubles back |
| Poor exploration | No additional benefit |

**The dilemma**: Optimal $L$ depends on the target distribution, varies across the parameter space, and is difficult to determine a priori.

### The U-Turn Intuition

Consider a 1D harmonic oscillator. The trajectory is periodic:
- Starting from $x_0$, the particle moves away
- It reaches maximum distance, then returns
- Eventually it comes back toward $x_0$

**Key insight**: Once the trajectory starts returning toward its origin, continuing provides no benefit. The "U-turn" signals that we should stop.

### NUTS Solution

NUTS automatically determines trajectory length by:
1. Building the trajectory incrementally (doubling)
2. Detecting when it starts to turn back
3. Selecting a point from the valid trajectory

---

## The U-Turn Criterion

### Basic Definition

A trajectory makes a **U-turn** when continuing would bring it closer to the starting point. For positions $\mathbf{x}^-$ and $\mathbf{x}^+$ at the ends of a trajectory with momenta $\mathbf{v}^-$ and $\mathbf{v}^+$:

**U-turn condition**:
$$
(\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{v}^+ < 0 \quad \text{or} \quad (\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{v}^- < 0
$$

**Interpretation**:
- $(\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{v}^+ < 0$: The forward end is moving backward
- $(\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{v}^- < 0$: The backward end is moving forward

Either condition indicates the trajectory is "turning around."

### Generalized Criterion

For trajectories with mass matrix $\mathbf{M}$, the criterion becomes:

$$
(\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{M}^{-1}\mathbf{v}^+ < 0 \quad \text{or} \quad (\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{M}^{-1}\mathbf{v}^- < 0
$$

This uses velocity $\mathbf{M}^{-1}\mathbf{v}$ rather than momentum $\mathbf{v}$.

### Why This Works

For a quadratic potential (Gaussian target), trajectories are elliptical. The U-turn criterion detects when the trajectory has completed approximately half an orbit—the point of maximum displacement from the origin.

For general targets, the criterion provides a robust heuristic for "enough exploration."

---

## Tree-Building Algorithm

### The Doubling Scheme

NUTS builds the trajectory as a **binary tree**:

1. **Initialize**: Start with a single point (depth 0)
2. **Double**: Randomly extend forward or backward
3. **Check**: Stop if U-turn detected or other termination
4. **Repeat**: Continue doubling until termination

```
Depth 0:        [x₀]                    (1 point)
Depth 1:    [x₋₁, x₀, x₁]              (2 points added)
Depth 2: [x₋₃..x₋₁, x₀, x₁..x₃]        (4 points added)
   ...
Depth j: 2ʲ points total
```

### Why Doubling?

**Advantages**:
- Trajectory length grows exponentially: $2^j$ steps at depth $j$
- Maintains time-reversibility (symmetric extension)
- Efficient: $O(\log L)$ depth for trajectory length $L$

**Reversibility**: By randomly choosing forward/backward at each doubling, and using symmetric tree structure, NUTS preserves detailed balance.

### Tree Structure

Each node in the tree represents a **subtree** containing:
- Position and momentum at left end: $(\mathbf{x}^-, \mathbf{v}^-)$
- Position and momentum at right end: $(\mathbf{x}^+, \mathbf{v}^+)$
- A candidate sample from the subtree
- Number of valid states in subtree
- Termination flag

---

## The NUTS Algorithm

### High-Level Pseudocode

```
function NUTS(x₀, ε, M):
    # Sample momentum
    v₀ ~ Normal(0, M)
    
    # Initialize tree
    x⁻ = x⁺ = x₀
    v⁻ = v⁺ = v₀
    j = 0  # tree depth
    n = 1  # number of acceptable states
    s = 1  # continue flag
    
    # Initial candidate
    x_sample = x₀
    
    while s == 1:
        # Choose direction uniformly
        direction ~ Uniform({-1, +1})
        
        # Build tree in chosen direction
        if direction == -1:
            x⁻, v⁻, _, _, x', n', s' = BuildTree(x⁻, v⁻, -1, j, ε)
        else:
            _, _, x⁺, v⁺, x', n', s' = BuildTree(x⁺, v⁺, +1, j, ε)
        
        # Maybe accept new sample
        if s' == 1 and Random() < n'/n:
            x_sample = x'
        
        # Update count and check U-turn
        n = n + n'
        s = s' AND NoUTurn(x⁻, x⁺, v⁻, v⁺)
        
        j = j + 1
    
    return x_sample
```

### BuildTree Function

```
function BuildTree(x, v, direction, depth, ε):
    if depth == 0:
        # Base case: single leapfrog step
        x', v' = Leapfrog(x, v, direction * ε)
        
        # Check energy
        valid = (H(x', v') - H(x₀, v₀) < Δmax)
        
        return x', v', x', v', x', valid, valid
    else:
        # Recursion: build left subtree
        x⁻, v⁻, x⁺, v⁺, x', n', s' = BuildTree(x, v, direction, depth-1, ε)
        
        if s' == 1:
            # Build right subtree
            if direction == -1:
                x⁻, v⁻, _, _, x'', n'', s'' = BuildTree(x⁻, v⁻, direction, depth-1, ε)
            else:
                _, _, x⁺, v⁺, x'', n'', s'' = BuildTree(x⁺, v⁺, direction, depth-1, ε)
            
            # Maybe update sample
            if Random() < n''/(n' + n''):
                x' = x''
            
            # Check for U-turn within subtree
            s' = s'' AND NoUTurn(x⁻, x⁺, v⁻, v⁺)
            n' = n' + n''
        
        return x⁻, v⁻, x⁺, v⁺, x', n', s'
```

### Multinomial Sampling

NUTS doesn't just return the final point—it samples uniformly from all valid points on the trajectory. This is implemented via:

```
if Random() < n''/(n' + n''):
    x' = x''
```

This "streaming" selection ensures uniform sampling without storing all points.

---

## Termination Criteria

### U-Turn Termination

The primary termination: stop when the trajectory makes a U-turn.

```python
def no_uturn(x_minus, x_plus, v_minus, v_plus, M_inv):
    delta_x = x_plus - x_minus
    return (np.dot(delta_x, M_inv @ v_plus) >= 0 and 
            np.dot(delta_x, M_inv @ v_minus) >= 0)
```

### Energy Termination

Stop if energy error becomes too large (numerical instability):

$$
H(\mathbf{x}', \mathbf{v}') - H(\mathbf{x}_0, \mathbf{v}_0) > \Delta_{\max}
$$

Typical value: $\Delta_{\max} = 1000$.

### Maximum Tree Depth

Prevent infinite loops with a maximum depth:

$$
j \leq j_{\max}
$$

Typical value: $j_{\max} = 10$ (up to $2^{10} = 1024$ leapfrog steps).

### Divergence Detection

A **divergent transition** occurs when the trajectory encounters numerical problems:
- Very large energy change
- NaN or Inf values
- Indicates problematic posterior geometry

```python
def is_divergent(H_new, H_old, delta_max=1000):
    return H_new - H_old > delta_max or np.isnan(H_new)
```

---

## Detailed Balance

### Why NUTS is Valid

NUTS maintains detailed balance through careful construction:

1. **Symmetric tree building**: Random direction at each doubling
2. **Multinomial sampling**: Uniform selection from valid states
3. **Consistent termination**: U-turn checked symmetrically

### The Slice Sampling View

NUTS can be viewed as **slice sampling** in trajectory space:

1. Define a "slice" of acceptable states: $\{(\mathbf{x}, \mathbf{v}) : H(\mathbf{x}, \mathbf{v}) < H_0 + u\}$ where $u \sim \text{Exp}(1)$
2. Build trajectory until it leaves the slice or U-turns
3. Sample uniformly from trajectory within slice

This perspective clarifies why multinomial sampling is correct.

### Acceptance Probability

Unlike standard HMC, NUTS has no explicit MH accept/reject step. Instead:
- Invalid states (high energy) are excluded from selection
- Valid states are sampled uniformly
- The slice sampling mechanism ensures correctness

Effective "acceptance rate" in NUTS refers to the fraction of trajectory that is valid.

---

## Implementation

### Complete NUTS Implementation

```python
import numpy as np

class NUTS:
    def __init__(self, log_prob, grad_log_prob, dim,
                 epsilon=0.1, M=None, max_depth=10, delta_max=1000):
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob
        self.dim = dim
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.delta_max = delta_max
        
        if M is None:
            self.M = np.eye(dim)
            self.M_inv = np.eye(dim)
            self.L_M = np.eye(dim)
        else:
            self.M = M
            self.M_inv = np.linalg.inv(M)
            self.L_M = np.linalg.cholesky(M)
    
    def hamiltonian(self, x, v):
        U = -self.log_prob(x)
        K = 0.5 * v @ self.M_inv @ v
        return U + K
    
    def leapfrog(self, x, v, direction):
        eps = direction * self.epsilon
        x, v = x.copy(), v.copy()
        
        v = v + 0.5 * eps * self.grad_log_prob(x)
        x = x + eps * self.M_inv @ v
        v = v + 0.5 * eps * self.grad_log_prob(x)
        
        return x, v
    
    def check_uturn(self, x_minus, x_plus, v_minus, v_plus):
        delta_x = x_plus - x_minus
        return (np.dot(delta_x, self.M_inv @ v_plus) >= 0 and
                np.dot(delta_x, self.M_inv @ v_minus) >= 0)
    
    def build_tree(self, x, v, direction, depth, H0):
        if depth == 0:
            # Base case: single leapfrog step
            x_new, v_new = self.leapfrog(x, v, direction)
            H_new = self.hamiltonian(x_new, v_new)
            
            # Check validity
            valid = (H_new - H0) < self.delta_max
            divergent = (H_new - H0) > self.delta_max
            
            # Weight for multinomial sampling
            log_weight = -H_new if valid else -np.inf
            
            return (x_new, v_new, x_new, v_new, x_new, 
                    log_weight, valid, divergent, 1)
        else:
            # Recursion
            (x_minus, v_minus, x_plus, v_plus, x_prime,
             log_weight, valid, divergent, n_steps) = \
                self.build_tree(x, v, direction, depth - 1, H0)
            
            if valid:
                if direction == -1:
                    (x_minus, v_minus, _, _, x_prime2,
                     log_weight2, valid2, divergent2, n_steps2) = \
                        self.build_tree(x_minus, v_minus, direction, 
                                       depth - 1, H0)
                else:
                    (_, _, x_plus, v_plus, x_prime2,
                     log_weight2, valid2, divergent2, n_steps2) = \
                        self.build_tree(x_plus, v_plus, direction,
                                       depth - 1, H0)
                
                # Multinomial sampling
                log_weight_sum = np.logaddexp(log_weight, log_weight2)
                if np.log(np.random.rand()) < log_weight2 - log_weight_sum:
                    x_prime = x_prime2
                
                # Update
                log_weight = log_weight_sum
                valid = valid2 and self.check_uturn(x_minus, x_plus, 
                                                     v_minus, v_plus)
                divergent = divergent or divergent2
                n_steps = n_steps + n_steps2
            
            return (x_minus, v_minus, x_plus, v_plus, x_prime,
                    log_weight, valid, divergent, n_steps)
    
    def step(self, x):
        # Sample momentum
        v = self.L_M @ np.random.randn(self.dim)
        H0 = self.hamiltonian(x, v)
        
        # Initialize tree
        x_minus = x_plus = x
        v_minus = v_plus = v
        depth = 0
        valid = True
        x_sample = x
        log_weight = -H0
        n_divergent = 0
        
        while valid and depth < self.max_depth:
            # Choose direction
            direction = 2 * (np.random.rand() < 0.5) - 1
            
            # Build tree
            if direction == -1:
                (x_minus, v_minus, _, _, x_prime,
                 log_weight_subtree, valid_subtree, divergent, _) = \
                    self.build_tree(x_minus, v_minus, direction, depth, H0)
            else:
                (_, _, x_plus, v_plus, x_prime,
                 log_weight_subtree, valid_subtree, divergent, _) = \
                    self.build_tree(x_plus, v_plus, direction, depth, H0)
            
            n_divergent += divergent
            
            # Maybe accept new sample
            if valid_subtree:
                if np.log(np.random.rand()) < log_weight_subtree - log_weight:
                    x_sample = x_prime
                log_weight = np.logaddexp(log_weight, log_weight_subtree)
            
            # Check overall U-turn
            valid = valid_subtree and self.check_uturn(x_minus, x_plus,
                                                        v_minus, v_plus)
            depth += 1
        
        return x_sample, depth, n_divergent > 0
    
    def sample(self, x0, n_samples, n_warmup=1000):
        x = x0.copy()
        samples = np.zeros((n_samples, self.dim))
        depths = []
        n_divergent = 0
        
        # Warmup (could add adaptation here)
        for _ in range(n_warmup):
            x, _, _ = self.step(x)
        
        # Sampling
        for i in range(n_samples):
            x, depth, divergent = self.step(x)
            samples[i] = x
            depths.append(depth)
            n_divergent += divergent
        
        return samples, np.mean(depths), n_divergent / n_samples
```

### Usage Example

```python
# 2D correlated Gaussian
mu = np.array([0, 0])
Sigma = np.array([[1, 0.8], [0.8, 1]])
Sigma_inv = np.linalg.inv(Sigma)

def log_prob(x):
    return -0.5 * (x - mu) @ Sigma_inv @ (x - mu)

def grad_log_prob(x):
    return -Sigma_inv @ (x - mu)

# Sample
nuts = NUTS(log_prob, grad_log_prob, dim=2, epsilon=0.1)
samples, avg_depth, div_rate = nuts.sample(np.zeros(2), 5000)

print(f"Average tree depth: {avg_depth:.1f}")
print(f"Divergence rate: {div_rate:.2%}")
print(f"Sample mean: {samples.mean(0)}")
```

---

## Diagnostics

### Tree Depth

The tree depth indicates trajectory length:

| Depth | Leapfrog Steps | Interpretation |
|-------|----------------|----------------|
| 1-3 | 2-8 | Short trajectories, possibly inefficient |
| 4-6 | 16-64 | Typical range |
| 7-10 | 128-1024 | Long trajectories, may indicate problems |
| 10 (max) | 1024 | Hit maximum, consider increasing |

**Average depth**: Should be moderate (4-8 typically). Very low suggests step size too large; very high suggests difficult geometry.

### Divergences

Divergent transitions indicate serious problems:

- **Few divergences** (< 1%): Usually acceptable, check parameter space
- **Many divergences** (> 5%): Problematic, need to address
- **Where do they occur?**: Plot divergent points to diagnose

**Common causes**:
- Step size too large
- Funnel-shaped posterior
- Boundaries or constraints
- Multi-scale geometry

### Energy Bayesian Fraction of Missing Information (E-BFMI)

BFMI measures how well momentum resampling explores energy levels:

$$
\text{E-BFMI} = \frac{\mathbb{E}[(E_n - E_{n-1})^2]}{\text{Var}(E_n)}
$$

where $E_n = H(\mathbf{x}^{(n)}, \mathbf{v}^{(n)})$.

- **E-BFMI > 0.3**: Good
- **E-BFMI < 0.2**: Problematic (poor energy exploration)

---

## NUTS vs Standard HMC

### Advantages of NUTS

1. **No $L$ tuning**: Automatic trajectory length
2. **Adaptive**: Adjusts to local geometry
3. **Robust**: Handles varying scales
4. **Efficient**: Often more efficient than fixed-$L$ HMC

### When Standard HMC Might Be Better

1. **Known optimal $L$**: If you know the right trajectory length
2. **Computational budget**: NUTS overhead may matter
3. **Very high dimensions**: Tree building has overhead
4. **Simple targets**: May not need adaptive complexity

### Empirical Comparison

For most practical problems, NUTS matches or exceeds tuned HMC:

| Metric | Standard HMC | NUTS |
|--------|--------------|------|
| Tuning effort | High ($\epsilon$, $L$, $\mathbf{M}$) | Lower ($\epsilon$, $\mathbf{M}$) |
| ESS/gradient | Depends on tuning | Consistently good |
| Robustness | Sensitive to $L$ | Robust |
| Implementation | Simpler | More complex |

---

## Practical Recommendations

### Step Size Adaptation

NUTS still requires step size $\epsilon$ tuning. Use dual averaging:

```python
def adapt_step_size(epsilon, accept_stat, target=0.8, 
                    gamma=0.05, t0=10, kappa=0.75, iteration=None):
    # Dual averaging for step size adaptation
    # (simplified version)
    if accept_stat > target:
        return epsilon * 1.02
    else:
        return epsilon * 0.98
```

Target acceptance statistic for NUTS: ~0.8 (higher than HMC's ~0.65).

### Maximum Tree Depth

Default $j_{\max} = 10$ is usually sufficient. Increase if:
- Average depth hits maximum frequently
- Target has very different scales
- Long-range correlations in posterior

### Mass Matrix with NUTS

Same principles as HMC:
- Adapt during warmup
- Diagonal usually sufficient
- Full matrix for strong correlations

### Warmup Schedule (Stan-style)

1. **Iterations 1-75**: Fast step size adaptation
2. **Iterations 76-975**: Step size + mass matrix adaptation
3. **Iterations 976-1000**: Final step size tuning
4. **Iterations 1001+**: Sampling (no adaptation)

---

## Summary

| Component | Description |
|-----------|-------------|
| **U-turn criterion** | Stop when trajectory reverses direction |
| **Tree doubling** | Exponentially grow trajectory |
| **Multinomial sampling** | Uniform selection from valid states |
| **Termination** | U-turn, energy error, or max depth |

NUTS transforms HMC from a method requiring careful tuning into a robust, nearly automatic sampler. By eliminating the trajectory length parameter, NUTS has become the default algorithm in modern probabilistic programming systems like Stan and PyMC.

---

## Exercises

1. **Implement basic NUTS**. Implement NUTS for a 2D Gaussian and verify it produces correct samples. Compare tree depths for different target geometries.

2. **U-turn visualization**. For a 2D target, visualize several NUTS trajectories showing where U-turns occur. How does this relate to the target shape?

3. **Depth analysis**. Run NUTS on targets with varying condition numbers. Plot average tree depth vs condition number. What pattern emerges?

4. **Divergence investigation**. Implement Neal's funnel ($y \sim \mathcal{N}(0, 3)$, $x \sim \mathcal{N}(0, e^y)$). Run NUTS and identify where divergences occur. How does step size affect divergence rate?

5. **NUTS vs HMC comparison**. Compare NUTS to HMC with various fixed $L$ values on a challenging target. Plot ESS per gradient evaluation for each method.

---

## References

1. Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo." *JMLR*, 15, 1593-1623.
2. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
3. Stan Development Team. "Stan Reference Manual."
4. Betancourt, M. (2016). "Diagnosing Biased Inference with Divergences." Stan Case Studies.
