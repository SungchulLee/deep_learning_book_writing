# Likelihood-Free Inference

Many scientific models can simulate data but cannot evaluate the likelihood of observations. Likelihood-free inference—also called simulation-based inference—provides methods to perform Bayesian inference in these settings. This section introduces the problem, motivates the ABC framework, and surveys the landscape of likelihood-free methods.

---

## The Intractable Likelihood Problem

### When Likelihoods Are Unavailable

Standard Bayesian inference requires evaluating the likelihood $p(\mathbf{y} | \theta)$:

$$
p(\theta | \mathbf{y}) = \frac{p(\mathbf{y} | \theta) p(\theta)}{p(\mathbf{y})}
$$

But many models are defined as **simulators**: given parameters $\theta$, we can generate synthetic data $\mathbf{x} \sim p(\cdot | \theta)$, but we cannot evaluate $p(\mathbf{y} | \theta)$ for observed data $\mathbf{y}$.

### Examples of Simulator Models

**Population genetics**: Coalescent models simulate genealogies and genetic variation, but the likelihood integrates over all possible genealogies—computationally intractable.

**Epidemiology**: Agent-based disease models simulate individual infections and contacts. The likelihood requires summing over all possible transmission histories.

**Ecology**: Individual-based models simulate animal movement, births, deaths, and interactions. No closed-form likelihood exists.

**Cosmology**: N-body simulations generate matter distributions. The likelihood of observed galaxy positions is intractable.

**Neuroscience**: Biophysical neuron models simulate spike trains. The likelihood involves integrating over unobserved internal states.

**Economics**: Agent-based market models simulate trader behavior. The likelihood of price time series is unavailable.

### The Simulator Abstraction

A **simulator** (or generative model) is a stochastic function:

$$
\mathbf{x} = f(\theta, \mathbf{u})
$$

where:
- $\theta$ are the parameters of interest
- $\mathbf{u}$ is random noise (often many random numbers)
- $\mathbf{x}$ is the simulated output

We can sample $\mathbf{x} | \theta$ by drawing $\mathbf{u}$ and computing $f$, but we cannot compute $p(\mathbf{x} | \theta)$.

---

## Why Standard Methods Fail

### MCMC Requires Likelihood Evaluation

Metropolis-Hastings needs to compute:

$$
\alpha = \min\left(1, \frac{p(\mathbf{y} | \theta') p(\theta')}{p(\mathbf{y} | \theta) p(\theta)} \cdot \frac{q(\theta | \theta')}{q(\theta' | \theta)}\right)
$$

Without $p(\mathbf{y} | \theta)$, we cannot compute $\alpha$.

### Importance Sampling Requires Likelihood

Importance sampling weights are:

$$
w(\theta) = \frac{p(\mathbf{y} | \theta) p(\theta)}{q(\theta)}
$$

Again, we need $p(\mathbf{y} | \theta)$.

### Variational Inference Requires Likelihood

The ELBO involves:

$$
\mathcal{L}(q) = \mathbb{E}_q[\log p(\mathbf{y} | \theta)] - D_{KL}(q(\theta) \| p(\theta))
$$

The first term requires the likelihood.

---

## The ABC Idea

### Core Insight

If we can't compute $p(\mathbf{y} | \theta)$, we can **simulate** from it. The ABC approach:

1. Propose $\theta$ from prior (or proposal)
2. Simulate $\mathbf{x} \sim p(\cdot | \theta)$
3. Compare $\mathbf{x}$ to observed $\mathbf{y}$
4. Accept $\theta$ if $\mathbf{x} \approx \mathbf{y}$

The accepted $\theta$ values form an approximate posterior sample.

### The Exact (Impractical) Version

If we accept only when $\mathbf{x} = \mathbf{y}$ exactly:

$$
p(\theta | \mathbf{x} = \mathbf{y}) = p(\theta | \mathbf{y})
$$

This gives the exact posterior! But for continuous data, $P(\mathbf{x} = \mathbf{y}) = 0$.

### The Approximate Version

Accept when $\mathbf{x}$ is "close enough" to $\mathbf{y}$:

$$
\rho(\mathbf{x}, \mathbf{y}) < \epsilon
$$

where $\rho$ is a distance metric and $\epsilon$ is a tolerance.

This targets the **ABC posterior**:

$$
p_{ABC}(\theta | \mathbf{y}) \propto p(\theta) \int p(\mathbf{x} | \theta) \mathbf{1}[\rho(\mathbf{x}, \mathbf{y}) < \epsilon] \, d\mathbf{x}
$$

---

## The ABC Posterior

### Interpretation

The ABC posterior is the posterior given that the simulated data is within $\epsilon$ of the observed data:

$$
p_{ABC}(\theta | \mathbf{y}) = p(\theta | \rho(\mathbf{X}, \mathbf{y}) < \epsilon)
$$

where $\mathbf{X} \sim p(\cdot | \theta)$.

### Relationship to True Posterior

As $\epsilon \to 0$:

$$
p_{ABC}(\theta | \mathbf{y}) \to p(\theta | \mathbf{y})
$$

For finite $\epsilon$, the ABC posterior is a smoothed/broadened version of the true posterior.

### The Bias-Variance Trade-off

| Small $\epsilon$ | Large $\epsilon$ |
|------------------|------------------|
| Less bias | More bias |
| Higher variance (rare accepts) | Lower variance (many accepts) |
| Closer to true posterior | Farther from true posterior |
| Computationally expensive | Computationally cheap |

---

## Summary Statistics

### The Curse of Dimensionality

Directly comparing high-dimensional $\mathbf{x}$ and $\mathbf{y}$ is problematic:
- Random $\mathbf{x}$ is almost never close to $\mathbf{y}$
- Acceptance rate becomes vanishingly small
- Need exponentially many simulations

### Dimension Reduction via Summary Statistics

Replace raw data with **summary statistics** $S(\mathbf{x})$:

$$
\rho(S(\mathbf{x}), S(\mathbf{y})) < \epsilon
$$

Now we compare lower-dimensional summaries.

### The Sufficiency Question

**Sufficient statistics**: $S$ is sufficient for $\theta$ if $p(\mathbf{y} | \theta) = p(\mathbf{y} | S(\mathbf{y}), \theta) p(S(\mathbf{y}) | \theta)$.

If $S$ is sufficient:
$$
p_{ABC}(\theta | S(\mathbf{y})) = p(\theta | S(\mathbf{y})) = p(\theta | \mathbf{y})
$$

ABC with sufficient statistics (and $\epsilon \to 0$) gives the exact posterior.

### The Problem: Sufficient Statistics Rarely Exist

For most complex models:
- No finite-dimensional sufficient statistics exist
- Must use approximately sufficient or heuristic summaries
- Information loss is inevitable

### Choosing Summary Statistics

**Domain knowledge**: Statistics that capture relevant features.

**Automatic methods**:
- Semi-automatic ABC (Fearnhead & Prangle, 2012)
- Neural network embeddings
- Information-theoretic selection

---

## The Distance Function

### Common Choices

**Euclidean distance**:
$$
\rho(\mathbf{x}, \mathbf{y}) = \|S(\mathbf{x}) - S(\mathbf{y})\|_2
$$

**Weighted Euclidean**:
$$
\rho(\mathbf{x}, \mathbf{y}) = \sqrt{(S(\mathbf{x}) - S(\mathbf{y}))^T W (S(\mathbf{x}) - S(\mathbf{y}))}
$$

where $W$ accounts for different scales.

**Mahalanobis distance**:
$$
\rho(\mathbf{x}, \mathbf{y}) = \sqrt{(S(\mathbf{x}) - S(\mathbf{y}))^T \Sigma^{-1} (S(\mathbf{x}) - S(\mathbf{y}))}
$$

where $\Sigma$ is the covariance of $S(\mathbf{X})$ under the prior predictive.

### Kernel ABC

Replace hard threshold with soft kernel:

$$
K_\epsilon(\mathbf{x}, \mathbf{y}) = K\left(\frac{\rho(\mathbf{x}, \mathbf{y})}{\epsilon}\right)
$$

Common kernels:
- Uniform: $K(u) = \mathbf{1}[u < 1]$
- Gaussian: $K(u) = \exp(-u^2/2)$
- Epanechnikov: $K(u) = (1 - u^2)\mathbf{1}[u < 1]$

---

## Theoretical Foundations

### Consistency

Under regularity conditions, ABC is **consistent**: as $n \to \infty$ (data size) and $\epsilon \to 0$ (appropriately):

$$
p_{ABC}(\theta | \mathbf{y}_n) \to \delta_{\theta_0}
$$

where $\theta_0$ is the true parameter.

### Convergence Rate

The convergence rate depends on:
- Dimension of summary statistics
- Smoothness of the model
- Choice of $\epsilon$ schedule

For $d$-dimensional sufficient statistics:
$$
\epsilon_n \sim n^{-1/(d+4)}
$$

gives optimal mean squared error.

### Asymptotic Normality

Under conditions, the ABC posterior is asymptotically normal:

$$
p_{ABC}(\theta | \mathbf{y}_n) \approx \mathcal{N}(\hat{\theta}_n, V_n)
$$

where $\hat{\theta}_n$ is a consistent estimator and $V_n \to 0$.

---

## Beyond Basic ABC

### The Landscape of Likelihood-Free Methods

ABC is one approach. The broader landscape includes:

**ABC variants**:
- ABC rejection sampling
- ABC-MCMC
- ABC-SMC (Sequential Monte Carlo)
- Regression adjustment

**Neural likelihood-free methods**:
- Neural posterior estimation (NPE)
- Neural likelihood estimation (NLE)
- Neural ratio estimation (NRE)

**Other approaches**:
- Synthetic likelihood
- Indirect inference
- Bayesian optimization for likelihood-free inference

### Neural Density Estimation

Train a neural network to approximate:

**The posterior** (NPE):
$$
q_\phi(\theta | \mathbf{y}) \approx p(\theta | \mathbf{y})
$$

**The likelihood** (NLE):
$$
q_\phi(\mathbf{y} | \theta) \approx p(\mathbf{y} | \theta)
$$

**The likelihood ratio** (NRE):
$$
r_\phi(\theta, \mathbf{y}) \approx \frac{p(\mathbf{y} | \theta)}{p(\mathbf{y})}
$$

These methods amortize inference: once trained, posterior samples for new observations are cheap.

### Synthetic Likelihood

Assume summary statistics are approximately Gaussian:

$$
S(\mathbf{X}) | \theta \approx \mathcal{N}(\mu(\theta), \Sigma(\theta))
$$

Estimate $\mu(\theta)$, $\Sigma(\theta)$ from simulations, then use this Gaussian likelihood in standard MCMC.

---

## When to Use Likelihood-Free Methods

### Good Candidates

✓ Model is a simulator (can generate, can't evaluate)
✓ Model is scientifically motivated (not just for fitting)
✓ Simulation is reasonably fast
✓ Informative summary statistics exist
✓ Prior is proper and not too diffuse

### Poor Candidates

✗ Likelihood is tractable (use standard methods!)
✗ Simulation is extremely slow
✗ No good summary statistics known
✗ Very high-dimensional parameter space
✗ Model is misspecified (garbage in, garbage out)

### Computational Considerations

| Factor | Impact |
|--------|--------|
| Simulation cost | Dominates runtime |
| Parameter dimension | Affects acceptance rate |
| Summary dimension | Trade-off: information vs. acceptance |
| Data size | More data → need smaller $\epsilon$ |

---

## Practical Workflow

### Step 1: Model Validation

Before inference, validate the simulator:
- Can it produce data resembling observations?
- Is the prior reasonable?
- Are there bugs in the simulation code?

```python
def prior_predictive_check(simulator, prior, n_sims=100):
    """Generate prior predictive samples."""
    samples = []
    for _ in range(n_sims):
        theta = prior.sample()
        x = simulator(theta)
        samples.append({'theta': theta, 'x': x})
    
    # Visualize: do any samples look like real data?
    return samples
```

### Step 2: Choose Summary Statistics

Start with domain-motivated summaries:

```python
def summary_statistics(x):
    """Example: time series summaries."""
    return np.array([
        np.mean(x),
        np.std(x),
        np.corrcoef(x[:-1], x[1:])[0, 1],  # Lag-1 autocorrelation
        np.percentile(x, [25, 50, 75]),
    ]).flatten()
```

### Step 3: Calibrate Tolerance

Run pilot simulations to understand the distance distribution:

```python
def calibrate_epsilon(simulator, prior, summary_fn, y_obs, n_pilot=1000):
    """Determine reasonable epsilon range."""
    distances = []
    
    s_obs = summary_fn(y_obs)
    
    for _ in range(n_pilot):
        theta = prior.sample()
        x = simulator(theta)
        s_x = summary_fn(x)
        distances.append(np.linalg.norm(s_x - s_obs))
    
    # Epsilon as quantile of prior predictive distances
    return {
        'q01': np.percentile(distances, 1),
        'q05': np.percentile(distances, 5),
        'q10': np.percentile(distances, 10),
    }
```

### Step 4: Run ABC

Start with rejection sampling, move to MCMC or SMC if needed.

### Step 5: Validate Results

Check posterior predictive:

```python
def posterior_predictive_check(simulator, posterior_samples, summary_fn, y_obs):
    """Check if posterior can reproduce observations."""
    s_obs = summary_fn(y_obs)
    
    for theta in posterior_samples:
        x = simulator(theta)
        s_x = summary_fn(x)
        # Compare s_x to s_obs
```

---

## Summary

| Concept | Description |
|---------|-------------|
| **Likelihood-free** | Can simulate, cannot evaluate likelihood |
| **ABC idea** | Accept parameters that produce similar data |
| **Summary statistics** | Reduce dimension for tractable comparison |
| **Tolerance $\epsilon$** | Trade-off: bias vs. variance |
| **ABC posterior** | Approximation to true posterior |
| **Consistency** | Exact as $\epsilon \to 0$, $n \to \infty$ |

Likelihood-free inference enables Bayesian analysis for complex simulator models where traditional methods fail. ABC provides a simple, widely applicable framework, while modern neural approaches offer improved efficiency for repeated inference tasks.

---

## Exercises

1. **Simulator example**. Implement a simple ecological model (e.g., Lotka-Volterra) as a simulator. Verify you can generate data but cannot evaluate the likelihood.

2. **ABC by hand**. For a normal mean inference problem (where likelihood is available), implement ABC rejection sampling. Compare the ABC posterior to the true posterior for various $\epsilon$.

3. **Summary statistic impact**. For a model of your choice, compare ABC with (a) sufficient statistics, (b) insufficient but informative statistics, (c) random statistics. How does the posterior change?

4. **Tolerance calibration**. Implement the calibration procedure above. How does the acceptance rate depend on $\epsilon$? What is a reasonable choice?

5. **Comparison to exact**. For a model where both ABC and exact inference are possible, quantify the ABC approximation error as a function of $\epsilon$.

---

## References

1. Beaumont, M. A., Zhang, W., & Balding, D. J. (2002). "Approximate Bayesian Computation in Population Genetics." *Genetics*.
2. Marin, J.-M., Pudlo, P., Robert, C. P., & Ryder, R. J. (2012). "Approximate Bayesian Computational Methods." *Statistics and Computing*.
3. Sisson, S. A., Fan, Y., & Beaumont, M. A. (2018). *Handbook of Approximate Bayesian Computation*. CRC Press.
4. Cranmer, K., Brehmer, J., & Louppe, G. (2020). "The Frontier of Simulation-Based Inference." *PNAS*.
5. Fearnhead, P., & Prangle, D. (2012). "Constructing Summary Statistics for Approximate Bayesian Computation: Semi-Automatic Approximate Bayesian Computation." *JRSS-B*.
