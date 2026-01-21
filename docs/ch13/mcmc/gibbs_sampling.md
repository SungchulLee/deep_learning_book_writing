# Gibbs Sampling as Metropolis-Hastings with α = 1

## The Surprising Result

Gibbs sampling can be viewed as a **special case** of Metropolis-Hastings where the **acceptance probability is always 1**—meaning we **never** reject a proposal.

This is not obvious! Let's prove it rigorously.

---

## Setup: Multivariate Target

Target distribution $\pi(\mathbf{x})$ where $\mathbf{x} = (x_1, \ldots, x_d) \in \mathbb{R}^d$.

Assume we can compute **full conditional distributions**:

$$
\pi_i(x_i | \mathbf{x}_{-i}) = \frac{\pi(\mathbf{x})}{\int \pi(\mathbf{x}) dx_i}
$$

where $\mathbf{x}_{-i} = (x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_d)$ denotes all variables except $x_i$.

---

## Gibbs Sampling Algorithm

At iteration $t$, cycle through coordinates:

**For $i = 1, \ldots, d$:**

$$
x_i^{(t+1)} \sim \pi_i(x_i | x_1^{(t+1)}, \ldots, x_{i-1}^{(t+1)}, x_{i+1}^{(t)}, \ldots, x_d^{(t)})
$$

Key: Use **most recent** values for conditioning.

---

## Gibbs as Metropolis-Hastings

### The Proposal Distribution

For coordinate $i$, define the **Gibbs proposal**:

$$
q_i(x_i' | \mathbf{x}) = \pi_i(x_i' | \mathbf{x}_{-i})
$$

That is: propose $x_i'$ from the **full conditional** given current values of other variables.

The full proposal for the state is:

$$
\mathbf{x}' = (x_1, \ldots, x_{i-1}, x_i', x_{i+1}, \ldots, x_d)
$$

Only coordinate $i$ changes; others remain fixed.

### The Metropolis-Hastings Acceptance Ratio

For this proposal:

$$
\alpha = \min\left(1, \frac{\pi(\mathbf{x}') q_i(\mathbf{x} | \mathbf{x}')}{\pi(\mathbf{x}) q_i(\mathbf{x}' | \mathbf{x})}\right)
$$

Let's substitute the Gibbs proposal:

$$
\alpha = \min\left(1, \frac{\pi(\mathbf{x}') \pi_i(x_i | \mathbf{x}_{-i}')}{\pi(\mathbf{x}) \pi_i(x_i' | \mathbf{x}_{-i})}\right)
$$

But $\mathbf{x}_{-i}' = \mathbf{x}_{-i}$ (other coordinates don't change), so:

$$
\alpha = \min\left(1, \frac{\pi(\mathbf{x}') \pi_i(x_i | \mathbf{x}_{-i})}{\pi(\mathbf{x}) \pi_i(x_i' | \mathbf{x}_{-i})}\right)
$$

---

## The Key Identity

Recall the definition of the full conditional:

$$
\pi_i(x_i | \mathbf{x}_{-i}) = \frac{\pi(\mathbf{x})}{\int \pi(x_1, \ldots, x_{i-1}, u, x_{i+1}, \ldots, x_d) du}
$$

Denote the marginal over $x_i$ as:

$$
m_{-i}(\mathbf{x}_{-i}) = \int \pi(\mathbf{x}) dx_i
$$

Then:

$$
\pi_i(x_i | \mathbf{x}_{-i}) = \frac{\pi(\mathbf{x})}{m_{-i}(\mathbf{x}_{-i})}
$$

Rearranging:

$$
\pi(\mathbf{x}) = \pi_i(x_i | \mathbf{x}_{-i}) \cdot m_{-i}(\mathbf{x}_{-i})
$$

---

## Proof that α = 1

Substitute the identity into the acceptance ratio:

$$
\alpha = \min\left(1, \frac{\pi_i(x_i' | \mathbf{x}_{-i}) \cdot m_{-i}(\mathbf{x}_{-i}) \cdot \pi_i(x_i | \mathbf{x}_{-i})}{\pi_i(x_i | \mathbf{x}_{-i}) \cdot m_{-i}(\mathbf{x}_{-i}) \cdot \pi_i(x_i' | \mathbf{x}_{-i})}\right)
$$

Note:
- $\mathbf{x}_{-i}' = \mathbf{x}_{-i}$ → the marginal $m_{-i}$ is the **same** in numerator and denominator
- The conditional $\pi_i(x_i | \mathbf{x}_{-i})$ appears in both numerator and denominator
- The conditional $\pi_i(x_i' | \mathbf{x}_{-i})$ appears in both numerator and denominator

**Everything cancels!**

$$
\alpha = \min\left(1, \frac{\cancel{\pi_i(x_i' | \mathbf{x}_{-i})} \cdot \cancel{m_{-i}(\mathbf{x}_{-i})} \cdot \cancel{\pi_i(x_i | \mathbf{x}_{-i})}}{\cancel{\pi_i(x_i | \mathbf{x}_{-i})} \cdot \cancel{m_{-i}(\mathbf{x}_{-i})} \cdot \cancel{\pi_i(x_i' | \mathbf{x}_{-i})}}\right) = \min(1, 1) = 1
$$

**Therefore**: $\alpha = 1$ for every Gibbs proposal! ✓

---

## Intuition: Why Acceptance is Always 1

### Proposal from the Conditional

The **magic** is that the proposal **is** the full conditional:

$$
q_i(x_i' | \mathbf{x}) = \pi_i(x_i' | \mathbf{x}_{-i})
$$

This means:
- We're proposing from the **exact** conditional distribution of $x_i$ given $\mathbf{x}_{-i}$
- The proposal **already respects** the target $\pi$ (conditional on $\mathbf{x}_{-i}$)

### Perfect Balance

For detailed balance, we need:

$$
\pi(\mathbf{x}) q(\mathbf{x}' | \mathbf{x}) = \pi(\mathbf{x}') q(\mathbf{x} | \mathbf{x}')
$$

With Gibbs proposals:

$$
\pi(\mathbf{x}) \pi_i(x_i' | \mathbf{x}_{-i}) = \pi(\mathbf{x}') \pi_i(x_i | \mathbf{x}_{-i})
$$

Using $\pi(\mathbf{x}) = \pi_i(x_i | \mathbf{x}_{-i}) \cdot m_{-i}(\mathbf{x}_{-i})$:

$$
\pi_i(x_i | \mathbf{x}_{-i}) \cdot m_{-i}(\mathbf{x}_{-i}) \cdot \pi_i(x_i' | \mathbf{x}_{-i}) = \pi_i(x_i' | \mathbf{x}_{-i}) \cdot m_{-i}(\mathbf{x}_{-i}) \cdot \pi_i(x_i | \mathbf{x}_{-i})
$$

This is **automatically satisfied** (both sides are identical)!

**Conclusion**: The proposal **perfectly** satisfies detailed balance → no correction needed → $\alpha = 1$.

---

## Alternative Proof: Direct from Detailed Balance

We can also prove this directly without using the MH formula.

**Goal**: Show that Gibbs transitions satisfy detailed balance.

**Transition kernel** for updating coordinate $i$:

$$
T_i(\mathbf{x}' | \mathbf{x}) = \begin{cases}
\pi_i(x_i' | \mathbf{x}_{-i}) & \text{if } \mathbf{x}_{-i}' = \mathbf{x}_{-i} \\
0 & \text{otherwise}
\end{cases}
$$

**Detailed balance** requires:

$$
\pi(\mathbf{x}) T_i(\mathbf{x}' | \mathbf{x}) = \pi(\mathbf{x}') T_i(\mathbf{x} | \mathbf{x}')
$$

**Left side**:

$$
\pi(\mathbf{x}) T_i(\mathbf{x}' | \mathbf{x}) = \pi(\mathbf{x}) \pi_i(x_i' | \mathbf{x}_{-i})
$$

Using $\pi(\mathbf{x}) = \pi_i(x_i | \mathbf{x}_{-i}) \cdot m_{-i}(\mathbf{x}_{-i})$:

$$
= \pi_i(x_i | \mathbf{x}_{-i}) \cdot m_{-i}(\mathbf{x}_{-i}) \cdot \pi_i(x_i' | \mathbf{x}_{-i})
$$

**Right side**:

$$
\pi(\mathbf{x}') T_i(\mathbf{x} | \mathbf{x}') = \pi(\mathbf{x}') \pi_i(x_i | \mathbf{x}_{-i}')
$$

Since $\mathbf{x}_{-i}' = \mathbf{x}_{-i}$:

$$
= \pi_i(x_i' | \mathbf{x}_{-i}) \cdot m_{-i}(\mathbf{x}_{-i}) \cdot \pi_i(x_i | \mathbf{x}_{-i})
$$

**Left = Right** ✓

Detailed balance holds **exactly** → no MH correction needed!

---

## What This Means Practically

### No Rejection Step

In Gibbs sampling:
1. Sample $x_i' \sim \pi_i(x_i | \mathbf{x}_{-i})$
2. **Accept** it (always!)
3. Move to next coordinate

There is **no** rejection step because $\alpha = 1$ guarantees acceptance.

### 100% Acceptance Rate

Unlike Metropolis-Hastings where:
- Acceptance rate typically 20%-90%
- Rejections mean wasted computation

Gibbs sampling:
- **100% acceptance rate**
- Every proposal is kept
- No "wasted" iterations

### Why This is Powerful

**Efficiency**: No rejections → every iteration moves the chain

**Simplicity**: No need to:
- Compute acceptance ratio
- Generate uniform random variable for accept/reject
- Tune proposal to balance acceptance vs exploration

**Automatic tuning**: The conditional $\pi_i(x_i | \mathbf{x}_{-i})$ **automatically** adapts to local geometry of $\pi$

---

## The Critical Requirement

### Must Sample from Conditional Exactly

Gibbs requires:

$$
x_i' \sim \pi_i(x_i | \mathbf{x}_{-i}) \quad \text{(exact sampling)}
$$

**Not** just:
- Evaluate $\pi_i(x_i | \mathbf{x}_{-i})$ (density evaluation)
- Approximate $\pi_i$ (MCMC within MCMC)

If we can only **evaluate** but not **sample** from $\pi_i$, we **cannot** use Gibbs. Must use Metropolis-Hastings instead.

### Examples Where Exact Sampling is Possible

**Conjugate models**:
- Beta-Binomial
- Gaussian-Gaussian
- Gamma-Poisson

**Standard distributions**:
- Gaussian
- Gamma
- Beta
- Dirichlet

**Conditionals with closed form**:
- Latent Dirichlet Allocation (LDA)
- Gaussian Mixture Models
- Hidden Markov Models

### Examples Where Exact Sampling is **Not** Possible

**Logistic regression**: $p(x_i | \mathbf{x}_{-i})$ is non-standard distribution

**Neural networks**: Conditionals are intractable

**Complex hierarchical models**: No closed-form conditionals

In these cases, use:
- **Metropolis-within-Gibbs**: Use MH for problematic coordinates
- **Hamiltonian Monte Carlo**: Avoids coordinate-wise updates entirely
- **Variational inference**: Deterministic approximation

---

## Gibbs vs Metropolis-Hastings: Summary

| Property | Gibbs | Metropolis-Hastings |
|----------|-------|---------------------|
| **Proposal** | $q = \pi_i(x_i \| \mathbf{x}_{-i})$ | Arbitrary $q(\mathbf{x}' \| \mathbf{x})$ |
| **Acceptance** | Always 1 | Varies (0 to 1) |
| **Requires** | Exact sampling | Only evaluation |
| **Tuning** | None | Proposal variance $\sigma$ |
| **Rejection** | Never | Sometimes |
| **Scope** | One coordinate | Entire state (or subset) |
| **When to use** | Tractable conditionals | General distributions |

---

## Computational Perspective

### Cost of α = 1

Gibbs acceptance of $\alpha = 1$ is **not free**:

**Cost shifted to**:
- **Sampling** from $\pi_i(x_i | \mathbf{x}_{-i})$ (may be expensive)
- Computing **normalization constant** of conditional (if needed for sampling)

**Metropolis-Hastings**:
- **Cheap** proposals (e.g., Gaussian)
- Some **rejections** (wasted computation)

**Gibbs**:
- **Expensive** proposals (exact sampling from conditional)
- **No** rejections (all computation used)

**Net effect**: Depends on problem structure!

### When Gibbs is More Efficient

If:
- Conditionals are **standard** distributions (cheap sampling)
- MH would have **low** acceptance (<50%)
- Strong **correlations** between variables (coordinate-wise updates work well)

Then: Gibbs often wins.

### When Metropolis-Hastings is More Efficient

If:
- Conditionals are **non-standard** (expensive or impossible to sample)
- MH achieves **high** acceptance (>70%)
- Weak **correlations** (random walk works well)

Then: MH often wins.

---

## The Geometric View

### Gibbs Moves on Axis-Aligned Slices

Each Gibbs update:
- Fixes $d-1$ coordinates
- Samples exactly from the **slice** of $\pi$ along remaining axis

This is like exploring a high-dimensional distribution by:
1. Choosing an axis
2. **Perfectly** sampling the 1D conditional along that axis
3. Repeat for next axis

**Perfect sampling** along each slice → no rejection needed!

### MH Explores via Proposals

MH:
- Proposes a move in **any direction** (or subset of coordinates)
- **Tests** whether the proposal improves fit to $\pi$
- **Rejects** if it worsens too much

This requires:
- Proposal design (choose direction/magnitude)
- Acceptance probability (balance exploration/exploitation)

**Trade-off**: More flexibility but requires tuning and rejections.

---

## Why Understanding This Matters

### Hybrid Samplers

Modern MCMC often uses **both**:

**Metropolis-within-Gibbs**:
- Use Gibbs for coordinates with tractable conditionals
- Use MH for coordinates without tractable conditionals

**Example**: Hierarchical model
- Latent variables: Gibbs (conjugate)
- Hyperparameters: MH (non-conjugate)

### Recognizing When Gibbs Applies

Knowing that Gibbs is MH with $\alpha = 1$ helps recognize:
- When you **can** use Gibbs (exact conditional sampling)
- When you **cannot** (no closed form → use MH instead)

### Theoretical Understanding

The $\alpha = 1$ result is a beautiful example of:
- **Special structure** (conditional proposals) → **automatic** detailed balance
- **Mathematical elegance**: no tuning, no rejections, just works!

---

## Historical Note

**Gibbs sampling** was named after Josiah Willard Gibbs (statistical mechanics), introduced to statistics by Geman & Geman (1984) for image restoration.

**Key insight**: If you can sample from full conditionals, you get MCMC **for free**—no MH overhead!

This revolutionized Bayesian computation, making complex hierarchical models tractable.

**Modern perspective**: Gibbs is a **special case** of the general MH framework, but one with particularly nice properties ($\alpha = 1$).

---

## The Bottom Line

**Gibbs sampling has acceptance ratio α = 1 because**:

1. The proposal distribution **is** the target conditional
2. This **automatically** satisfies detailed balance
3. No MH correction needed → always accept
4. Requires exact conditional sampling (not just evaluation)

This is both a **strength** (no rejections, no tuning) and a **limitation** (only works when conditionals are tractable).

Understanding this connection deepens our appreciation of both Gibbs sampling and the general Metropolis-Hastings framework!
