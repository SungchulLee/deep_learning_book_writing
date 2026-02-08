# 31.5.4 Property Optimization

## Overview

Generating valid, novel molecules is only the first step in molecular design. The ultimate goal is to generate molecules that optimize one or more **target properties**—binding affinity to a disease target, synthetic accessibility, drug-likeness, selectivity, or a combination thereof. Property optimization sits at the intersection of generative modeling and optimization, and draws on techniques from reinforcement learning, Bayesian optimization, and gradient-based search in latent spaces.

## Problem Formulation

Given a property oracle $f: \mathcal{M} \to \mathbb{R}$ (or a vector of properties $\mathbf{f}: \mathcal{M} \to \mathbb{R}^k$), the goal is to find:

$$\mathcal{M}^* = \arg\max_{\mathcal{M} \in \mathcal{V}} f(\mathcal{M})$$

where $\mathcal{V}$ is the set of valid molecules. In practice, the oracle may be expensive (e.g., a docking simulation or wet-lab experiment), and the chemical space is combinatorially vast ($\sim 10^{60}$ drug-like molecules), making exhaustive search impossible.

Key challenges include:

**Multi-objective optimization**: Real drug candidates must satisfy multiple constraints simultaneously—high binding affinity, low toxicity, good oral bioavailability, synthesizability. These objectives often conflict.

**Sample efficiency**: When the oracle is expensive, each evaluation counts. Methods must find good molecules with as few oracle calls as possible.

**Diversity**: Finding a diverse set of high-scoring molecules is more useful than many copies of a single optimum.

**Mode collapse**: Generative models may converge to a narrow region of chemical space, producing slight variations of the same scaffold.

## Latent Space Optimization

If a generative model (VAE, diffusion model) learns a continuous latent space $\mathbf{z}$, properties can be optimized by searching this space:

**Gradient ascent**: Train a property predictor $g_\phi(\mathbf{z})$ on the latent space, then follow the gradient:

$$\mathbf{z}_{t+1} = \mathbf{z}_t + \eta \nabla_\mathbf{z} g_\phi(\mathbf{z}_t)$$

Decode $\mathbf{z}_{t+1}$ to obtain an improved molecule. The smoothness of the latent space determines how well this works—a well-organized latent space where nearby points decode to similar molecules enables effective gradient-based search.

**Bayesian optimization**: Model the property as a Gaussian process over the latent space, and use an acquisition function (Expected Improvement, Upper Confidence Bound) to select the next point to evaluate:

$$\mathbf{z}_{\text{next}} = \arg\max_\mathbf{z} \alpha(\mathbf{z} \mid \mathcal{D})$$

This is particularly appropriate when the oracle is expensive, as Bayesian optimization is designed for sample-efficient black-box optimization.

**CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**: A derivative-free optimization algorithm that maintains a multivariate Gaussian over the search space and iteratively adapts its mean and covariance to focus on high-scoring regions.

## Reinforcement Learning Approaches

RL provides a natural framework for molecular optimization when generation is sequential (SMILES, autoregressive graph construction):

**REINFORCE**: Treat the generator as a policy $\pi_\theta$ that produces molecules (actions). The reward is the property score. The policy gradient is:

$$\nabla_\theta J = \mathbb{E}_{\mathcal{M} \sim \pi_\theta}\left[(R(\mathcal{M}) - b) \nabla_\theta \log \pi_\theta(\mathcal{M})\right]$$

**PPO / TRPO**: Constrained policy updates prevent the generator from collapsing to a narrow mode. The KL divergence between the updated and previous policy serves as a regularizer.

**KL-regularized optimization**: To prevent the optimized model from diverging too far from chemical reality, add a KL penalty between the optimized policy and the pre-trained prior:

$$J_\text{reg} = \mathbb{E}_{\pi_\theta}[R(\mathcal{M})] - \beta\, \text{KL}(\pi_\theta \| \pi_\text{prior})$$

This balances exploitation (optimizing properties) with prior knowledge (generating chemically reasonable molecules).

## Genetic Algorithms for Molecules

Evolutionary approaches operate directly on molecular representations:

**Graph-based genetic algorithm (GB-GA)**:
1. Maintain a population of molecules
2. **Crossover**: Combine fragments from two parent molecules (cut at random bonds, swap fragments)
3. **Mutation**: Add/remove/change atoms or bonds
4. **Selection**: Keep high-scoring molecules for the next generation

GB-GA is simple, parallelizable, and surprisingly competitive with sophisticated deep learning approaches. Its main weakness is sample inefficiency—it requires many oracle evaluations.

**SELFIES-based GA**: Operate on SELFIES strings, where any mutation produces a valid molecule. This eliminates the need for validity checking during evolution.

## Multi-Objective Optimization

Real drug design involves optimizing multiple conflicting objectives. The Pareto front contains all molecules where no single property can be improved without worsening another:

$$\text{Pareto}(\mathcal{M}) = \{\mathcal{M} : \nexists\, \mathcal{M}' \text{ s.t. } f_i(\mathcal{M}') \geq f_i(\mathcal{M})\;\forall i,\; f_j(\mathcal{M}') > f_j(\mathcal{M})\;\exists j\}$$

**Scalarization**: Combine objectives with weights: $R = \sum_i w_i f_i(\mathcal{M})$. Simple but requires specifying weights a priori.

**Hypervolume maximization**: Optimize the hypervolume dominated by the Pareto front. This automatically discovers diverse trade-off solutions.

**Conditional generation**: Train the generator conditioned on a desired property vector $\mathbf{y}$, then vary $\mathbf{y}$ to explore the Pareto front.

## Constrained Optimization

In addition to optimizing properties, molecules must satisfy hard constraints:

**Valency constraints**: Satisfied by construction (autoregressive with masking) or by post-hoc correction.

**Substructure constraints**: The molecule must contain (or exclude) specific functional groups. Enforced by fragment-based generation or by filtering.

**Similarity constraints**: The optimized molecule must remain similar to a reference (e.g., Tanimoto > 0.4 to a lead compound). This ensures the optimization doesn't venture into completely unfamiliar chemical space.

## Quantitative Finance Parallel

The optimization framework for molecular generation maps directly to quantitative finance problems:

| Molecular Design | Portfolio/Network Design |
|-----------------|------------------------|
| Molecule → Property score | Portfolio → Sharpe ratio |
| Validity constraints | Regulatory constraints |
| Latent space search | Factor space navigation |
| Multi-objective Pareto | Risk-return trade-off |
| Oracle cost | Backtest cost |
| Diversity requirement | Diversification constraint |

This cross-domain insight motivates the application of molecular optimization techniques to financial network generation (Section 31.6).
