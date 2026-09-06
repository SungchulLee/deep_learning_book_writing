# 34.5.4 Model-Based Reinforcement Learning
## Introduction

Model-based RL learns a dynamics model $\hat{P}(s'|s,a)$ of the environment and uses it to improve sample efficiency. By planning or generating synthetic data through the learned model, agents can learn effective policies with far fewer real environment interactions.

## Approaches

### Dyna-Style Methods
Interleave real environment interaction with model-based rollouts:

1. Interact with real environment, store transitions
2. Train dynamics model on real data
3. Generate synthetic transitions from the model
4. Train policy on both real and synthetic data

### Model Predictive Control (MPC)
Use the learned model for online planning:

1. At each state, simulate multiple action sequences through the model
2. Select the action sequence with highest predicted return
3. Execute the first action, re-plan at next step

### Analytic Gradients
Differentiate through the learned model to compute policy gradients:

$$\nabla_\theta J \approx \nabla_\theta \sum_t \hat{r}(s_t, \pi_\theta(s_t))$$

where trajectories are unrolled through the differentiable model.

## Key Algorithms

### MBPO (Model-Based Policy Optimization)
Janner et al., 2019:

- Train an ensemble of dynamics models for uncertainty estimation
- Generate short model rollouts from real data start states
- Train SAC on a mix of real and model data
- Short rollout horizons mitigate model error compounding

### Dreamer
Hafner et al., 2020:

- Learn a world model in latent space
- Train actor-critic entirely in imagination (latent rollouts)
- Achieve strong performance from pixels with minimal real data

### PETS (Probabilistic Ensemble Trajectory Sampling)
- Ensemble of neural network models
- CEM-based planning through ensemble predictions
- Uncertainty quantification via ensemble disagreement

## Model Architecture

Dynamics models predict $(\hat{s}', \hat{r}) = f_\psi(s, a)$:

- **Deterministic**: Direct prediction of next state
- **Probabilistic**: Output Gaussian parameters $(\mu, \sigma)$
- **Ensemble**: $K$ models for uncertainty estimation
- **Latent space**: Learn compact representations for planning

## Challenges

1. **Model error compounding**: Small per-step errors accumulate over long horizons
2. **Distribution shift**: Model trained on data from old policies
3. **Computational cost**: Planning through the model is expensive
4. **Exploration**: Models may be inaccurate in unexplored regions

## Finance Applications

Model-based RL is particularly appealing for finance:

- **Sample efficiency**: Real market data is limited and expensive
- **Market simulators**: Models can generate realistic market scenarios
- **Risk assessment**: Model uncertainty provides risk estimates
- **Regime changes**: Models can adapt to changing market dynamics

## Summary

Model-based RL dramatically improves sample efficiency by leveraging learned dynamics models. The key challenge is managing model errors, addressed through ensembles, short rollout horizons, and mixing real and model data. For finance applications where data is scarce, model-based approaches offer compelling advantages.

## Exercises

**Exercise 1.**
Derive the policy gradient for the method described in this section. Clearly state which terms require estimation and which can be computed exactly.

??? success "Solution to Exercise 1"
    The policy gradient takes the form $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t]$ where $\hat{A}_t$ is the advantage estimate. The log-probability gradient $\nabla_\theta \log \pi_\theta$ can be computed exactly via automatic differentiation. The advantage $\hat{A}_t$ must be estimated from sampled trajectories, introducing variance. The expectation is approximated by averaging over a batch of trajectories. Variance reduction via baselines preserves unbiasedness while reducing the estimation noise. $\square$

---

**Exercise 2.**
Compare the sample efficiency of this method with a value-based approach (e.g., DQN) on a continuous control task. Explain the theoretical reasons for any observed differences.

??? success "Solution to Exercise 2"
    Policy-based methods are generally less sample-efficient than value-based methods because they use on-policy data (each trajectory is used once). DQN reuses data via experience replay, achieving better sample efficiency. However, policy methods handle continuous actions naturally (no argmax over action space needed), converge to stochastic policies when optimal, and provide monotonic improvement guarantees under trust regions. Off-policy actor-critic methods (DDPG, SAC) bridge this gap by combining policy optimization with experience replay. $\square$

---

**Exercise 3.**
Implement this method for a simple continuous control task (e.g., Pendulum-v1). Report hyperparameter sensitivity with respect to the learning rate and the key method-specific parameter.

??? success "Solution to Exercise 3"
    For Pendulum-v1 with a Gaussian policy, typical performance: learning rate $3 \times 10^{-4}$ achieves convergence in $\sim$500 episodes; $10^{-3}$ causes oscillation; $10^{-5}$ converges too slowly. The method-specific parameter (e.g., clipping range for PPO, KL constraint for TRPO) controls the trade-off between update aggressiveness and stability. Too aggressive leads to performance collapse; too conservative wastes samples. The optimal operating point balances these, typically found via grid search over a small range. $\square$

---

**Exercise 4.**
Discuss how this method could be applied to portfolio optimization where the action space is a simplex (portfolio weights summing to 1) and the reward is risk-adjusted return.

??? success "Solution to Exercise 4"
    The action space is the $(n-1)$-dimensional simplex $\Delta^{n-1} = \{w \in \mathbb{R}^n : w_i \geq 0, \sum_i w_i = 1\}$. The policy can use a Dirichlet distribution or softmax-transformed Gaussian. The reward is the Sharpe ratio or differential Sharpe ratio of the resulting portfolio. Challenges include: high-dimensional action space (many assets), transaction costs penalizing frequent rebalancing, and non-stationarity of market returns. The method from this section addresses these through its specific mechanism for stable policy updates. $\square$
