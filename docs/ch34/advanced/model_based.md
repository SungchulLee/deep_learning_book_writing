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
