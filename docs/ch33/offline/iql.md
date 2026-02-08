# 33.5.4 Implicit Q-Learning (IQL)

## Core Idea

**IQL** (Kostrikov et al., 2022) avoids querying out-of-distribution actions entirely by learning Q-values using only actions present in the dataset. It uses **expectile regression** to approximate the maximum over actions without ever evaluating unseen actions.

## The Key Insight

Standard Q-learning requires $\max_a Q(s', a)$ in the Bellman target, which queries OOD actions. IQL replaces this with an expectile-based value function:

$$\mathcal{L}_V(\psi) = \mathbb{E}_{(s,a) \sim \mathcal{D}}\left[L_2^\tau(Q_\theta(s, a) - V_\psi(s))\right]$$

where $L_2^\tau$ is the **asymmetric squared loss** (expectile loss):

$$L_2^\tau(u) = |\tau - \mathbf{1}(u < 0)| \cdot u^2$$

For $\tau > 0.5$, this loss penalizes underestimation more heavily, causing $V_\psi(s)$ to approximate $\max_a Q(s, a)$ using only in-sample actions.

## Algorithm

IQL trains three networks:

1. **Q-network** $Q_\theta(s, a)$: Trained with standard Bellman loss using $V_\psi$ instead of $\max_a Q$:
   $$\mathcal{L}_Q(\theta) = \mathbb{E}_\mathcal{D}\left[(r + \gamma V_\psi(s') - Q_\theta(s, a))^2\right]$$

2. **Value network** $V_\psi(s)$: Trained with expectile regression on Q-values:
   $$\mathcal{L}_V(\psi) = \mathbb{E}_\mathcal{D}\left[L_2^\tau(Q_{\hat{\theta}}(s, a) - V_\psi(s))\right]$$

3. **Policy** $\pi_\phi(a|s)$: Extracted via advantage-weighted regression:
   $$\mathcal{L}_\pi(\phi) = \mathbb{E}_\mathcal{D}\left[\exp(\beta \cdot A(s, a)) \cdot \log \pi_\phi(a|s)\right]$$
   where $A(s, a) = Q(s, a) - V(s)$

## Why Expectile Regression Works

- At $\tau = 0.5$: $V(s)$ approximates $\mathbb{E}_\mu[Q(s, a)]$ (mean)
- At $\tau \to 1$: $V(s)$ approximates $\max_{a \in \text{support}(\mu)} Q(s, a)$ (in-sample max)
- Typical $\tau = 0.7$–$0.9$: Good balance between optimism and staying in-distribution

## Advantages

1. **No OOD queries**: Never evaluates Q at unseen actions
2. **Simple**: No behavior model, no constrained optimization, no min-max
3. **Effective**: State-of-the-art on D4RL benchmarks
4. **Stable**: Avoids the optimization challenges of CQL's min-max objective

## Hyperparameters

| Parameter | Typical | Notes |
|-----------|---------|-------|
| $\tau$ (expectile) | 0.7–0.9 | Higher = more optimistic |
| $\beta$ (advantage temperature) | 3.0–10.0 | Higher = more selective policy |
| LR (all networks) | $3 \times 10^{-4}$ | Standard |

## Finance Application

IQL is particularly attractive for finance because:
- No risk of evaluating extreme positions not in historical data
- The expectile parameter $\tau$ can be interpreted as a quantile preference
- Advantage-weighted policy extraction naturally favors profitable trades seen in data
