"""
Chapter 34.1.1: Policy Parameterization
========================================
Implementations of various policy parameterization strategies for
both discrete and continuous action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta
import numpy as np
import gymnasium as gym


# ---------------------------------------------------------------------------
# Weight initialization utilities
# ---------------------------------------------------------------------------

def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Orthogonal initialization following PPO best practices."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# Discrete Policy (Softmax / Categorical)
# ---------------------------------------------------------------------------

class DiscretePolicy(nn.Module):
    """
    Categorical policy for discrete action spaces.
    
    The network outputs logits that are converted to a categorical
    distribution via softmax. Log-softmax is used for numerical stability.
    
    Parameters
    ----------
    obs_dim : int
        Dimension of the observation space.
    act_dim : int
        Number of discrete actions.
    hidden_dim : int
        Hidden layer size.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),  # Small init for near-uniform
        )
    
    def forward(self, obs: torch.Tensor):
        """Return action logits."""
        return self.network(obs)
    
    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        """Return categorical distribution over actions."""
        logits = self.forward(obs)
        return Categorical(logits=logits)
    
    def get_action(self, obs: torch.Tensor):
        """Sample action and return action, log_prob, entropy."""
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate log_prob and entropy for given state-action pairs."""
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


# ---------------------------------------------------------------------------
# Gaussian Policy (Continuous, unbounded)
# ---------------------------------------------------------------------------

class GaussianPolicy(nn.Module):
    """
    Diagonal Gaussian policy for continuous action spaces.
    
    Two variants:
    - State-independent log_std: learnable parameter (default, used in PPO)
    - State-dependent log_std: output by the network
    
    Parameters
    ----------
    obs_dim : int
        Dimension of the observation space.
    act_dim : int
        Dimension of the continuous action space.
    hidden_dim : int
        Hidden layer size.
    state_dependent_std : bool
        If True, std depends on state via the network.
    log_std_init : float
        Initial value for log standard deviation.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 64,
        state_dependent_std: bool = False,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        self.act_dim = act_dim
        
        # Shared feature extractor
        self.features = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        
        # Mean head
        self.mean_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        
        if state_dependent_std:
            # State-dependent: network outputs log_std
            self.log_std_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        else:
            # State-independent: learnable parameter
            self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
    
    def forward(self, obs: torch.Tensor):
        """Return mean and log_std of the Gaussian policy."""
        features = self.features(obs)
        mean = self.mean_head(features)
        
        if self.state_dependent_std:
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, min=-20, max=2)  # Stability clamp
        else:
            log_std = self.log_std.expand_as(mean)
        
        return mean, log_std
    
    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """Return Normal distribution for the policy."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        return Normal(mean, std)
    
    def get_action(self, obs: torch.Tensor):
        """Sample action and return action, log_prob, entropy."""
        dist = self.get_distribution(obs)
        action = dist.sample()
        # Sum log_prob across action dimensions for multivariate
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate log_prob and entropy for given state-action pairs."""
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ---------------------------------------------------------------------------
# Squashed Gaussian Policy (SAC-style, bounded actions)
# ---------------------------------------------------------------------------

class SquashedGaussianPolicy(nn.Module):
    """
    Squashed Gaussian policy for bounded continuous action spaces.
    
    Samples from a Gaussian and applies tanh to bound actions to [-1, 1].
    Applies the change-of-variables correction to the log-probability.
    
    Used in SAC (Soft Actor-Critic).
    
    Parameters
    ----------
    obs_dim : int
        Dimension of the observation space.
    act_dim : int
        Dimension of the action space.
    hidden_dim : int
        Hidden layer size.
    action_scale : float
        Scale factor for actions (to map to actual action bounds).
    action_bias : float
        Bias for actions.
    """
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    EPS = 1e-6
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.mean_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.log_std_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        
        # Action rescaling
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))
    
    def forward(self, obs: torch.Tensor):
        """Return mean and log_std."""
        features = self.features(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample action with tanh squashing.
        
        Returns
        -------
        action : Tensor
            Squashed and rescaled action.
        log_prob : Tensor
            Log probability with change-of-variables correction.
        mean : Tensor
            Mean action (for deterministic evaluation).
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        
        if deterministic:
            u = mean
        else:
            u = dist.rsample()  # Reparameterized sample for gradient flow
        
        # Tanh squashing
        action = torch.tanh(u)
        
        # Log-probability with change-of-variables correction
        log_prob = dist.log_prob(u)
        # Correction: log|det(da/du)| = sum(log(1 - tanh^2(u)))
        log_prob -= torch.log(1 - action.pow(2) + self.EPS)
        log_prob = log_prob.sum(dim=-1)
        
        # Rescale to actual action bounds
        action = action * self.action_scale + self.action_bias
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean_action


# ---------------------------------------------------------------------------
# Beta Policy (naturally bounded)
# ---------------------------------------------------------------------------

class BetaPolicy(nn.Module):
    """
    Beta distribution policy for bounded continuous action spaces [0, 1].
    
    Avoids the need for tanh squashing and log-probability corrections.
    
    Parameters
    ----------
    obs_dim : int
        Dimension of the observation space.
    act_dim : int
        Dimension of the action space.
    hidden_dim : int
        Hidden layer size.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.alpha_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
    
    def forward(self, obs: torch.Tensor):
        """Return alpha and beta parameters."""
        features = self.network(obs)
        alpha = F.softplus(self.alpha_head(features)) + 1.0  # > 1 for unimodal
        beta = F.softplus(self.beta_head(features)) + 1.0
        return alpha, beta
    
    def get_distribution(self, obs: torch.Tensor) -> Beta:
        alpha, beta = self.forward(obs)
        return Beta(alpha, beta)
    
    def get_action(self, obs: torch.Tensor):
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


# ---------------------------------------------------------------------------
# Actor-Critic with shared backbone
# ---------------------------------------------------------------------------

class ActorCriticShared(nn.Module):
    """
    Shared-backbone actor-critic network.
    
    Uses a common feature extractor with separate policy and value heads.
    Supports both discrete and continuous action spaces.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 64,
        continuous: bool = False,
    ):
        super().__init__()
        self.continuous = continuous
        
        # Shared feature extractor
        self.features = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        
        # Policy head
        if continuous:
            self.mean_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        else:
            self.policy_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        
        # Value head
        self.value_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.features(obs)).squeeze(-1)
    
    def get_action_and_value(self, obs: torch.Tensor, action=None):
        features = self.features(obs)
        value = self.value_head(features).squeeze(-1)
        
        if self.continuous:
            mean = self.mean_head(features)
            std = self.log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            logits = self.policy_head(features)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Separate Actor-Critic networks
# ---------------------------------------------------------------------------

class ActorCriticSeparate(nn.Module):
    """
    Separate-network actor-critic.
    
    Uses independent networks for policy and value to avoid
    gradient interference between the two objectives.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 64,
        continuous: bool = False,
    ):
        super().__init__()
        self.continuous = continuous
        
        # Actor network
        if continuous:
            self.actor = GaussianPolicy(obs_dim, act_dim, hidden_dim)
        else:
            self.actor = DiscretePolicy(obs_dim, act_dim, hidden_dim)
        
        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)
    
    def get_action_and_value(self, obs: torch.Tensor, action=None):
        value = self.get_value(obs)
        
        if action is None:
            action, log_prob, entropy = self.actor.get_action(obs)
        else:
            log_prob, entropy = self.actor.evaluate_actions(obs, action)
        
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Temperature-scaled policy
# ---------------------------------------------------------------------------

class TemperatureScaledPolicy(nn.Module):
    """
    Wrapper that applies temperature scaling to a discrete policy.
    
    Lower temperature → more greedy (exploitation)
    Higher temperature → more uniform (exploration)
    """
    
    def __init__(self, base_policy: DiscretePolicy, temperature: float = 1.0):
        super().__init__()
        self.base_policy = base_policy
        self.temperature = temperature
    
    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        logits = self.base_policy(obs)
        scaled_logits = logits / self.temperature
        return Categorical(logits=scaled_logits)
    
    def get_action(self, obs: torch.Tensor):
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_discrete_policy():
    """Demonstrate discrete policy with CartPole."""
    print("=" * 60)
    print("Discrete Policy Demo (CartPole-v1)")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    policy = DiscretePolicy(obs_dim, act_dim)
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    obs, _ = env.reset()
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    
    action, log_prob, entropy = policy.get_action(obs_t)
    print(f"\nSampled action: {action.item()}")
    print(f"Log probability: {log_prob.item():.4f}")
    print(f"Entropy: {entropy.item():.4f}")
    
    dist = policy.get_distribution(obs_t)
    print(f"Action probabilities: {dist.probs.detach().numpy().round(4)}")
    
    env.close()


def demo_gaussian_policy():
    """Demonstrate Gaussian policy with Pendulum."""
    print("\n" + "=" * 60)
    print("Gaussian Policy Demo (Pendulum-v1)")
    print("=" * 60)
    
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    policy = GaussianPolicy(obs_dim, act_dim)
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    obs, _ = env.reset()
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    
    action, log_prob, entropy = policy.get_action(obs_t)
    mean, log_std = policy(obs_t)
    
    print(f"\nMean: {mean.detach().numpy().round(4)}")
    print(f"Std: {log_std.exp().detach().numpy().round(4)}")
    print(f"Sampled action: {action.detach().numpy().round(4)}")
    print(f"Log probability: {log_prob.item():.4f}")
    print(f"Entropy: {entropy.item():.4f}")
    
    env.close()


def demo_squashed_gaussian():
    """Demonstrate squashed Gaussian policy."""
    print("\n" + "=" * 60)
    print("Squashed Gaussian Policy Demo")
    print("=" * 60)
    
    obs_dim, act_dim = 3, 1
    policy = SquashedGaussianPolicy(obs_dim, act_dim)
    
    obs = torch.randn(1, obs_dim)
    action, log_prob, mean_action = policy.get_action(obs)
    
    print(f"Sampled action (bounded): {action.detach().numpy().round(4)}")
    print(f"Mean action: {mean_action.detach().numpy().round(4)}")
    print(f"Log probability (with correction): {log_prob.item():.4f}")
    print(f"Action in [-1, 1]: {(action.abs() <= 1.0).all().item()}")


def demo_actor_critic():
    """Demonstrate actor-critic architectures."""
    print("\n" + "=" * 60)
    print("Actor-Critic Architecture Demo")
    print("=" * 60)
    
    obs_dim, act_dim = 4, 2
    
    # Shared backbone
    shared = ActorCriticShared(obs_dim, act_dim, continuous=False)
    obs = torch.randn(8, obs_dim)  # Batch of 8
    action, log_prob, entropy, value = shared.get_action_and_value(obs)
    print(f"\nShared Actor-Critic:")
    print(f"  Actions shape: {action.shape}")
    print(f"  Log probs shape: {log_prob.shape}")
    print(f"  Values shape: {value.shape}")
    print(f"  Params: {sum(p.numel() for p in shared.parameters()):,}")
    
    # Separate networks
    separate = ActorCriticSeparate(obs_dim, act_dim, continuous=False)
    action, log_prob, entropy, value = separate.get_action_and_value(obs)
    print(f"\nSeparate Actor-Critic:")
    print(f"  Actions shape: {action.shape}")
    print(f"  Log probs shape: {log_prob.shape}")
    print(f"  Values shape: {value.shape}")
    print(f"  Params: {sum(p.numel() for p in separate.parameters()):,}")


if __name__ == "__main__":
    demo_discrete_policy()
    demo_gaussian_policy()
    demo_squashed_gaussian()
    demo_actor_critic()
