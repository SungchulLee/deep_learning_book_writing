"""
Chapter 34.6.3: Observation Normalization
==========================================
Running normalization utilities for observations, rewards,
and advantages in RL training.
"""

import torch
import numpy as np
from typing import Optional


class RunningMeanStd:
    """Welford's online algorithm for running mean and variance."""
    
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        if x.ndim == 1 and self.mean.shape:
            batch_mean = x
            batch_var = np.zeros_like(self.mean)
            batch_count = 1
        
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = M2 / total
        self.count = total


class ObservationNormalizer:
    """
    Normalize observations using running statistics.
    
    Parameters
    ----------
    shape : tuple
        Observation shape.
    clip : float
        Clip normalized values to [-clip, clip].
    """
    
    def __init__(self, shape, clip=10.0):
        self.rms = RunningMeanStd(shape)
        self.clip = clip
        self.training = True
    
    def normalize(self, obs):
        if self.training:
            if isinstance(obs, np.ndarray):
                x = obs if obs.ndim > 1 else obs.reshape(1, -1)
                self.rms.update(x)
            
        normalized = (obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)
    
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True


class RewardNormalizer:
    """
    Normalize rewards by running standard deviation (not mean).
    
    Only divides by std to preserve reward sign and relative magnitude.
    """
    
    def __init__(self, clip=10.0):
        self.rms = RunningMeanStd(())
        self.clip = clip
        self.training = True
    
    def normalize(self, reward):
        if self.training:
            self.rms.update(np.array([reward]))
        
        normalized = reward / (np.sqrt(self.rms.var) + 1e-8)
        return np.clip(normalized, -self.clip, self.clip)
    
    def eval(self):
        self.training = False


class VecNormalizer:
    """
    Combined normalizer for vectorized environments.
    Handles observation and reward normalization together.
    """
    
    def __init__(self, obs_shape, n_envs, clip_obs=10.0, clip_rew=10.0,
                 normalize_obs=True, normalize_rew=True, gamma=0.99):
        self.normalize_obs = normalize_obs
        self.normalize_rew = normalize_rew
        
        if normalize_obs:
            self.obs_rms = RunningMeanStd(obs_shape)
        if normalize_rew:
            self.ret_rms = RunningMeanStd(())
            self.returns = np.zeros(n_envs)
        
        self.clip_obs = clip_obs
        self.clip_rew = clip_rew
        self.gamma = gamma
        self.training = True
    
    def normalize_obs_fn(self, obs):
        if not self.normalize_obs:
            return obs
        if self.training:
            self.obs_rms.update(obs)
        normalized = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(normalized, -self.clip_obs, self.clip_obs).astype(np.float32)
    
    def normalize_reward_fn(self, rewards, dones):
        if not self.normalize_rew:
            return rewards
        if self.training:
            self.returns = self.returns * self.gamma + rewards
            self.ret_rms.update(self.returns)
            self.returns[dones.astype(bool)] = 0.0
        
        normalized = rewards / (np.sqrt(self.ret_rms.var) + 1e-8)
        return np.clip(normalized, -self.clip_rew, self.clip_rew)


def normalize_advantages(advantages, eps=1e-8):
    """Normalize advantages to zero mean, unit variance (per minibatch)."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def demo_normalization():
    print("=" * 60)
    print("Observation and Reward Normalization Demo")
    print("=" * 60)
    
    # Simulate observations with different scales
    np.random.seed(42)
    obs_dim = 4
    
    # Feature 0: large scale, Feature 1: small scale, etc.
    scales = np.array([100.0, 0.01, 50.0, 0.001])
    offsets = np.array([500.0, -0.05, 25.0, 0.0])
    
    normalizer = ObservationNormalizer(shape=(obs_dim,))
    
    print("\nBefore normalization (raw observations):")
    raw_obs = []
    for _ in range(100):
        obs = np.random.randn(obs_dim) * scales + offsets
        normalizer.normalize(obs)  # Update stats
        raw_obs.append(obs)
    
    raw_obs = np.array(raw_obs)
    print(f"  Mean:  {raw_obs.mean(axis=0).round(3)}")
    print(f"  Std:   {raw_obs.std(axis=0).round(3)}")
    
    # Now normalize
    norm_obs = np.array([normalizer.normalize(o) for o in raw_obs])
    print("\nAfter normalization:")
    print(f"  Mean:  {norm_obs.mean(axis=0).round(3)}")
    print(f"  Std:   {norm_obs.std(axis=0).round(3)}")
    print(f"  Range: [{norm_obs.min():.2f}, {norm_obs.max():.2f}]")
    
    # Reward normalization
    print("\n" + "-" * 40)
    print("Reward Normalization:")
    
    rew_normalizer = RewardNormalizer()
    rewards = np.random.randn(50) * 100 + 50  # Large scale rewards
    norm_rewards = [rew_normalizer.normalize(r) for r in rewards]
    
    print(f"  Raw:  mean={rewards.mean():.1f}, std={rewards.std():.1f}")
    print(f"  Norm: mean={np.mean(norm_rewards):.3f}, std={np.std(norm_rewards):.3f}")
    
    # Advantage normalization
    print("\n" + "-" * 40)
    print("Advantage Normalization:")
    advantages = torch.randn(32) * 5 + 2
    norm_adv = normalize_advantages(advantages)
    print(f"  Raw:  mean={advantages.mean():.3f}, std={advantages.std():.3f}")
    print(f"  Norm: mean={norm_adv.mean():.6f}, std={norm_adv.std():.3f}")


if __name__ == "__main__":
    demo_normalization()
