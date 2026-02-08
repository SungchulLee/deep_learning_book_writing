"""
Chapter 34.2.3: Asynchronous Advantage Actor-Critic (A3C)
==========================================================
A3C implementation using PyTorch multiprocessing with shared
memory for global model and optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import List
import os


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

class A3CNetwork(nn.Module):
    """Actor-critic network for A3C with shared backbone."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)
    
    def forward(self, obs):
        features = self.features(obs)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value
    
    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ---------------------------------------------------------------------------
# Shared Adam Optimizer
# ---------------------------------------------------------------------------

class SharedAdam(torch.optim.Adam):
    """
    Adam optimizer with shared state for multiprocessing.
    
    Shares optimizer state tensors across processes using
    shared memory, enabling Hogwild-style updates.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        # Initialize state and share memory
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                
                # Share memory
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


# ---------------------------------------------------------------------------
# A3C Worker
# ---------------------------------------------------------------------------

def a3c_worker(
    rank: int,
    global_model: A3CNetwork,
    optimizer: SharedAdam,
    env_id: str,
    global_episode_counter: mp.Value,
    global_rewards: mp.Manager,
    max_episodes: int,
    gamma: float = 0.99,
    n_steps: int = 20,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 40.0,
):
    """
    A3C worker process.
    
    Each worker:
    1. Syncs local model from global
    2. Collects n_steps of experience
    3. Computes gradients locally
    4. Applies gradients to global model
    """
    torch.manual_seed(rank + 42)
    
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Local model (not shared)
    local_model = A3CNetwork(obs_dim, act_dim)
    
    obs, _ = env.reset(seed=rank)
    episode_reward = 0.0
    
    while True:
        # Check if training is done
        with global_episode_counter.get_lock():
            if global_episode_counter.value >= max_episodes:
                break
        
        # Sync local model from global
        local_model.load_state_dict(global_model.state_dict())
        
        # Collect n-step rollout
        states, actions, rewards, dones = [], [], [], []
        log_probs, values, entropies = [], [], []
        
        for _ in range(n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, entropy, value = local_model.get_action_and_value(obs_t)
            
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states.append(obs)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                with global_episode_counter.get_lock():
                    global_episode_counter.value += 1
                    ep_num = global_episode_counter.value
                
                global_rewards.append(episode_reward)
                
                if ep_num % 100 == 0:
                    recent = list(global_rewards)[-100:]
                    print(
                        f"Worker {rank} | Episode {ep_num} | "
                        f"Reward: {episode_reward:.1f} | "
                        f"Avg(100): {np.mean(recent):.1f}"
                    )
                
                episode_reward = 0.0
                obs, _ = env.reset()
                
                if ep_num >= max_episodes:
                    break
        
        # Compute returns and advantages
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        
        # Bootstrap value for last state
        with torch.no_grad():
            if dones[-1]:
                R = 0.0
            else:
                _, last_value = local_model(torch.FloatTensor(obs).unsqueeze(0))
                R = last_value.item()
        
        returns = []
        for t in reversed(range(len(rewards))):
            R = rewards[t] + gamma * R * (1 - dones[t])
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        # Forward pass with gradients
        _, log_probs_t, entropies_t, values_t = local_model.get_action_and_value(
            states_t, actions_t
        )
        
        # Advantages
        advantages = returns - values_t.detach()
        
        # Losses
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = F.mse_loss(values_t, returns)
        entropy_loss = -entropies_t.mean()
        
        total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        
        # Compute gradients on local model
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)
        
        # Transfer gradients to global model
        for local_param, global_param in zip(
            local_model.parameters(), global_model.parameters()
        ):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad.copy_(local_param.grad)
        
        # Apply gradients to global model (Hogwild-style)
        optimizer.step()
    
    env.close()


# ---------------------------------------------------------------------------
# A3C Trainer
# ---------------------------------------------------------------------------

class A3CTrainer:
    """
    A3C training coordinator.
    
    Spawns worker processes and manages global model/optimizer.
    """
    
    def __init__(
        self,
        env_id: str = "CartPole-v1",
        n_workers: int = 4,
        n_steps: int = 20,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 128,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        self.env_id = env_id
        self.n_workers = n_workers
        self.n_steps = n_steps
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Determine dimensions
        env = gym.make(env_id)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        env.close()
        
        # Global model (shared memory)
        self.global_model = A3CNetwork(obs_dim, act_dim, hidden_dim)
        self.global_model.share_memory()
        
        # Shared optimizer
        self.optimizer = SharedAdam(self.global_model.parameters(), lr=lr)
    
    def train(self, max_episodes: int = 2000) -> List[float]:
        """Train using multiple worker processes."""
        mp.set_start_method("spawn", force=True)
        
        # Shared counters
        global_episode_counter = mp.Value("i", 0)
        manager = mp.Manager()
        global_rewards = manager.list()
        
        # Spawn workers
        processes = []
        for rank in range(self.n_workers):
            p = mp.Process(
                target=a3c_worker,
                args=(
                    rank,
                    self.global_model,
                    self.optimizer,
                    self.env_id,
                    global_episode_counter,
                    global_rewards,
                    max_episodes,
                    self.gamma,
                    self.n_steps,
                    self.entropy_coef,
                    self.value_coef,
                ),
            )
            p.start()
            processes.append(p)
        
        # Wait for all workers
        for p in processes:
            p.join()
        
        return list(global_rewards)
    
    def evaluate(self, n_episodes: int = 10) -> float:
        """Evaluate the global model."""
        env = gym.make(self.env_id)
        rewards = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    logits, _ = self.global_model(obs_t)
                    action = logits.argmax(dim=-1).item()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            rewards.append(total_reward)
        
        env.close()
        return np.mean(rewards)


# ---------------------------------------------------------------------------
# Simplified single-process A3C simulation
# ---------------------------------------------------------------------------

class SimulatedA3C:
    """
    Simplified A3C simulation in a single process.
    
    Mimics A3C behavior by maintaining multiple environment instances
    and updating sequentially, useful for demonstration without
    multiprocessing complexity.
    """
    
    def __init__(
        self,
        env_id: str = "CartPole-v1",
        n_workers: int = 4,
        n_steps: int = 20,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 128,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        self.n_workers = n_workers
        self.n_steps = n_steps
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Create environments
        self.envs = [gym.make(env_id) for _ in range(n_workers)]
        obs_dim = self.envs[0].observation_space.shape[0]
        act_dim = self.envs[0].action_space.n
        
        # Global model
        self.global_model = A3CNetwork(obs_dim, act_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=lr)
        
        # Local models
        self.local_models = [
            A3CNetwork(obs_dim, act_dim, hidden_dim) for _ in range(n_workers)
        ]
    
    def train(self, max_episodes: int = 1000, print_interval: int = 100):
        all_rewards = []
        recent_rewards = deque(maxlen=100)
        episode_count = 0
        
        # Initialize observations
        obs_list = [env.reset(seed=i)[0] for i, env in enumerate(self.envs)]
        episode_rewards = [0.0] * self.n_workers
        
        while episode_count < max_episodes:
            # Each worker collects and updates
            for w in range(self.n_workers):
                # Sync local from global
                self.local_models[w].load_state_dict(self.global_model.state_dict())
                
                # Collect rollout
                states, actions, rewards, dones = [], [], [], []
                
                for _ in range(self.n_steps):
                    obs_t = torch.FloatTensor(obs_list[w]).unsqueeze(0)
                    with torch.no_grad():
                        action, _, _, _ = self.local_models[w].get_action_and_value(obs_t)
                    
                    next_obs, reward, terminated, truncated, _ = self.envs[w].step(action.item())
                    done = terminated or truncated
                    
                    states.append(obs_list[w])
                    actions.append(action.item())
                    rewards.append(reward)
                    dones.append(done)
                    episode_rewards[w] += reward
                    
                    obs_list[w] = next_obs
                    
                    if done:
                        all_rewards.append(episode_rewards[w])
                        recent_rewards.append(episode_rewards[w])
                        episode_count += 1
                        episode_rewards[w] = 0.0
                        obs_list[w], _ = self.envs[w].reset()
                        
                        if episode_count % print_interval == 0:
                            print(
                                f"Episode {episode_count} | "
                                f"Avg(100): {np.mean(recent_rewards):.1f}"
                            )
                        
                        if episode_count >= max_episodes:
                            break
                
                if episode_count >= max_episodes:
                    break
                
                # Compute returns
                states_t = torch.FloatTensor(np.array(states))
                actions_t = torch.LongTensor(actions)
                
                with torch.no_grad():
                    if dones[-1]:
                        R = 0.0
                    else:
                        _, last_v = self.local_models[w](
                            torch.FloatTensor(obs_list[w]).unsqueeze(0)
                        )
                        R = last_v.item()
                
                returns_list = []
                for t in reversed(range(len(rewards))):
                    R = rewards[t] + self.gamma * R * (1 - dones[t])
                    returns_list.insert(0, R)
                returns_t = torch.FloatTensor(returns_list)
                
                # Forward + loss
                _, lp, ent, val = self.local_models[w].get_action_and_value(states_t, actions_t)
                adv = returns_t - val.detach()
                
                loss = (
                    -(lp * adv).mean()
                    + self.value_coef * F.mse_loss(val, returns_t)
                    - self.entropy_coef * ent.mean()
                )
                
                # Apply gradients to global model
                self.optimizer.zero_grad()
                loss.backward()
                
                for local_p, global_p in zip(
                    self.local_models[w].parameters(),
                    self.global_model.parameters()
                ):
                    if global_p.grad is None:
                        global_p.grad = local_p.grad.clone()
                    else:
                        global_p.grad.copy_(local_p.grad)
                
                nn.utils.clip_grad_norm_(self.global_model.parameters(), 40.0)
                self.optimizer.step()
        
        for env in self.envs:
            env.close()
        
        return all_rewards


from collections import deque


def demo_simulated_a3c():
    """Demo A3C in single-process simulation."""
    print("=" * 60)
    print("Simulated A3C on CartPole-v1")
    print("=" * 60)
    
    agent = SimulatedA3C(
        env_id="CartPole-v1",
        n_workers=4,
        n_steps=20,
        lr=1e-3,
        gamma=0.99,
        hidden_dim=128,
        entropy_coef=0.01,
        value_coef=0.5,
    )
    
    rewards = agent.train(max_episodes=1000, print_interval=200)
    
    if len(rewards) >= 100:
        print(f"\nFinal avg reward (last 100): {np.mean(rewards[-100:]):.1f}")


if __name__ == "__main__":
    demo_simulated_a3c()
