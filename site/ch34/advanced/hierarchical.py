"""
Chapter 34.5.2: Hierarchical Reinforcement Learning
=====================================================
Simple hierarchical RL with options framework and
goal-conditioned policies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from collections import deque


class OptionPolicy(nn.Module):
    """Intra-option policy for a single option."""
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, obs):
        return Categorical(logits=self.net(obs))


class TerminationFunction(nn.Module):
    """Option termination probability β(s)."""
    def __init__(self, obs_dim, n_options, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_options),
        )
    def forward(self, obs):
        return torch.sigmoid(self.net(obs))


class OptionCritic(nn.Module):
    """
    Option-Critic Architecture (Bacon et al., 2017).
    
    Learns options (intra-option policies + termination functions)
    and a policy over options simultaneously.
    """
    def __init__(self, obs_dim, act_dim, n_options=4, hidden=64):
        super().__init__()
        self.n_options = n_options
        
        # Policy over options
        self.option_policy = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_options),
        )
        
        # Intra-option policies
        self.options = nn.ModuleList([
            OptionPolicy(obs_dim, act_dim, hidden) for _ in range(n_options)
        ])
        
        # Termination functions
        self.termination = TerminationFunction(obs_dim, n_options, hidden)
        
        # Value functions Q(s, ω) for each option
        self.q_options = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_options),
        )
    
    def get_option(self, obs):
        """Select option using epsilon-greedy over Q(s, ω)."""
        q = self.q_options(obs)
        return q.argmax(dim=-1)
    
    def get_action(self, obs, option_idx):
        """Get action from intra-option policy."""
        dist = self.options[option_idx](obs)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def should_terminate(self, obs, option_idx):
        """Check if option should terminate."""
        beta = self.termination(obs)
        terminate_prob = beta[:, option_idx]
        return torch.bernoulli(terminate_prob).bool()


class OptionCriticAgent:
    """Option-Critic training agent."""
    
    def __init__(self, env, n_options=4, lr=1e-3, gamma=0.99, hidden=64):
        self.env = env
        self.gamma = gamma
        self.n_options = n_options
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        self.model = OptionCritic(obs_dim, act_dim, n_options, hidden)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epsilon = 0.1
    
    def train(self, n_episodes=1000, print_interval=100):
        rewards_history = []
        recent = deque(maxlen=100)
        option_usage = np.zeros(self.n_options)
        
        for ep in range(1, n_episodes + 1):
            obs, _ = self.env.reset()
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            
            # Select initial option
            if np.random.random() < self.epsilon:
                current_option = np.random.randint(self.n_options)
            else:
                with torch.no_grad():
                    current_option = self.model.get_option(obs_t).item()
            
            total_reward = 0.0
            done = False
            transitions = []
            
            while not done:
                # Get action from current option
                action, log_prob = self.model.get_action(obs_t, current_option)
                
                next_obs, reward, term, trunc, _ = self.env.step(action.item())
                done = term or trunc
                next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0)
                
                transitions.append((obs_t, current_option, action, log_prob, reward, next_obs_t, done))
                option_usage[current_option] += 1
                total_reward += reward
                
                # Check termination
                if not done:
                    with torch.no_grad():
                        should_term = self.model.should_terminate(next_obs_t, current_option).item()
                    
                    if should_term:
                        if np.random.random() < self.epsilon:
                            current_option = np.random.randint(self.n_options)
                        else:
                            with torch.no_grad():
                                current_option = self.model.get_option(next_obs_t).item()
                
                obs_t = next_obs_t
            
            # Update
            self._update(transitions)
            
            rewards_history.append(total_reward)
            recent.append(total_reward)
            
            if ep % print_interval == 0:
                usage = option_usage / option_usage.sum() * 100
                print(
                    f"Episode {ep:>5d} | Avg(100): {np.mean(recent):>7.1f} | "
                    f"Options: {usage.round(1)}"
                )
        
        return rewards_history
    
    def _update(self, transitions):
        total_loss = torch.tensor(0.0)
        
        for obs, opt, act, lp, reward, next_obs, done in transitions:
            with torch.no_grad():
                q_next = self.model.q_options(next_obs).squeeze(0)
                beta_next = self.model.termination(next_obs).squeeze(0)[opt]
                
                # Option value: (1-β)Q(s',ω) + β max_ω' Q(s',ω')
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * (
                        (1 - beta_next) * q_next[opt] + beta_next * q_next.max()
                    )
            
            q_current = self.model.q_options(obs).squeeze(0)[opt]
            q_loss = (q_current - target).pow(2)
            
            # Policy loss
            advantage = (target - q_current).detach()
            policy_loss = -lp * advantage
            
            # Termination loss
            beta = self.model.termination(obs).squeeze(0)[opt]
            q_omega = self.model.q_options(obs).squeeze(0)
            term_advantage = q_omega[opt] - q_omega.max()
            term_loss = beta * term_advantage.detach()
            
            total_loss = total_loss + q_loss + policy_loss + 0.01 * term_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()


# ---------------------------------------------------------------------------
# Goal-Conditioned Policy
# ---------------------------------------------------------------------------

class GoalConditionedPolicy(nn.Module):
    """Policy conditioned on a goal: π(a|s, g)."""
    
    def __init__(self, obs_dim, goal_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
    
    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)
        return Categorical(logits=self.net(x))


def demo_option_critic():
    print("=" * 60)
    print("Option-Critic on CartPole-v1")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    agent = OptionCriticAgent(env, n_options=4, lr=1e-3, gamma=0.99)
    rewards = agent.train(n_episodes=500, print_interval=100)
    env.close()
    
    if len(rewards) >= 100:
        print(f"\nFinal avg (last 100): {np.mean(rewards[-100:]):.1f}")


if __name__ == "__main__":
    demo_option_critic()
