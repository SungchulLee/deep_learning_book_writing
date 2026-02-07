# GANs in Reinforcement Learning

GANs and reinforcement learning (RL) share fundamental connections through their adversarial training dynamics. This section explores how GANs enhance RL systems and vice versa.

## GAN-RL Connections

### Theoretical Links

| GAN Component | RL Analogue |
|---------------|-------------|
| Generator | Policy |
| Discriminator | Reward function |
| Latent vector | State |
| Generated sample | Action/Trajectory |
| Adversarial loss | Reward signal |

### Key Applications

1. **Imitation Learning**: Learn policies from demonstrations
2. **World Models**: Generate environment simulations
3. **Reward Learning**: Infer rewards from expert behavior
4. **Data Augmentation**: Generate training experiences

## Generative Adversarial Imitation Learning (GAIL)

GAIL learns policies by matching expert demonstrations without explicit reward engineering:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    """Policy network (Generator in GAIL)."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Mean and log_std for Gaussian policy
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        features = self.net(state)
        mean = self.mean(features)
        std = self.log_std.exp()
        return mean, std
    
    def sample(self, state):
        """Sample action from policy."""
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()  # Reparameterized sample
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def log_prob(self, state, action):
        """Compute log probability of action."""
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)


class GAILDiscriminator(nn.Module):
    """
    Discriminator for GAIL.
    
    Distinguishes expert (state, action) pairs from policy-generated pairs.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
    
    def reward(self, state, action):
        """
        Compute reward signal for RL.
        
        Uses -log(1 - D(s, a)) as reward to encourage
        state-action pairs that look like expert demonstrations.
        """
        with torch.no_grad():
            logits = self.forward(state, action)
            # Reward: -log(1 - sigmoid(logits))
            reward = -F.logsigmoid(-logits)
        return reward.squeeze(-1)


class GAIL:
    """
    Generative Adversarial Imitation Learning.
    
    Combines GAN discriminator with policy gradient RL.
    """
    
    def __init__(self, state_dim, action_dim, expert_data, 
                 hidden_dim=256, lr=3e-4, device='cpu'):
        self.device = device
        
        # Policy (generator)
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Discriminator
        self.discriminator = GAILDiscriminator(state_dim, action_dim, hidden_dim).to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        
        # Expert demonstrations
        self.expert_states = torch.FloatTensor(expert_data['states']).to(device)
        self.expert_actions = torch.FloatTensor(expert_data['actions']).to(device)
    
    def collect_trajectories(self, env, num_steps):
        """Collect trajectories using current policy."""
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        state = env.reset()
        
        for _ in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob = self.policy.sample(state_tensor)
            
            action_np = action.cpu().numpy()[0]
            next_state, _, done, _ = env.step(action_np)
            
            # Get reward from discriminator
            reward = self.discriminator.reward(state_tensor, action)
            
            states.append(state)
            actions.append(action_np)
            rewards.append(reward.item())
            log_probs.append(log_prob.item())
            
            state = next_state if not done else env.reset()
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'log_probs': np.array(log_probs),
        }
    
    def update_discriminator(self, policy_data, batch_size=64):
        """Update discriminator to distinguish expert from policy."""
        # Sample from expert data
        expert_idx = np.random.choice(len(self.expert_states), batch_size)
        expert_states = self.expert_states[expert_idx]
        expert_actions = self.expert_actions[expert_idx]
        
        # Sample from policy data
        policy_idx = np.random.choice(len(policy_data['states']), batch_size)
        policy_states = torch.FloatTensor(policy_data['states'][policy_idx]).to(self.device)
        policy_actions = torch.FloatTensor(policy_data['actions'][policy_idx]).to(self.device)
        
        # Discriminator outputs
        expert_logits = self.discriminator(expert_states, expert_actions)
        policy_logits = self.discriminator(policy_states, policy_actions)
        
        # Binary cross-entropy loss
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones_like(expert_logits)
        )
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_logits, torch.zeros_like(policy_logits)
        )
        
        disc_loss = expert_loss + policy_loss
        
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()
        
        return disc_loss.item()
    
    def update_policy(self, trajectories):
        """Update policy using PPO-style update with discriminator rewards."""
        states = torch.FloatTensor(trajectories['states']).to(self.device)
        actions = torch.FloatTensor(trajectories['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(trajectories['log_probs']).to(self.device)
        rewards = torch.FloatTensor(trajectories['rewards']).to(self.device)
        
        # Compute returns (simplified - no value function baseline)
        returns = self._compute_returns(rewards, gamma=0.99)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update
        for _ in range(10):  # Multiple epochs
            new_log_probs = self.policy.log_prob(states, actions)
            ratio = (new_log_probs - old_log_probs).exp()
            
            # Clipped surrogate objective
            clip_ratio = torch.clamp(ratio, 0.8, 1.2)
            policy_loss = -torch.min(ratio * returns, clip_ratio * returns).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
        return policy_loss.item()
    
    def _compute_returns(self, rewards, gamma):
        """Compute discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def train(self, env, num_iterations=1000, steps_per_iter=2048):
        """Main training loop."""
        for iteration in range(num_iterations):
            # Collect trajectories
            trajectories = self.collect_trajectories(env, steps_per_iter)
            
            # Update discriminator
            disc_loss = self.update_discriminator(trajectories)
            
            # Update policy
            policy_loss = self.update_policy(trajectories)
            
            if iteration % 10 == 0:
                avg_reward = np.mean(trajectories['rewards'])
                print(f'Iter {iteration}: Disc Loss={disc_loss:.4f}, '
                      f'Policy Loss={policy_loss:.4f}, Avg Reward={avg_reward:.4f}')
```

## World Model GANs

GANs can learn environment dynamics for model-based RL:

```python
class WorldModelGAN:
    """
    GAN-based world model for environment simulation.
    
    Generator: Predicts next state given current state and action
    Discriminator: Distinguishes real transitions from predicted ones
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, device='cpu'):
        self.device = device
        
        # Transition generator
        self.generator = TransitionGenerator(state_dim, action_dim, hidden_dim).to(device)
        
        # Transition discriminator
        self.discriminator = TransitionDiscriminator(state_dim, action_dim, hidden_dim).to(device)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=1e-4)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4)
    
    def train_step(self, real_transitions):
        """
        Train world model on batch of real transitions.
        
        Args:
            real_transitions: Dict with 'states', 'actions', 'next_states'
        """
        states = torch.FloatTensor(real_transitions['states']).to(self.device)
        actions = torch.FloatTensor(real_transitions['actions']).to(self.device)
        next_states = torch.FloatTensor(real_transitions['next_states']).to(self.device)
        
        batch_size = states.size(0)
        
        # =====================================================================
        # Train Discriminator
        # =====================================================================
        self.d_optimizer.zero_grad()
        
        # Real transitions
        real_logits = self.discriminator(states, actions, next_states)
        real_loss = F.binary_cross_entropy_with_logits(
            real_logits, torch.ones(batch_size, 1, device=self.device)
        )
        
        # Fake transitions (generated next states)
        with torch.no_grad():
            fake_next_states = self.generator(states, actions)
        
        fake_logits = self.discriminator(states, actions, fake_next_states)
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros(batch_size, 1, device=self.device)
        )
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # =====================================================================
        # Train Generator
        # =====================================================================
        self.g_optimizer.zero_grad()
        
        fake_next_states = self.generator(states, actions)
        
        # Adversarial loss
        fake_logits = self.discriminator(states, actions, fake_next_states)
        g_adv_loss = F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones(batch_size, 1, device=self.device)
        )
        
        # Reconstruction loss (for stability)
        g_recon_loss = F.mse_loss(fake_next_states, next_states)
        
        g_loss = g_adv_loss + 10.0 * g_recon_loss
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'recon_loss': g_recon_loss.item(),
        }
    
    def imagine_trajectory(self, initial_state, policy, horizon):
        """
        Generate imagined trajectory using world model.
        
        Args:
            initial_state: Starting state
            policy: Policy network
            horizon: Number of steps to imagine
        
        Returns:
            Imagined trajectory
        """
        states = [initial_state]
        actions = []
        
        state = torch.FloatTensor(initial_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(horizon):
                # Get action from policy
                action, _ = policy.sample(state)
                actions.append(action.cpu().numpy()[0])
                
                # Predict next state
                next_state = self.generator(state, action)
                states.append(next_state.cpu().numpy()[0])
                
                state = next_state
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
        }


class TransitionGenerator(nn.Module):
    """Predicts next state given state and action."""
    
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)
        return state + delta  # Predict residual


class TransitionDiscriminator(nn.Module):
    """Distinguishes real from generated transitions."""
    
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state, action, next_state):
        x = torch.cat([state, action, next_state], dim=-1)
        return self.net(x)
```

## Reward Learning with GANs

```python
class InverseRLGAN:
    """
    Learn reward function from expert demonstrations using GAN.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, device='cpu'):
        self.device = device
        
        # Reward network
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)
        
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=1e-4)
    
    def compute_reward(self, state, action):
        """Compute learned reward."""
        x = torch.cat([state, action], dim=-1)
        return self.reward_net(x)
    
    def update(self, expert_sa, policy_sa):
        """
        Update reward function to separate expert from policy behavior.
        
        Learned reward should be high for expert, low for policy.
        """
        self.optimizer.zero_grad()
        
        expert_rewards = self.compute_reward(expert_sa['states'], expert_sa['actions'])
        policy_rewards = self.compute_reward(policy_sa['states'], policy_sa['actions'])
        
        # MaxEnt IRL objective: maximize expert reward, minimize policy reward
        loss = -expert_rewards.mean() + torch.logsumexp(policy_rewards, dim=0)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

## Summary

| Application | GAN Role | RL Role |
|-------------|----------|---------|
| **GAIL** | Learn reward from demos | Optimize policy |
| **World Models** | Generate environment | Plan in imagination |
| **Inverse RL** | Learn reward function | Define task |
| **Data Augmentation** | Generate experiences | Train more efficiently |

GANs provide powerful tools for RL systems, enabling learning from demonstrations, model-based planning, and efficient reward specification.
