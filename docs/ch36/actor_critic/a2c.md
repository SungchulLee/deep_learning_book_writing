# 이점 행위자-비평가(A2C)

이점 행위자-비평가(A2C)는 행위자-비평가 방법에서 종요로운 생각이다. n걸음 돌아옴과 행위자-비평가를 함께 익히는 것을 다룬다. 이 구현은 여기에 걸린 종요로운 알고리즘과 자료 얼개를 손으로 만져 보이며, 이치 바탕과 실제로 서비스에 올릴 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""
34.2.2장: 이점 행위자-비평가(A2C)
=============================================
벡터 둘레, n걸음 돌아옴, 행위자-비평가를 함께 익히는
온전한 A2C 구현.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import List
from collections import deque

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# 행위자-비평가 그물
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class A2CNetwork(nn.Module):
    """A2C를 위한, 함께 쓰는 행위자-비평가 그물."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
    
    def forward(self, obs):
        features = self.features(obs)
        return self.actor(features), self.critic(features).squeeze(-1)
    
    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
    
    def get_value(self, obs):
        return self.forward(obs)[1]


# ---------------------------------------------------------------------------
# 벡터 둘레 감싸개
# ---------------------------------------------------------------------------

class VecEnv:
    """
    쉬운 발맞춘 벡터 둘레.
    
    둘레 N개를 나란히 돌리며 발맞추어 걸음을 옮긴다.
    """
    
    def __init__(self, env_id: str, n_envs: int, seed: int = 0):
        self.envs = [gym.make(env_id) for _ in range(n_envs)]
        self.n_envs = n_envs
        for i, env in enumerate(self.envs):
            env.reset(seed=seed + i)
    
    def reset(self):
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return np.array(obs_list)
    
    def step(self, actions):
        obs_list, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 끝나면 저절로 되돌린다
            if done:
                info["terminal_obs"] = obs
                info["episode_reward"] = info.get("episode", {}).get("r", None)
                obs, _ = env.reset()
            
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return np.array(obs_list), np.array(rewards), np.array(dones), infos
    
    @property
    def observation_space(self):
        return self.envs[0].observation_space
    
    @property
    def action_space(self):
        return self.envs[0].action_space
    
    def close(self):
        for env in self.envs:
            env.close()


# ---------------------------------------------------------------------------
# A2C 부림꾼
# ---------------------------------------------------------------------------

class A2C:
    """
    발맞춘 벡터 둘레를 쓰는 이점 행위자-비평가(A2C).
    
    매개변수
    ----------
    env_id : str
        Gymnasium 둘레 아이디.
    n_envs : int
        나란한 둘레의 개수.
    n_steps : int
        둘레마다 굴림 길이.
    lr : float
        배움률.
    gamma : float
        깎기 인자.
    value_coef : float
        값 손실 계수.
    entropy_coef : float
        엔트로피 덤 계수.
    max_grad_norm : float
        자르기를 위한 기울기 노름의 최댓값.
    use_gae : bool
        GAE를 쓸지(True) n걸음 돌아옴을 쓸지(False).
    gae_lambda : float
        GAE 람다 매개변수.
    """
    
    def __init__(
        self,
        env_id: str = "CartPole-v1",
        n_envs: int = 8,
        n_steps: int = 5,
        lr: float = 7e-4,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        use_gae: bool = False,
        gae_lambda: float = 0.95,
        seed: int = 0,
    ):
        self.gamma = gamma
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        
        # 둘레 갖추기
        self.envs = VecEnv(env_id, n_envs, seed=seed)
        obs_dim = self.envs.observation_space.shape[0]
        act_dim = self.envs.action_space.n
        
        # 그물 갖추기
        self.network = A2CNetwork(obs_dim, act_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
    
    def collect_rollout(self, obs):
        """
        온 둘레에서 n_steps만큼의 겪음을 모은다.
        
        돌려주는 값
        -------
        obs : ndarray (이어 가기 위한 다음 봄)
        rollout : 텐서 사전
        episode_rewards : 마친 에피소드 보상의 목록
        """
        # 곳간
        mb_obs = np.zeros((self.n_steps, self.n_envs) + self.envs.observation_space.shape)
        mb_actions = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
        mb_rewards = np.zeros((self.n_steps, self.n_envs))
        mb_dones = np.zeros((self.n_steps, self.n_envs))
        mb_values = np.zeros((self.n_steps, self.n_envs))
        mb_log_probs = np.zeros((self.n_steps, self.n_envs))
        
        episode_rewards = []
        
        for step in range(self.n_steps):
            mb_obs[step] = obs
            
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs)
                action, log_prob, _, value = self.network.get_action_and_value(obs_t)
            
            mb_actions[step] = action.numpy()
            mb_values[step] = value.numpy()
            mb_log_probs[step] = log_prob.numpy()
            
            obs, rewards, dones, infos = self.envs.step(action.numpy())
            mb_rewards[step] = rewards
            mb_dones[step] = dones
            
            # 에피소드가 마친 것을 좇는다
            for i, done in enumerate(dones):
                if done:
                    # 갈무리한 보상으로 에피소드 보상을 셈한다
                    ep_reward = sum(mb_rewards[s, i] for s in range(step + 1))
                    episode_rewards.append(ep_reward)
        
        rollout = {
            "obs": torch.FloatTensor(mb_obs.reshape(-1, *self.envs.observation_space.shape)),
            "actions": torch.LongTensor(mb_actions.reshape(-1)),
            "rewards": mb_rewards,
            "dones": mb_dones,
            "values": mb_values,
            "log_probs": mb_log_probs,
        }
        
        return obs, rollout, episode_rewards
    
    def compute_returns_and_advantages(self, rollout, last_obs):
        """돌아옴 과녁과 이점을 셈한다."""
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"]
        
        with torch.no_grad():
            last_value = self.network.get_value(
                torch.FloatTensor(last_obs)
            ).numpy()
        
        if self.use_gae:
            # GAE 셈하기
            advantages = np.zeros_like(rewards)
            last_gae = 0
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_values = last_value
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_values = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
                advantages[t] = last_gae = \
                    delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            
            returns = advantages + values
        else:
            # n걸음 돌아옴(여느 A2C)
            returns = np.zeros_like(rewards)
            R = last_value
            for t in reversed(range(self.n_steps)):
                R = rewards[t] + self.gamma * R * (1.0 - dones[t])
                returns[t] = R
            advantages = returns - values
        
        returns = torch.FloatTensor(returns.reshape(-1))
        advantages = torch.FloatTensor(advantages.reshape(-1))
        
        # 이점을 고르게 한다
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, rollout, returns, advantages):
        """A2C 고침을 벌인다."""
        obs = rollout["obs"]
        actions = rollout["actions"]
        
        # 앞으로 지나가기
        _, new_log_probs, entropy, new_values = \
            self.network.get_action_and_value(obs, actions)
        
        # 방침 손실
        policy_loss = -(new_log_probs * advantages).mean()
        
        # 값 손실
        value_loss = nn.functional.mse_loss(new_values, returns)
        
        # 엔트로피 손실
        entropy_loss = -entropy.mean()
        
        # 온 손실
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": total_loss.item(),
        }
    
    def train(self, total_steps: int = 200000, print_interval: int = 10000) -> List[float]:
        """A2C 부림꾼을 익힌다."""
        obs = self.envs.reset()
        all_rewards = []
        recent_rewards = deque(maxlen=100)
        steps = 0
        updates = 0
        
        while steps < total_steps:
            obs, rollout, episode_rewards = self.collect_rollout(obs)
            returns, advantages = self.compute_returns_and_advantages(rollout, obs)
            metrics = self.update(rollout, returns, advantages)
            
            steps += self.n_steps * self.n_envs
            updates += 1
            
            for r in episode_rewards:
                all_rewards.append(r)
                recent_rewards.append(r)
            
            if steps % print_interval < self.n_steps * self.n_envs and len(recent_rewards) > 0:
                print(
                    f"Step {steps:>8d} | "
                    f"Avg(100): {np.mean(recent_rewards):>7.1f} | "
                    f"Policy: {metrics['policy_loss']:>7.4f} | "
                    f"Value: {metrics['value_loss']:>7.4f} | "
                    f"H: {metrics['entropy']:>5.3f}"
                )
        
        self.envs.close()
        return all_rewards
    
    def evaluate(self, env_id: str, n_episodes: int = 10) -> float:
        """익힌 방침을 따진다."""
        env = gym.make(env_id)
        rewards = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    logits, _ = self.network(obs_t)
                    action = logits.argmax(dim=-1).item()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            rewards.append(total_reward)
        
        env.close()
        return np.mean(rewards)


# ---------------------------------------------------------------------------
# 보여 주기
# ---------------------------------------------------------------------------

def demo_a2c():
    """CartPole에서 A2C를 익힌다."""
    print("=" * 60)
    print("A2C on CartPole-v1")
    print("=" * 60)
    
    agent = A2C(
        env_id="CartPole-v1",
        n_envs=8,
        n_steps=5,
        lr=7e-4,
        gamma=0.99,
        hidden_dim=64,
        value_coef=0.5,
        entropy_coef=0.01,
        use_gae=False,
        seed=42,
    )
    
    rewards = agent.train(total_steps=200000, print_interval=20000)
    
    # 따지기
    eval_reward = agent.evaluate("CartPole-v1", n_episodes=20)
    print(f"\nEvaluation reward (20 episodes): {eval_reward:.1f}")
    
    return rewards


def demo_a2c_with_gae():
    """CartPole에서 GAE를 쓰는 A2C를 익힌다."""
    print("\n" + "=" * 60)
    print("A2C with GAE on CartPole-v1")
    print("=" * 60)
    
    agent = A2C(
        env_id="CartPole-v1",
        n_envs=8,
        n_steps=128,
        lr=2.5e-4,
        gamma=0.99,
        hidden_dim=64,
        value_coef=0.5,
        entropy_coef=0.01,
        use_gae=True,
        gae_lambda=0.95,
        seed=42,
    )
    
    rewards = agent.train(total_steps=200000, print_interval=20000)
    
    eval_reward = agent.evaluate("CartPole-v1", n_episodes=20)
    print(f"\nEvaluation reward (20 episodes): {eval_reward:.1f}")
    
    return rewards


if __name__ == "__main__":
    demo_a2c()
    demo_a2c_with_gae()
```

## 2. 논의

이 구현은 이점 행위자-비평가(A2C)의 한가운데 논리를 담은 `A2CNetwork`, `VecEnv`, `A2C` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

보여 주기 함수는 이 조각들을 여느 힘 북돋우는 배움 잣대에 실제로 써 보인다. 그 출력을 살피면 매개변수 고름과 문제 얼개에 따라 알고리즘의 됨됨이가 어떻게 달라지는지 볼 수 있다.

쓰임의 눈으로 보면 이 구현은 날 성능보다 또렷함을 앞세운다. 서비스 시스템은 묶음 셈하기, GPU 빠르게 하기, 더 야무진 매개변수 벼리기 같은 다듬기를 더 넣는 것이 보통이다. 그렇더라도 여기서 보인 한가운데 알고리즘 생각은 큰 잣대의 쓰임새에 그대로 옮겨 간다.

## 연습문제

**연습문제 1.**
보여 주기 코드를 돌리고 종요로운 출력 재기를 적어라. 매개변수 하나(배움률, 숨은 차원, 켜 개수 따위)를 고쳐 열매가 어떻게 달라지는지 밝혀라.

??? success "연습문제 1 풀이"
    보여 주기를 돌린 뒤 다른 것을 붙박아 두고 고른 매개변수만 짜임 있게 바꾼다. 보기로 숨은 차원을 곱절로 늘리면 나타내는 그릇이 커지지만 셈하는 때가 는다. 배움률은 한결같지 않은 결과를 낳는다. 너무 작으면 더디게 모이고 너무 크면 들쭉날쭉해진다. 고른 매개변수의 서로 다른 값 적어도 셋에 대해 또렷한 수를 적어 두라.

---

**연습문제 2.**
이 구현에서 종요로운 얼개 고름이 맡은 몫을 풀어라. 왜 그런 활성 함수, 고르게 하기 꾀, 손실 함수를 쓰는가? 다른 것으로 바꾸면 무슨 일이 생기는가?

??? success "연습문제 2 풀이"
    이 얼개 고름은 행위자-비평가 방법에서 자리 잡은 좋은 버릇을 비춘다. 보기로 ReLU 활성은 곧지 않음을 주면서 0보다 큰 들임에서 기울기가 사라지는 것을 막는다. 손실 함수는 일감 갈래에 맞추어 고른다(갈래 나누기에는 사귐 엔트로피, 되돌이에는 평균 제곱 잘못). 다른 것으로 바꾸면(보기로 시그모이드 활성, L1 손실) 가장 좋게 하기 지형이 바뀌어 됨됨이가 나빠질 수 있으나, 어떤 자리에서는 바꾸는 것이 이로울 수도 있다.

---

**연습문제 3.**
이 구현을 더 만만치 않은 자리로 넓혀라. 더 큰 자료 뭉치, 다른 문제 갈래, 덧붙인 기능 가운데 하나를 고르라. 고친 바를 밝히고 됨됨이에 미친 바를 따져라.

??? success "연습문제 3 풀이"
    절로 떠오르는 넓히기 하나는 정칙화(드롭아웃, 무게 삭임)나 더 야무진 얼개(켜 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓히기를 만들고 같은 자료로 익힌 뒤 앞뒤의 재기를 견주어라. 이 넓히기는 처음 알고리즘과 고친 바의 이치 밑뜻을 모두 아는 것을 보여야 한다.

## 정리하며

**다룬 것** — 이점 행위자-비평가(A2C)

이 구현은 이점 행위자-비평가(A2C)의 한가운데 논리를 담은 `A2CNetwork`, `VecEnv`, `A2C` 클래스를 축으로 삼는다.

고갱이 갈래는 `A2CNetwork`, `VecEnv`, `A2C`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
