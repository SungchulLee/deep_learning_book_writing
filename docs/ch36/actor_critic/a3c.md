# 발 안 맞춘 이점 행위자-비평가(A3C)

발 안 맞춘 이점 행위자-비평가(A3C)는 행위자-비평가 방법에서 종요로운 생각이다. 온 세상 모형과 가장 좋게 하는 개를 위한 함께 쓰는 기억을 다룬다. 이 구현은 여기에 걸린 종요로운 알고리즘과 자료 얼개를 손으로 만져 보이며, 이치 바탕과 실제로 서비스에 올릴 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""
34.2.3장: 발 안 맞춘 이점 행위자-비평가(A3C)
==========================================================
온 세상 모형과 가장 좋게 하는 개를 위한 함께 쓰는 기억을
쓰는, 파이토치 여러 프로세스 A3C 구현.
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

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# 행위자-비평가 그물
# ---------------------------------------------------------------------------

class A3CNetwork(nn.Module):
    """등뼈를 함께 쓰는 A3C용 행위자-비평가 그물."""
    
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
        
        # 무게 첫 값 매기기
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
# 함께 쓰는 Adam 가장 좋게 하는 개
# ---------------------------------------------------------------------------

class SharedAdam(torch.optim.Adam):
    """
    여러 프로세스를 위해 상태를 함께 쓰는 Adam.
    
    함께 쓰는 기억으로 가장 좋게 하는 개의 상태 텐서를 프로세스
    사이에서 함께 써 Hogwild 꼴 고침을 이루게 한다.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        # 상태에 첫 값을 매기고 기억을 함께 쓴다
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                
                # 기억 함께 쓰기
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


# ---------------------------------------------------------------------------
# A3C 일꾼
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
    A3C 일꾼 프로세스.
    
    일꾼마다:
    1. 제 모형을 온 세상 모형에 맞춘다
    2. n_steps만큼 겪음을 모은다
    3. 제자리에서 기울기를 셈한다
    4. 그 기울기를 온 세상 모형에 매긴다
    """
    torch.manual_seed(rank + 42)
    
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # 제 모형(함께 쓰지 않는다)
    local_model = A3CNetwork(obs_dim, act_dim)
    
    obs, _ = env.reset(seed=rank)
    episode_reward = 0.0
    
    while True:
        # 익힘이 끝났는지 살핀다
        with global_episode_counter.get_lock():
            if global_episode_counter.value >= max_episodes:
                break
        
        # 제 모형을 온 세상 모형에 맞춘다
        local_model.load_state_dict(global_model.state_dict())
        
        # n걸음 굴림을 모은다
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
        
        # 돌아옴과 이점을 셈한다
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        
        # 마지막 상태의 부트스트랩 값
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
        
        # 기울기를 켜고 앞으로 지나가기
        _, log_probs_t, entropies_t, values_t = local_model.get_action_and_value(
            states_t, actions_t
        )
        
        # 이점
        advantages = returns - values_t.detach()
        
        # 손실
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = F.mse_loss(values_t, returns)
        entropy_loss = -entropies_t.mean()
        
        total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        
        # 제 모형에서 기울기를 셈한다
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)
        
        # 기울기를 온 세상 모형으로 옮긴다
        for local_param, global_param in zip(
            local_model.parameters(), global_model.parameters()
        ):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad.copy_(local_param.grad)
        
        # 온 세상 모형에 기울기를 매긴다(Hogwild 꼴)
        optimizer.step()
    
    env.close()


# ---------------------------------------------------------------------------
# A3C 익힘 이끄개
# ---------------------------------------------------------------------------

class A3CTrainer:
    """
    A3C 익힘을 이끄는 것.
    
    일꾼 프로세스를 돋아나게 하고 온 세상 모형과 가장 좋게 하는
    개를 다룬다.
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
        
        # 차원을 알아낸다
        env = gym.make(env_id)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        env.close()
        
        # 온 세상 모형(함께 쓰는 기억)
        self.global_model = A3CNetwork(obs_dim, act_dim, hidden_dim)
        self.global_model.share_memory()
        
        # 함께 쓰는 가장 좋게 하는 개
        self.optimizer = SharedAdam(self.global_model.parameters(), lr=lr)
    
    def train(self, max_episodes: int = 2000) -> List[float]:
        """일꾼 프로세스 여럿으로 익힌다."""
        mp.set_start_method("spawn", force=True)
        
        # 함께 쓰는 세개
        global_episode_counter = mp.Value("i", 0)
        manager = mp.Manager()
        global_rewards = manager.list()
        
        # 일꾼을 돋아나게 한다
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
        
        # 온 일꾼을 기다린다
        for p in processes:
            p.join()
        
        return list(global_rewards)
    
    def evaluate(self, n_episodes: int = 10) -> float:
        """온 세상 모형을 따진다."""
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
# 프로세스 하나로 쉽게 흉내 낸 A3C
# ---------------------------------------------------------------------------

class SimulatedA3C:
    """
    프로세스 하나에서 쉽게 흉내 낸 A3C.
    
    둘레를 여럿 지니고 차례대로 고쳐 A3C의 거동을 흉내 낸다.
    여러 프로세스의 얽힘 없이 보여 주기에 쓸모 있다.
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
        
        # 둘레를 만든다
        self.envs = [gym.make(env_id) for _ in range(n_workers)]
        obs_dim = self.envs[0].observation_space.shape[0]
        act_dim = self.envs[0].action_space.n
        
        # 온 세상 모형
        self.global_model = A3CNetwork(obs_dim, act_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=lr)
        
        # 제 모형들
        self.local_models = [
            A3CNetwork(obs_dim, act_dim, hidden_dim) for _ in range(n_workers)
        ]
    
    def train(self, max_episodes: int = 1000, print_interval: int = 100):
        all_rewards = []
        recent_rewards = deque(maxlen=100)
        episode_count = 0
        
        # 봄에 첫 값을 매긴다
        obs_list = [env.reset(seed=i)[0] for i, env in enumerate(self.envs)]
        episode_rewards = [0.0] * self.n_workers
        
        while episode_count < max_episodes:
            # 일꾼마다 모으고 고친다
            for w in range(self.n_workers):
                # 제 모형을 온 세상 모형에 맞춘다
                self.local_models[w].load_state_dict(self.global_model.state_dict())
                
                # 굴림을 모은다
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
                
                # 돌아옴을 셈한다
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
                
                # 앞으로 지나가기 + 손실
                _, lp, ent, val = self.local_models[w].get_action_and_value(states_t, actions_t)
                adv = returns_t - val.detach()
                
                loss = (
                    -(lp * adv).mean()
                    + self.value_coef * F.mse_loss(val, returns_t)
                    - self.entropy_coef * ent.mean()
                )
                
                # 온 세상 모형에 기울기를 매긴다
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
    """프로세스 하나로 흉내 낸 A3C를 보여 준다."""
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
```

## 2. 논의

이 구현은 발 안 맞춘 이점 행위자-비평가(A3C)의 한가운데 논리를 담은 `A3CNetwork`, `SharedAdam`, `A3CTrainer` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

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

**다룬 것** — 발 안 맞춘 이점 행위자-비평가(A3C)

이 구현은 발 안 맞춘 이점 행위자-비평가(A3C)의 한가운데 논리를 담은 `A3CNetwork`, `SharedAdam`, `A3CTrainer` 클래스를 축으로 삼는다.

고갱이 갈래는 `A3CNetwork`, `SharedAdam`, `A3CTrainer`, `SimulatedA3C`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
