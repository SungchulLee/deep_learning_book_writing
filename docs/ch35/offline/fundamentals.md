# 오프라인 힘 북돋우는 배움의 바탕

오프라인 힘 북돋우는 배움은 둘레와 더 주고받지 않고 앞서 모은 자료 뭉치만으로 쓸모 있는 방침을 배우려 한다. 이 자리에는 남다른 어려움이 있다. DQN 같은 여느 알고리즘을 어수룩하게 쓰면 분포 밖 움직임의 값이 지나치게 어림되어 방침이 나빠진다. 이 짜기는 품질이 다른 움직임 방침으로 자료 뭉치를 만들고, 매임 없는 오프라인 DQN이 Q 값 발산으로 왜 안 되는지 보이며, 오프라인 알고리즘을 따질 밑그림을 세운다.

## 1. 코드

```python
"""
33.5.1 묶음 힘 북돋우는 배움의 바탕
================================

자료 뭉치 만들기, 묶음 값 매기기, 그리고 묶음 환경에서 어설픈 DQN이
왜 틀리는지 보이기.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
from typing import Dict, List, Tuple
import random

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# 묶음 자료 뭉치 만들기
# ---------------------------------------------------------------------------

def collect_dataset(env_name: str = 'CartPole-v1', n_transitions: int = 10000,
                    policy_type: str = 'medium', seed: int = 42) -> Dict[str, np.ndarray]:
    """정한 움직임 방침의 질로 묶음 자료 뭉치를 모은다.
    
    policy_type: 'random', 'medium', 'expert', 'mixed'
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make(env_name)
    sd = env.observation_space.shape[0]
    ad = env.action_space.n

    # 정한 질까지 Q 그물 익히기
    q_net = nn.Sequential(nn.Linear(sd, 64), nn.ReLU(),
                          nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, ad))

    if policy_type in ('medium', 'expert', 'mixed'):
        target_net = nn.Sequential(nn.Linear(sd, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, ad))
        target_net.load_state_dict(q_net.state_dict())
        opt = optim.Adam(q_net.parameters(), lr=1e-3)
        buf_s, buf_a, buf_r, buf_ns, buf_d = [], [], [], [], []

        n_train = {'medium': 100, 'expert': 500, 'mixed': 200}[policy_type]
        step = 0
        for ep in range(n_train):
            s, _ = env.reset(); done = False
            while not done:
                step += 1
                eps = max(0.05, 1.0 - step / 3000)
                if random.random() < eps:
                    a = env.action_space.sample()
                else:
                    with torch.no_grad():
                        a = q_net(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
                ns, r, term, trunc, _ = env.step(a)
                done = term or trunc
                buf_s.append(s); buf_a.append(a); buf_r.append(r)
                buf_ns.append(ns); buf_d.append(float(done))
                if len(buf_s) > 64:
                    idx = np.random.randint(0, len(buf_s), 64)
                    st = torch.FloatTensor(np.array(buf_s)[idx])
                    at = torch.LongTensor(np.array(buf_a)[idx])
                    rt = torch.FloatTensor(np.array(buf_r)[idx])
                    nst = torch.FloatTensor(np.array(buf_ns)[idx])
                    dt = torch.FloatTensor(np.array(buf_d)[idx])
                    q = q_net(st).gather(1, at.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        nq = target_net(nst).max(1)[0]
                        tgt = rt + (1 - dt) * 0.99 * nq
                    loss = nn.functional.mse_loss(q, tgt)
                    opt.zero_grad(); loss.backward(); opt.step()
                    if step % 200 == 0:
                        target_net.load_state_dict(q_net.state_dict())
                s = ns

    # 익힌(또는 아무) 방침으로 자료 뭉치 모으기
    states, actions, rewards, next_states, dones = [], [], [], [], []
    count = 0
    while count < n_transitions:
        s, _ = env.reset(); done = False
        while not done and count < n_transitions:
            if policy_type == 'random':
                a = env.action_space.sample()
            elif policy_type == 'mixed':
                if random.random() < 0.3:
                    a = env.action_space.sample()
                else:
                    with torch.no_grad():
                        a = q_net(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            else:
                eps = 0.1 if policy_type == 'medium' else 0.01
                if random.random() < eps:
                    a = env.action_space.sample()
                else:
                    with torch.no_grad():
                        a = q_net(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()

            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            states.append(s); actions.append(a); rewards.append(r)
            next_states.append(ns); dones.append(float(done))
            s = ns; count += 1

    env.close()
    return {
        'states': np.array(states, dtype=np.float32),
        'actions': np.array(actions, dtype=np.int64),
        'rewards': np.array(rewards, dtype=np.float32),
        'next_states': np.array(next_states, dtype=np.float32),
        'dones': np.array(dones, dtype=np.float32),
    }


def dataset_statistics(dataset: Dict[str, np.ndarray]) -> Dict:
    """묶음 자료 뭉치의 셈밝힘을 셈한다."""
    n = len(dataset['rewards'])
    ep_rewards = []
    current = 0
    for i in range(n):
        current += dataset['rewards'][i]
        if dataset['dones'][i] > 0.5:
            ep_rewards.append(current)
            current = 0
    if current > 0:
        ep_rewards.append(current)

    unique_actions, action_counts = np.unique(dataset['actions'], return_counts=True)
    return {
        'n_transitions': n,
        'n_episodes': len(ep_rewards),
        'mean_episode_reward': np.mean(ep_rewards) if ep_rewards else 0,
        'std_episode_reward': np.std(ep_rewards) if ep_rewards else 0,
        'action_distribution': dict(zip(unique_actions.tolist(), 
                                        (action_counts / action_counts.sum()).tolist())),
        'state_mean': dataset['states'].mean(axis=0),
        'state_std': dataset['states'].std(axis=0),
    }


# ---------------------------------------------------------------------------
# 어설픈 묶음 DQN(틀림을 보이려고)
# ---------------------------------------------------------------------------

def train_offline_dqn(dataset: Dict, n_steps: int = 5000, lr: float = 1e-3,
                      batch_size: int = 64, gamma: float = 0.99,
                      target_freq: int = 200) -> nn.Module:
    """고정된 자료 뭉치에서 순전히 묶음으로 DQN을 익힌다."""
    sd = dataset['states'].shape[1]
    ad = int(dataset['actions'].max()) + 1
    n = len(dataset['rewards'])

    q_net = nn.Sequential(nn.Linear(sd, 128), nn.ReLU(),
                          nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, ad))
    target_net = nn.Sequential(nn.Linear(sd, 128), nn.ReLU(),
                                nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, ad))
    target_net.load_state_dict(q_net.state_dict())
    opt = optim.Adam(q_net.parameters(), lr=lr)

    losses = []
    q_means = []
    for step in range(n_steps):
        idx = np.random.randint(0, n, batch_size)
        s = torch.FloatTensor(dataset['states'][idx])
        a = torch.LongTensor(dataset['actions'][idx])
        r = torch.FloatTensor(dataset['rewards'][idx])
        ns = torch.FloatTensor(dataset['next_states'][idx])
        d = torch.FloatTensor(dataset['dones'][idx])

        q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = target_net(ns).max(1)[0]
            tgt = r + (1 - d) * gamma * nq
        loss = nn.functional.mse_loss(q, tgt)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        opt.step()

        losses.append(loss.item())
        q_means.append(q.mean().item())
        if step % target_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

    return q_net, losses, q_means


def evaluate_policy(q_net: nn.Module, env_name: str = 'CartPole-v1',
                    n_episodes: int = 20) -> Dict:
    env = gym.make(env_name)
    returns = []
    for _ in range(n_episodes):
        s, _ = env.reset(); total = 0; done = False
        while not done:
            with torch.no_grad():
                a = q_net(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc; total += r
        returns.append(total)
    env.close()
    return {'mean': np.mean(returns), 'std': np.std(returns),
            'min': np.min(returns), 'max': np.max(returns)}


# ---------------------------------------------------------------------------
# 시연
# ---------------------------------------------------------------------------

def demo_offline_fundamentals():
    print("=" * 60)
    print("Offline RL Fundamentals Demo")
    print("=" * 60)

    # --- 자료 뭉치 만들기 ---
    print("\n--- Dataset Generation ---")
    for ptype in ['random', 'medium', 'expert']:
        dataset = collect_dataset(policy_type=ptype, n_transitions=5000)
        stats = dataset_statistics(dataset)
        print(f"\n  {ptype.upper()} dataset:")
        print(f"    Transitions: {stats['n_transitions']}")
        print(f"    Episodes: {stats['n_episodes']}")
        print(f"    Avg episode reward: {stats['mean_episode_reward']:.1f} "
              f"± {stats['std_episode_reward']:.1f}")
        print(f"    Action dist: {stats['action_distribution']}")

    # --- 어설픈 DQN의 틀림 보이기 ---
    print("\n--- Naive Offline DQN ---")
    for ptype in ['random', 'medium', 'expert']:
        dataset = collect_dataset(policy_type=ptype, n_transitions=5000)
        q_net, losses, q_means = train_offline_dqn(dataset, n_steps=3000)
        eval_result = evaluate_policy(q_net)
        print(f"\n  {ptype.upper()} data → Offline DQN:")
        print(f"    Eval: {eval_result['mean']:.1f} ± {eval_result['std']:.1f}")
        print(f"    Final loss: {np.mean(losses[-100:]):.4f}")
        print(f"    Q-value range: [{min(q_means):.2f}, {max(q_means):.2f}]")
        if max(q_means) > 50:
            print(f"    ⚠ Q-values diverging! (max={max(q_means):.1f})")

    print("\nOffline RL fundamentals demo complete!")


if __name__ == "__main__":
    demo_offline_fundamentals()
```

## 2. 논의

이 짜기는 바탕의 핵심 연산을 짜는 여러 연장 함수를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

보여 주기 함수는 이 조각들을 여느 힘 북돋우는 배움 잣대에 실제로 쓰는 모습을 보인다. 내놓기를 살피면 웃잡 고름과 문제 짜임에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실제 관점에서 이 짜기는 순수한 성능보다 또렷함을 앞세운다. 실제로 쓰는 얼개는 보통 묶음 셈, GPU 빠르게 하기, 더 정교한 윗매개변수 맞추기 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 생각은 큰 규모의 쓰임새로 곧바로 옮겨 간다.

## 연습문제

**연습문제 1.**
보여 주기 코드를 돌려 핵심 내놓기 잣대를 적어라. 윗매개변수 하나(배움 빠르기, 숨은 차원, 층 개수 같은 것)를 고치고 결과가 어떻게 바뀌는지 적어라.

??? success "연습문제 1 풀이"
    보여 주기를 돌린 뒤 나머지를 붙박아 두고 고른 윗매개변수를 차근히 바꾼다. 보기로 숨은 차원을 두 배로 하면 보통 나타냄 담이가 늘지만 셈 시간이 커진다. 배움 빠르기는 단조롭지 않은 영향을 준다. 너무 작으면 느리게 모이고 너무 크면 흔들린다. 고른 윗매개변수의 서로 다른 값을 적어도 셋 잡아 구체적인 수를 적어 두라.

---

**연습문제 2.**
이 짜기에서 핵심 얼개 고르기의 몫을 밝혀라. 왜 그 깨움 함수, 고르게 맞추기 셈속, 손실 함수를 쓰는가? 다른 것으로 바꾸면 어떻게 되는가?

??? success "연습문제 2 풀이"
    이 얼개 고르기는 오프라인 힘 북돋우는 배움에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 오프라인 힘 북돋우는 배움의 바탕

이 짜기는 바탕의 핵심 연산을 짜는 여러 연장 함수를 한가운데 둔다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
