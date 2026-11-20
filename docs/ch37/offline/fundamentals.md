# 오프라인 강화학습의 기초

오프라인 강화학습은 둘레와 더 주고받지 않고 앞서 모은 데이터셋만으로 쓸모 있는 폴리시를 배우려 한다. 이 설정에는 고유한 어려움이 있다. DQN 같은 일반 알고리즘을 단순하게 적용하면 분포 밖 액션의 값이 과대추정되어 폴리시가 나빠진다. 이 구현은 품질이 다른 행동 폴리시로 데이터셋을 만들고, 제약 없는 오프라인 DQN이 Q 값 발산으로 실패할 수 있음을 살피며, 오프라인 알고리즘을 평가할 기반을 세운다.

## 1. 코드

```python
"""
35.5.1 배치 힘 북돋우는 러닝의 바탕
================================

자료 뭉치 만들기, 배치 값 매기기, 그리고 배치 환경에서 어설픈 DQN이
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
# 배치 자료 뭉치 만들기
# ---------------------------------------------------------------------------

def collect_dataset(env_name: str = 'CartPole-v1', n_transitions: int = 10000,
                    policy_type: str = 'medium', seed: int = 42) -> Dict[str, np.ndarray]:
    """정한 액션 폴리시의 질로 배치 자료 뭉치를 모은다.
    
    policy_type: 'random', 'medium', 'expert', 'mixed'
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make(env_name)
    sd = env.observation_space.shape[0]
    ad = env.action_space.n

    # 정한 질까지 Q 네트워크 익히기
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

    # 익힌(또는 아무) 폴리시로 자료 뭉치 모으기
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
    """배치 자료 뭉치의 셈밝힘을 셈한다."""
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
# 어설픈 배치 DQN(틀림을 보이려고)
# ---------------------------------------------------------------------------

def train_offline_dqn(dataset: Dict, n_steps: int = 5000, lr: float = 1e-3,
                      batch_size: int = 64, gamma: float = 0.99,
                      target_freq: int = 200) -> nn.Module:
    """고정된 자료 뭉치에서 순전히 배치로 DQN을 익힌다."""
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

이 구현은 네 함수로 되어 있다. `collect_dataset` 은 CartPole 에서 행동 폴리시의 품질을 넷으로 나눠 고정 데이터셋을 만든다. `'random'` 은 무작위 행동이고, `'medium'`·`'expert'`·`'mixed'` 는 먼저 DQN 을 각각 100·500·200 에피소드 온라인으로 학습한 뒤 그 Q 네트워크로 $\varepsilon = 0.1$, $\varepsilon = 0.01$, 30% 무작위 섞기로 데이터를 모은다. 이름은 품질을 노린 것이지 보장하지는 않는다. 「expert」가 정말 전문가 수준인지는 출력의 평균 에피소드 리워드로 확인해야 한다. `dataset_statistics` 는 전이 수, 에피소드 수, 평균 에피소드 리워드, 행동 비율을 센다. `train_offline_dqn` 은 둘레와 전혀 주고받지 않고 데이터셋에서만 미니배치를 뽑아 일반 DQN 업데이트를 되풀이한다. `evaluate_policy` 는 학습한 Q 네트워크의 그리디 폴리시를 실제 둘레에서 20 에피소드 돌려 평가한다.

데모 함수는 두 부분이다. 먼저 random·medium·expert 데이터셋을 전이 5,000 개씩 만들어 통계를 찍는다. 다음으로 같은 세 데이터셋(같은 시드 42 로 다시 만들므로 앞과 똑같다)에서 오프라인 DQN 을 3,000 스텝 학습하고, 평가 리턴, 마지막 로스, 업데이트마다 기록한 배치 평균 Q 값의 범위를 찍는다. 평균 Q 가 50 을 넘으면 「Q-values diverging」 경고를 내는데, 이 기준은 CartPole 에서 맞지 않는다(연습문제 3). 실제 오프라인 강화학습에서는 `evaluate_policy` 처럼 둘레에서 평가하는 것 자체가 허락되지 않을 때가 많다는 점도 기억해 두자. 이 데모의 평가는 진단을 위한 것이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`collect_dataset` 은 전이가 `n_transitions` 개가 되면 에피소드 중간이라도 멈춘다. `dataset_statistics` 는 이 마지막 미완성 에피소드를 어떻게 다루는가? 그것이 `Avg episode reward` 를 어느 쪽으로 치우치게 하며, 그 치우침은 random 데이터셋과 expert 데이터셋 가운데 어디서 더 큰가?

</div>

??? success "풀이"
    `dataset_statistics` 는 `done` 을 만날 때마다 누적 리워드를 에피소드 하나로 닫고, 끝에 남은 `current` 가 0 보다 크면 그것도 에피소드 하나로 넣는다. 그런데 이 마지막 조각은 중간에 잘린 에피소드라 리워드가 실제보다 작으므로, 평균을 **아래쪽**으로 끌어내린다. 치우침의 크기는 에피소드 수에 반비례한다. random 폴리시는 에피소드가 평균 20 스텝 남짓이라 5,000 전이에 에피소드가 200 개쯤이어서, 조각 하나의 영향이 작다. 폴리시가 좋아 에피소드가 500 스텝까지 가면 5,000 전이에 에피소드가 10 개 안팎뿐이라, 잘린 조각 하나가 평균을 눈에 띄게 낮출 수 있다. 고치려면 마지막 조각을 빼거나, 데이터셋을 에피소드 단위로 끝내도록 수집을 바꾼다. `n_episodes` 도 같은 이유로 하나 많게 세어진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
expert 데이터셋은 $\varepsilon = 0.01$ 로 모았다. 어떤 상태에서 학습된 폴리시가 고르지 **않는** 행동이 데이터에 나올 확률은 얼마인가? 이것이 `train_offline_dqn` 의 타깃 $r + \gamma\max_{a'}Q(s', a')$ 에 어떤 문제를 일으키는지, random 데이터셋과 견주어 설명하라.

</div>

??? success "풀이"
    $\varepsilon = 0.01$ 이면 1% 의 확률로 두 행동 가운데 하나를 무작위로 고르므로, 그리디가 아닌 행동이 나올 확률은 $0.01 \times \tfrac12 = 0.005$ 다. 전이 5,000 개에서 25 개쯤뿐이다. 그러면 대부분의 상태에서 $Q(s, a_{\text{비그리디}})$ 는 데이터로 한 번도 맞춰지지 않고, 네트워크가 이웃 상태에서 외삽한 값을 그대로 갖는다. 타깃의 $\max_{a'}$ 는 이 맞춰지지 않은 값이 우연히 크면 바로 그것을 고르고, 그 큰 값이 부트스트랩으로 다른 상태의 Q 로 번진다. 둘레에서 그 행동을 해 보고 바로잡을 길이 없으므로 오차가 고쳐지지 않는다. 이것을 외삽 오차(extrapolation error) 또는 분포 밖 액션 문제라 한다. random 데이터셋은 두 행동이 반반이라 모든 상태–행동이 고루 덮이지만, 막대가 금방 쓰러지는 상태만 담겨 있어 좋은 폴리시가 방문할 상태를 거의 모른다. 데이터의 품질과 덮음이 서로 다른 축이라는 점이 오프라인 강화학습의 출발점이다. BCQ·CQL·IQL 은 모두 데이터에 없는 행동의 값을 믿지 않도록 타깃이나 로스를 바꾼다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
데모는 배치 평균 Q 값이 50 을 넘으면 「Q-values diverging」 경고를 낸다. CartPole 에서 $\gamma = 0.99$ 일 때 참 Q 값이 가질 수 있는 범위를 구하고, 이 기준이 왜 맞지 않는지 설명하라. 발산을 제대로 진단하는 방법을 제안하라.

</div>

??? success "풀이"
    매 스텝 리워드가 1 이므로 참 Q 값은 $\sum_{k=0}^{T-1}\gamma^k = (1 - \gamma^T)/(1 - \gamma)$ 로 1 이상 100 미만이다. 500 스텝에서 잘려도 약 99.3 이다. 좋은 폴리시가 모은 데이터라면 대부분의 상태에서 참값이 90 을 넘으므로, 평균 Q 가 50 을 넘는 것은 오히려 **제대로 배우고 있다는** 신호일 수 있다. 반대로 random 데이터에서는 에피소드가 20 스텝 남짓이라 참값이 20 이하인데, 그곳에서 Q 가 40 이 되어도 경고가 나지 않는다. 고정된 문턱 대신 두 가지를 본다. 첫째, 이론적 상한 $1/(1-\gamma) = 100$ 을 넘는 Q 는 확실한 과대추정이다. 둘째, 데이터셋의 각 전이에서 에피소드 끝까지의 할인 리턴을 계산해, 데이터에 있는 $(s, a)$ 에서의 $Q(s, a)$ 와 견준다. 여기에 더해 데이터에 **없는** 행동의 $Q(s, a')$ 이 데이터에 있는 행동보다 체계적으로 큰지를 보면, 연습문제 2 의 외삽 오차가 실제로 일어나는지 확인할 수 있다. 업데이트가 거듭될수록 이 차이가 벌어지면 발산이다.

## 정리하며

**다룬 것** — 오프라인 강화학습의 기초

이 구현은 품질이 다른 오프라인 데이터셋을 만드는 `collect_dataset`, 그 통계를 보는 `dataset_statistics`, 제약 없이 데이터셋만으로 학습하는 `train_offline_dqn`, 그리고 진단용 `evaluate_policy` 를 담는다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
