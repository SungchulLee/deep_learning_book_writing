# 익히기 굽은 줄

익히기 굽은 줄은 힘 북돋우는 배움 따지기 방법의 중요한 개념이다. 힘 북돋우는 배움 익히기 굽은 줄을 기록하고 매끄럽게 하고 그리는 연장을 다룬다. 이 짜기는 얽힌 핵심 알고리즘과 자료 얼개를 손으로 만져 보게 하며 이론 바탕과 실제로 펼칠 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""
33.6.1 익히기 굽은 금
========================

힘 북돋우는 배움 익히기 굽은 금을 적고 매끄럽게 하고 그리는 연장.
"""

import numpy as np
import matplotlib

# ========================================================================
# 메인
# ========================================================================
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import json
import os


class TrainingLogger:
    """힘 북돋우는 배움 실험을 위한 두루 살피는 익히기 기록개."""

    def __init__(self, name: str = "experiment"):
        self.name = name
        self.data: Dict[str, List[float]] = {
            'episode_rewards': [], 'episode_lengths': [],
            'losses': [], 'q_values': [], 'grad_norms': [],
            'epsilons': [], 'eval_rewards': [], 'eval_episodes': [],
        }

    def log_episode(self, reward, length, epsilon=None):
        self.data['episode_rewards'].append(reward)
        self.data['episode_lengths'].append(length)
        if epsilon is not None:
            self.data['epsilons'].append(epsilon)

    def log_step(self, loss=None, q_value=None, grad_norm=None):
        if loss is not None: self.data['losses'].append(loss)
        if q_value is not None: self.data['q_values'].append(q_value)
        if grad_norm is not None: self.data['grad_norms'].append(grad_norm)

    def log_eval(self, episode, reward):
        self.data['eval_episodes'].append(episode)
        self.data['eval_rewards'].append(reward)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in self.data.items()}, f)

    @classmethod
    def load(cls, path: str) -> 'TrainingLogger':
        logger = cls()
        with open(path) as f:
            logger.data = json.load(f)
        return logger


def moving_average(data: List[float], window: int = 100) -> np.ndarray:
    """간단한 움직이는 평균."""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')


def exponential_moving_average(data: List[float], beta: float = 0.99) -> np.ndarray:
    """지수 움직이는 평균."""
    ema = np.zeros(len(data))
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = beta * ema[i-1] + (1 - beta) * data[i]
    return ema


def percentile_bands(data: List[float], window: int = 100) -> Dict[str, np.ndarray]:
    """흐르는 백분위 띠를 셈한다."""
    n = len(data)
    medians, lows, highs = [], [], []
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = data[start:i+1]
        medians.append(np.median(chunk))
        lows.append(np.percentile(chunk, 25))
        highs.append(np.percentile(chunk, 75))
    return {'median': np.array(medians), 'p25': np.array(lows), 'p75': np.array(highs)}


def plot_training_curves(logger: TrainingLogger, save_path: str = 'training_curves.png'):
    """두루 살피는 익히기 굽은 금 그림을 만든다."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Curves: {logger.name}', fontsize=14)

    # 1. 마당 보상
    ax = axes[0, 0]
    rewards = logger.data['episode_rewards']
    if rewards:
        ax.plot(rewards, alpha=0.2, color='blue')
        if len(rewards) >= 50:
            ma = moving_average(rewards, 50)
            ax.plot(range(49, len(rewards)), ma, color='red', label='MA(50)')
        ax.set_title('Episode Rewards'); ax.set_xlabel('Episode'); ax.set_ylabel('Return')
        ax.legend()

    # 2. 손실
    ax = axes[0, 1]
    losses = logger.data['losses']
    if losses:
        ax.plot(losses, alpha=0.2, color='orange')
        if len(losses) >= 100:
            ma = moving_average(losses, 100)
            ax.plot(range(99, len(losses)), ma, color='red')
        ax.set_title('Training Loss'); ax.set_xlabel('Step'); ax.set_ylabel('Loss')
        ax.set_yscale('log')

    # 3. Q 값
    ax = axes[0, 2]
    qvals = logger.data['q_values']
    if qvals:
        ax.plot(qvals, alpha=0.3, color='green')
        if len(qvals) >= 100:
            ma = moving_average(qvals, 100)
            ax.plot(range(99, len(qvals)), ma, color='darkgreen')
        ax.set_title('Mean Q-Value'); ax.set_xlabel('Step')

    # 4. 마당 길이
    ax = axes[1, 0]
    lengths = logger.data['episode_lengths']
    if lengths:
        ax.plot(lengths, alpha=0.2, color='purple')
        if len(lengths) >= 50:
            ma = moving_average(lengths, 50)
            ax.plot(range(49, len(lengths)), ma, color='darkviolet')
        ax.set_title('Episode Length'); ax.set_xlabel('Episode')

    # 5. 기울기 크기
    ax = axes[1, 1]
    gnorms = logger.data['grad_norms']
    if gnorms:
        ax.plot(gnorms, alpha=0.2, color='brown')
        if len(gnorms) >= 100:
            ma = moving_average(gnorms, 100)
            ax.plot(range(99, len(gnorms)), ma, color='darkred')
        ax.set_title('Gradient Norm'); ax.set_xlabel('Step')

    # 6. 값 매김 보상
    ax = axes[1, 2]
    eval_ep = logger.data['eval_episodes']
    eval_r = logger.data['eval_rewards']
    if eval_ep:
        ax.plot(eval_ep, eval_r, 'o-', color='teal')
        ax.set_title('Evaluation Return'); ax.set_xlabel('Episode')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_multi_seed(all_rewards: List[List[float]], labels: List[str] = None,
                    window: int = 50, save_path: str = 'multi_seed.png'):
    """여러 번 돌린 것을 평균과 표준편차 띠로 그린다."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, rewards in enumerate(all_rewards):
        label = labels[i] if labels else f'Run {i+1}'
        color = colors[i % len(colors)]
        smoothed = moving_average(rewards, window)
        x = range(window - 1, len(rewards))
        ax.plot(x, smoothed, color=color, label=label)
        ax.fill_between(x, smoothed - np.std(rewards[-len(smoothed):]),
                        smoothed + np.std(rewards[-len(smoothed):]),
                        alpha=0.1, color=color)

    ax.set_xlabel('Episode'); ax.set_ylabel('Return')
    ax.set_title('Multi-Seed Comparison'); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Multi-seed plot saved to {save_path}")


def demo_training_curves():
    print("=" * 60)
    print("Training Curves Demo")
    print("=" * 60)

    # 흉내 익히기 자료 만들기
    np.random.seed(42)
    n_episodes = 500
    logger = TrainingLogger("DQN-CartPole")

    for ep in range(n_episodes):
        # 나아지는 보상 흉내
        base = min(200, 20 + ep * 0.4)
        reward = max(10, base + np.random.randn() * 30)
        length = int(reward)
        epsilon = max(0.01, 1.0 - ep / 200)

        logger.log_episode(reward, length, epsilon)

        for _ in range(length):
            loss = max(0.001, 1.0 / (1 + ep * 0.01) + np.random.randn() * 0.1)
            q_val = base * 0.5 + np.random.randn() * 2
            grad_norm = np.abs(np.random.randn()) * 2
            logger.log_step(loss, q_val, grad_norm)

        if ep % 50 == 0:
            logger.log_eval(ep, base + np.random.randn() * 10)

    # 그림
    plot_training_curves(logger, 'training_curves.png')

    # 매끄럽게 하기 견주기
    print("\n--- Smoothing Methods ---")
    rewards = logger.data['episode_rewards']
    ma50 = moving_average(rewards, 50)
    ema99 = exponential_moving_average(rewards, 0.99)
    bands = percentile_bands(rewards, 50)
    print(f"  Raw: final = {rewards[-1]:.1f}")
    print(f"  MA(50): final = {ma50[-1]:.1f}")
    print(f"  EMA(0.99): final = {ema99[-1]:.1f}")
    print(f"  Median: final = {bands['median'][-1]:.1f}")

    # 여러 씨앗
    all_rewards = []
    for seed in range(3):
        np.random.seed(seed)
        rews = [max(10, min(200, 20 + ep*0.4) + np.random.randn()*30) for ep in range(300)]
        all_rewards.append(rews)
    plot_multi_seed(all_rewards, ['Seed 0', 'Seed 1', 'Seed 2'])

    print("\nTraining curves demo complete!")


if __name__ == "__main__":
    demo_training_curves()
```

## 2. 논의

이 짜기는 익히기 굽은 줄의 핵심 논리를 감싼 `TrainingLogger` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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
    이 얼개 고르기는 힘 북돋우는 배움 따지기 방법에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 익히기 굽은 줄

이 짜기는 익히기 굽은 줄의 핵심 논리를 감싼 `TrainingLogger` 갈래를 한가운데 둔다.

고갱이 갈래는 `TrainingLogger`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
