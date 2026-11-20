# 학습 곡선

강화학습의 에피소드 리턴은 한 에피소드만 보면 잡음이 너무 커서, 학습이 나아지고 있는지는 곡선을 매끄럽게 해야 보인다. 그런데 매끄럽게 하는 방법마다 곡선이 늦게 따라오는 정도가 달라, 같은 데이터에서도 「마지막 성능」이 다르게 읽힌다. 이 쪽은 학습 기록을 모으는 `TrainingLogger`, 곡선을 매끄럽게 하는 세 방법(이동 평균, 지수 이동 평균, 이동 백분위 띠), 그리고 그림을 그리는 함수 둘을 다룬다.

## 1. 코드

```python
"""
35.6.1 익히기 굽은 금
========================

힘 북돋우는 러닝 익히기 굽은 금을 적고 매끄럽게 하고 그리는 연장.
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
    """힘 북돋우는 러닝 실험을 위한 두루 살피는 익히기 기록개."""

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

    # 1. 마당 리워드
    ax = axes[0, 0]
    rewards = logger.data['episode_rewards']
    if rewards:
        ax.plot(rewards, alpha=0.2, color='blue')
        if len(rewards) >= 50:
            ma = moving_average(rewards, 50)
            ax.plot(range(49, len(rewards)), ma, color='red', label='MA(50)')
        ax.set_title('Episode Rewards'); ax.set_xlabel('Episode'); ax.set_ylabel('Return')
        ax.legend()

    # 2. 로스
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

    # 5. 그래디언트 크기
    ax = axes[1, 1]
    gnorms = logger.data['grad_norms']
    if gnorms:
        ax.plot(gnorms, alpha=0.2, color='brown')
        if len(gnorms) >= 100:
            ma = moving_average(gnorms, 100)
            ax.plot(range(99, len(gnorms)), ma, color='darkred')
        ax.set_title('Gradient Norm'); ax.set_xlabel('Step')

    # 6. 값 매김 리워드
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
        # 나아지는 리워드 흉내
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

**출력:**

```
============================================================
Training Curves Demo
============================================================
Training curves saved to training_curves.png

--- Smoothing Methods ---
  Raw: final = 161.2
  MA(50): final = 205.0
  EMA(0.99): final = 179.3
  Median: final = 205.5
Multi-seed plot saved to multi_seed.png

Training curves demo complete!
```

## 2. 논의

`TrainingLogger` 는 에피소드 단위(리턴, 길이, $\varepsilon$), 업데이트 단위(로스, Q 값, 그래디언트 노름), 평가 단위(평가 리턴) 기록을 모으고 JSON 으로 저장하고 불러온다. `moving_average` 는 마지막 `window` 개의 단순 평균, `exponential_moving_average` 는 $m_t = \beta m_{t-1} + (1-\beta)x_t$ 로 옛 값의 가중치를 지수적으로 줄이는 평균, `percentile_bands` 는 마지막 `window` 개의 중앙값과 25·75 백분위다. `plot_training_curves` 는 기록 여섯 가지를 2×3 그림으로, `plot_multi_seed` 는 여러 번 돌린 리턴 곡선을 한 그림으로 그려 PNG 로 저장한다.

데모는 실제 학습을 하지 않는다. 기댓값이 에피소드마다 0.4 씩 올라 450 번째 에피소드에서 200 에 닿고 그 뒤로 머무는 리턴에 표준편차 30 의 가우스 잡음을 더해 500 에피소드를 지어내고, 같은 리턴 열에 세 방법을 적용해 마지막 값을 찍는다. 출력에서 한 에피소드의 값(161.2), 이동 평균(205.0), 중앙값(205.5), 지수 이동 평균(179.3)이 서로 크게 다른 것이 이 쪽이 보이려는 점이다. 지어낸 데이터라 참값을 알고 있으므로, 어느 방법이 얼마나 틀렸는지 계산으로 확인할 수 있다(연습문제 1, 2). 그림 파일은 실행한 디렉터리에 저장되며 이 쪽에는 싣지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모의 마지막 50 에피소드는 모두 기댓값이 200 이고 잡음의 표준편차가 30 이다. `MA(50)` 의 마지막 값의 기댓값과 표준편차를 구하고, 출력의 205.0 이 이것과 맞는지 따져라. `Raw: final = 161.2` 로 성능을 말하면 무엇이 문제인가?

</div>

??? success "풀이"
    마지막 50 개는 서로 독립이고 평균 200, 표준편차 30 이다(`max(10, …)` 의 자르기는 200 근처에서 거의 일어나지 않는다). 그 평균은 기댓값 200, 표준편차 $30/\sqrt{50} \approx 4.2$ 다. 205.0 은 200 에서 약 1.2 표준편차 떨어져 있으니 이상할 것 없는 값이다. 한 에피소드의 값은 표준편차가 30 이라, 161.2 는 기댓값보다 1.3 표준편차쯤 낮게 나온 것뿐이다. 같은 에이전트라도 마지막 에피소드에 따라 160 으로도 240 으로도 보고될 수 있으므로, 한 에피소드의 리턴은 성능 지표가 될 수 없다. 이동 평균이나 여러 평가 에피소드의 평균을 쓰고, 그 표준오차를 함께 적는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
출력에서 `EMA(0.99)` 의 마지막 값(179.3)은 `MA(50)`(205.0)보다 훨씬 낮다. 기울기 $s$ 로 꾸준히 오르는 신호에서 창 $w$ 인 이동 평균과 계수 $\beta$ 인 지수 이동 평균이 각각 몇 에피소드만큼 늦게 따라오는지 구하라. 이것으로 데모의 지어낸 리턴에서 `EMA(0.99)` 마지막 값의 기댓값을 어림해 출력과 견주어라.

</div>

??? success "풀이"
    이동 평균은 마지막 $w$ 개를 고르게 평균하므로 그 무게 중심이 $(w-1)/2$ 에피소드 전에 있고, 그만큼 늦는다. $w = 50$ 이면 24.5 에피소드다. 지수 이동 평균의 가중치는 $k$ 에피소드 전 값에 $(1-\beta)\beta^k$ 이고 그 무게 중심은 $\sum_k k(1-\beta)\beta^k = \beta/(1-\beta)$ 이므로, $\beta = 0.99$ 이면 약 99 에피소드 늦는다. 데모의 기댓값은 450 번째 에피소드까지 0.4 씩 오르고 마지막 50 에피소드는 200 이다. 이동 평균(창 50)은 평평한 구간만 보므로 기댓값이 정확히 200 이다. 지수 이동 평균은 $k \ge 50$ 인 옛 값에 기댓값이 $200 - 0.4(k - 49)$ 인 오르막 구간을 담으므로, 기댓값이 $200 - 0.4\sum_{k\ge50}(1-\beta)\beta^k(k-49) = 200 - 0.4 \times 0.99^{50} \times 100 \approx 200 - 24.2 = 175.8$ 이다. 출력의 179.3 은 잡음을 생각하면 이 값과 맞는다. 곧 179.3 은 에이전트가 나빠서가 아니라 매끄럽게 하는 방법이 100 에피소드쯤 늦게 따라와서 나온 값이다. 지수 이동 평균은 「얼마나 오래된 값까지 섞였는가」를 $1/(1-\beta)$ 로 따져 보고 읽어야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`plot_multi_seed` 의 독스트링은 「여러 번 돌린 것을 평균과 표준편차 띠로 그린다」고 하지만, 코드는 시드마다 곡선을 따로 그리고 띠의 폭으로 `np.std(rewards[-len(smoothed):])` 를 쓴다. 이 띠는 무엇을 나타내며 왜 시드 사이의 흩어짐이 아닌가? 시드 사이의 평균과 흩어짐을 바르게 그리도록 고쳐라.

</div>

??? success "풀이"
    `np.std(rewards[-len(smoothed):])` 는 한 시드의 리턴 열 거의 전체에 대한 표준편차 하나, 곧 **상수**다. 게다가 데모가 이 함수에 넘기는 300 에피소드짜리 리턴은 20 에서 140 남짓까지 오르는 추세를 품고 있어, 이 표준편차에는 잡음(30)뿐 아니라 추세에 따른 변화까지 들어가 실제 잡음보다 훨씬 크다. 그러니 이 띠는 시간에 따라 변하지도 않고, 시드끼리 얼마나 다른지도 말해 주지 않는다. 바르게 하려면 시드 $S$ 개의 매끄럽게 한 곡선을 `(S, T)` 배열로 쌓은 뒤 축 0 에 대해 평균과 표준편차(또는 표준오차 $\sigma/\sqrt S$)를 에피소드마다 구해, 평균 곡선 하나와 그 주위의 띠를 그린다. 시드가 셋뿐이면 표준편차 추정 자체가 불안정하므로 각 시드의 곡선을 옅게 함께 그려 두는 편이 정직하고, 띠가 표준편차인지 표준오차인지 그림에 적는다. 두 알고리즘을 견줄 때는 이 띠가 겹치는지를 먼저 보아야 한다.

## 정리하며

**다룬 것** — 학습 곡선

이 구현은 학습 기록을 모으는 `TrainingLogger` 와, 이동 평균·지수 이동 평균·이동 백분위 띠라는 세 가지 매끄럽게 하기, 그리고 그 결과를 그리는 두 함수를 다룬다. 매끄럽게 하는 방법마다 늦게 따라오는 정도가 다르다는 것이 이 쪽의 요점이다.

핵심 클래스는 `TrainingLogger`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
