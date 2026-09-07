# 잣대 시험

잣대 시험은 힘 북돋우는 배움 따지기 방법의 중요한 개념이다. 잣대 따지기 연장과 표준 점수 매기기를 다룬다. 이 짜기는 얽힌 핵심 알고리즘과 자료 얼개를 손으로 만져 보게 하며 이론 바탕과 실제로 펼칠 때 살필 것을 함께 보여 준다.

## 코드

```python
"""
33.6.2 잣대
==================

잣대 값 매기기 연장과 표준 점수 매기기.
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from typing import Dict, List, Callable

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# 사람 기준 점수 매기기
# ---------------------------------------------------------------------------

ATARI_BASELINES = {
    'Breakout': {'random': 1.7, 'human': 30.5},
    'Pong': {'random': -20.7, 'human': 14.6},
    'SpaceInvaders': {'random': 148.0, 'human': 1668.7},
    'Seaquest': {'random': 68.4, 'human': 42054.7},
    'Qbert': {'random': 163.9, 'human': 13455.0},
}


def human_normalized_score(agent_score: float, game: str) -> float:
    """아타리 놀이의 사람 기준 점수를 셈한다."""
    if game not in ATARI_BASELINES:
        raise ValueError(f"Unknown game: {game}")
    rand = ATARI_BASELINES[game]['random']
    human = ATARI_BASELINES[game]['human']
    return (agent_score - rand) / (human - rand) * 100


def d4rl_normalized_score(agent_score: float, random_score: float,
                          expert_score: float) -> float:
    """D4RL 고른 점수: 0은 마구잡이, 100은 전문가."""
    return (agent_score - random_score) / (expert_score - random_score + 1e-8) * 100


# ---------------------------------------------------------------------------
# 잣대 값 매기기
# ---------------------------------------------------------------------------

def evaluate_agent(env_name: str, select_action_fn: Callable,
                   n_episodes: int = 100, max_steps: int = 10000,
                   seed: int = 42) -> Dict[str, float]:
    """두루 살피는 부림꾼 값 매기기."""
    env = gym.make(env_name)
    returns, lengths = [], []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_r, steps, done = 0.0, 0, False
        while not done and steps < max_steps:
            action = select_action_fn(state)
            state, r, term, trunc, _ = env.step(action)
            total_r += r; steps += 1
            done = term or trunc
        returns.append(total_r)
        lengths.append(steps)

    env.close()
    returns = np.array(returns)
    lengths = np.array(lengths)
    return {
        'mean_return': returns.mean(),
        'std_return': returns.std(),
        'median_return': np.median(returns),
        'min_return': returns.min(),
        'max_return': returns.max(),
        'iqr_return': np.percentile(returns, 75) - np.percentile(returns, 25),
        'mean_length': lengths.mean(),
        'n_episodes': n_episodes,
    }


def benchmark_suite(env_name: str, agents: Dict[str, Callable],
                    n_episodes: int = 50) -> Dict[str, Dict]:
    """같은 둘레에서 여러 부림꾼을 견준다."""
    results = {}
    for name, action_fn in agents.items():
        result = evaluate_agent(env_name, action_fn, n_episodes)
        results[name] = result
    return results


def print_benchmark_table(results: Dict[str, Dict]):
    """잣대 견주기를 보기 좋게 찍는다."""
    header = f"{'Agent':<25s} {'Mean':>8s} {'Std':>8s} {'Median':>8s} {'Min':>6s} {'Max':>6s}"
    print(header)
    print("-" * len(header))
    for name, r in sorted(results.items(), key=lambda x: -x[1]['mean_return']):
        print(f"{name:<25s} {r['mean_return']:>8.1f} {r['std_return']:>8.1f} "
              f"{r['median_return']:>8.1f} {r['min_return']:>6.0f} {r['max_return']:>6.0f}")


# ---------------------------------------------------------------------------
# 둘레 앎
# ---------------------------------------------------------------------------

CLASSIC_CONTROL = {
    'CartPole-v1': {'solved': 475, 'max_steps': 500},
    'MountainCar-v0': {'solved': -110, 'max_steps': 200},
    'LunarLander-v2': {'solved': 200, 'max_steps': 1000},
    'Acrobot-v1': {'solved': -100, 'max_steps': 500},
}


def check_solved(env_name: str, mean_return: float) -> bool:
    """여느 잣대로 환경이 '풀렸는지' 살핀다."""
    if env_name in CLASSIC_CONTROL:
        return mean_return >= CLASSIC_CONTROL[env_name]['solved']
    return False


# ---------------------------------------------------------------------------
# 시연
# ---------------------------------------------------------------------------

def demo_benchmarks():
    print("=" * 60)
    print("Benchmarks Demo")
    print("=" * 60)

    # 사람 기준 점수
    print("\n--- Human-Normalized Scoring ---")
    for game, scores in ATARI_BASELINES.items():
        for agent_score in [scores['random'], scores['human'], scores['human'] * 1.5]:
            hn = human_normalized_score(agent_score, game)
            print(f"  {game}: score={agent_score:.0f} → {hn:.1f}% human-normalized")

    # CartPole 잣대 재기
    print("\n--- CartPole Benchmark ---")
    env = gym.make('CartPole-v1')
    sd, ad = env.observation_space.shape[0], env.action_space.n
    env.close()

    agents = {
        'Random': lambda s: np.random.randint(ad),
        'Always Left': lambda s: 0,
        'Always Right': lambda s: 1,
    }

    results = benchmark_suite('CartPole-v1', agents, n_episodes=50)
    print_benchmark_table(results)

    # 풀렸는지 살핌
    for name, r in results.items():
        solved = check_solved('CartPole-v1', r['mean_return'])
        print(f"  {name}: {'SOLVED' if solved else 'not solved'} "
              f"(need ≥ {CLASSIC_CONTROL['CartPole-v1']['solved']})")

    # D4RL 점수 매기기
    print("\n--- D4RL Normalized Scoring ---")
    for agent_s in [10, 30, 50, 80, 100]:
        norm = d4rl_normalized_score(agent_s, random_score=10, expert_score=100)
        print(f"  Agent score={agent_s}: D4RL normalized = {norm:.1f}")

    print("\nBenchmarks demo complete!")


if __name__ == "__main__":
    demo_benchmarks()```

## 논의

이 짜기는 잣대 시험의 핵심 연산을 짜는 여러 연장 함수를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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
