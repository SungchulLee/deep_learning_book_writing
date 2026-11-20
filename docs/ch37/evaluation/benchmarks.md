# 벤치마크

벤치마크는 강화학습 평가 방법의 중요한 개념이다. 벤치마크 평가 도구와 정규화 점수 계산을 다룬다. 이 구현은 복잡한 핵심 알고리즘과 자료 구조를 직접 다뤄 보게 하며 이론적 기초와 실제 적용 시 고려할 점을 함께 보여 준다.

## 1. 코드

```python
"""
35.6.2 잣대
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
    """두루 살피는 에이전트 값 매기기."""
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
    """같은 둘레에서 여러 에이전트를 견준다."""
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
    demo_benchmarks()
```

## 2. 논의

이 구현은 벤치마크의 핵심 연산을 구현하는 여러 도구 함수를 중심에 둔다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

데모 함수는 이 구성 요소들을 전형적인 강화학습 벤치마크에 실제로 적용하는 모습을 보인다. 출력을 살피면 하이퍼파라미터 선택과 문제 구조에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실용적 관점에서 이 구현은 순수한 성능보다 명료함을 앞세운다. 실제로 쓰는 시스템은 보통 배치 계산, GPU 가속, 더 정교한 하이퍼파라미터 튜닝 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 아이디어는 대규모 응용으로 그대로 옮겨 간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모 코드를 돌려 핵심 출력 지표를 기록하라. 하이퍼파라미터 하나(러닝레이트, 은닉 차원, 레이어 개수 같은 것)를 바꾸고 결과가 어떻게 바뀌는지 기록하라.

</div>

??? success "풀이"
    데모를 돌린 뒤 나머지를 고정하고 고른 하이퍼파라미터를 차례로 바꾼다. 예를 들어 은닉 차원을 두 배로 하면 보통 표현력이 늘지만 계산 시간이 커진다. 러닝레이트는 단조롭지 않은 영향을 준다. 너무 작으면 느리게 수렴하고 너무 크면 진동한다. 고른 하이퍼파라미터의 서로 다른 값을 적어도 셋 잡아 구체적인 수를 기록해 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 핵심 구조 선택의 역할을 설명하라. 왜 그 활성화 함수, 정규화 전략, 로스 함수를 쓰는가? 다른 것으로 바꾸면 어떻게 되는가?

</div>

??? success "풀이"
    이 구조 선택은 강화학습 평가 방법에서 자리 잡은 모범 사례를 따른다. 예를 들어 ReLU 활성화는 비선형성을 주면서 양의 입력에서 그래디언트가 사라지는 것을 피한다. 로스 함수는 과제 유형에 맞게 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(예를 들어 시그모이드 활성화, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
더 어려운 상황을 다루도록 구현을 확장하라. 더 큰 데이터셋, 다른 문제 변형, 피처 추가 가운데 하나를 고른다. 수정 사항을 적고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 정칙화(드롭아웃, 웨이트 감쇠)나 더 정교한 구조(레이어 추가, 스킵 연결)를 더하는 것이다. 고른 확장을 구현하고 같은 데이터로 학습한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 벤치마크

이 구현은 벤치마크의 핵심 연산을 구현하는 여러 도구 함수를 중심에 둔다.

앞의 연습문제 3개로 스스로 점검해 볼 수 있다.
