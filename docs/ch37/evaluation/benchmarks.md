# 벤치마크

**벤치마크**는 강화학습 알고리즘을 같은 조건에서 견주기 위해 정해 둔 기준 과제와 점수 체계다. 이 쪽은 네 가지를 구현한다. 아타리 게임의 **사람 기준 정규화 점수**(human-normalized score) — 무작위 정책의 점수를 0, 사람의 점수를 100으로 놓고 에이전트 점수를 다시 매긴 값이다 — 와 오프라인 강화학습 벤치마크 D4RL 의 정규화 점수, 여러 에피소드에 걸쳐 리턴을 모으는 평가 도구, 그리고 고전 제어 환경이 「풀렸는지」 가리는 기준이다.

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

두 정규화 점수는 같은 형태의 식이다 — 에이전트 점수에서 무작위 기준을 빼고, 기준 사이의 폭(사람 − 무작위, 전문가 − 무작위)으로 나눈다. 그래서 게임마다 수십에서 수만까지 제각각인 원점수가 한 눈금에 놓인다. 원점수로는 시퀘스트의 4만 점과 퐁의 14.6점을 견줄 수 없다.

무작위 기준을 빼기 때문에 「사람 점수의 1.5배」가 늘 150% 가 되지는 않는다. 무작위 점수가 양수인 브레이크아웃에서는 150% 보다 조금 크고, 무작위 점수가 음수인 퐁에서는 150% 보다 한참 작다. 데모의 첫 부분이 이것을 게임별로 찍는다.

CartPole 부분은 무작위·항상 왼쪽·항상 오른쪽 세 정책을 50 에피소드씩 돌려 평균·표준편차·중앙값·최솟값·최댓값을 표로 찍는다. 셋 다 풀림 기준(평균 리턴 475 이상)에 한참 못 미친다. 이 표는 트레인한 에이전트가 넘어야 할 바닥을 보여 준다.

이 구현은 빠르기보다 읽기 쉬움을 앞세웠다. 실제 벤치마크 보고에서는 에피소드 수를 정하는 것만큼 **여러 시드의 흩어짐**을 함께 적는 것이 중요하다. 평균 하나로 우열을 가리는 위험은 다음 쪽의 통계적 검정에서 다룬다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모 코드를 돌려 CartPole 표의 세 정책의 평균·표준편차를 기록하라. `n_episodes` 를 10, 50, 200 으로 바꾸면 평균 리턴이 얼마나 흔들리는지도 기록하라.

</div>

??? success "풀이"
    평균의 표준오차는 $\sigma/\sqrt{n}$ 이므로 에피소드 수를 네 배로 늘리면 평균의 흔들림이 절반쯤으로 준다. 환경은 `seed + ep` 로 고정되지만 무작위 정책의 `np.random.randint` 는 시드가 박혀 있지 않으므로, 그 행은 같은 `n_episodes` 에서도 돌릴 때마다 조금씩 달라진다. 항상 왼쪽·항상 오른쪽 정책은 결정적이라 환경 시드가 같으면 같은 수가 나온다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
아타리 게임 57개의 성적을 한 수로 요약할 때 왜 원점수가 아니라 사람 기준 정규화 점수를 쓰는가? 또 정규화 점수를 게임 전체에 걸쳐 요약할 때 평균보다 중앙값을 흔히 보고하는 까닭은 무엇인가?

</div>

??? success "풀이"
    원점수는 게임마다 눈금이 다르다. `ATARI_BASELINES` 만 봐도 사람 점수가 퐁은 14.6, 시퀘스트는 42054.7 이라 원점수를 평균하면 점수가 큰 몇 게임이 결과를 정한다. 정규화 점수는 무작위 0, 사람 100 이라는 같은 눈금에 놓아 이 문제를 없앤다. 그래도 정규화 점수에는 위로 상한이 없어서, 에이전트가 몇몇 게임에서 사람의 수십 배를 내면 평균이 그 게임들에 끌려간다. 중앙값은 그런 극단값에 덜 흔들리므로 「대부분의 게임에서 얼마나 하는가」를 더 잘 요약한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`check_solved` 는 평균 리턴 하나로 판정한다. 50 에피소드 평균이 476 인 에이전트를 「풀었다」고 말해도 되는가? `evaluate_agent` 의 결과로 평균 리턴의 95% 신뢰 구간을 구하고, 구간의 **하한**이 기준을 넘을 때만 풀렸다고 판정하는 함수를 써라.

</div>

??? success "풀이"
    정규 근사로는 구간이 $\bar R \pm 1.96\, s/\sqrt{n}$ 이다. 표준편차가 50 이고 $n=50$ 이면 반폭이 약 13.9 라서, 평균 476 의 하한은 약 462 로 기준 475 에 못 미친다 — 평균이 기준을 1 넘은 것은 우연으로 설명된다. 다만 CartPole 의 리턴은 500 에서 잘리므로 분포가 한쪽으로 몰려 정규 근사가 잘 맞지 않는다. 이럴 때는 다음 쪽의 `bootstrap_ci` 처럼 부트스트랩으로 구간을 내는 편이 낫다. `evaluate_agent` 의 `std_return` 은 `ddof=0` 으로 계산되므로 표본 표준편차가 필요하면 $\sqrt{n/(n-1)}$ 을 곱한다.

## 정리하며

**다룬 것** — 벤치마크

정규화 점수는 무작위 기준을 빼고 기준 사이의 폭으로 나누어 게임마다 다른 눈금을 하나로 맞춘다. 평가 도구는 여러 에피소드의 리턴을 모아 평균과 흩어짐을 함께 보고, 풀림 판정은 그 평균을 환경의 기준과 견준다.

앞의 연습문제 3개로 스스로 점검해 볼 수 있다.
