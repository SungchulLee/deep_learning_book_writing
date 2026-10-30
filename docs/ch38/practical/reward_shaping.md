# 리워드 셰이핑

리워드 다듬기는 실제 강화학습 재주에서 중요한 생각이다. 금융에 딸린 리워드 함수를 다룬다. 이 구현은 여기에 걸린 중요한 알고리즘과 자료 구조를 손으로 만져 보이며, 이치 기초와 실제로 서비스에 올릴 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""
34.6.1장: 리워드 다듬기
================================
퍼텐셜에 바탕을 둔 리워드 다듬기, 리워드 고르게 하기, 금융에 딸린
리워드 함수.
"""

import torch
import numpy as np
import gymnasium as gym
from collections import deque

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# 퍼텐셜에 바탕을 둔 리워드 다듬기
# ---------------------------------------------------------------------------

class PotentialBasedShaping:
    """
    퍼텐셜에 바탕을 둔 리워드 다듬기: r' = r + γΦ(s') - Φ(s)
    가장 좋은 폴리시가 지켜짐이 보장된다.
    """
    
    def __init__(self, potential_fn, gamma=0.99):
        self.potential_fn = potential_fn
        self.gamma = gamma
    
    def shape(self, reward, state, next_state):
        phi_s = self.potential_fn(state)
        phi_s_next = self.potential_fn(next_state)
        return reward + self.gamma * phi_s_next - phi_s


class RewardNormalizer:
    """웰퍼드의 알고리즘으로 리워드를 흐르게 고르게 한다."""
    
    def __init__(self, clip=10.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self.clip = clip
    
    def update(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += (delta * delta2 - self.var) / self.count
    
    def normalize(self, reward):
        self.update(reward)
        return np.clip(reward / (np.sqrt(self.var) + 1e-8), -self.clip, self.clip)


class RewardClipper:
    """리워드를 [-clip, clip] 너비로 자른다."""
    def __init__(self, clip=1.0):
        self.clip = clip
    
    def __call__(self, reward):
        return np.clip(reward, -self.clip, self.clip)


# ---------------------------------------------------------------------------
# 금융 리워드 함수
# ---------------------------------------------------------------------------

class SharpeReward:
    """
    밑천 다루기를 위한 샤프 비 바탕 리워드.
    흐르는 샤프 비를 리워드 신호로 셈한다.
    """
    
    def __init__(self, window=20, risk_free_rate=0.0, annualize=252):
        self.returns = deque(maxlen=window)
        self.risk_free_rate = risk_free_rate
        self.annualize = annualize
    
    def __call__(self, portfolio_return):
        self.returns.append(portfolio_return)
        if len(self.returns) < 5:
            return portfolio_return
        
        returns = np.array(self.returns)
        excess = returns - self.risk_free_rate / self.annualize
        
        mean_return = excess.mean()
        std_return = excess.std() + 1e-8
        
        sharpe = mean_return / std_return * np.sqrt(self.annualize)
        return sharpe


class DrawdownPenalty:
    """밑천 내림폭에 벌을 준다."""
    
    def __init__(self, penalty_coef=1.0):
        self.peak = 1.0
        self.penalty_coef = penalty_coef
    
    def __call__(self, portfolio_value, base_reward):
        self.peak = max(self.peak, portfolio_value)
        drawdown = (self.peak - portfolio_value) / self.peak
        return base_reward - self.penalty_coef * drawdown


def demo_reward_shaping():
    """서로 다른 리워드 신호로 익힘을 견준다."""
    print("=" * 60)
    print("Reward Shaping Comparison")
    print("=" * 60)
    
    # 서로 다른 리워드 다듬기를 쓰는 CartPole
    env = gym.make("CartPole-v1")
    
    # 퍼텐셜을 정한다: 가운데에 가까울수록 퍼텐셜이 높다
    def center_potential(obs):
        return -abs(obs[0]) - abs(obs[2]) * 0.5  # 자리와 각도에 벌을 준다
    
    shaper = PotentialBasedShaping(center_potential, gamma=0.99)
    normalizer = RewardNormalizer()
    
    # 넘어감을 얼마쯤 모아 다듬은 리워드를 보인다
    obs, _ = env.reset()
    print(f"\n{'Step':>4} {'Raw':>8} {'Shaped':>8} {'Normalized':>10}")
    print("-" * 34)
    
    for step in range(10):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        
        shaped = shaper.shape(reward, obs, next_obs)
        normalized = normalizer.normalize(reward)
        
        print(f"{step:>4} {reward:>8.3f} {shaped:>8.3f} {normalized:>10.3f}")
        
        obs = next_obs
        if term or trunc:
            break
    
    env.close()
    
    # 금융 리워드 보여 주기
    print("\n" + "-" * 40)
    print("Finance Reward Functions")
    print("-" * 40)
    
    sharpe_reward = SharpeReward(window=20)
    dd_penalty = DrawdownPenalty(penalty_coef=2.0)
    
    np.random.seed(42)
    portfolio_value = 100.0
    
    print(f"\n{'Day':>4} {'Return':>8} {'Sharpe':>8} {'DD Penalty':>12} {'Value':>8}")
    print("-" * 45)
    
    for day in range(30):
        daily_return = np.random.normal(0.001, 0.02)
        portfolio_value *= (1 + daily_return)
        
        sr = sharpe_reward(daily_return)
        dd_r = dd_penalty(portfolio_value, daily_return)
        
        if day % 5 == 0:
            print(f"{day:>4} {daily_return:>8.4f} {sr:>8.3f} {dd_r:>12.4f} {portfolio_value:>8.2f}")


if __name__ == "__main__":
    demo_reward_shaping()
```

## 2. 논의

이 구현은 리워드 다듬기의 한가운데 논리를 담은 `PotentialBasedShaping`, `RewardNormalizer`, `RewardClipper` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

보여 주기 함수는 이 조각들을 여느 강화학습 잣대에 실제로 써 보인다. 그 출력을 살피면 파라메터 공정성과 문제 구조에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

쓰임의 눈으로 보면 이 구현은 날 성능보다 또렷함을 앞세운다. 서비스 시스템은 배치 계산, GPU 빠르게 하기, 더 야무진 파라메터 벼리기 같은 다듬기를 더 넣는 것이 보통이다. 그렇더라도 여기서 보인 한가운데 알고리즘 생각은 큰 잣대의 쓰임새에 그대로 옮겨 간다.

<div class="storybox" markdown>

**그림으로 보는 AI 1.** 셰이핑의 꼴이 정해져 있는 까닭

리워드 셰이핑에는 "이렇게 해도 된다"가 아니라 **"이 꼴만 된다"** 는 정리가 있다. 1999년에 앤드루 응(Andrew Ng)·다이시 하라다(Daishi Harada)·스튜어트 러셀(Stuart Russell)이 보인 것으로, 최적 폴리시를 지키려면 셰이핑이 **포텐셜 꼴**이어야 한다는 것이다.

$$
F(s, a, s') \;=\; \gamma\,\Phi(s') - \Phi(s)
$$

이 꼴이면 **언제나** 최적 폴리시가 그대로이고(충분조건), 더 센 쪽으로 — 환경의 전이나 리워드 구조에 따로 가정을 두지 않는 한 이 꼴이 **아니면 지켜지지 않는다**(필요조건). 셰이핑을 고를 자유가 생각보다 훨씬 좁다는 뜻이다.

왜 좁아야 하는지는 반례 하나면 충분하다. 복도 끝에 목표를 두고, 누구나 떠올릴 법한 셰이핑을 걸어 보자 — **앞으로 한 칸 가면 $+c$.**

![왼쪽은 칸 여섯 개짜리 복도를 두 줄로 그린 그림이다. 위는 원래 리워드와 포텐셜 꼴 셰이핑일 때로 화살표가 모두 목표 쪽을 향한다. 아래는 앞으로 가면 상을 주는 셰이핑일 때로, 마지막 칸에서 화살표가 양쪽을 가리키며 되돌아간다. 오른쪽은 상의 크기에 따라 에이전트가 생각하는 첫 칸의 값이 어떻게 부풀어 오르는지 그린 것으로, 참값 0.815에서 시작해 c가 1일 때 11.646까지 올라가고 c가 0.12를 넘으면 폴리시가 바뀐다고 세로선이 그어져 있다.](figures/shaping_potential.svg)

값 반복으로 정확히 풀어 보면 이렇다. 포텐셜 꼴 셰이핑은 폴리시를 **하나도 바꾸지 않는다.** 그런데 "앞으로 가면 $+c$" 는 $c$ 가 **0.12** 를 넘는 순간 폴리시를 바꾼다. 바뀐 폴리시가 하는 일이 고약하다 — **마지막 칸에서 목표로 들어가지 않고 되돌아간다.** 앞뒤로 오가며 $+c$ 를 계속 타는 쪽이, 목표에 닿아 $+1$ 을 받고 끝나는 쪽보다 낫기 때문이다.

에이전트가 생각하는 첫 칸의 값도 참값 0.815에서 **11.646** 까지 부푼다. 이 수가 올라가는 것은 과제를 잘 풀게 되어서가 아니라 **스스로에게 주는 상을 세고 있어서**다.

**여기서 셰이핑을 보는 눈이 하나 바뀐다.** 셰이핑은 "힌트를 주는 일"처럼 들리지만 실제로는 **과제를 바꾸는 일**이고, 포텐셜 꼴은 그중 과제를 바꾸지 않는 유일한 꼴이다. 위 구현에서 셰이핑을 걸고 학습 곡선이 좋아졌다면 물어야 할 것이 하나 더 있다 — **올라간 것이 원래 리워드인가, 아니면 셰이핑까지 더한 수인가.** 둘을 따로 기록해 두지 않으면 구별할 방법이 없다.

_확인한 곳 — A. Y. Ng, D. Harada, S. Russell, "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping", ICML 1999. 포텐셜 꼴이 최적 폴리시를 지키는 충분조건이자, 환경에 더 가정을 두지 않는 한 (양의 아핀 변환을 빼면) 필요조건이기도 하다는 것이 그 글의 결과다. 그림의 수는 그림을 그리는 코드가 작은 복도 MDP 를 값 반복으로 정확히 풀어 얻으며, 포텐셜 꼴이 실제로 폴리시를 지키는지를 먼저 확인한 뒤에만 그린다._

</div>

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
보여 주기 코드를 돌리고 중요한 출력 재기를 적어라. 파라메터 하나(러닝레이트, 숨은 차원, 켜 개수 따위)를 고쳐 열매가 어떻게 달라지는지 밝혀라.

</div>

??? success "풀이"
    보여 주기를 돌린 뒤 다른 것을 붙박아 두고 고른 파라메터만 구조 있게 바꾼다. 보기로 숨은 차원을 곱절로 늘리면 나타내는 그릇이 커지지만 셈하는 때가 는다. 러닝레이트는 한결같지 않은 결과를 낳는다. 너무 작으면 더디게 모이고 너무 크면 들쭉날쭉해진다. 고른 파라메터의 서로 다른 값 적어도 셋에 대해 또렷한 수를 적어 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 중요한 구조 공정성이 맡은 몫을 풀어라. 왜 그런 엑티베이션 함수, 고르게 하기 꾀, 로스 함수를 쓰는가? 다른 것으로 바꾸면 무슨 일이 생기는가?

</div>

??? success "풀이"
    이 구조 공정성은 실제 강화학습 재주에서 자리 잡은 좋은 버릇을 비춘다. 보기로 ReLU 활성은 곧지 않음을 주면서 0보다 큰 입력에서 그래디언트가 사라지는 것을 막는다. 로스 함수는 과제 클래스에 맞추어 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(보기로 시그모이드 활성, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있으나, 어떤 자리에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 구현을 더 만만치 않은 자리로 넓혀라. 더 큰 데이터셋, 다른 문제 클래스, 덧붙인 기능 가운데 하나를 고르라. 고친 바를 밝히고 성능에 미친 바를 따져라.

</div>

??? success "풀이"
    절로 떠오르는 넓히기 하나는 정규화(드롭아웃, 웨이트 삭임)나 더 야무진 구조(켜 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓히기를 만들고 같은 데이터로 학습된 뒤 앞뒤의 재기를 비교하여라. 이 넓히기는 처음 알고리즘과 고친 바의 이치 밑뜻을 모두 아는 것을 보여야 한다.

## 정리하며

**다룬 것** — 리워드 다듬기

이 구현은 리워드 다듬기의 한가운데 논리를 담은 `PotentialBasedShaping`, `RewardNormalizer`, `RewardClipper` 클래스를 축으로 삼는다.

핵심 클래스는 `PotentialBasedShaping`, `RewardNormalizer`, `RewardClipper`, `SharpeReward`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
