# 보상 다듬기

보상 다듬기는 실제 힘 북돋우는 배움 재주에서 종요로운 생각이다. 금융에 딸린 보상 함수를 다룬다. 이 구현은 여기에 걸린 종요로운 알고리즘과 자료 얼개를 손으로 만져 보이며, 이치 바탕과 실제로 서비스에 올릴 때 살필 것을 함께 보여 준다.

## 코드

```python
"""
34.6.1장: 보상 다듬기
================================
퍼텐셜에 바탕을 둔 보상 다듬기, 보상 고르게 하기, 금융에 딸린
보상 함수.
"""

import torch
import numpy as np
import gymnasium as gym
from collections import deque

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# 퍼텐셜에 바탕을 둔 보상 다듬기
# ---------------------------------------------------------------------------

class PotentialBasedShaping:
    """
    퍼텐셜에 바탕을 둔 보상 다듬기: r' = r + γΦ(s') - Φ(s)
    가장 좋은 방침이 지켜짐이 보장된다.
    """
    
    def __init__(self, potential_fn, gamma=0.99):
        self.potential_fn = potential_fn
        self.gamma = gamma
    
    def shape(self, reward, state, next_state):
        phi_s = self.potential_fn(state)
        phi_s_next = self.potential_fn(next_state)
        return reward + self.gamma * phi_s_next - phi_s


class RewardNormalizer:
    """웰퍼드의 알고리즘으로 보상을 흐르게 고르게 한다."""
    
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
    """보상을 [-clip, clip] 너비로 자른다."""
    def __init__(self, clip=1.0):
        self.clip = clip
    
    def __call__(self, reward):
        return np.clip(reward, -self.clip, self.clip)


# ---------------------------------------------------------------------------
# 금융 보상 함수
# ---------------------------------------------------------------------------

class SharpeReward:
    """
    밑천 다루기를 위한 샤프 비 바탕 보상.
    흐르는 샤프 비를 보상 신호로 셈한다.
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
    """서로 다른 보상 신호로 익힘을 견준다."""
    print("=" * 60)
    print("Reward Shaping Comparison")
    print("=" * 60)
    
    # 서로 다른 보상 다듬기를 쓰는 CartPole
    env = gym.make("CartPole-v1")
    
    # 퍼텐셜을 정한다: 가운데에 가까울수록 퍼텐셜이 높다
    def center_potential(obs):
        return -abs(obs[0]) - abs(obs[2]) * 0.5  # 자리와 각도에 벌을 준다
    
    shaper = PotentialBasedShaping(center_potential, gamma=0.99)
    normalizer = RewardNormalizer()
    
    # 넘어감을 얼마쯤 모아 다듬은 보상을 보인다
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
    
    # 금융 보상 보여 주기
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
    demo_reward_shaping()```

## 논의

이 구현은 보상 다듬기의 한가운데 논리를 담은 `PotentialBasedShaping`, `RewardNormalizer`, `RewardClipper` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

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
    이 얼개 고름은 실제 힘 북돋우는 배움 재주에서 자리 잡은 좋은 버릇을 비춘다. 보기로 ReLU 활성은 곧지 않음을 주면서 0보다 큰 들임에서 기울기가 사라지는 것을 막는다. 손실 함수는 일감 갈래에 맞추어 고른다(갈래 나누기에는 사귐 엔트로피, 되돌이에는 평균 제곱 잘못). 다른 것으로 바꾸면(보기로 시그모이드 활성, L1 손실) 가장 좋게 하기 지형이 바뀌어 됨됨이가 나빠질 수 있으나, 어떤 자리에서는 바꾸는 것이 이로울 수도 있다.

---

**연습문제 3.**
이 구현을 더 만만치 않은 자리로 넓혀라. 더 큰 자료 뭉치, 다른 문제 갈래, 덧붙인 기능 가운데 하나를 고르라. 고친 바를 밝히고 됨됨이에 미친 바를 따져라.

??? success "연습문제 3 풀이"
    절로 떠오르는 넓히기 하나는 정칙화(드롭아웃, 무게 삭임)나 더 야무진 얼개(켜 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓히기를 만들고 같은 자료로 익힌 뒤 앞뒤의 재기를 견주어라. 이 넓히기는 처음 알고리즘과 고친 바의 이치 밑뜻을 모두 아는 것을 보여야 한다.
