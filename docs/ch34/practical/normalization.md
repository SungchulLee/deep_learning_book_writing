# 봄 고르게 하기

봄 고르게 하기는 실제 힘 북돋우는 배움 재주에서 종요로운 생각이다. 힘 북돋우는 배움 익힘에서의 이점을 다룬다. 이 구현은 여기에 걸린 종요로운 알고리즘과 자료 얼개를 손으로 만져 보이며, 이치 바탕과 실제로 서비스에 올릴 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""
34.6.3장: 봄 고르게 하기
==========================================
힘 북돋우는 배움 익힘에서 봄, 보상, 이점에 쓰는 흐르는 고르게
하기 도구.
"""

import torch
import numpy as np
from typing import Optional

# ========================================================================
# 메인
# ========================================================================


class RunningMeanStd:
    """흐르는 평균과 흩어짐을 위한 웰퍼드의 잇단 알고리즘."""
    
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        if x.ndim == 1 and self.mean.shape:
            batch_mean = x
            batch_var = np.zeros_like(self.mean)
            batch_count = 1
        
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = M2 / total
        self.count = total


class ObservationNormalizer:
    """
    흐르는 통계로 봄을 고르게 한다.
    
    매개변수
    ----------
    shape : tuple
        봄의 꼴.
    clip : float
        고르게 한 값을 [-clip, clip]으로 자른다.
    """
    
    def __init__(self, shape, clip=10.0):
        self.rms = RunningMeanStd(shape)
        self.clip = clip
        self.training = True
    
    def normalize(self, obs):
        if self.training:
            if isinstance(obs, np.ndarray):
                x = obs if obs.ndim > 1 else obs.reshape(1, -1)
                self.rms.update(x)
            
        normalized = (obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)
    
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True


class RewardNormalizer:
    """
    (평균이 아니라) 흐르는 표준편차로 보상을 고르게 한다.
    
    보상의 부호와 서로의 크기를 지키려고 표준편차로만 나눈다.
    """
    
    def __init__(self, clip=10.0):
        self.rms = RunningMeanStd(())
        self.clip = clip
        self.training = True
    
    def normalize(self, reward):
        if self.training:
            self.rms.update(np.array([reward]))
        
        normalized = reward / (np.sqrt(self.rms.var) + 1e-8)
        return np.clip(normalized, -self.clip, self.clip)
    
    def eval(self):
        self.training = False


class VecNormalizer:
    """
    벡터 둘레를 위한 엮은 고르게 하는 개.
    봄 고르게 하기와 보상 고르게 하기를 함께 다룬다.
    """
    
    def __init__(self, obs_shape, n_envs, clip_obs=10.0, clip_rew=10.0,
                 normalize_obs=True, normalize_rew=True, gamma=0.99):
        self.normalize_obs = normalize_obs
        self.normalize_rew = normalize_rew
        
        if normalize_obs:
            self.obs_rms = RunningMeanStd(obs_shape)
        if normalize_rew:
            self.ret_rms = RunningMeanStd(())
            self.returns = np.zeros(n_envs)
        
        self.clip_obs = clip_obs
        self.clip_rew = clip_rew
        self.gamma = gamma
        self.training = True
    
    def normalize_obs_fn(self, obs):
        if not self.normalize_obs:
            return obs
        if self.training:
            self.obs_rms.update(obs)
        normalized = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(normalized, -self.clip_obs, self.clip_obs).astype(np.float32)
    
    def normalize_reward_fn(self, rewards, dones):
        if not self.normalize_rew:
            return rewards
        if self.training:
            self.returns = self.returns * self.gamma + rewards
            self.ret_rms.update(self.returns)
            self.returns[dones.astype(bool)] = 0.0
        
        normalized = rewards / (np.sqrt(self.ret_rms.var) + 1e-8)
        return np.clip(normalized, -self.clip_rew, self.clip_rew)


def normalize_advantages(advantages, eps=1e-8):
    """이점을 평균 0, 흩어짐 1로 고르게 한다(작은 묶음마다)."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def demo_normalization():
    print("=" * 60)
    print("Observation and Reward Normalization Demo")
    print("=" * 60)
    
    # 잣대가 다른 봄을 흉내 낸다
    np.random.seed(42)
    obs_dim = 4
    
    # 특징 0: 큰 잣대, 특징 1: 작은 잣대, 그렇게 이어진다
    scales = np.array([100.0, 0.01, 50.0, 0.001])
    offsets = np.array([500.0, -0.05, 25.0, 0.0])
    
    normalizer = ObservationNormalizer(shape=(obs_dim,))
    
    print("\nBefore normalization (raw observations):")
    raw_obs = []
    for _ in range(100):
        obs = np.random.randn(obs_dim) * scales + offsets
        normalizer.normalize(obs)  # 통계를 고친다
        raw_obs.append(obs)
    
    raw_obs = np.array(raw_obs)
    print(f"  Mean:  {raw_obs.mean(axis=0).round(3)}")
    print(f"  Std:   {raw_obs.std(axis=0).round(3)}")
    
    # 이제 고르게 한다
    norm_obs = np.array([normalizer.normalize(o) for o in raw_obs])
    print("\nAfter normalization:")
    print(f"  Mean:  {norm_obs.mean(axis=0).round(3)}")
    print(f"  Std:   {norm_obs.std(axis=0).round(3)}")
    print(f"  Range: [{norm_obs.min():.2f}, {norm_obs.max():.2f}]")
    
    # 보상 고르게 하기
    print("\n" + "-" * 40)
    print("Reward Normalization:")
    
    rew_normalizer = RewardNormalizer()
    rewards = np.random.randn(50) * 100 + 50  # 잣대가 큰 보상
    norm_rewards = [rew_normalizer.normalize(r) for r in rewards]
    
    print(f"  Raw:  mean={rewards.mean():.1f}, std={rewards.std():.1f}")
    print(f"  Norm: mean={np.mean(norm_rewards):.3f}, std={np.std(norm_rewards):.3f}")
    
    # 이점 고르게 하기
    print("\n" + "-" * 40)
    print("Advantage Normalization:")
    advantages = torch.randn(32) * 5 + 2
    norm_adv = normalize_advantages(advantages)
    print(f"  Raw:  mean={advantages.mean():.3f}, std={advantages.std():.3f}")
    print(f"  Norm: mean={norm_adv.mean():.6f}, std={norm_adv.std():.3f}")


if __name__ == "__main__":
    demo_normalization()```

## 2. 논의

이 구현은 봄 고르게 하기의 한가운데 논리를 담은 `RunningMeanStd`, `ObservationNormalizer`, `RewardNormalizer` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

보여 주기 함수는 이 조각들을 종요로운 거동이 드러나는 지어낸 자료에 실제로 써 보인다. 그 출력을 살피면 매개변수 고름과 문제 얼개에 따라 알고리즘의 됨됨이가 어떻게 달라지는지 볼 수 있다.

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

## 정리하며

**다룬 것** — 봄 고르게 하기

이 구현은 봄 고르게 하기의 한가운데 논리를 담은 `RunningMeanStd`, `ObservationNormalizer`, `RewardNormalizer` 클래스를 축으로 삼는다.

고갱이 갈래는 `RunningMeanStd`, `ObservationNormalizer`, `RewardNormalizer`, `VecNormalizer`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
