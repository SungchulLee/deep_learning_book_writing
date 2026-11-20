# 관측 정규화

관찰 정규화는 실전 강화학습 기법에서 중요한 아이디어다. 강화학습 학습에서의 어드밴티지를 다룬다. 이 구현은 관련된 중요한 알고리즘과 자료 구조를 직접 다뤄 보이며, 이론적 기초와 실제 배포 시 고려할 점을 함께 보여 준다.

## 1. 코드

```python
"""
34.6.3장: 봄 고르게 하기
==========================================
힘 북돋우는 러닝 익힘에서 봄, 리워드, 이점에 쓰는 흐르는 고르게
하기 도구.
"""

import torch
import numpy as np
from typing import Optional
np.random.seed(42)

# 쪽 전체가 한 프로그램이다. 모듈 수준에서 한 번 박아 두어야
# 아래 수를 다시 얻을 수 있다.
torch.manual_seed(42)

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
    
    파라메터
    ----------
    shape : tuple
        봄의 형태.
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
    (평균이 아니라) 흐르는 표준편차로 리워드를 고르게 한다.
    
    리워드의 부호와 서로의 크기를 지키려고 표준편차로만 나눈다.
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
    봄 고르게 하기와 리워드 고르게 하기를 함께 다룬다.
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
    """이점을 평균 0, 흩어짐 1로 고르게 한다(작은 배치마다)."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def demo_normalization():
    print("=" * 60)
    print("Observation and Reward Normalization Demo")
    print("=" * 60)
    
    # 잣대가 다른 봄을 흉내 낸다
    np.random.seed(42)
    obs_dim = 4
    
    # 피처 0: 큰 잣대, 피처 1: 작은 잣대, 그렇게 이어진다
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
    
    # 리워드 고르게 하기
    print("\n" + "-" * 40)
    print("Reward Normalization:")
    
    rew_normalizer = RewardNormalizer()
    rewards = np.random.randn(50) * 100 + 50  # 잣대가 큰 리워드
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
    demo_normalization()
```

**출력:**

```
============================================================
Observation and Reward Normalization Demo
============================================================

Before normalization (raw observations):
  Mean:  [ 4.99019e+02 -5.00000e-02  2.61250e+01  0.00000e+00]
  Std:   [8.6371e+01 9.0000e-03 5.1939e+01 1.0000e-03]

After normalization:
  Mean:  [ 0.013  0.016  0.03  -0.003]
  Std:   [1.018 0.978 0.997 0.751]
  Range: [-3.06, 3.83]

----------------------------------------
Reward Normalization:
  Raw:  mean=45.1, std=106.0
  Norm: mean=0.336, std=1.839

----------------------------------------
Advantage Normalization:
  Raw:  mean=3.090, std=5.413
  Norm: mean=0.000000, std=1.000
```

## 2. 논의

이 구현은 관찰 정규화의 핵심 논리를 담은 `RunningMeanStd`, `ObservationNormalizer`, `RewardNormalizer` 클래스를 축으로 삼는다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

데모 함수는 이 구성 요소들을 중요한 동작이 드러나는 합성 데이터에 실제로 적용해 보인다. 그 출력을 살피면 하이퍼파라미터 선택과 문제 구조에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실용적 관점에서 이 구현은 순수한 성능보다 명료함을 앞세운다. 프로덕션 시스템은 배치 계산, GPU 가속, 더 정교한 하이퍼파라미터 튜닝 같은 최적화를 더 넣는 것이 보통이다. 그렇더라도 여기서 보인 핵심 알고리즘 아이디어는 대규모 응용에 그대로 옮겨 간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모 코드를 돌리고 중요한 출력 지표를 기록하라. 하이퍼파라미터 하나(러닝레이트, 은닉 차원, 레이어 개수 따위)를 바꿔 결과가 어떻게 달라지는지 설명하라.

</div>

??? success "풀이"
    데모를 돌린 뒤 다른 것은 고정하고 고른 하이퍼파라미터만 체계적으로 바꾼다. 예를 들어 은닉 차원을 두 배로 늘리면 표현력이 커지지만 계산 시간이 는다. 러닝레이트는 단조롭지 않은 결과를 낳는다. 너무 작으면 더디게 수렴하고 너무 크면 진동한다. 고른 하이퍼파라미터의 서로 다른 값 적어도 셋에 대해 구체적인 수를 기록해 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 중요한 구조 선택이 맡은 역할을 설명하라. 왜 그런 활성화 함수, 정규화 전략, 로스 함수를 쓰는가? 다른 것으로 바꾸면 무슨 일이 생기는가?

</div>

??? success "풀이"
    이 구조 선택은 실전 강화학습 기법에서 자리 잡은 모범 사례를 따른다. 예를 들어 ReLU 활성화는 비선형성을 주면서 양의 입력에서 그래디언트가 사라지는 것을 막는다. 로스 함수는 과제 유형에 맞추어 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(예를 들어 시그모이드 활성화, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있으나, 어떤 상황에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 구현을 더 어려운 상황으로 확장하라. 더 큰 데이터셋, 다른 문제 유형, 새 기능 가운데 하나를 고르라. 수정 사항을 설명하고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 정칙화(드롭아웃, 웨이트 감쇠)나 더 정교한 구조(레이어 추가, 스킵 연결)를 더하는 것이다. 고른 확장을 구현하고 같은 데이터로 학습한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 관찰 정규화

이 구현은 관찰 정규화의 핵심 논리를 담은 `RunningMeanStd`, `ObservationNormalizer`, `RewardNormalizer` 클래스를 축으로 삼는다.

핵심 클래스는 `RunningMeanStd`, `ObservationNormalizer`, `RewardNormalizer`, `VecNormalizer`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
