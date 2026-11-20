# 관측 정규화

관측 정규화는 관측의 각 성분에서 실행 평균을 빼고 실행 표준편차로 나누어, 스케일이 제각각인 입력을 비슷한 크기로 맞추는 실전 강화학습 기법이다. 이 쪽은 웰퍼드 방식의 실행 통계(`RunningMeanStd`)를 바탕으로 관측 정규화, 리워드 정규화, 벡터 둘레용 정규화, 그리고 미니배치 단위의 어드밴티지 정규화를 구현하고 합성 데이터에서 그 효과를 잰다.

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

이 구현은 관측 정규화의 핵심 논리를 담은 `RunningMeanStd`, `ObservationNormalizer`, `RewardNormalizer` 클래스를 축으로 삼는다. 코드는 알고리즘 구성 요소를 데모 논리와 분리하는 모듈식 설계를 따른다.

데모 함수는 이 구성 요소들을 스케일이 10만 배까지 차이 나는 합성 관측에 적용한다. 정규화 전에는 표준편차가 86.371부터 0.001까지 벌어져 있지만, 정규화 뒤에는 평균이 모두 0 근처(0.03 이하)이고 표준편차는 0.751~1.018로 모인다. 넷째 성분만 0.751로 1에 못 미치는 것은 `RunningMeanStd` 가 처음에 분산 1을 의사 개수 1e-4로 품고 시작하기 때문이다. 다른 성분에서는 이 몫이 무시할 만하지만, 표준편차가 0.001인 성분에서는 100개 표본의 제곱합과 같은 크기라 분산 추정을 두 배쯤 부풀린다. 리워드 정규화 뒤의 표준편차가 1.839로 1보다 큰 것은 처음 몇 걸음에서 표준편차 추정이 아직 작아 값이 크게(클리핑 상한 10까지) 나오기 때문이다. 어드밴티지 정규화는 한 미니배치 안에서 바로 계산하므로 평균 0, 표준편차 1이 정확히 나온다.

실용적 관점에서 이 구현은 순수한 성능보다 명료함을 앞세운다. 프로덕션 시스템은 배치 계산, GPU 가속, 더 정교한 하이퍼파라미터 튜닝 같은 최적화를 더 넣는 것이 보통이다. 그렇더라도 여기서 보인 핵심 알고리즘 아이디어는 대규모 응용에 그대로 옮겨 간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모 코드를 돌리고 중요한 출력 지표를 기록하라. 하이퍼파라미터 하나(클리핑 값, 통계를 모으는 표본 수, 초기 의사 개수 따위)를 바꿔 결과가 어떻게 달라지는지 설명하라.

</div>

??? success "풀이"
    데모를 돌린 뒤 다른 것은 고정하고 고른 하이퍼파라미터만 체계적으로 바꾼다. 통계를 모으는 표본 수를 100에서 1000으로 늘리면 초기 분산 1의 영향이 묽어져 넷째 성분의 정규화 뒤 표준편차가 1에 가까워진다. 클리핑 값을 3으로 줄이면 정규화 뒤 범위가 [-3, 3]으로 잘리고, 리워드 정규화의 초반 큰 값도 함께 잘려 표준편차가 줄어든다. 고른 하이퍼파라미터의 서로 다른 값 적어도 셋에 대해 구체적인 수를 기록해 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 중요한 구조 선택이 맡은 역할을 설명하라. 왜 관측은 평균을 빼고 표준편차로 나누는데 리워드는 표준편차로만 나누는가? `VecNormalizer` 가 리워드 대신 할인 리턴의 통계를 쓰는 까닭은 무엇인가?

</div>

??? success "풀이"
    관측은 네트워크 입력이므로 평균을 빼서 0 근처에 모아야 첫 레이어의 활성화가 한쪽으로 쏠리지 않는다. 리워드에서 평균을 빼면 부호가 바뀌어, 예를 들어 매 스텝 +1을 받는 생존 과제가 「살아 있으면 벌점」 과제로 뒤집힐 수 있다. 그래서 리워드는 표준편차로만 나누어 부호와 상대 크기를 지킨다. `VecNormalizer` 가 할인 리턴의 분산을 쓰는 것은, 밸류 함수가 맞히는 대상이 리워드 하나가 아니라 할인 리턴이기 때문이다. 리턴의 스케일을 1 근처로 맞추면 밸류 타깃의 크기가 안정되고, 할인율이 높아 리턴이 리워드보다 훨씬 커지는 과제에서도 같은 러닝레이트가 통한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 구현을 더 어려운 상황으로 확장하라. 실제 학습, 다른 문제 유형, 새 기능 가운데 하나를 고르라. 수정 사항을 설명하고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 이 정규화기를 실제 학습 루프(예를 들어 PPO)에 걸어 정규화 유무에 따른 학습 곡선을 견주는 것이다. 이때 평가 단계에서는 `eval()` 로 통계 갱신을 멈추고, 학습 때의 통계를 모델과 함께 저장해야 한다. 고른 확장을 구현하고 같은 둘레와 시드로 학습한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 관측 정규화

이 구현은 관측 정규화의 핵심 논리를 담은 `RunningMeanStd`, `ObservationNormalizer`, `RewardNormalizer` 클래스를 축으로 삼는다.

벡터 둘레용 `VecNormalizer` 까지 포함해, 앞의 연습문제 세 개로 이 구성 요소들을 직접 바꿔 보며 확인할 수 있다.
