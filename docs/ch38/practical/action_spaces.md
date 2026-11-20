# 액션 공간

액션 공간은 실전 강화학습 기법에서 중요한 아이디어다. 액션 변환 도구를 다룬다. 이 구현은 관련된 중요한 알고리즘과 자료 구조를 직접 다뤄 보이며, 이론적 기초와 실제 배포 시 고려할 점을 함께 보여 준다.

## 1. 코드

```python
"""
34.6.2장: 액션 공간
===============================
액션 공간 감싸개, 밑천 매임, 액션 바꾸기 도구.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

# 이 쪽이 싣는 수가 돌릴 때마다 달라지지 않게 씨앗을 박는다.
torch.manual_seed(0)
np.random.seed(0)


# ========================================================================
# 메인
# ========================================================================


class PortfolioActionHead(nn.Module):
    """
    클래스별 매임을 지닌 밑천 나누기 액션 머리.
    
    클래스:
    - 'softmax': 사기만 하며 무게 합이 1이다
    - 'long_short': 무게 합이 0이다(저자에 치우치지 않음)
    - 'leverage': L1 노름이 max_leverage로 매인다
    """
    
    def __init__(self, hidden_dim, n_assets, mode="softmax", max_leverage=1.0):
        super().__init__()
        self.mode = mode
        self.max_leverage = max_leverage
        self.head = nn.Linear(hidden_dim, n_assets)
    
    def forward(self, features):
        raw = self.head(features)
        
        if self.mode == "softmax":
            return F.softmax(raw, dim=-1)
        
        elif self.mode == "long_short":
            weights = torch.tanh(raw)
            weights = weights - weights.mean(dim=-1, keepdim=True)
            return weights
        
        elif self.mode == "leverage":
            weights = torch.tanh(raw)
            l1_norm = weights.abs().sum(dim=-1, keepdim=True)
            scale = torch.clamp(l1_norm / self.max_leverage, min=1.0)
            return weights / scale
        
        return raw


class DiscreteActionWrapper:
    """이어진 액션 공간을 칸으로 따로 떼어 놓는다."""
    
    def __init__(self, n_bins_per_dim, low, high):
        self.n_bins = n_bins_per_dim
        self.low = np.array(low)
        self.high = np.array(high)
        self.dim = len(low)
        self.total_actions = n_bins_per_dim ** self.dim
    
    def discrete_to_continuous(self, action_idx):
        """따로 떨어진 액션 번호를 이어진 액션으로 바꾼다."""
        indices = []
        idx = action_idx
        for _ in range(self.dim):
            indices.append(idx % self.n_bins)
            idx //= self.n_bins
        
        continuous = np.array([
            self.low[d] + (self.high[d] - self.low[d]) * i / (self.n_bins - 1)
            for d, i in enumerate(indices)
        ])
        return continuous


class ContinuousActionRescaler:
    """액션을 [-1, 1]에서 참 매임으로 다시 잣댄다."""
    
    def __init__(self, low, high):
        self.low = torch.FloatTensor(low)
        self.high = torch.FloatTensor(high)
        self.center = (self.high + self.low) / 2
        self.scale = (self.high - self.low) / 2
    
    def scale_action(self, action):
        """[-1, 1] → [low, high]"""
        return action * self.scale + self.center
    
    def unscale_action(self, action):
        """[low, high] → [-1, 1]"""
        return (action - self.center) / self.scale


def demo_action_spaces():
    print("=" * 60)
    print("Action Space Demonstrations")
    print("=" * 60)
    
    # 밑천 액션 머리
    features = torch.randn(4, 64)  # 배치 크기 4
    n_assets = 5
    
    for mode in ["softmax", "long_short", "leverage"]:
        head = PortfolioActionHead(64, n_assets, mode=mode, max_leverage=1.5)
        weights = head(features)
        print(f"\n{mode} mode:")
        print(f"  Weights[0]: {weights[0].detach().numpy().round(3)}")
        print(f"  Sum: {weights[0].sum().item():.4f}")
        print(f"  L1 norm: {weights[0].abs().sum().item():.4f}")
    
    # 따로 떼어 놓기
    print("\n" + "-" * 40)
    wrapper = DiscreteActionWrapper(n_bins_per_dim=5, low=[-1, -1], high=[1, 1])
    print(f"Discrete actions: {wrapper.total_actions} (5 bins × 2 dims)")
    for idx in [0, 6, 12, 24]:
        cont = wrapper.discrete_to_continuous(idx)
        print(f"  Action {idx:>2d} → {cont.round(3)}")
    
    # 다시 잣대기
    print("\n" + "-" * 40)
    rescaler = ContinuousActionRescaler(low=[-2.0], high=[2.0])
    for a in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        scaled = rescaler.scale_action(torch.tensor([a]))
        print(f"  {a:>5.1f} → {scaled.item():>5.1f}")


if __name__ == "__main__":
    demo_action_spaces()
```

**출력:**

```
============================================================
Action Space Demonstrations
============================================================

softmax mode:
  Weights[0]: [0.284 0.176 0.184 0.123 0.233]
  Sum: 1.0000
  L1 norm: 1.0000

long_short mode:
  Weights[0]: [ 0.175  0.391 -0.415  0.347 -0.498]
  Sum: 0.0000
  L1 norm: 1.8246

leverage mode:
  Weights[0]: [ 0.264 -0.163  0.441 -0.23   0.402]
  Sum: 0.7140
  L1 norm: 1.5000

----------------------------------------
Discrete actions: 25 (5 bins × 2 dims)
  Action  0 → [-1. -1.]
  Action  6 → [-0.5 -0.5]
  Action 12 → [0. 0.]
  Action 24 → [1. 1.]

----------------------------------------
   -1.0 →  -2.0
   -0.5 →  -1.0
    0.0 →   0.0
    0.5 →   1.0
    1.0 →   2.0
```

## 2. 논의

이 구현은 액션 공간의 핵심 논리를 담은 `PortfolioActionHead`, `DiscreteActionWrapper`, `ContinuousActionRescaler` 클래스를 축으로 삼는다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

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

**다룬 것** — 액션 공간

이 구현은 액션 공간의 핵심 논리를 담은 `PortfolioActionHead`, `DiscreteActionWrapper`, `ContinuousActionRescaler` 클래스를 축으로 삼는다.

핵심 클래스는 `PortfolioActionHead`, `DiscreteActionWrapper`, `ContinuousActionRescaler`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
