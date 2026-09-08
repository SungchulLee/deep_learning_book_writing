# 움직임 공간

움직임 공간은 실제 힘 북돋우는 배움 재주에서 종요로운 생각이다. 움직임 바꾸기 도구를 다룬다. 이 구현은 여기에 걸린 종요로운 알고리즘과 자료 얼개를 손으로 만져 보이며, 이치 바탕과 실제로 서비스에 올릴 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""
34.6.2장: 움직임 공간
===============================
움직임 공간 감싸개, 밑천 매임, 움직임 바꾸기 도구.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

# ========================================================================
# 메인
# ========================================================================


class PortfolioActionHead(nn.Module):
    """
    갈래별 매임을 지닌 밑천 나누기 움직임 머리.
    
    갈래:
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
    """이어진 움직임 공간을 칸으로 따로 떼어 놓는다."""
    
    def __init__(self, n_bins_per_dim, low, high):
        self.n_bins = n_bins_per_dim
        self.low = np.array(low)
        self.high = np.array(high)
        self.dim = len(low)
        self.total_actions = n_bins_per_dim ** self.dim
    
    def discrete_to_continuous(self, action_idx):
        """따로 떨어진 움직임 번호를 이어진 움직임으로 바꾼다."""
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
    """움직임을 [-1, 1]에서 참 매임으로 다시 잣댄다."""
    
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
    
    # 밑천 움직임 머리
    features = torch.randn(4, 64)  # 묶음 크기 4
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
    demo_action_spaces()```

## 2. 논의

이 구현은 움직임 공간의 한가운데 논리를 담은 `PortfolioActionHead`, `DiscreteActionWrapper`, `ContinuousActionRescaler` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

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

**다룬 것** — 움직임 공간

이 구현은 움직임 공간의 한가운데 논리를 담은 `PortfolioActionHead`, `DiscreteActionWrapper`, `ContinuousActionRescaler` 클래스를 축으로 삼는다.

고갱이 갈래는 `PortfolioActionHead`, `DiscreteActionWrapper`, `ContinuousActionRescaler`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
