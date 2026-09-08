# Set2Set 읽어내기

Set2Set 읽어내기는 그래프 켜 헤아리기 일의 중요한 개념이다. 이 짜기는 얽힌 핵심 알고리즘과 자료 얼개를 손으로 만져 보게 하며 이론 바탕과 실제로 펼칠 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""29.5.5: Set2Set 읽어내기"""
import torch, torch.nn as nn, torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

class Set2Set(nn.Module):
    def __init__(self, in_ch, processing_steps=3):
        super().__init__()
        self.steps = processing_steps
        self.lstm = nn.LSTMCell(2 * in_ch, in_ch)
        self.in_ch = in_ch
    def forward(self, x, batch):
        ng = batch.max().item() + 1
        q = torch.zeros(ng, self.in_ch, device=x.device)
        c = torch.zeros(ng, self.in_ch, device=x.device)
        r = torch.zeros(ng, self.in_ch, device=x.device)
        for _ in range(self.steps):
            q, c = self.lstm(torch.cat([q, r], dim=-1), (q, c))
            e = (x * q[batch]).sum(dim=-1)
            e_max = torch.full((ng,), float('-inf'), device=x.device)
            e_max.scatter_reduce_(0, batch, e, reduce='amax')
            a = torch.exp(e - e_max[batch])
            a_sum = torch.zeros(ng, device=x.device)
            a_sum.scatter_add_(0, batch, a)
            a = a / a_sum[batch].clamp(min=1e-10)
            r = torch.zeros(ng, self.in_ch, device=x.device)
            r.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x * a.unsqueeze(1))
        return torch.cat([q, r], dim=-1)

def demo():
    print("=" * 60); print("Set2Set Readout"); print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(10, 8)
    batch = torch.tensor([0,0,0,0,1,1,1,2,2,2])
    s2s = Set2Set(8, processing_steps=3)
    out = s2s(x, batch)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Output dim = 2 * input_dim = {2*8}")
    # 단순 모으기와 견주기
    pool = torch.zeros(3, 8)
    pool.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
    print(f"Sum pool output: {pool.shape}")
    print(f"Set2Set output (richer): {out.shape}")

if __name__ == "__main__":
    demo()```

## 2. 논의

이 짜기는 Set2Set 읽어내기의 핵심 논리를 감싼 `Set2Set` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

보여 주기 함수는 핵심 움직임을 도드라지게 하는 만든 자료에서 이 조각들의 실제 쓰임을 보인다. 내놓기를 살펴보면 윗매개변수를 어떻게 고르고 문제를 어떻게 차리느냐에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

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
    이 얼개 고르기는 그래프 켜 헤아리기 일에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — Set2Set 읽어내기

이 짜기는 Set2Set 읽어내기의 핵심 논리를 감싼 `Set2Set` 갈래를 한가운데 둔다.

고갱이 갈래는 `Set2Set`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
