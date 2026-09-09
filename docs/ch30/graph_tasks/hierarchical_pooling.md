# 켜진 모으기

켜진 모으기는 그래프 켜 헤아리기 일의 중요한 개념이다. 이 짜기는 얽힌 핵심 알고리즘과 자료 얼개를 손으로 만져 보게 하며 이론 바탕과 실제로 펼칠 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""29.5.4: 켜진 모으기 - TopK과 DiffPool"""
import torch, torch.nn as nn, torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

class TopKPool(nn.Module):
    """배운 점수로 위 k개 마디를 고른다."""
    def __init__(self, in_ch, ratio=0.5):
        super().__init__()
        self.score = nn.Linear(in_ch, 1)
        self.ratio = ratio
    def forward(self, x, edge_index):
        scores = self.score(x).squeeze(-1)
        k = max(1, int(self.ratio * x.shape[0]))
        _, idx = scores.topk(k)
        idx = idx.sort()[0]
        x_pool = x[idx] * torch.sigmoid(scores[idx]).unsqueeze(-1)
        # 변 거르기
        mask = torch.zeros(x.shape[0], dtype=torch.bool)
        mask[idx] = True
        node_map = torch.full((x.shape[0],), -1, dtype=torch.long)
        node_map[idx] = torch.arange(k)
        src, dst = edge_index[0], edge_index[1]
        edge_mask = mask[src] & mask[dst]
        new_src = node_map[src[edge_mask]]; new_dst = node_map[dst[edge_mask]]
        return x_pool, torch.stack([new_src, new_dst])

class SimpleDiffPool(nn.Module):
    """간단히 한 DiffPool: 부드러운 뭉치기."""
    def __init__(self, in_ch, n_clusters):
        super().__init__()
        self.assign = nn.Sequential(nn.Linear(in_ch, n_clusters), nn.Softmax(dim=-1))
    def forward(self, x, adj):
        S = self.assign(x)  # [n, k]
        x_pool = S.T @ x     # [k, d]
        adj_pool = S.T @ adj @ S  # [k, k]
        return x_pool, adj_pool, S

def demo():
    print("=" * 60); print("Hierarchical Pooling"); print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(10, 8)
    ei = torch.tensor([[0,1,1,2,2,3,3,4,5,6,6,7,7,8,8,9],[1,0,2,1,3,2,4,3,6,5,7,6,8,7,9,8]], dtype=torch.long)
    # 위 K개
    topk = TopKPool(8, ratio=0.5)
    x_new, ei_new = topk(x, ei)
    print(f"TopK: {x.shape[0]} -> {x_new.shape[0]} nodes, {ei.shape[1]} -> {ei_new.shape[1]} edges")
    # DiffPool
    adj = torch.zeros(10, 10); adj[ei[0], ei[1]] = 1
    dp = SimpleDiffPool(8, 3)
    x_p, adj_p, S = dp(x, adj)
    print(f"DiffPool: {x.shape[0]} -> {x_p.shape[0]} clusters")
    print(f"  Assignment (first 3 nodes): {S[:3].detach().round(decimals=2)}")

if __name__ == "__main__":
    demo()
```

**출력:**

```
============================================================
Hierarchical Pooling
============================================================
TopK: 10 -> 5 nodes, 16 -> 2 edges
DiffPool: 10 -> 3 clusters
  Assignment (first 3 nodes): tensor([[0.3800, 0.4400, 0.1800],
        [0.2600, 0.4000, 0.3400],
        [0.3300, 0.1900, 0.4800]])
```

## 2. 논의

이 짜기는 켜진 모으기의 핵심 논리를 감싼 `TopKPool`, `SimpleDiffPool` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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

**다룬 것** — 켜진 모으기

이 짜기는 켜진 모으기의 핵심 논리를 감싼 `TopKPool`, `SimpleDiffPool` 갈래를 한가운데 둔다.

고갱이 갈래는 `TopKPool`, `SimpleDiffPool`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
