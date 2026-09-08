# 그래프 되돌이 맞춤

그래프 되돌이 맞춤은 그래프 켜 헤아리기 일의 중요한 개념이다. 이 짜기는 얽힌 핵심 알고리즘과 자료 얼개를 손으로 만져 보게 하며 이론 바탕과 실제로 펼칠 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""29.5.2: 그래프 되돌이 맞춤"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

# ========================================================================
# 메인
# ========================================================================

class GraphRegressor(nn.Module):
    def __init__(self, in_ch, hidden_ch, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([nn.Linear(in_ch if i==0 else hidden_ch, hidden_ch) for i in range(num_layers)])
        self.predictor = nn.Sequential(nn.Linear(hidden_ch, hidden_ch), nn.ReLU(), nn.Linear(hidden_ch, 1))
    def forward(self, x, edge_index, batch=None):
        n = x.shape[0]
        if batch is None: batch = torch.zeros(n, dtype=torch.long, device=x.device)
        src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n, device=x.device)
        src_a = torch.cat([src, loop]); dst_a = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst_a, torch.ones(dst_a.shape[0], device=x.device))
        norm = (deg[src_a]*deg[dst_a]).pow(-0.5); norm[norm==float('inf')]=0
        for lin in self.convs:
            h = lin(x); msg = h[src_a]*norm.unsqueeze(1)
            out = torch.zeros(n, h.shape[1], device=x.device)
            out.scatter_add_(0, dst_a.unsqueeze(1).expand_as(msg), msg)
            x = F.relu(out)
        ng = batch.max().item()+1
        pool = torch.zeros(ng, x.shape[1], device=x.device)
        pool.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
        cnt = torch.zeros(ng, device=x.device)
        cnt.scatter_add_(0, batch, torch.ones(n, device=x.device))
        pool = pool / cnt.clamp(min=1).unsqueeze(1)
        return self.predictor(pool).squeeze(-1)

def demo():
    print("=" * 60); print("Graph Regression"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    graphs = []
    for i in range(200):
        n = np.random.randint(5, 15); x = torch.randn(n, 4)
        edges = [(j, j+1) for j in range(n-1)]
        for _ in range(n//3):
            u,v = np.random.choice(n, 2, replace=False); edges.append((u,v))
        src = [e[0] for e in edges]+[e[1] for e in edges]
        dst = [e[1] for e in edges]+[e[0] for e in edges]
        ei = torch.tensor([src, dst], dtype=torch.long)
        target = x.sum().item() * 0.1 + len(edges) * 0.05 + np.random.randn() * 0.1
        graphs.append((x, ei, target))

    model = GraphRegressor(4, 32); opt = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    for epoch in range(50):
        total_loss = 0
        for x, ei, y in graphs[:160]:
            opt.zero_grad()
            pred = model(x, ei)
            loss = F.mse_loss(pred, torch.tensor([y]))
            loss.backward(); opt.step(); total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            model.eval(); test_loss = sum(F.mse_loss(model(x,ei), torch.tensor([y])).item() for x,ei,y in graphs[160:])
            print(f"  Epoch {epoch+1}: Train={total_loss/160:.4f}, Test={test_loss/40:.4f}")
            model.train()

if __name__ == "__main__":
    demo()```

## 2. 논의

이 짜기는 그래프 되돌이 맞춤의 핵심 논리를 감싼 `GraphRegressor` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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

**다룬 것** — 그래프 되돌이 맞춤

이 짜기는 그래프 되돌이 맞춤의 핵심 논리를 감싼 `GraphRegressor` 갈래를 한가운데 둔다.

고갱이 갈래는 `GraphRegressor`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
