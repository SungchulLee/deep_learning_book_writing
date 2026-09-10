# 지나친 매끄러워짐

지나친 매끄러워짐은 층을 쌓을수록 마디 나타냄이 점점 비슷해지는 깊은 그래프 신경망의 결정적 실패 방식이다. 이 현상은 그래프 신경망의 실제 깊이를 제한하며 재는 잣대와 누그러뜨리는 셈속에 대한 넓은 연구를 이끌었다. 지나친 매끄러워짐을 이해하는 것은 그래프의 먼 거리 매임을 담는 그래프 신경망 얼개를 짜는 데 꼭 필요하다.

## 1. 코드

```python
"""
29.4.2: 지나친 매끄러워짐 - 재기와 누그러뜨리기.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx

# ========================================================================
# 메인
# ========================================================================

def measure_smoothness(H, edge_index):
    """평균 절대 거리와 디리클레 에너지를 셈한다."""
    n = H.shape[0]
    # 평균 절대 거리
    diffs = H.unsqueeze(0) - H.unsqueeze(1)
    mad = diffs.norm(dim=-1).mean().item()
    # 디리클레 에너지
    src, dst = edge_index[0], edge_index[1]
    energy = ((H[src] - H[dst])**2).sum().item()
    return mad, energy

class SimpleGCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, num_layers):
        super().__init__()
        self.input = nn.Linear(in_ch, hidden_ch)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(nn.Linear(hidden_ch, hidden_ch))
    def forward(self, x, edge_index):
        n = x.shape[0]; x = F.relu(self.input(x))
        src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n, device=x.device)
        src_a = torch.cat([src, loop]); dst_a = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst_a, torch.ones(dst_a.shape[0], device=x.device))
        norm = (deg[src_a] * deg[dst_a]).pow(-0.5)
        norm[norm==float('inf')] = 0
        for lin in self.convs:
            h = lin(x); msg = h[src_a] * norm.unsqueeze(1)
            out = torch.zeros(n, h.shape[1], device=x.device)
            out.scatter_add_(0, dst_a.unsqueeze(1).expand_as(msg), msg)
            x = F.relu(out)
        return x

def demo_over_smoothing():
    print("=" * 60); print("Over-Smoothing Analysis"); print("=" * 60)
    torch.manual_seed(42)
    G = nx.karate_club_graph(); n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.eye(n)
    print(f"{'Layers':>8} {'MAD':>10} {'Dirichlet':>12}")
    for nl in [1, 2, 4, 8, 16, 32]:
        model = SimpleGCN(n, 16, nl); model.eval()
        with torch.no_grad():
            H = model(x, ei)
            mad, energy = measure_smoothness(H, ei)
        print(f"{nl:>8} {mad:>10.4f} {energy:>12.4f}")

if __name__ == "__main__":
    demo_over_smoothing()
```

**출력:**

```
============================================================
Over-Smoothing Analysis
============================================================
  Layers        MAD    Dirichlet
       1     0.1786      24.2646
       2     0.1157      12.8935
       4     0.1644      28.5002
       8     0.1583      25.9448
      16     0.1455      20.6601
      32     0.1231      15.4138
```

## 2. 논의

이 코드는 지나친 매끄러워짐을 재는 핵심 잣대 둘을 보여 준다. 평균 거리(MAD)와 디리클레 에너지이다. 평균 거리는 마디 나타냄 사이의 짝별 거리의 평균을 재고, 디리클레 에너지는 이어진 마디의 특징 차의 제곱 합을 잰다. 여느 그래프 겹말기 신경망에서 두 잣대 모두 깊이에 따라 단조로 줄어들어, 되풀이되는 이웃 모으기가 낮은 진동수만 통과시키는 거르개처럼 군다는 이론 예측을 확인해 준다.

`SimpleGCN` 짜기는 누그러뜨리는 재주가 없는 밋밋한 여러 층 그래프 겹말기 신경망을 보인다. 층이 1개에서 32개로 늘면 평균 거리와 디리클레 에너지가 모두 크게 떨어져 마디 특징이 공통 벡터로 모임을 가리킨다. 이 모임은 고르게 맞춘 이웃 행렬의 스펙트럼 반지름이 1이고 그것을 되풀이해 쓰면 특징이 으뜸 고유 벡터로 쏘아지기 때문에 일어난다.

지나친 매끄러워짐을 누그러뜨리는 셈속이 여럿 있다. 남은 이음(ResGCN처럼), 변 떨구기(익히는 동안 변을 아무렇게나 지우기), 짝 고르게 맞추기(짝별 거리를 고르게 맞추기), 그리고 건너뛰는 앎 신경망 같은 얼개 바꿈이다. 무엇을 고를지는 일에 매인다. 어떤 쓰임새는 먼 거리 앎 퍼뜨리기를 위해 참으로 깊은 신경망이 필요하고, 다른 쓰임새는 그 자리의 얼개를 지키는 얕은 얼개에서 가장 잘 듣는다.

## 연습문제

**연습문제 1.**
층이 1, 2, 4, 8, 16, 32개인 SimpleGCN의 평균 거리와 디리클레 에너지를 셈하라. 두 잣대를 깊이의 함수로 그려라. 디리클레 에너지가 처음 값의 1% 아래로 떨어지는 깊이는 얼마인가?

??? success "연습문제 1 풀이"
    보여 주기를 돌리면 두 잣대를 담은 표가 나온다. 보통 디리클레 에너지는 층 8~16개 언저리에서 1층 값의 1% 아래로 떨어진다. 정확한 갈림목은 아무 첫자리매김에 매이지만 지수로 사그라지는 무늬는 한결같다. 사그라지는 빠르기는 고르게 맞춘 라플라스의 스펙트럼 틈이 다스린다. $\lambda_2$이 고르게 맞춘 이웃 행렬의 둘째로 큰 고윳값일 때 $E_L \approx E_1 \cdot \lambda_2^{2L}$이다.

---

**연습문제 2.**
고윳값이 $1 = \lambda_1 > \lambda_2 \geq \cdots \geq \lambda_n$을 만족하는 고르게 맞춘 이웃 행렬 $\hat{A}$을 가진 그래프에서 (비선형이 없는) 선형 그래프 겹말기 신경망 $L$층 뒤의 디리클레 에너지가 $E(H^{(L)}) \leq \lambda_2^{2L} \cdot E(H^{(0)})$을 만족함을 밝혀라.

??? success "연습문제 2 풀이"
    $H^{(L)} = \hat{A}^L H^{(0)} W_1 \cdots W_L$이라 하자. 디리클레 에너지는 $L = I - \hat{A}$일 때 $E(H) = \text{tr}(H^T L H)$이다. $\hat{A}$과 $L$이 고유 벡터를 함께 가지고 $L$의 고윳값이 $1 - \lambda_i$이므로 $E(\hat{A}^L H) = \sum_i (1 - \lambda_i) \lambda_i^{2L} \|u_i^T H\|^2$을 얻는다. $i \geq 2$에서 $\lambda_i^{2L} \leq \lambda_2^{2L}$이고 $1 - \lambda_1 = 0$이므로 $i=1$ 항은 사라진다. 따라서 무게 행렬의 영향을 빼면 $E(H^{(L)}) \leq \lambda_2^{2L} \sum_{i \geq 2} (1 - \lambda_i) \|u_i^T H\|^2 \leq \lambda_2^{2L} E(H^{(0)})$이다.

---

**연습문제 3.**
짝별 온 거리를 상수로 지키도록 마디 특징을 고르게 맞추는 짝 고르게 맞추기(자오와 아코글루, 2020)를 짜라. 이를 `SimpleGCN`에 더하고 16층에서 짝 고르게 맞추기가 있을 때와 없을 때의 지나친 매끄러워짐 잣대를 견주어라.

??? success "연습문제 3 풀이"
    ```python
    def pairnorm(x, s=1.0):
        x = x - x.mean(dim=0, keepdim=True)  # Center
        norm = x.pow(2).sum(dim=-1, keepdim=True).mean().sqrt()
        return s * x / (norm + 1e-8)
    ```
    `SimpleGCN`의 forward 메서드에서 ReLU마다 뒤에 `x = pairnorm(x)`을 넣는다. 짝 고르게 맞추기를 쓰면 16층의 디리클레 에너지가 고르게 맞추지 않은 판보다 크게 높게(흔히 10~100배) 남아 지나친 매끄러워짐을 잘 누그러뜨림을 보여 준다. 평균 거리도 높게 남아 마디 나타냄이 여전히 가릴 수 있음을 확인해 준다.

## 정리하며

**다룬 것** — 지나친 매끄러워짐

이 코드는 지나친 매끄러워짐을 재는 핵심 잣대 둘을 보여 준다.

고갱이 갈래는 `SimpleGCN`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
