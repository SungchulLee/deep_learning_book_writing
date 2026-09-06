# 그래프 모으기

마디 켜의 그래프 신경망 층은 마디마다의 박아 넣기를 내놓지만, 분자 성질
헤아리기, 그래프 가름, 사회 그물 살피기 같은 많은 일은 그래프 전체에 대한
붙박인 크기의 나타냄 하나를 바란다. 그래프 모으기는 마디 박아 넣기를
그래프 켜의 벡터로 모아 이 틈을 잇는다.

## 납작한(온 자리) 모으기

가장 단순한 길은 모든 마디 박아 넣기 $\{\mathbf{h}_v : v \in V\}$에
자리바꿈에 안 바뀌는 모으기 함수를 쓰는 것이다.

### 합, 평균, 최대 모으기

$$
\mathbf{h}_G^{\text{sum}} = \sum_{v \in V} \mathbf{h}_v, \quad
\mathbf{h}_G^{\text{mean}} = \frac{1}{|V|} \sum_{v \in V} \mathbf{h}_v, \quad
\mathbf{h}_G^{\text{max}} = \max_{v \in V} \mathbf{h}_v
$$

변형마다 지키는 앎이 다르다:

| 방법 | 지키는 것 | 잃는 것 |
|---|---|---|
| 합 | 그래프 크기, 온 신호 | 마디마다의 세부 |
| 평균 | 평균 분포 | 그래프 크기 |
| 최대 | 두드러진 특징 | 잦음 앎 |

!!! tip "여러 모으기 셈속"
    여러 모으기 내놓기를 이어 붙이면 서로 보완하는 앎을 담는다:
    $\mathbf{h}_G = [\mathbf{h}_G^{\text{sum}} \| \mathbf{h}_G^{\text{mean}} \| \mathbf{h}_G^{\text{max}}]$.

### 눈길 모으기

모으기 전에 마디마다 배운 중요도 점수를 매긴다:

$$
\mathbf{h}_G = \sum_{v \in V} \alpha_v \, \mathbf{h}_v
$$

여기서 눈길 무게는 다음과 같다

$$
\alpha_v = \frac{\exp\!\bigl(f(\mathbf{h}_v)\bigr)}{\sum_{u \in V} \exp\!\bigl(f(\mathbf{h}_u)\bigr)}
$$

그리고 $f$은 배울 수 있는 점수 함수(보통 작은 여러 층 신경망)이다. 이는 모델이
앎이 가장 많은 마디에 집중하게 한다.

## 켜진 모으기

납작한 모으기는 모든 마디를 한 걸음에 눌러 담아 그래프 얼개를 버린다.
켜진 방법은 그래프를 차츰 성글게 하여 여러 잣수의 얼개 앎을
지킨다.

### DiffPool

DiffPool은 층마다 부드러운 배정 행렬 $S^{(l)} \in \mathbb{R}^{n_l \times n_{l+1}}$을 배워
마디 $n_l$개를 뭉치 마디 $n_{l+1}$개로 옮긴다:

$$
X^{(l+1)} = S^{(l)\top} Z^{(l)}, \quad A^{(l+1)} = S^{(l)\top} A^{(l)} S^{(l)}
$$

여기서 $Z^{(l)}$은 층 $l$의 그래프 신경망 박아 넣기이고 $A^{(l)}$은
이웃 행렬이다. 배정 행렬은 따로 둔 그래프 신경망이 만든다:

$$
S^{(l)} = \text{softmax}\!\bigl(\text{GNN}_{\text{pool}}(A^{(l)}, X^{(l)})\bigr)
$$

DiffPool은 온전히 미분할 수 있지만 켜마다 뭉치의 개수를 밝혀야 하고
빽빽한 배정 행렬 때문에 기억 비용이
$O(n^2)$이다.

### SAGPool(스스로 눈길 그래프 모으기)

SAGPool은 배운 점수로 가장 중요한 마디의 부분 모임을 고른다:

$$
\mathbf{s} = \text{GNN}(A, X) \in \mathbb{R}^n
$$

점수가 높은 위 $k$개의 마디를 남긴다:

$$
\text{idx} = \text{top-}k(\mathbf{s}, \lceil k \cdot n \rceil), \quad
X' = X_{\text{idx}} \odot \sigma(\mathbf{s}_{\text{idx}})
$$

여기서 $\odot$은 문을 지난 점수와의 원소마다 곱을 뜻한다.
이웃 행렬도 남은 마디에 맞게 줄인다.

SAGPool은 빽빽한 배정 행렬을 피하므로 DiffPool보다 기억을
아낀다.

### TopKPool

TopKPool은 더 단순한 변형이다. 마디마다 낱값 쏘기 점수를 셈해
위 $k$ 몫을 남긴다. 점수 벡터는 다음과 같다

$$
\mathbf{s} = \frac{X \mathbf{p}}{\|\mathbf{p}\|}
$$

여기서 $\mathbf{p}$은 배울 수 있는 쏘기 벡터이다.

## 구현

```python
"""
그래프 신경망의 그래프 모으기 연산.

합, 평균, 눈길, 위 k개 모으기를 보여 준다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# === 납작한 모으기 ===
def global_sum_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """묶음 번호로 묶어 마디에 대해 합 모으기."""
    num_graphs = batch.max().item() + 1
    out = torch.zeros(num_graphs, x.size(1), device=x.device)
    out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
    return out


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """묶음 번호로 묶어 마디에 대해 평균 모으기."""
    sums = global_sum_pool(x, batch)
    counts = torch.zeros(sums.size(0), device=x.device)
    counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
    return sums / counts.unsqueeze(1).clamp(min=1)


# === 눈길 모으기 ===
class AttentionPool(nn.Module):
    """눈길 무게를 준 온 자리 모으기."""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        scores = self.score_fn(x)
        # 그래프마다 소프트맥스
        max_scores = torch.zeros(batch.max() + 1, 1, device=x.device)
        max_scores.scatter_reduce_(0, batch.unsqueeze(1), scores, reduce="amax")
        scores = scores - max_scores[batch]
        exp_scores = scores.exp()
        sum_exp = torch.zeros(batch.max() + 1, 1, device=x.device)
        sum_exp.scatter_add_(0, batch.unsqueeze(1), exp_scores)
        alpha = exp_scores / sum_exp[batch].clamp(min=1e-8)
        weighted = x * alpha
        return global_sum_pool(weighted, batch)


# === 보기 ===
if __name__ == "__main__":
    torch.manual_seed(42)
    # 그래프 둘을 묶음: 0번 그래프는 마디 3개, 1번 그래프는 마디 2개
    x = torch.randn(5, 8)
    batch = torch.tensor([0, 0, 0, 1, 1])

    print("Sum pool shape:", global_sum_pool(x, batch).shape)    # [2, 8]
    print("Mean pool shape:", global_mean_pool(x, batch).shape)  # [2, 8]

    attn_pool = AttentionPool(in_dim=8)
    print("Attn pool shape:", attn_pool(x, batch).shape)         # [2, 8]
```

## 모으기 방법 고르기

| 방법 | 복잡도 | 지키는 얼개 | 알맞은 곳 |
|---|---|---|---|
| 합/평균/최대 | $O(n)$ | 없음 | 작은 그래프, 밑그림 |
| 눈길 | $O(n)$ | 배운 중요도 | 크기가 바뀌는 그래프 |
| DiffPool | $O(n^2)$ | 켜진 뭉치 | 빽빽하고 짜임새 있는 그래프 |
| SAGPool | $O(n)$ | 마디 고르기 | 큰 그래프, 효율 |

## 참고 문헌

- Ying, R. et al. "Hierarchical Graph Representation Learning with
  Differentiable Pooling." NeurIPS 2018.
- Lee, J. et al. "Self-Attention Graph Pooling." ICML 2019.
- Xu, K. et al. "How Powerful are Graph Neural Networks?" ICLR 2019.


## 연습문제

**연습문제 1.**
그래프 켜 일에서 그래프 모으기가 왜 필요한지 밝히고 겹말기 신경망의 모으기와 견주어라.

??? success "연습문제 1 풀이"
    그래프 모으기는 마디가 $n$개인 그래프를 그래프 켜 헤아리기를 위해 벡터 하나(또는 더 작은 그래프)로 줄인다. 겹말기 신경망에서는 모으기가 고른 격자 위 자리 차원을 줄인다. 그래프는 붙박인 자리 얼개가 없고 크기도 제각각이라 그래프 모으기가 더 어렵다. 흔한 길: (1) 온 자리 모으기(모든 마디 특징의 합/평균/최대), (2) 켜진 모으기(그래프를 차츰 성글게 함), (3) 눈길 바탕 모으기(어느 마디가 중요한지 배움).

---

**연습문제 2.**
DiffPool 켜진 모으기 방법과 그 배정 얼개를 밝혀라.

??? success "연습문제 2 풀이"
    DiffPool(Ying et al., 2018)은 마디 $n$개를 뭉치 $k$개로 옮기는 부드러운 배정 행렬 $S \in \mathbb{R}^{n \times k}$을 배운다. 그래프 신경망 둘이 나란히 돈다. 하나는 마디 박아 넣기 $Z$을 내놓고 다른 하나는 배정 $S = \text{softmax}(\text{GNN}_{pool}(A, X))$을 내놓는다. 성글게 한 그래프의 이웃 관계는 $A' = S^T A S$이고 특징은 $X' = S^T Z$이다. 이를 켜지게 되풀이해 층마다 그래프 크기를 줄인다. 배정은 미분할 수 있어 끝에서 끝까지 익힐 수 있다. 손실 항이 뭉치의 이어짐과 엔트로피 다잡기를 북돋운다.

---

**연습문제 3.**
납작한 모으기(읽어내기)와 켜진 모으기의 차이는 무엇인가?

??? success "연습문제 3 풀이"
    납작한 모으기(읽어내기)는 모든 마디에 모으기 하나를 쓴다: $h_G = \text{AGG}(\{h_v : v \in V\})$이고 여기서 AGG는 합, 평균, 최대, 또는 눈길 무게 합이다. 얼개의 켜는 지키지 못한다. 켜진 모으기는 여러 켜를 거쳐 그래프를 차츰 성글게 하여 여러 잣수의 얼개를 지킨다. 납작한 모으기가 더 단순하고 작은 그래프에 잘 맞는다. 켜진 모으기는 큰 그래프의 더 풍부한 얼개 앎을 담지만 복잡해지고 성글게 하는 걸음마다 앎을 잃을 수 있다.

---

**연습문제 4.**
Set2Set 모으기를 밝히고 왜 단순한 평균/합 모으기보다 나타냄 힘이 셀 수 있는지 말하여라.

??? success "연습문제 4 풀이"
    Set2Set은 장단기 기억망 바탕 눈길 얼개로 그래프 특징을 읽어낸다. 마디 박아 넣기의 모임을 여러 눈길 걸음으로 다루며 걸음마다 다른 마디를 살피는 맥락 벡터를 쌓는다. 마지막 나타냄은 모든 맥락 벡터를 이어 붙인 것이다. 평균/합 모으기는 자리바꿈에 안 바뀌지만 모으기를 한 번만 쓰는 반면, Set2Set은 걸음마다 다른 마디 무리에 집중해 더 복잡한 서로 작용을 담을 수 있다. 분자 성질 헤아리기 일에서 성능을 높이는 것이 보여졌다.