# 그래프 만들기 들머리
## 만들기 문제

그래프 만들기는 익히기 모임 $\{\mathcal{G}_1, \ldots, \mathcal{G}_N\}$에서 그래프에 대한 분포 $p_\theta(\mathcal{G})$을 배우고 익히기 분포와 통계로 가려낼 수 없는 새 그래프를 뽑으려 한다. 이 문제는 그래프에만 있는 세 가지 얼개 성질 때문에 그림이나 차례를 만드는 것보다 근본에서 더 어렵다.

**얽음의 내놓기 자리.** 이름표가 붙은 마디 $n$개의 방향 없는 그래프는 두 값 변 변수 $\binom{n}{2}$개를 밝혀야 하며 가능한 그래프가 $2^{\binom{n}{2}}$개다. $n = 50$이면 $10^{368}$을 넘어 볼 수 있는 우주의 원자 수보다 훨씬 많다. 다룰 만한 만들기 방법은 이 자리에 얼개를 두어야 한다.

**자리바꿈 맞섬.** 마디가 $n$개인 그래프 $\mathcal{G}$은 마디 이름표를 바꾸면 같은 뜻의 나타냄이 많게는 $n!$개다. 짓는 모델은 다음 가운데 하나를 해야 한다:

1. 드러나게 자리바꿈에 안 바뀐다: 모든 $\pi \in S_n$에 대해 $p_\theta(\mathcal{G}) = p_\theta(\pi(\mathcal{G}))$
2. 맞섬을 깨는 표준 차례를 쓴다
3. 자리바꿈에 대해 주변화한다(흔히 다룰 수 없다)

**차원이 제각각.** 눈금이 붙박인 그림과 달리 자료 뭉치의 그래프는 마디와 변의 수가 서로 다를 수 있다. 모델은 크기에 조건을 걸거나 자연스러운 멈춤 잣대가 있는 얼개를 써서 이 제각각임을 다루어야 한다.

## 엄밀한 문제 서술

그래프마다 $\mathcal{G}_i = (\mathcal{V}_i, \mathcal{E}_i, \mathbf{X}_i, \mathbf{E}_i)$이 마디 $\mathcal{V}_i$, 변 $\mathcal{E}_i$, 마디 특징 $\mathbf{X}_i \in \mathbb{R}^{|\mathcal{V}_i| \times d_n}$, 변 특징 $\mathbf{E}_i \in \mathbb{R}^{|\mathcal{E}_i| \times d_e}$으로 이루어진 자료 뭉치 $\mathcal{D} = \{\mathcal{G}_i\}_{i=1}^N$이 주어질 때 목표는 다음과 같다:

$$
\theta^* = \arg\max_\theta \sum_{i=1}^N \log p_\theta(\mathcal{G}_i)
$$

가능도 $p_\theta(\mathcal{G})$은 마디 자리바꿈에 안 바뀌어야 한다:

$$
p_\theta(\mathcal{G}) = p_\theta(\mathbf{P}\mathcal{G}\mathbf{P}^\top) \quad \forall \mathbf{P} \in \Pi_n
$$

여기서 $\mathbf{P}\mathcal{G}\mathbf{P}^\top$은 이웃 행렬과 마디 특징의 자리를 한꺼번에 바꾸는 것을 뜻한다.

## 두 가지 틀

### 자기 되돌이 쪼개기

고른 차례 $\sigma$에 대해 사슬 규칙으로 함께 분포를 인수 분해한다:

$$
p_\theta(\mathcal{G}) = \prod_{t=1}^{T} p_\theta(a_t \mid a_{1:t-1})
$$

여기서 $a_t$은 $t$번째 만들기 움직임(마디, 변, 덩이 더하기)을 나타낸다. 끝냄 움직임으로 크기가 제각각인 그래프를 자연스럽게 다루지만 차례에 매이게 된다.

### 한 번에 만들기

마디와 변을 모두 한꺼번에 만든다:

$$
p_\theta(\mathcal{G}) = p_\theta(\mathbf{A}, \mathbf{X})
$$

이는 온전히 나란하지만 붙박인 최대 크기 $n_{\max}$을 다루고 자리바꿈에 안 바뀜을 보장해야 한다. 퍼짐 바탕 방법이 되풀이 다듬기로 이 틀을 넓힌다.

## 만들기 흐름

```
Training Data ──→ Graph Encoder ──→ Latent Space ──→ Graph Decoder ──→ Generated Graph
                        │                                    │
                   마디 차례              올바름 지키기
                   특징 뽑기         뒷손질
                   크기 고르게 맞추기         매임 채우기
```

여느 그래프 만들기 흐름은 다음을 담는다:

1. **앞손질**: 마디 차례를 표준으로 맞추고(보기로 너비 우선, 깊이 우선) 크기를 고르게 채운다
2. **담기**: 그래프 신경망으로 들임 그래프를 숨은 나타냄으로 옮긴다
3. **숨은 값 나타내기**: 숨은 분포를 배운다(변분 스스로 담개, 흐름, 퍼짐)
4. **풀기**: 숨은 표본을 그래프 얼개로 옮긴다
5. **뒷손질**: 올바름 매임을 지키고 채운 것을 없앤다

## 계량 금융과의 이음

금융 그래프 만들기는 위험 다루기의 결정적인 바탕 시설이다. 은행 사이 빌려주기 그물을 보자. 은행마다 속성(자산, 부채, 자본 비율)을 가진 마디이고 방향 변마다 금액과 만기를 가진 빌려주기 노출을 나타낸다. 참 그물은 비밀이고 일부만 볼 수 있다. 이 그물을 그럴듯하게 채우는 일은 다음에 꼭 필요하다:

- **옮아감 나타내기**: 여러 그물 위상에서 못 갚음이 줄줄이 번지는 것을 흉내 내기
- **꾸러미 버거움 시험**: 지어낸 시장 국면에서 꾸러미 성과 따지기
- **거래 상대 위험**: 만들어 낸 그물 길로 에두른 이음에 대한 노출 어림하기

이 장의 방법들이 이 모든 쓰임새의 기술 바탕을 준다.

## 짜기: 그래프 만들기 바탕 갈래

```python
"""
그래프 만들기 바탕 갈래와 연장.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class GeneratedGraph:
    """만들어 낸 그래프를 담는 그릇."""
    adjacency: torch.Tensor       # (n, n) 두 값 또는 무게
    node_features: Optional[torch.Tensor] = None  # (n, d_n)
    edge_features: Optional[torch.Tensor] = None   # (n, n, d_e)
    num_nodes: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.num_nodes == 0:
            self.num_nodes = self.adjacency.size(0)
        if self.metadata is None:
            self.metadata = {}

    @property
    def num_edges(self) -> int:
        if self.adjacency.dim() == 2:
            return int(self.adjacency.sum().item()) // 2
        return 0

    @property
    def density(self) -> float:
        n = self.num_nodes
        max_edges = n * (n - 1) / 2
        return self.num_edges / max_edges if max_edges > 0 else 0.0


class GraphGenerator(ABC, nn.Module):
    """그래프 만들개의 추상 바탕 갈래."""

    def __init__(self, max_nodes: int, node_feature_dim: int = 0):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim

    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        """손실을 돌려주는 익히기 앞으로 가기."""
        ...

    @abstractmethod
    @torch.no_grad()
    def generate(self, num_graphs: int = 1, **kwargs) -> list[GeneratedGraph]:
        """새 그래프를 만든다."""
        ...

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def adjacency_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    """이웃 행렬을 PyG의 edge_index 꼴로 바꾼다."""
    row, col = torch.where(adj > 0)
    return torch.stack([row, col], dim=0)


def edge_index_to_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """PyG의 edge_index을 이웃 행렬로 바꾼다."""
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


if __name__ == "__main__":
    # GeneratedGraph 쓰임 보이기
    n = 10
    adj = torch.zeros(n, n)
    # 빽빽함 30%쯤의 아무 그래프를 만든다
    mask = torch.rand(n, n) < 0.3
    adj = (mask | mask.t()).float()
    adj.fill_diagonal_(0)

    graph = GeneratedGraph(
        adjacency=adj,
        node_features=torch.randn(n, 4),
        metadata={"generator": "random", "density_target": 0.3},
    )
    print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
    print(f"Density: {graph.density:.3f}")
    print(f"Edge index shape: {adjacency_to_edge_index(adj).shape}")
```

## 연습문제

**연습문제 1.**
자리바꿈에 안 바뀜 문제 때문에 그래프 만들기가 그림 만들기보다 왜 근본에서 더 어려운지 밝혀라. 이름표가 붙은 마디 $n$개의 그래프에는 같은 뜻의 이웃 행렬 나타냄이 몇 개 있는가?

??? success "연습문제 1 풀이"
    마디 $n$개의 그래프는 서로 다른 이웃 행렬 $n!$개로 나타낼 수 있다(마디 이름표의 자리바꿈마다 하나). $n = 10$이면 같은 뜻의 나타냄이 $10! = 3{,}628{,}800$개다. 짓는 모델은 다음 가운데 하나를 해야 한다. (1) 이 겹침을 줄이려 표준 차례를 배운다, (2) 자리바꿈에 안 바뀌는 손실 함수를 쓴다(보기로 그래프 짝짓기), (3) 본디 안 바뀌는 나타냄에서 돈다(보기로 스펙트럼 특징). 그림 만들기는 픽셀에 붙박인 자리 차례가 있어 이 문제를 겪지 않는다. $\square$

---

**연습문제 2.**
커지기, 품질, 올바름 매임을 지킬 수 있음의 면에서 그래프 만들기의 자기 되돌이 길과 한 번에 만들기 길을 견주어라.

??? success "연습문제 2 풀이"
    자기 되돌이 방법은 마디와 변을 차례로 만들어 걸음마다 그 자리 매임(보기로 원자가)을 자연스럽게 지키지만, 이웃 결정이 늘어나 $O(n^2)$으로 커진다. 한 번에 만들기 방법은 이웃 행렬 전체를 한꺼번에 만들어 나란한 셈에는 더 잘 커지지만 띄엄띄엄한 얼개와 온 자리 매임에 애를 먹는다. 자기 되돌이 방법은 작은 크기($n < 100$)에서 대개 더 좋은 그래프를 내지만 큰 그래프에서는 느려진다. 자리바꿈에 안 바뀌는 손실을 쓰는 한 번에 만들기 방법은 더 빠르지만 짝짓기 문제를 조심스레 다루어야 한다. $\square$

---

**연습문제 3.**
자기 되돌이 인수 분해 $p(G) = \prod_{i=1}^n p(\text{node}_i | \text{nodes}_{<i}) \prod_{j<i} p(e_{ij} | \text{node}_i, \text{nodes}_{<i})$ 아래에서 그래프의 가능도를 이끌어 내어라.

??? success "연습문제 3 풀이"
    붙박인 마디 차례 $\pi$ 아래에서 함께 확률은 $p(G | \pi) = \prod_{i=1}^n p(v_{\pi(i)} | G_{\pi(<i)}) \prod_{j < i} p(e_{\pi(i),\pi(j)} | v_{\pi(i)}, G_{\pi(<i)})$으로 인수 분해된다. 여기서 $G_{\pi(<i)}$은 앞서 만든 마디의 부분 그래프다. 주변 가능도 $p(G) = \frac{1}{n!}\sum_\pi p(G | \pi)$은 모든 차례를 더하지만 다룰 수 없다. 실제로는 표준 차례(보기로 너비 우선)를 써서 가능도를 그 차례에 매인 것으로 만든다. 차례의 품질이 만들기 품질에 영향을 준다. $\square$

---

**연습문제 4.**
분포 잣대를 넘어 화학의 올바름과 약다움을 재는, 만들어 낸 분자 그래프의 따지기 규약을 내놓아라.

??? success "연습문제 4 풀이"
    분포 잣대(차수 분포, 뭉침 계수)를 넘어 다음을 따진다. (1) 화학의 올바름 -- 원자가 매임을 만족하는 분자의 몫(RDKit으로 확인), (2) 하나뿐임 -- 서로 다른 올바른 분자의 몫, (3) 새로움 -- 익히기 모임에 없는 몫, (4) 약다움 -- QED 점수, 리핀스키의 다섯 규칙 지킴, (5) 만들기 쉬움 -- 합성이 얼마나 쉬운지 나타내는 SA 점수, (6) 성질 가장 좋게 하기 -- 바란 과녁 성질과 얻은 성질의 얽힘. 여러 번 만들어 얻은 믿음 구간과 함께 모든 잣대를 알린다. $\square$
