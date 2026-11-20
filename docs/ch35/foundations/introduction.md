# 그래프 생성 입문

---

## 1. 생성 문제

그래프 생성은 학습 집합 $\{\mathcal{G}_1, \ldots, \mathcal{G}_N\}$에서 그래프에 대한 분포 $p_\theta(\mathcal{G})$를 배우고 학습 분포와 통계로 가려낼 수 없는 새 그래프를 뽑으려 한다. 이 문제는 그래프에만 있는 세 가지 구조 성질 때문에 이미지나 시퀀스를 생성하는 것보다 근본에서 더 어렵다.

**조합적 출력 공간.** 레이블이 붙은 노드 $n$개의 방향 없는 그래프는 이진 엣지 변수 $\binom{n}{2}$개를 밝혀야 하며 가능한 그래프가 $2^{\binom{n}{2}}$개다. $n = 50$이면 $10^{368}$을 넘어 볼 수 있는 우주의 원자 수보다 훨씬 많다. 다룰 만한 생성 방법은 이 공간에 어떤 구조를 가정해야 한다.

**순열 대칭.** 노드가 $n$개인 그래프 $\mathcal{G}$는 노드 레이블을 바꾸면 같은 뜻의 표현이 많게는 $n!$개다. 생성 모델은 다음 가운데 하나를 해야 한다:

1. 명시적으로 순열 불변이다: 모든 $\pi \in S_n$에 대해 $p_\theta(\mathcal{G}) = p_\theta(\pi(\mathcal{G}))$
2. 대칭을 깨는 표준 순서를 쓴다
3. 순열에 대해 주변화한다(흔히 다룰 수 없다)

**차원이 제각각.** 격자가 고정된 이미지와 달리 데이터셋의 그래프는 노드와 엣지의 수가 서로 다를 수 있다. 모델은 크기에 조건을 걸거나 자연스러운 종료 기준이 있는 구조를 써서 이 가변성을 다루어야 한다.

---

## 2. 엄밀한 문제 서술

그래프마다 $\mathcal{G}_i = (\mathcal{V}_i, \mathcal{E}_i, \mathbf{X}_i, \mathbf{E}_i)$이 노드 $\mathcal{V}_i$, 엣지 $\mathcal{E}_i$, 노드 피처 $\mathbf{X}_i \in \mathbb{R}^{|\mathcal{V}_i| \times d_n}$, 엣지 피처 $\mathbf{E}_i \in \mathbb{R}^{|\mathcal{E}_i| \times d_e}$으로 이루어진 데이터셋 $\mathcal{D} = \{\mathcal{G}_i\}_{i=1}^N$이 주어질 때 목표는 다음과 같다:

$$
\theta^* = \arg\max_\theta \sum_{i=1}^N \log p_\theta(\mathcal{G}_i)
$$

우도 $p_\theta(\mathcal{G})$는 노드 순열에 불변이어야 한다:

$$
p_\theta(\mathcal{G}) = p_\theta(\mathbf{P}\mathcal{G}\mathbf{P}^\top) \quad \forall \mathbf{P} \in \Pi_n
$$

여기서 $\mathbf{P}\mathcal{G}\mathbf{P}^\top$는 인접 행렬과 노드 피처의 자리를 한꺼번에 바꾸는 것을 뜻한다.

---

## 3. 두 가지 틀

### 자기회귀 분해

고른 순서 $\sigma$에 대해 사슬 규칙으로 결합 분포를 인수 분해한다:

$$
p_\theta(\mathcal{G}) = \prod_{t=1}^{T} p_\theta(a_t \mid a_{1:t-1})
$$

여기서 $a_t$는 $t$번째 생성 액션(노드, 엣지, 블록 더하기)을 나타낸다. 종료 액션으로 크기가 제각각인 그래프를 자연스럽게 다루지만 순서에 의존하게 된다.

### 원샷 생성

노드와 엣지를 모두 한꺼번에 만든다:

$$
p_\theta(\mathcal{G}) = p_\theta(\mathbf{A}, \mathbf{X})
$$

이는 완전히 병렬이지만 고정된 최대 크기 $n_{\max}$를 정해 두어야 하고 순열 불변성을 보장해야 한다. 확산 기반 방법이 반복 정제로 이 틀을 넓힌다.

---

## 4. 생성 파이프라인

```
Training Data ──→ Graph Encoder ──→ Latent Space ──→ Graph Decoder ──→ Generated Graph
                        │                                    │
                   마디 차례              올바름 지키기
                   피처 뽑기         뒷손질
                   크기 고르게 맞추기         매임 채우기
```

일반적인 그래프 생성 파이프라인은 다음을 담는다:

1. **전처리**: 노드 순서를 표준으로 맞추고(보기로 너비 우선, 깊이 우선) 패딩으로 크기를 맞춘다
2. **인코딩**: 그래프 뉴럴 네트워크로 입력 그래프를 잠재 표현으로 옮긴다
3. **잠재 표현 학습**: 잠재 분포를 배운다(변분 오토인코더, 정규화 흐름, 확산)
4. **디코딩**: 잠재 표본을 그래프 구조로 옮긴다
5. **후처리**: 유효성 제약을 지키고 패딩을 없앤다

---

## 5. 계량 금융과의 연결

금융 그래프 생성은 위험 관리의 핵심 기반이다. 은행 간 대출 네트워크를 보자. 은행마다 속성(자산, 부채, 자본 비율)을 가진 노드이고 방향 엣지마다 금액과 만기를 가진 대출 노출을 나타낸다. 참 네트워크는 비밀이고 일부만 볼 수 있다. 이 네트워크를 그럴듯하게 채우는 일은 다음에 꼭 필요하다:

- **전염 모델링**: 여러 네트워크 위상에서 채무 불이행이 연쇄적으로 번지는 과정을 시뮬레이션하기
- **꾸러미 스트레스 테스트**: 합성 시장 국면에서 꾸러미 성과 평가
- **거래 상대 위험**: 생성한 네트워크의 경로를 따라 간접 연결에 대한 노출 추정하기

이 장의 방법들이 이 모든 쓰임새의 기술 기초를 준다.

---

## 6. 구현: 그래프 생성 기반 클래스

```python
"""
그래프 만들기 바탕 클래스와 연장.
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
    """그래프 생성기의 추상 바탕 클래스."""

    def __init__(self, max_nodes: int, node_feature_dim: int = 0):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim

    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        """로스를 돌려주는 익히기 앞으로 가기."""
        ...

    @abstractmethod
    @torch.no_grad()
    def generate(self, num_graphs: int = 1, **kwargs) -> list[GeneratedGraph]:
        """새 그래프를 만든다."""
        ...

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def adjacency_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    """이웃 행렬을 PyG의 edge_index 형태로 바꾼다."""
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

**출력:**

```
Nodes: 10, Edges: 19
Density: 0.422
Edge index shape: torch.Size([2, 38])
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
순열 불변성 문제 때문에 그래프 생성이 이미지 생성보다 왜 근본에서 더 어려운지 밝혀라. 레이블이 붙은 노드 $n$개의 그래프에는 같은 뜻의 인접 행렬 표현이 몇 개 있는가?

</div>

??? success "풀이"
    노드 $n$개의 그래프는 서로 다른 인접 행렬 최대 $n!$개로 나타낼 수 있다(노드 레이블의 순열마다 하나, 자기동형 사상이 있으면 그보다 적다). $n = 10$이면 같은 뜻의 표현이 최대 $10! = 3{,}628{,}800$개다. 생성 모델은 다음 가운데 하나를 해야 한다. (1) 이 겹침을 줄이려 표준 순서를 배운다, (2) 순열 불변인 로스 함수를 쓴다(보기로 그래프 매칭), (3) 본래 불변인 표현 위에서 작동한다(보기로 스펙트럼 피처). 이미지 생성은 픽셀에 고정된 위치 순서가 있어 이 문제를 겪지 않는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
확장성, 품질, 유효성 제약을 지키는 능력의 면에서 그래프 생성의 자기회귀 길과 원샷 생성 길을 비교하여라.

</div>

??? success "풀이"
    자기회귀 방법은 노드와 엣지를 차례로 만들어 스텝마다 국소 제약(보기로 원자가)을 자연스럽게 지키지만, 엣지마다 결정을 내려야 해서 스텝 수가 $O(n^2)$으로 커진다. 원샷 생성 방법은 인접 행렬 전체를 한꺼번에 만들어 병렬 계산으로 확장하기 쉽지만 이산적인 구조와 전역 제약에 애를 먹는다. 자기회귀 방법은 작은 크기($n < 100$)에서 대개 더 좋은 그래프를 내지만 큰 그래프에서는 느려진다. 순열 불변인 로스를 쓰는 원샷 생성 방법은 더 빠르지만 매칭 문제를 조심스레 다루어야 한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
자기회귀 인수 분해 $p(G) = \prod_{i=1}^n p(\text{node}_i | \text{nodes}_{<i}) \prod_{j<i} p(e_{ij} | \text{node}_i, \text{nodes}_{<i})$ 아래에서 그래프의 우도를 이끌어 내어라.

</div>

??? success "풀이"
    고정된 노드 순서 $\pi$ 아래에서 결합 확률은 $p(G | \pi) = \prod_{i=1}^n p(v_{\pi(i)} | G_{\pi(<i)}) \prod_{j < i} p(e_{\pi(i),\pi(j)} | v_{\pi(i)}, G_{\pi(<i)})$으로 인수 분해된다. 여기서 $G_{\pi(<i)}$은 앞서 만든 노드의 부분 그래프다. 주변 우도 $p(G) = \frac{1}{n!}\sum_\pi p(G | \pi)$은 모든 순서를 더하지만 다룰 수 없다. 실제로는 표준 순서(보기로 너비 우선)를 써서 우도를 그 순서에 의존하는 것으로 만든다. 순서의 품질이 생성 품질에 영향을 준다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
분포 지표를 넘어 화학적 유효성과 약물 유사성을 재는, 생성한 분자 그래프의 평가 규약을 내놓아라.

</div>

??? success "풀이"
    분포 지표(차수 분포, 군집 계수)를 넘어 다음을 평가한다. (1) 화학적 유효성 -- 원자가 제약을 만족하는 분자의 몫(RDKit으로 확인), (2) 고유성 -- 서로 다른 유효한 분자의 몫, (3) 참신성 -- 학습 집합에 없는 몫, (4) 약물 유사성 -- QED 점수, 리핀스키의 다섯 규칙 준수, (5) 합성 용이성 -- 합성이 얼마나 쉬운지 나타내는 SA 점수, (6) 성질 최적화 -- 바란 타깃 성질과 얻은 성질의 상관. 여러 번 만들어 얻은 신뢰 구간과 함께 모든 지표를 알린다. $\square$

## 정리하며

이 절에서는 생성 문제, 엄밀한 문제 서술, 두 가지 틀, 생성 파이프라인을 살펴보았다.
