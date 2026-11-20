# 생성을 위한 그래프 표현

그래프를 어떻게 나타내느냐가 어떤 생성 전략을 쓸 수 있는지를 근본에서 정한다. 표현은 표현력(그래프의 모든 성질을 담음), 간결함(효율적인 학습), 뉴럴 네트워크 구조와의 호환성을 저울질해야 한다. 이 쪽은 그래프 생성에 쓰이는 주요 표현과 그것이 모델 설계에 주는 뜻을 훑는다.

---

## 1. 인접 행렬 표현

노드가 $n$개인 그래프 $\mathcal{G}$를 가장 곧바로 나타내는 것은 인접 행렬 $\mathbf{A} \in \{0,1\}^{n \times n}$이며 엣지 $(i,j)$가 있으면 $A_{ij} = 1$이다. 속성이 있는 그래프에서는 노드 피처 $\mathbf{X} \in \mathbb{R}^{n \times d_n}$과 엣지 피처 $\mathbf{E} \in \mathbb{R}^{n \times n \times d_e}$를 더한다.

**성질:**

- 공간 복잡도: 엣지에 $O(n^2)$, 노드 피처에 $O(n \cdot d_n)$
- 방향 없는 그래프에서는 자연스럽게 대칭이다: $\mathbf{A} = \mathbf{A}^\top$
- 실제로는 희소하다: 대부분의 실제 그래프는 $|\mathcal{E}| \ll n^2$이다

**순열의 모호성.** 같은 그래프라도 노드 레이블을 바꾸면 인접 행렬 표현이 최대 $n!$개다. 순열 $\pi$에 대해 자리를 바꾼 인접 행렬은 다음과 같다:

$$
\mathbf{A}' = \mathbf{P}_\pi \mathbf{A} \mathbf{P}_\pi^\top
$$

여기서 $\mathbf{P}_\pi$는 순열 행렬이다. 이 모호성이 원샷 생성 방법의 한가운데 놓인 어려움이다.

---

## 2. 위쪽 삼각 표현

자기 루프가 없는 방향 없는 그래프에서는 $\mathbf{A}$의 위쪽 삼각 칸만 있으면 된다. 이 삼각을 한 줄로 펼치면 이진 벡터 $\mathbf{a} \in \{0,1\}^{\binom{n}{2}}$이 나온다:

$$
\mathbf{a} = \text{triu\_flatten}(\mathbf{A}) = (A_{12}, A_{13}, \ldots, A_{1n}, A_{23}, \ldots, A_{(n-1)n})
$$

이는 출력 차원을 반으로 줄이고 전체 행렬의 중복을 없앤다. 이 펼친 벡터에 대한 자기회귀 모델은 엣지를 고정된 순서로 만든다.

---

## 3. 엣지 목록 표현

엣지 목록 $\mathcal{E} = \{(i_1, j_1), \ldots, (i_m, j_m)\}$은 있는 엣지만 담는다. 희소 그래프($m \ll n^2$)에서 메모리를 아끼며 PyTorch Geometric의 `edge_index` 형태와 자연스럽게 맞물린다:

$$
\texttt{edge\_index} = \begin{bmatrix} i_1 & i_2 & \cdots & i_m \\ j_1 & j_2 & \cdots & j_m \end{bmatrix} \in \mathbb{N}^{2 \times m}
$$

생성에서 엣지 목록 표현은 엣지의 개수와 양 끝 짝을 모두 예측해야 하므로 자기회귀 방식이 자연스럽다.

---

## 4. 표준 순서

순열 대칭을 깨려고 자기회귀 방법은 표준 노드 순서를 둔다. 흔한 선택:

**너비 우선 순서.** 임의의 노드에서 시작해 시작점에서 가까운 거리 단계부터 차례로 이웃을 들른다. 새로 더한 노드가 순서상 가까운 앞선 노드에만 이어지는 순서를 만들어 인접 행렬의 실제 대역폭을 줄인다.

**깊이 우선 순서.** 임의의 노드에서 시작해 되돌아가기 전에 되도록 깊이 들어간다. 국소성이 강한 순서를 만들어, 엣지 구조가 $\mathbf{A}$의 대각선 근처에 모인다.

**차수 순서.** 노드를 차수 내림차순으로 줄 세운다. 허브 노드가 먼저 나와, 차수가 높은 노드가 뼈대를 이루는 구조적인 인접 관계가 생긴다.

너비 우선 순서 아래 그래프의 너비 우선 대역폭 $B$가 엣지 결정에서 되돌아볼 최대 거리를 정한다:

$$
B = \max_{(i,j) \in \mathcal{E}} |i - j|
$$

GraphRNN은 이를 살려 엣지 예측을 크기 $B$의 창으로 잘라 복잡도를 $O(n^2)$에서 $O(n \cdot B)$로 줄인다.

---

## 5. 시퀀스 표현

어떤 방법은 그래프를 토큰의 시퀀스로 나타낸다:

**SMILES 문자열**은 분자 그래프를 문자열로 담는다(보기로 아스피린은 `CC(=O)Oc1ccccc1C(=O)O`). 생성이 언어 모델링으로 줄어들지만 여러 문자열이 한 그래프에 대응하는 다대일 매핑이고 문법적 유효성이 보장되지 않는다.

**인접 시퀀스.** 인접 행렬을 행 순서대로 펼쳐 길이 $n^2$(위쪽 삼각이면 $\binom{n}{2}$)의 이진 시퀀스를 얻는다. 위치마다 앞선 위치에 조건부인 베르누이 변수다.

---

## 6. 스펙트럼 표현

$\mathbf{A}$를 직접 다루기보다 스펙트럼 방법은 그래프 라플라시안의 고유 분해로 그래프를 나타낸다:

$$
\mathbf{L} = \mathbf{D} - \mathbf{A} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top
$$

고윳값 $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$은 전역 구조 성질(연결성, 확장 성질, 군집)을 담고 고유 벡터 $\mathbf{U}$는 스펙트럼 임베딩 공간에서 노드의 위치를 담는다. $(\boldsymbol{\Lambda}, \mathbf{U})$를 만들고 $\mathbf{A} = \mathbf{D} - \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^\top$으로 복원하면 자연스럽게 순열 등변인 표현이 나온다.

---

## 7. 구현: 표현 바꾸기

```python
"""
만들기 흐름을 위한 그래프 나타냄 바꾸기.
"""
import torch
import numpy as np
from collections import deque
from typing import Optional

# 이 쪽이 싣는 수가 돌릴 때마다 달라지지 않게 씨앗을 박는다.
torch.manual_seed(0)
np.random.seed(0)


def adjacency_to_upper_triangular(adj: torch.Tensor) -> torch.Tensor:
    """
    이웃 행렬의 위쪽 삼각을 벡터로 납작하게 편다.
    
    인수:
        adj: (n, n) 이웃 행렬
        
    반환값:
        (n*(n-1)/2,) binary vector
    """
    n = adj.size(0)
    indices = torch.triu_indices(n, n, offset=1)
    return adj[indices[0], indices[1]]

def upper_triangular_to_adjacency(vec: torch.Tensor, n: int) -> torch.Tensor:
    """
    위쪽 삼각 벡터에서 이웃 행렬을 되짓는다.
    
    인수:
        vec: (n*(n-1)/2,) 벡터
        n: 마디 개수
        
    반환값:
        (n, n) 대칭 이웃 행렬
    """
    adj = torch.zeros(n, n, dtype=vec.dtype, device=vec.device)
    indices = torch.triu_indices(n, n, offset=1)
    adj[indices[0], indices[1]] = vec
    adj = adj + adj.t()
    return adj

def bfs_ordering(adj: torch.Tensor, start: Optional[int] = None) -> list[int]:
    """
    마디의 너비 우선 차례를 셈한다.
    
    인수:
        adj: (n, n) 이웃 행렬
        start: 비롯하는 마디(None이면 마구잡이)
        
    반환값:
        너비 우선 차례의 마디 번호 목록
    """
    n = adj.size(0)
    if start is None:
        # 차수가 가장 높은 마디에서 시작한다
        start = adj.sum(dim=1).argmax().item()

    visited = set()
    order = []
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        order.append(node)
        neighbors = torch.where(adj[node] > 0)[0].tolist()
        # 정해진 결과를 위해 이웃을 차수 내림차순으로 줄 세운다
        neighbors.sort(key=lambda x: -adj[x].sum().item())
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    # 끊긴 마디를 더한다
    for i in range(n):
        if i not in visited:
            order.append(i)

    return order

def permute_adjacency(adj: torch.Tensor, perm: list[int]) -> torch.Tensor:
    """순열에 따라 이웃 행렬을 다시 늘어놓는다."""
    perm_tensor = torch.tensor(perm)
    return adj[perm_tensor][:, perm_tensor]

def compute_bfs_bandwidth(adj: torch.Tensor) -> int:
    """
    너비 우선 띠너비를 셈한다: 변 (i, j)에 대한 max |i - j|
    너비 우선 차례 아래에서.
    """
    order = bfs_ordering(adj)
    adj_bfs = permute_adjacency(adj, order)
    n = adj_bfs.size(0)
    max_bw = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj_bfs[i, j] > 0:
                max_bw = max(max_bw, j - i)
    return max_bw

def laplacian_eigendecomposition(
    adj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    그래프 라플라스의 고유 분해를 셈한다.
    
    반환값:
        eigenvalues: (n,) 줄 세운 고윳값
        eigenvectors: (n, n) 그에 맞는 고유 벡터
    """
    degree = adj.sum(dim=1)
    laplacian = torch.diag(degree) - adj
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    return eigenvalues, eigenvectors

def reconstruct_from_spectrum(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    스펙트럼 분해에서 이웃 행렬을 되짓는다.
    
    라플라스 행렬은 L = U Λ U^T이고 A = D - L이다.
    D을 모르므로 되지은 L에서 어림한다.
    """
    laplacian = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()
    # i != j일 때 A_ij = -L_ij
    adj = -laplacian
    adj.fill_diagonal_(0)
    # 임계를 걸어 두 값으로
    adj = (adj > threshold).float()
    return adj

if __name__ == "__main__":
    # 보기 그래프를 만든다(에르되시-레니)
    n = 8
    p = 0.4
    adj = (torch.rand(n, n) < p).float()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.t()

    print("=== Adjacency Matrix ===")
    print(adj.int())

    # 위쪽 삼각
    vec = adjacency_to_upper_triangular(adj)
    print(f"\nUpper-tri vector ({vec.shape[0]} entries): {vec.int().tolist()}")
    adj_reconstructed = upper_triangular_to_adjacency(vec, n)
    assert torch.allclose(adj, adj_reconstructed), "Reconstruction failed"

    # 너비 우선 차례
    order = bfs_ordering(adj)
    print(f"\nBFS order: {order}")
    bw = compute_bfs_bandwidth(adj)
    print(f"BFS bandwidth: {bw}")

    # 스펙트럼
    eigenvalues, eigenvectors = laplacian_eigendecomposition(adj)
    print(f"\nLaplacian eigenvalues: {eigenvalues.numpy().round(3)}")
    adj_spectral = reconstruct_from_spectrum(eigenvalues, eigenvectors)
    match = (adj == adj_spectral).float().mean()
    print(f"Spectral reconstruction accuracy: {match:.1%}")
```

**출력:**

```
=== Adjacency Matrix ===
tensor([[0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.int32)

Upper-tri vector (28 entries): [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]

BFS order: [3, 2, 6, 0, 4, 5, 1, 7]
BFS bandwidth: 5

Laplacian eigenvalues: [0.    0.819 1.865 3.    3.405 4.755 5.46  6.696]
Spectral reconstruction accuracy: 100.0%
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

이 절에서는 인접 행렬 표현, 위쪽 삼각 표현, 엣지 목록 표현, 표준 순서, 시퀀스 표현과 스펙트럼 표현을 살펴보았다.
