# 만들기를 위한 그래프 나타내기
## 개요

그래프를 어떻게 나타내느냐가 어떤 만들기 셈속을 쓸 수 있는지를 근본에서 정한다. 나타냄은 나타냄 힘(그래프의 모든 성질을 담음), 간결함(효율 좋은 배움), 신경망 얼개와의 어울림을 저울질해야 한다. 이 마디는 그래프 만들기에 쓰이는 주요 나타냄과 그것이 모델 설계에 주는 뜻을 훑는다.

## 이웃 행렬 나타내기

마디가 $n$개인 그래프 $\mathcal{G}$을 가장 곧바로 나타내는 것은 이웃 행렬 $\mathbf{A} \in \{0,1\}^{n \times n}$이며 변 $(i,j)$이 있으면 $A_{ij} = 1$이다. 속성이 있는 그래프에서는 마디 특징 $\mathbf{X} \in \mathbb{R}^{n \times d_n}$과 변 특징 $\mathbf{E} \in \mathbb{R}^{n \times n \times d_e}$을 더한다.

**성질:**

- 공간 복잡도: 변에 $O(n^2)$, 마디 특징에 $O(n \cdot d_n)$
- 방향 없는 그래프에서는 자연스럽게 맞섬이다: $\mathbf{A} = \mathbf{A}^\top$
- 실제로 성기다: 대부분의 실제 그래프는 $|\mathcal{E}| \ll n^2$이다

**자리바꿈의 아리송함.** 같은 그래프가 마디 이름표를 바꾸면 이웃 행렬 나타냄이 $n!$개다. 자리바꿈 $\pi$에 대해 자리를 바꾼 이웃 행렬은 다음과 같다:

$$
\mathbf{A}' = \mathbf{P}_\pi \mathbf{A} \mathbf{P}_\pi^\top
$$

여기서 $\mathbf{P}_\pi$은 자리바꿈 행렬이다. 이 아리송함이 한 번에 만들기 방법의 한가운데 놓인 어려움이다.

## 위쪽 삼각 나타내기

스스로 이음이 없는 방향 없는 그래프에서는 $\mathbf{A}$의 위쪽 삼각 칸만 있으면 된다. 이 삼각을 납작하게 펴면 두 값 벡터 $\mathbf{a} \in \{0,1\}^{\binom{n}{2}}$이 나온다:

$$
\mathbf{a} = \text{triu\_flatten}(\mathbf{A}) = (A_{12}, A_{13}, \ldots, A_{1n}, A_{23}, \ldots, A_{(n-1)n})
$$

이는 내놓기 차원을 반으로 줄이고 온 행렬의 겹침을 없앤다. 이 납작한 벡터에 대한 자기 되돌이 모델은 변을 붙박인 차례로 만든다.

## 변 목록 나타내기

변 목록 $\mathcal{E} = \{(i_1, j_1), \ldots, (i_m, j_m)\}$은 있는 변만 담는다. 성긴 그래프($m \ll n^2$)에서 기억을 아끼며 PyTorch Geometric의 `edge_index` 꼴과 자연스럽게 맞물린다:

$$
\texttt{edge\_index} = \begin{bmatrix} i_1 & i_2 & \cdots & i_m \\ j_1 & j_2 & \cdots & j_m \end{bmatrix} \in \mathbb{N}^{2 \times m}
$$

만들기에서 변 목록 나타냄은 변의 개수와 양 끝 짝을 모두 헤아려야 하므로 자기 되돌이 길이 자연스럽다.

## 표준 차례

자리바꿈 맞섬을 깨려고 자기 되돌이 방법은 표준 마디 차례를 둔다. 흔한 고름:

**너비 우선 차례.** 아무 마디에서 시작해 이웃을 켜마다 들른다. 최근에 더한 마디가 가까운 앞선 마디에 이어지는 차례를 내어 이웃 행렬의 실제 띠너비를 줄인다.

**깊이 우선 차례.** 아무 마디에서 시작해 되돌아가기 전에 되도록 깊이 들어간다. 가까움이 센 차례를 낸다. 변 얼개가 $\mathbf{A}$의 대각 언저리에 모인다.

**차수 차례.** 마디를 차수 내림차순으로 줄 세운다. 바퀴통이 먼저 나와 차수가 높은 마디가 등뼈를 세우는 짜임새 있는 이웃 관계가 생긴다.

너비 우선 차례 아래 그래프의 너비 우선 띠너비 $B$이 변 결정에서 되돌아볼 최대 거리를 정한다:

$$
B = \max_{(i,j) \in \mathcal{E}} |i - j|
$$

GraphRNN은 이를 살려 변 헤아림을 크기 $B$의 창으로 잘라 복잡도를 $O(n^2)$에서 $O(n \cdot B)$으로 줄인다.

## 차례 나타내기

어떤 방법은 그래프를 토막의 차례로 나타낸다:

**SMILES 문자열**은 분자 그래프를 글자 차례로 담는다(보기로 아스피린은 `CC(=O)Oc1ccccc1C(=O)O`). 만들기가 말 나타내기로 줄어들지만 문자열에서 그래프로의 옮김이 여럿 대 하나이고 문법의 올바름이 보장되지 않는다.

**이웃 차례.** 이웃 행렬을 줄줄이 납작하게 펴 길이 $n^2$(위쪽 삼각이면 $\binom{n}{2}$)의 두 값 차례를 얻는다. 자리마다 앞선 자리에 매인 베르누이 변수다.

## 스펙트럼 나타내기

$\mathbf{A}$에서 곧바로 돌기보다 스펙트럼 방법은 그래프 라플라스의 고유 분해로 그래프를 나타낸다:

$$
\mathbf{L} = \mathbf{D} - \mathbf{A} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top
$$

고윳값 $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$은 온 자리 얼개 성질(이어짐, 넓힘, 뭉침)을 담고 고유 벡터 $\mathbf{U}$은 스펙트럼 박아 넣기 자리에서 마디의 자리를 담는다. $(\boldsymbol{\Lambda}, \mathbf{U})$을 만들고 $\mathbf{A} = \mathbf{D} - \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^\top$으로 되지으면 자연스럽게 자리바꿈에 같이 바뀌는 나타냄이 나온다.

## 짜기: 나타냄 바꾸기

```python
"""
만들기 흐름을 위한 그래프 나타냄 바꾸기.
"""
import torch
import numpy as np
from collections import deque
from typing import Optional


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
        (n, n) symmetric adjacency matrix
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
        start: starting node (random if None)
        
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
    """자리바꿈에 따라 이웃 행렬을 다시 늘어놓는다."""
    perm_tensor = torch.tensor(perm)
    return adj[perm_tensor][:, perm_tensor]


def compute_bfs_bandwidth(adj: torch.Tensor) -> int:
    """
    Compute the BFS bandwidth: max |i - j| for edges (i, j) 
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
    
    The Laplacian L = U Λ U^T, and A = D - L.
    D을 모르므로 되지은 L에서 어림한다.
    """
    laplacian = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()
    # i != j일 때 A_ij = -L_ij
    adj = -laplacian
    adj.fill_diagonal_(0)
    # 문턱을 걸어 두 값으로
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
