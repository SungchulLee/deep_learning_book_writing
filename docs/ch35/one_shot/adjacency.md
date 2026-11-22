# 한 번에 인접 행렬 생성

한 번에 만드는 그래프 생성은 인접 행렬 $\mathbf{A} \in \{0,1\}^{n \times n}$과 노드 피처 $\mathbf{X} \in \mathbb{R}^{n \times d}$ 전체를 한 번의 순전파에 내놓아 자기회귀 방법의 순차 병목을 피한다. 핵심 어려움은 원샷 생성이 순열 불변성, 이산적인 구조, 제각각인 그래프 크기를 한꺼번에 다루어야 한다는 점이다. 자기회귀 방법은 순서 분해로 이 문제를 회피한다.

---

## 1. 정식화

최대 그래프 크기 $n_{\max}$가 주어질 때 원샷 생성 방법은 잠재 벡터 $\mathbf{z} \in \mathbb{R}^{d_z}$에서 그래프로의 매핑을 배운다:

$$
(\hat{\mathbf{A}}, \hat{\mathbf{X}}) = f_\theta(\mathbf{z}), \quad \mathbf{z} \sim p(\mathbf{z})
$$

여기서 $\hat{\mathbf{A}} \in [0,1]^{n_{\max} \times n_{\max}}$는 인접 행렬을 연속값으로 완화한 것이고 $\hat{\mathbf{X}} \in \mathbb{R}^{n_{\max} \times d}$는 예측한 노드 피처다. 이산 그래프는 임곗값을 걸거나 샘플링해 얻는다: $A_{ij} \sim \text{Bernoulli}(\hat{A}_{ij})$.

---

## 2. 순열 불변성의 어려움

원샷 생성의 근본 어려움은 그래프 우도가 노드 순열과 무관하게 정의된다는 것이다.

$$
p(\mathcal{G}) = \frac{1}{n!} \sum_{\pi \in S_n} p(\mathbf{P}_\pi \mathbf{A} \mathbf{P}_\pi^\top, \mathbf{P}_\pi \mathbf{X})
$$

순열 $n!$개 모두에 대한 이 주변화는 아주 작은 그래프가 아니면 계산할 수 없다. 이를 다루는 전략은 셋이다.

**표준 순서.** 정해진 순서(보기로 차수순)를 고정해 그래프마다 하나뿐인 행렬 표현을 갖게 한다. 이는 문제를 일반적인 밀도 추정으로 바꾸지만, 고른 순서에서 오는 인공적인 편향이 생길 수 있다.

**순열 등변인 구조.** $\mathbf{z}$(또는 그 구조화된 형태)의 자리를 바꾸면 출력도 그에 맞게 자리가 바뀌도록 디코더를 설계한다. 이로써 모델이 같은 그래프의 모든 표현에 같은 확률을 준다.

**매칭 기반 로스.** 트레인하는 동안 로스를 계산하기 전에, 생성한 그래프를 타깃에 가장 잘 맞추는 순열 $\pi^*$를 찾는다.

$$
\pi^* = \arg\min_{\pi \in S_n} \|\hat{\mathbf{A}} - \mathbf{P}_\pi \mathbf{A}^* \mathbf{P}_\pi^\top\|_F^2
$$

이는 (일반적으로 NP-난해인) 그래프 매칭 문제와 같지만, 노드 피처에 헝가리 알고리즘을 쓰는 실용적 근사가 쓸 만한 해를 준다.

---

## 3. 디코더 구조

### 다층 뉴럴 네트워크 디코더

가장 단순한 방법은 잠재 벡터를 인접 행렬의 상삼각 부분을 펼친 벡터로 매핑한다.

$$
\hat{\mathbf{a}} = \sigma\left(\text{MLP}(\mathbf{z})\right) \in [0,1]^{\binom{n_{\max}}{2}}
$$

이는 마지막 레이어에 파라미터 $O(n_{\max}^2 \cdot d_z)$개가 들고, 그래프 구조에 대한 어떤 사전 지식도 활용하지 못한다.

### 인수로 나눈 디코더

파라미터 수를 줄이려 엣지 예측을 노드 임베딩으로 인수 분해한다.

$$
\mathbf{Z}_{\text{nodes}} = \text{MLP}_{\text{node}}(\mathbf{z}) \in \mathbb{R}^{n_{\max} \times d_h}
$$

$$
\hat{A}_{ij} = \sigma\left(\mathbf{z}_i^\top \mathbf{W} \mathbf{z}_j + b\right)
$$

여기서 $\mathbf{z}_i$는 $\mathbf{Z}_{\text{nodes}}$의 $i$번째 행이다. 이 쌍선형 형태는 파라미터가 $O(n_{\max} \cdot d_h + d_h^2)$개만 들며, $\mathbf{W}$가 대칭이면 자연스럽게 대칭 인접 행렬을 낸다.

### 그래프 조건부 디코더

잠재 노드 표현에 그래프 뉴럴 네트워크를 적용해 엣지 예측을 반복해서 다듬는다.

$$
\mathbf{Z}^{(\ell+1)} = \text{GNN}^{(\ell)}(\mathbf{Z}^{(\ell)}, \hat{\mathbf{A}}^{(\ell)})
$$

여기서 $\hat{\mathbf{A}}^{(\ell)}$는 반복마다 $\mathbf{Z}^{(\ell)}$에서 계산한다. 이 반복 정제 덕에 엣지 예측이 지금까지 형성된 그래프 구조에 조건화될 수 있다.

---

## 4. 크기 다루기

원샷 생성 방법은 크기가 다른 그래프를 다루어야 한다. 흔한 방법은 다음과 같다.

**노드 마스크로 패딩하기.** 크기가 $n_{\max}$로 고정된 그래프를 만들고 노드가 있는지 나타내는 마스크 $\mathbf{m} \in [0,1]^{n_{\max}}$를 예측한다. 실제 그래프는 $m_i > 0.5$인 노드를 쓴다. 가려진 노드가 낀 엣지는 0으로 만든다:

$$
\hat{A}_{ij}^{\text{final}} = \hat{A}_{ij} \cdot m_i \cdot m_j
$$

**크기에 조건 걸기.** 경험적 크기 분포에서 그래프 크기 $n \sim p(n)$을 뽑고 $n$을 조건으로 생성한다. 디코더에 $n$을 추가 입력으로 주어 구현할 수 있다.

---

## 5. 트레인 로스의 조각

온전한 원샷 생성 로스는 보통 다음을 아우른다:

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}} + \lambda_{\text{match}} \mathcal{L}_{\text{match}}
$$

여기서 $\mathcal{L}_{\text{recon}}$은 엣지 복원 로스(이진 교차 엔트로피), $\mathcal{L}_{\text{KL}}$은 변분 오토인코더 기반 방법의 쿨백-라이블러 발산, $\mathcal{L}_{\text{match}}$는 순열 정렬을 다룬다.

---

## 6. 금융 응용: 한 시점의 네트워크 스냅숏

원샷 생성은 금융 네트워크의 완전한 스냅숏을 만드는 데 자연스럽게 알맞다. 보기로 규제 보고일 시점의 은행 간 익스포저 전체 단면을 생성한다. 네트워크가 생기는 과정을 나타내는 자기회귀 방법과 달리 원샷 생성은 균형 구조를 곧바로 내며, 시간에 따른 형성 과정이 관심 밖일 때 알맞다.

---

## 7. 구현: 한 번에 만드는 인접 행렬 생성기

```python
"""
여러 디코더 구조로 하는 한 번에 이웃 행렬 만들기.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from scipy.optimize import linear_sum_assignment
import numpy as np

class MLPDecoder(nn.Module):
    """단순한 다층 퍼셉트론 푸는개: 숨은 값 -> 펼친 위 삼각 이웃 행렬."""

    def __init__(self, latent_dim: int, max_nodes: int, hidden_dim: int = 256):
        super().__init__()
        self.max_nodes = max_nodes
        n_edges = max_nodes * (max_nodes - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_edges),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        인수:
            z: (B, latent_dim)
        반환값:
            adj: (B, max_nodes, max_nodes) 헤아린 이웃 확률
        """
        B = z.size(0)
        n = self.max_nodes
        edge_logits = self.mlp(z)  # (B, n_edges)

        # 맞섬 이웃 행렬을 되짓는다
        adj = torch.zeros(B, n, n, device=z.device)
        idx = torch.triu_indices(n, n, offset=1)
        adj[:, idx[0], idx[1]] = torch.sigmoid(edge_logits)
        adj = adj + adj.transpose(1, 2)

        return adj

class FactoredDecoder(nn.Module):
    """
    인수로 나눈 푸는개: 숨은 값 -> 마디 임베딩 -> 쌍선형 변 예측.
    여러 레이어 뉴럴 네트워크 디코더보다 잡을 아낀다.
    """

    def __init__(
        self,
        latent_dim: int,
        max_nodes: int,
        node_embed_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_embed_dim = node_embed_dim

        # 숨은 값에서 마디 임베딩으로
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * node_embed_dim),
        )

        # 쌍선형 변 헤아리개
        self.edge_weight = nn.Parameter(
            torch.randn(node_embed_dim, node_embed_dim) * 0.01
        )
        self.edge_bias = nn.Parameter(torch.zeros(1))

        # 마디가 있는지 헤아리개
        self.node_mask_predictor = nn.Sequential(
            nn.Linear(node_embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        반환값:
            adj: (B, n, n) 이웃 확률
            node_mask: (B, n) 마디가 있을 확률
        """
        B = z.size(0)
        n = self.max_nodes

        # 마디 임베딩을 푼다
        node_embeds = self.node_decoder(z).view(B, n, self.node_embed_dim)

        # 쌍선형 변 헤아리기: A_ij = σ(z_i^T W z_j + b)
        # 방향 없는 그래프를 위해 W을 맞섬으로 만든다
        W_sym = (self.edge_weight + self.edge_weight.t()) / 2
        edge_logits = torch.einsum("bid,de,bje->bij", node_embeds, W_sym, node_embeds)
        edge_logits = edge_logits + self.edge_bias
        adj = torch.sigmoid(edge_logits)

        # 대각을 0으로
        mask = 1 - torch.eye(n, device=z.device).unsqueeze(0)
        adj = adj * mask

        # 마디가 있는지 나타내는 마스크
        node_mask = torch.sigmoid(
            self.node_mask_predictor(node_embeds).squeeze(-1)
        )

        # 이웃 행렬에 마디 마스크를 쓴다
        adj = adj * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

        return adj, node_mask

def hungarian_matching(
    adj_pred: torch.Tensor,
    adj_target: torch.Tensor,
) -> torch.Tensor:
    """
    헝가리 알고리즘으로 가장 좋은 마디 순열을 찾는다.
    
    인수:
        adj_pred: (n, n) 헤아린 이웃 행렬
        adj_target: (n, n) 타깃 이웃 행렬
        
    반환값:
        perm: (n,) 순열 번호
    """
    n = adj_pred.size(0)
    # 비용 행렬: 모든 (i, j) 짝에 대해 ||pred_i - target_j||^2
    # 이웃 행렬의 줄을 마디 피처로 쓴다
    cost = torch.cdist(adj_pred.float(), adj_target.float(), p=2)
    cost_np = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    perm = torch.tensor(col_ind, dtype=torch.long, device=adj_pred.device)
    return perm

def permutation_aligned_loss(
    adj_pred: torch.Tensor,
    adj_target: torch.Tensor,
    use_matching: bool = True,
) -> torch.Tensor:
    """
    순열 맞추기를 골라 쓰며 되짓기 로스를 셈한다.
    
    인수:
        adj_pred: (B, n, n) 헤아린 이웃 행렬
        adj_target: (B, n, n) 타깃 이웃 행렬
        use_matching: 헝가리 짝짓기를 쓸지
    """
    B = adj_pred.size(0)
    total_loss = 0.0

    for b in range(B):
        pred = adj_pred[b]
        target = adj_target[b]

        if use_matching:
            perm = hungarian_matching(pred, target)
            target = target[perm][:, perm]

        # 위쪽 삼각에 대한 두 값 교차 엔트로피
        n = pred.size(0)
        idx = torch.triu_indices(n, n, offset=1)
        pred_edges = pred[idx[0], idx[1]]
        target_edges = target[idx[0], idx[1]]

        loss = F.binary_cross_entropy(
            pred_edges.clamp(1e-6, 1 - 1e-6),
            target_edges,
            reduction="mean",
        )
        total_loss += loss

    return total_loss / B

if __name__ == "__main__":
    torch.manual_seed(42)
    max_n = 12
    latent_dim = 32

    # 익히기 자료를 만든다
    print("=== One-Shot Adjacency Generation Demo ===\n")

    train_adjs = []
    for _ in range(100):
        n = torch.randint(6, max_n + 1, (1,)).item()
        adj = torch.zeros(max_n, max_n)
        # 아무 그래프
        for i in range(n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.25:
                    adj[i, j] = adj[j, i] = 1
        train_adjs.append(adj)

    train_adjs = torch.stack(train_adjs)
    print(f"Training data: {train_adjs.shape}")

    # 여러 레이어 뉴럴 네트워크 디코더 시험
    print("\n--- MLP Decoder ---")
    mlp_dec = MLPDecoder(latent_dim, max_n, hidden_dim=128)
    z = torch.randn(5, latent_dim)
    adj_pred = mlp_dec(z)
    print(f"Output shape: {adj_pred.shape}")
    print(f"Symmetry check: {torch.allclose(adj_pred, adj_pred.transpose(1,2))}")
    print(f"Value range: [{adj_pred.min():.3f}, {adj_pred.max():.3f}]")

    # 인수로 나눈 디코더 시험
    print("\n--- Factored Decoder ---")
    fac_dec = FactoredDecoder(latent_dim, max_n, node_embed_dim=32)
    adj_pred, node_mask = fac_dec(z)
    print(f"Adj shape: {adj_pred.shape}, Mask shape: {node_mask.shape}")
    print(f"Avg predicted nodes: {(node_mask > 0.5).float().sum(1).mean():.1f}")

    # 헝가리 짝짓기 시험
    print("\n--- Hungarian Matching ---")
    adj_a = torch.zeros(6, 6)
    adj_a[0, 1] = adj_a[1, 0] = 1
    adj_a[1, 2] = adj_a[2, 1] = 1
    adj_a[2, 3] = adj_a[3, 2] = 1

    # 자리를 바꾼 판
    perm = [3, 2, 1, 0, 4, 5]
    adj_b = adj_a[perm][:, perm]

    recovered_perm = hungarian_matching(adj_a, adj_b)
    adj_recovered = adj_b[recovered_perm][:, recovered_perm]
    print(f"Match quality: {(adj_a == adj_recovered).float().mean():.1%}")

    # 자리 맞춘 로스 시험
    print("\n--- Permutation-Aligned Loss ---")
    pred_batch = torch.rand(4, max_n, max_n) * 0.3
    pred_batch = (pred_batch + pred_batch.transpose(1, 2)) / 2
    pred_batch.diagonal(dim1=1, dim2=2).zero_()

    loss_no_match = permutation_aligned_loss(pred_batch, train_adjs[:4], use_matching=False)
    loss_match = permutation_aligned_loss(pred_batch, train_adjs[:4], use_matching=True)
    print(f"Loss without matching: {loss_no_match:.4f}")
    print(f"Loss with matching: {loss_match:.4f}")
    print(f"Matching reduces loss: {loss_match < loss_no_match}")
```

**출력:**

```
=== One-Shot Adjacency Generation Demo ===

Training data: torch.Size([100, 12, 12])

--- MLP Decoder ---
Output shape: torch.Size([5, 12, 12])
Symmetry check: True
Value range: [0.000, 0.570]

--- Factored Decoder ---
Adj shape: torch.Size([5, 12, 12]), Mask shape: torch.Size([5, 12])
Avg predicted nodes: 4.0

--- Hungarian Matching ---
Match quality: 100.0%

--- Permutation-Aligned Loss ---
Loss without matching: 0.5271
Loss with matching: 0.4930
Matching reduces loss: True
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
순열 불변성 문제 때문에 그래프 생성이 이미지 생성보다 왜 근본적으로 더 어려운지 설명하여라. 레이블이 붙은 노드 $n$개의 그래프에는 같은 뜻의 인접 행렬 표현이 몇 개 있는가?

</div>

??? success "풀이"
    노드 $n$개의 그래프는 서로 다른 인접 행렬 $n!$개로 나타낼 수 있다(노드 레이블의 순열마다 하나). $n = 10$이면 같은 뜻의 표현이 $10! = 3{,}628{,}800$개다. 생성 모델은 다음 가운데 하나를 해야 한다. (1) 이 중복을 줄이려 표준 순서를 정한다, (2) 순열 불변인 로스 함수를 쓴다(보기로 그래프 매칭), (3) 처음부터 불변인 표현 위에서 다룬다(보기로 스펙트럼 피처). 이미지 생성은 픽셀에 고정된 위치 순서가 있어 이 문제를 겪지 않는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
확장성, 품질, 유효성 제약 준수의 측면에서 그래프 생성의 자기회귀 길과 원샷 생성 길을 비교하여라.

</div>

??? success "풀이"
    자기회귀 방법은 노드와 엣지를 차례로 만들어 스텝마다 국소 제약(보기로 원자가)을 자연스럽게 지키지만, 엣지 결정의 수가 $O(n^2)$으로 커진다. 원샷 생성 방법은 인접 행렬 전체를 한꺼번에 만들어 병렬 계산에는 더 잘 커지지만 이산적인 구조와 전역 제약에 애를 먹는다. 자기회귀 방법은 작은 크기($n < 100$)에서 대개 더 좋은 그래프를 내지만 큰 그래프에서는 느려진다. 순열 불변인 로스를 쓰는 원샷 생성 방법은 더 빠르지만 매칭 문제를 조심스레 다루어야 한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
자기회귀 인수 분해 $p(G) = \prod_{i=1}^n p(\text{node}_i | \text{nodes}_{<i}) \prod_{j<i} p(e_{ij} | \text{node}_i, \text{nodes}_{<i})$ 아래에서 그래프의 우도를 이끌어 내어라.

</div>

??? success "풀이"
    고정된 노드 순서 $\pi$ 아래에서 결합 확률은 $p(G | \pi) = \prod_{i=1}^n p(v_{\pi(i)} | G_{\pi(<i)}) \prod_{j < i} p(e_{\pi(i),\pi(j)} | v_{\pi(i)}, G_{\pi(<i)})$로 인수 분해된다. 여기서 $G_{\pi(<i)}$는 앞서 만든 노드의 부분 그래프다. 주변 우도 $p(G) = \frac{1}{n!}\sum_\pi p(G | \pi)$는 모든 순서에 대한 합이라 계산할 수 없다. 실제로는 표준 순서(보기로 너비 우선)를 써서 우도를 그 순서에 의존하는 것으로 만든다. 순서의 품질이 생성 품질에 영향을 준다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
분포 지표를 넘어 화학적 유효성과 약물 유사성을 재는, 생성한 분자 그래프의 평가 규약을 내놓아라.

</div>

??? success "풀이"
    분포 지표(차수 분포, 군집 계수)를 넘어 다음을 평가한다. (1) 화학적 유효성 -- 원자가 제약을 만족하는 분자의 몫(RDKit으로 확인), (2) 고유성 -- 서로 다른 유효한 분자의 몫, (3) 참신성 -- 트레인 집합에 없는 몫, (4) 약물 유사성 -- QED 점수, 리핀스키의 다섯 규칙 준수, (5) 합성 용이성 -- 합성이 얼마나 쉬운지 나타내는 SA 점수, (6) 성질 최적화 -- 목표 성질과 얻은 성질의 상관. 여러 번 만들어 얻은 신뢰 구간과 함께 모든 지표를 알린다. $\square$

## 정리하며

이 절에서는 정식화, 순열 불변성의 어려움, 디코더 구조, 크기 다루기를 살펴보았다.
