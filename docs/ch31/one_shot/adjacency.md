# 한 번에 이웃 행렬 만들기
## 개요

한 번에 만드는 그래프 만들기는 이웃 행렬 $\mathbf{A} \in \{0,1\}^{n \times n}$과 마디 특징 $\mathbf{X} \in \mathbb{R}^{n \times d}$ 전체를 한 번의 앞으로 가기에 내놓아 자기 되돌이 방법의 차례 병목을 피한다. 핵심 어려움은 한 번에 만들기가 자리바꿈에 안 바뀜, 띄엄띄엄한 얼개, 제각각인 그래프 크기를 한꺼번에 다루어야 한다는 점이다. 자기 되돌이 방법은 차례 쪼개기로 이 문제를 비껴간다.

## 적기

최대 그래프 크기 $n_{\max}$이 주어질 때 한 번에 만들기 방법은 숨은 벡터 $\mathbf{z} \in \mathbb{R}^{d_z}$에서 그래프로의 옮김을 배운다:

$$
(\hat{\mathbf{A}}, \hat{\mathbf{X}}) = f_\theta(\mathbf{z}), \quad \mathbf{z} \sim p(\mathbf{z})
$$

여기서 $\hat{\mathbf{A}} \in [0,1]^{n_{\max} \times n_{\max}}$은 이웃 행렬을 이어진 값으로 느슨히 한 것이고 $\hat{\mathbf{X}} \in \mathbb{R}^{n_{\max} \times d}$은 헤아린 마디 특징이다. 만들어진 그래프는 문턱을 걸거나 뽑아 얻는다: $A_{ij} \sim \text{Bernoulli}(\hat{A}_{ij})$.

## 자리바꿈에 안 바뀜의 어려움

한 번에 만들기의 근본 어려움은 그래프 가능도가 자리바꿈을 빼고 뜻매김된다는 것이다:

$$
p(\mathcal{G}) = \frac{1}{n!} \sum_{\pi \in S_n} p(\mathbf{P}_\pi \mathbf{A} \mathbf{P}_\pi^\top, \mathbf{P}_\pi \mathbf{X})
$$

자리바꿈 $n!$개 모두에 대한 이 주적화는 아주 작은 그래프가 아니면 다룰 수 없다. 셈속 셋이 이를 다룬다:

**표준 차례.** 정해진 차례(보기로 차수순)를 붙박아 그래프마다 하나뿐인 행렬 나타냄을 갖게 한다. 이는 문제를 여느 밀도 어림으로 바꾸지만 고른 차례에서 오는 군더더기가 생길 수 있다.

**자리바꿈에 같이 바뀌는 얼개.** $\mathbf{z}$(또는 짜임 있는 판)의 자리를 바꾸면 내놓기도 그에 맞게 자리가 바뀌도록 풀개를 설계한다. 이로써 모델이 같은 그래프의 모든 나타냄에 같은 확률을 준다.

**짝짓기 바탕 손실.** 익히는 동안 손실을 셈하기 전에 만들어 낸 그래프를 과녁에 맞추는 가장 좋은 자리바꿈 $\pi^*$을 찾는다:

$$
\pi^* = \arg\min_{\pi \in S_n} \|\hat{\mathbf{A}} - \mathbf{P}_\pi \mathbf{A}^* \mathbf{P}_\pi^\top\|_F^2
$$

이는 (일반으로 NP어려움인) 그래프 짝짓기 문제와 같지만 마디 특징에 헝가리 알고리즘을 쓰는 실제 어림이 쓸 만한 풀이를 준다.

## 풀개 얼개

### 여러 층 신경망 풀개

가장 단순한 길은 숨은 벡터를 납작하게 편 위쪽 삼각 이웃 벡터로 옮긴다:

$$
\hat{\mathbf{a}} = \sigma\left(\text{MLP}(\mathbf{z})\right) \in [0,1]^{\binom{n_{\max}}{2}}
$$

이는 마지막 층에 잡 $O(n_{\max}^2 \cdot d_z)$개가 들고 그래프에 대한 어떤 얼개 사전 믿음도 살리지 못한다.

### 인수로 나눈 풀개

잡의 수를 줄이려 이웃 헤아리기를 마디 박아 넣기로 인수 분해한다:

$$
\mathbf{Z}_{\text{nodes}} = \text{MLP}_{\text{node}}(\mathbf{z}) \in \mathbb{R}^{n_{\max} \times d_h}
$$

$$
\hat{A}_{ij} = \sigma\left(\mathbf{z}_i^\top \mathbf{W} \mathbf{z}_j + b\right)
$$

여기서 $\mathbf{z}_i$은 $\mathbf{Z}_{\text{nodes}}$의 $i$번째 줄이다. 이 겹선형 꼴은 잡이 $O(n_{\max} \cdot d_h + d_h^2)$개만 들며 $\mathbf{W}$이 맞섬이면 자연스럽게 맞섬 이웃 행렬을 낸다.

### 그래프에 매인 풀개

숨은 마디 나타냄에 그래프 신경망을 써서 변 헤아림을 되풀이해 다듬는다:

$$
\mathbf{Z}^{(\ell+1)} = \text{GNN}^{(\ell)}(\mathbf{Z}^{(\ell)}, \hat{\mathbf{A}}^{(\ell)})
$$

여기서 $\hat{\mathbf{A}}^{(\ell)}$은 되풀이마다 $\mathbf{Z}^{(\ell)}$에서 셈한다. 이 되풀이 다듬기 덕에 변 헤아림이 드러나는 그래프 얼개에 매일 수 있다.

## 크기 다루기

한 번에 만들기 방법은 크기가 다른 그래프를 다루어야 한다. 흔한 길:

**마디 가리개로 채우기.** 크기가 붙박인 $n_{\max}$의 그래프를 만들고 마디가 있는지 나타내는 가리개 $\mathbf{m} \in [0,1]^{n_{\max}}$을 헤아린다. 실제 그래프는 $m_i > 0.5$인 마디를 쓴다. 가려진 마디가 낀 변은 0으로 만든다:

$$
\hat{A}_{ij}^{\text{final}} = \hat{A}_{ij} \cdot m_i \cdot m_j
$$

**크기에 조건 걸기.** 겪어 얻은 크기 분포에서 그래프 크기 $n \sim p(n)$을 뽑고 $n$에 매어 만든다. 풀개에 $n$을 들임으로 더 주어 짤 수 있다.

## 익히기 손실의 조각

온전한 한 번에 만들기 손실은 보통 다음을 아우른다:

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}} + \lambda_{\text{match}} \mathcal{L}_{\text{match}}
$$

여기서 $\mathcal{L}_{\text{recon}}$은 변 되짓기 손실(두 값 교차 엔트로피), $\mathcal{L}_{\text{KL}}$은 변분 스스로 담개 바탕 방법의 쿨백-라이블러 어긋남, $\mathcal{L}_{\text{match}}$은 자리바꿈 맞추기를 다룬다.

## 금융 쓰임새: 한때의 그물 스냅숏

한 번에 만들기는 온전한 금융 그물 스냅숏을 내놓는 데 자연스럽게 알맞다. 보기로 규제 보고일의 은행 사이 노출을 온 단면으로 만들어 낸다. 그물이 생기는 과정을 나타내는 자기 되돌이 방법과 달리 한 번에 만들기는 균형 짜임을 곧바로 내며, 때에 따른 생김 과정이 관심 밖일 때 알맞다.

## 짜기: 한 번에 만드는 이웃 행렬 만들개

```python
"""
여러 풀개 얼개로 하는 한 번에 이웃 행렬 만들기.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from scipy.optimize import linear_sum_assignment
import numpy as np


class MLPDecoder(nn.Module):
    """Simple MLP decoder: latent -> flattened upper-triangular adjacency."""

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
    Factored decoder: latent -> node embeddings -> bilinear edge prediction.
    여러 층 신경망 풀개보다 잡을 아낀다.
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

        # 숨은 값에서 마디 박아 넣기로
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * node_embed_dim),
        )

        # 겹선형 변 헤아리개
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

        # 마디 박아 넣기를 푼다
        node_embeds = self.node_decoder(z).view(B, n, self.node_embed_dim)

        # 겹선형 변 헤아리기: A_ij = σ(z_i^T W z_j + b)
        # 방향 없는 그래프를 위해 W을 맞섬으로 만든다
        W_sym = (self.edge_weight + self.edge_weight.t()) / 2
        edge_logits = torch.einsum("bid,de,bje->bij", node_embeds, W_sym, node_embeds)
        edge_logits = edge_logits + self.edge_bias
        adj = torch.sigmoid(edge_logits)

        # 대각을 0으로
        mask = 1 - torch.eye(n, device=z.device).unsqueeze(0)
        adj = adj * mask

        # 마디가 있는지 나타내는 가리개
        node_mask = torch.sigmoid(
            self.node_mask_predictor(node_embeds).squeeze(-1)
        )

        # 이웃 행렬에 마디 가리개를 쓴다
        adj = adj * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

        return adj, node_mask


def hungarian_matching(
    adj_pred: torch.Tensor,
    adj_target: torch.Tensor,
) -> torch.Tensor:
    """
    헝가리 알고리즘으로 가장 좋은 마디 자리바꿈을 찾는다.
    
    인수:
        adj_pred: (n, n) 헤아린 이웃 행렬
        adj_target: (n, n) 과녁 이웃 행렬
        
    반환값:
        perm: (n,) 자리바꿈 번호
    """
    n = adj_pred.size(0)
    # 비용 행렬: 모든 (i, j) 짝에 대해 ||pred_i - target_j||^2
    # 이웃 행렬의 줄을 마디 특징으로 쓴다
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
    자리바꿈 맞추기를 골라 쓰며 되짓기 손실을 셈한다.
    
    인수:
        adj_pred: (B, n, n) 헤아린 이웃 행렬
        adj_target: (B, n, n) 과녁 이웃 행렬
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

    # 여러 층 신경망 풀개 시험
    print("\n--- MLP Decoder ---")
    mlp_dec = MLPDecoder(latent_dim, max_n, hidden_dim=128)
    z = torch.randn(5, latent_dim)
    adj_pred = mlp_dec(z)
    print(f"Output shape: {adj_pred.shape}")
    print(f"Symmetry check: {torch.allclose(adj_pred, adj_pred.transpose(1,2))}")
    print(f"Value range: [{adj_pred.min():.3f}, {adj_pred.max():.3f}]")

    # 인수로 나눈 풀개 시험
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

    # 자리 맞춘 손실 시험
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
