# GraphVAE
## 개요

GraphVAE(Simonovsky & Komodakis, 2018)은 변분 스스로 담개 틀을 한 번에 만드는 그래프 만들기에 쓴다. 담개는 그래프 신경망으로 그래프를 숨은 분포로 옮기고, 풀개는 뽑은 숨은 벡터에서 이웃 행렬 전체와 마디 특징을 되짓는다. 핵심 새로움은 (1) 표준 차례 없이 자리바꿈에 안 바뀜을 다루는 그래프 짝짓기 손실과 (2) 그래프 위상과 마디·변 속성을 한꺼번에 만드는 확률 풀개다.

## 구조

### 부호기

담개는 들임 그래프 $\mathcal{G} = (\mathbf{A}, \mathbf{X})$을 숨은 자리의 가우스 사후 분포로 옮긴다. 그래프 신경망이 마디 박아 넣기를 셈하고 이를 그래프 켜의 나타냄으로 모은다:

$$
\mathbf{H} = \text{GNN}_{\text{enc}}(\mathbf{A}, \mathbf{X}) \in \mathbb{R}^{n \times d_h}
$$

$$
\mathbf{h}_{\mathcal{G}} = \text{READOUT}(\mathbf{H}) \in \mathbb{R}^{d_h}
$$

사후 분포의 잡은 그래프 박아 넣기에서 셈한다:

$$
\boldsymbol{\mu} = \text{MLP}_\mu(\mathbf{h}_{\mathcal{G}}), \quad \log \boldsymbol{\sigma}^2 = \text{MLP}_\sigma(\mathbf{h}_{\mathcal{G}})
$$

$$
q_\phi(\mathbf{z} \mid \mathcal{G}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))
$$

### 풀개

풀개는 숨은 벡터 $\mathbf{z}$을 마디가 $n_{\max}$개인 확률 그래프로 옮긴다. 조각 셋을 헤아린다:

**마디가 있음.** 어느 마디가 있는지 나타내는 확률 벡터 $\hat{\mathbf{m}} \in [0,1]^{n_{\max}}$:

$$
\hat{\mathbf{m}} = \sigma(\text{MLP}_{\text{node}}(\mathbf{z}))
$$

**이웃 행렬.** 모든 마디 짝의 변 확률:

$$
\hat{\mathbf{A}} = \sigma(\text{MLP}_{\text{edge}}(\mathbf{z})) \in [0,1]^{n_{\max} \times n_{\max}}
$$

**마디 특징.** 마디마다 헤아린 특징:

$$
\hat{\mathbf{X}} = \text{MLP}_{\text{feat}}(\mathbf{z}) \in \mathbb{R}^{n_{\max} \times d_x}
$$

## 그래프 짝짓기 손실

GraphVAE의 한가운데 이바지는 드러난 그래프 짝짓기로 자리바꿈에 안 바뀜을 다루는 것이다. 헤아린 그래프 $\hat{\mathcal{G}}$과 과녁 그래프 $\mathcal{G}^*$이 주어지면 최대 무게 두 쪽 짝짓기 문제를 풀어 가장 좋은 자리바꿈 $\pi^*$을 찾는다.

마디 닮음 행렬 $\mathbf{S} \in \mathbb{R}^{n_{\max} \times n_{\max}}$을 뜻매김한다:

$$
S_{ij} = \mathbf{1}[\hat{m}_i > 0.5] \cdot \mathbf{1}[j \leq n^*] \cdot \left( \lambda_x \cdot \text{sim}(\hat{\mathbf{x}}_i, \mathbf{x}_j^*) + \lambda_e \cdot \text{sim}(\hat{\mathbf{a}}_i, \mathbf{a}_j^*) \right)
$$

여기서 $\hat{\mathbf{a}}_i$과 $\mathbf{a}_j^*$은 각각 헤아린 이웃 행렬의 $i$번째 줄과 과녁 이웃 행렬의 $j$번째 줄이다. 가장 좋은 짝짓기는 다음과 같다:

$$
\pi^* = \arg\max_{\pi} \sum_{i} S_{i, \pi(i)}
$$

헝가리 알고리즘으로 때 $O(n_{\max}^3)$에 셈한다.

## 증거 아래 가둠 목표

익히기 손실은 그래프 짝짓기 되짓기 항을 가진 증거 아래 가둠이다:

$$
\mathcal{L} = \underbrace{-\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathcal{G})} \left[ \log p_\theta(\mathcal{G} \mid \mathbf{z}) \right]}_{\text{reconstruction}} + \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathcal{G}) \| p(\mathbf{z}))}_{\text{regularization}}
$$

되짓기 항은 (짝짓기 뒤에) 다음으로 쪼개진다:

$$
\log p_\theta(\mathcal{G} \mid \mathbf{z}) = \underbrace{\sum_{i} \log p(\text{node}_i \mid \mathbf{z})}_{\text{node existence}} + \underbrace{\sum_{i<j} \log p(A_{ij} \mid \mathbf{z})}_{\text{edge reconstruction}} + \underbrace{\sum_{i} \log p(\mathbf{x}_i \mid \mathbf{z})}_{\text{feature reconstruction}}
$$

항마다 짝짓기 걸음에서 자리를 맞춘 과녁을 쓴다.

## 한계

**커지기.** 이웃 내놓기가 $O(n_{\max}^2)$이고 짝짓기 셈이 $O(n_{\max}^3)$이어서 GraphVAE은 비교적 작은 그래프(보통 $n_{\max} \leq 40$)에 갇힌다.

**독립이라는 여김.** 풀개는 $\mathbf{z}$이 주어지면 변마다 따로 헤아려 변 사이의 매임을 무시한다. 이 때문에 그 자리 얼개가 어긋나는 그래프가 나올 수 있다.

**사후 분포 무너짐.** 여느 변분 스스로 담개처럼 모델이 숨은 변수를 무시하고 평균처럼 보이는 그래프를 낼 수 있다. 풀개가 너무 세거나 쿨백-라이블러의 $\beta$ 무게가 너무 클 때 특히 그렇다.

## 금융 쓰임새: 금융 그물의 숨은 자리

GraphVAE의 숨은 자리는 금융 그물 위상의 이어진 나타냄을 준다. 담긴 금융 그물 둘 $\mathbf{z}_1$과 $\mathbf{z}_2$ 사이의 사이 값을 내면 짜임 사이를 매끄럽게 오가는 중간 위상이 나온다. 그물 얼개를 여느 국면에서 버거운 국면으로 차츰 옮기는 버거움 시험 상황에 쓸모 있다.

## 짜기: GraphVAE

```python
"""
GraphVAE: 한 번에 그래프를 만드는 변분 스스로 담개.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


class GraphEncoder(nn.Module):
    """GraphVAE의 그래프 신경망 바탕 그래프 담개."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        adj: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        인수:
            adj: (B, n, n) 이웃 행렬
            x: (B, n, d) 마디 특징
            node_mask: (B, n) 올바른 마디의 두 값 가리개
        """
        h = self.input_proj(x)  # (B, n, hidden)

        for conv, norm in zip(self.conv_layers, self.norms):
            # 단순한 GCN 방식 쪽지 건네기: h = σ(A h W)
            h_msg = torch.bmm(adj, h)  # (B, n, hidden)
            h = norm(F.relu(conv(h_msg)) + h)

        # 가린 평균 모으기
        mask = node_mask.unsqueeze(-1)  # (B, n, 1)
        h_graph = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        mu = self.mu_head(h_graph)
        logvar = self.logvar_head(h_graph)
        return mu, logvar


class GraphDecoder(nn.Module):
    """이웃 행렬, 특징, 마디 가리개를 만드는 풀개."""

    def __init__(
        self,
        latent_dim: int,
        max_nodes: int,
        node_feature_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        n_edges = max_nodes * (max_nodes - 1) // 2

        # 함께 쓰는 등뼈
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 이웃 행렬 머리
        self.adj_head = nn.Linear(hidden_dim, n_edges)

        # 마디가 있는지 머리
        self.node_head = nn.Linear(hidden_dim, max_nodes)

        # 마디 특징 머리
        self.feat_head = nn.Linear(hidden_dim, max_nodes * node_feature_dim)
        self.node_feature_dim = node_feature_dim

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        반환값:
            adj_prob: (B, n, n)
            node_prob: (B, n)
            node_feat: (B, n, d)
        """
        B = z.size(0)
        n = self.max_nodes
        h = self.backbone(z)

        # 이웃 행렬
        edge_logits = self.adj_head(h)
        adj = torch.zeros(B, n, n, device=z.device)
        idx = torch.triu_indices(n, n, offset=1)
        adj[:, idx[0], idx[1]] = torch.sigmoid(edge_logits)
        adj = adj + adj.transpose(1, 2)

        # 마디 가리개
        node_prob = torch.sigmoid(self.node_head(h))

        # 마디 특징
        node_feat = self.feat_head(h).view(B, n, self.node_feature_dim)

        return adj, node_prob, node_feat


class GraphVAE(nn.Module):
    """
    그래프 짝짓기 손실을 갖춘 온전한 GraphVAE 모델.
    """

    def __init__(
        self,
        max_nodes: int,
        node_feature_dim: int = 1,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        beta: float = 1.0,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = GraphEncoder(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
        self.decoder = GraphDecoder(
            latent_dim=latent_dim,
            max_nodes=max_nodes,
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _compute_matching(
        self,
        adj_pred: torch.Tensor,
        adj_target: torch.Tensor,
        node_prob: torch.Tensor,
        n_target: int,
    ) -> torch.Tensor:
        """그래프 짝 하나의 헝가리 짝짓기."""
        n = self.max_nodes

        # 이웃 행렬 줄의 닮음으로 비용 행렬을 짓는다
        with torch.no_grad():
            cost = torch.zeros(n, n)
            for i in range(n):
                for j in range(n_target):
                    # 이웃 닮음과 마디가 있음을 합친다
                    adj_sim = -F.mse_loss(
                        adj_pred[i], adj_target[j], reduction="sum"
                    )
                    node_sim = node_prob[i] if j < n_target else 0
                    cost[i, j] = adj_sim + node_sim

            cost_np = (-cost).cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

        perm = torch.tensor(col_ind, dtype=torch.long, device=adj_pred.device)
        return perm

    def forward(
        self,
        adj: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        익히기 앞으로 가기.
        
        인수:
            adj: (B, n, n) 채운 이웃 행렬
            x: (B, n, d) 채운 마디 특징
            node_mask: (B, n) 올바른 마디 표시
        """
        B = adj.size(0)

        # 부호화
        mu, logvar = self.encoder(adj, x, node_mask)
        z = self.reparameterize(mu, logvar)

        # 디코딩
        adj_pred, node_pred, feat_pred = self.decoder(z)

        # KL 벌어짐
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        # 짝짓기와 함께 되짓기
        recon_loss = torch.tensor(0.0, device=adj.device)
        for b in range(B):
            n_actual = int(node_mask[b].sum().item())

            # 가장 좋은 짝짓기를 찾는다
            perm = self._compute_matching(
                adj_pred[b], adj[b], node_pred[b], n_actual
            )

            # 자리 맞춘 과녁
            adj_aligned = adj[b][perm][:, perm]
            mask_aligned = node_mask[b][perm]

            # 변 되짓기 손실
            idx = torch.triu_indices(self.max_nodes, self.max_nodes, offset=1)
            edge_pred = adj_pred[b][idx[0], idx[1]]
            edge_target = adj_aligned[idx[0], idx[1]]
            recon_loss = recon_loss + F.binary_cross_entropy(
                edge_pred.clamp(1e-6, 1 - 1e-6),
                edge_target,
                reduction="mean",
            )

            # 마디가 있는지 손실
            recon_loss = recon_loss + F.binary_cross_entropy(
                node_pred[b].clamp(1e-6, 1 - 1e-6),
                mask_aligned,
                reduction="mean",
            )

        recon_loss = recon_loss / B
        total_loss = recon_loss + self.beta * kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        num_graphs: int = 1,
        temperature: float = 1.0,
    ) -> list[torch.Tensor]:
        """사전 분포에서 그래프를 만든다."""
        self.eval()
        z = torch.randn(num_graphs, self.latent_dim) * temperature
        adj_pred, node_pred, _ = self.decoder(z)

        graphs = []
        for b in range(num_graphs):
            # 살아 있는 마디를 정한다
            active = node_pred[b] > 0.5
            n_active = active.sum().item()
            if n_active < 2:
                n_active = 2
                active[:2] = True

            # 살아 있는 마디의 변을 뽑는다
            adj_b = adj_pred[b]
            adj_b = (adj_b > 0.5).float()
            adj_b = adj_b * active.unsqueeze(0).float() * active.unsqueeze(1).float()
            adj_b.fill_diagonal_(0)

            # 부분 그래프를 뽑아낸다
            active_idx = torch.where(active)[0]
            sub_adj = adj_b[active_idx][:, active_idx]
            graphs.append(sub_adj)

        return graphs

    @torch.no_grad()
    def interpolate(
        self,
        adj1: torch.Tensor,
        x1: torch.Tensor,
        mask1: torch.Tensor,
        adj2: torch.Tensor,
        x2: torch.Tensor,
        mask2: torch.Tensor,
        steps: int = 5,
    ) -> list[torch.Tensor]:
        """숨은 자리에서 그래프 둘 사이의 사이 값을 낸다."""
        self.eval()
        mu1, _ = self.encoder(adj1.unsqueeze(0), x1.unsqueeze(0), mask1.unsqueeze(0))
        mu2, _ = self.encoder(adj2.unsqueeze(0), x2.unsqueeze(0), mask2.unsqueeze(0))

        graphs = []
        for alpha in torch.linspace(0, 1, steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            adj_pred, node_pred, _ = self.decoder(z)
            adj_b = (adj_pred[0] > 0.5).float()
            active = node_pred[0] > 0.5
            adj_b = adj_b * active.unsqueeze(0).float() * active.unsqueeze(1).float()
            adj_b.fill_diagonal_(0)
            active_idx = torch.where(active)[0]
            if len(active_idx) >= 2:
                graphs.append(adj_b[active_idx][:, active_idx])
            else:
                graphs.append(adj_b[:2, :2])

        return graphs


if __name__ == "__main__":
    torch.manual_seed(42)

    max_n = 12
    feat_dim = 4

    # 익히기 자료를 만든다
    print("=== GraphVAE Demo ===\n")
    train_data = []
    for _ in range(150):
        n = torch.randint(4, max_n, (1,)).item()
        adj = torch.zeros(max_n, max_n)
        for i in range(n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.3:
                    adj[i, j] = adj[j, i] = 1
        x = torch.randn(max_n, feat_dim)
        mask = torch.zeros(max_n)
        mask[:n] = 1.0
        train_data.append((adj, x, mask))

    adjs = torch.stack([d[0] for d in train_data])
    feats = torch.stack([d[1] for d in train_data])
    masks = torch.stack([d[2] for d in train_data])

    # 학습
    print("Training GraphVAE...")
    model = GraphVAE(
        max_nodes=max_n,
        node_feature_dim=feat_dim,
        hidden_dim=64,
        latent_dim=16,
        beta=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    batch_size = 32
    for epoch in range(30):
        model.train()
        idx = torch.randperm(len(train_data))[:batch_size]
        result = model(adjs[idx], feats[idx], masks[idx])
        loss = result["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f} "
                  f"(recon={result['recon_loss'].item():.4f}, "
                  f"kl={result['kl_loss'].item():.4f})")

    # 생성
    print("\n=== Generation ===")
    generated = model.generate(num_graphs=10)
    for i, g in enumerate(generated):
        n = g.size(0)
        e = int(g.sum().item()) // 2
        print(f"Graph {i}: {n} nodes, {e} edges")

    # 사이 값 내기
    print("\n=== Latent Interpolation ===")
    interp = model.interpolate(
        adjs[0], feats[0], masks[0],
        adjs[1], feats[1], masks[1],
        steps=5,
    )
    for i, g in enumerate(interp):
        n = g.size(0)
        e = int(g.sum().item()) // 2
        density = 2 * e / (n * (n - 1)) if n > 1 else 0
        print(f"Step {i}: {n} nodes, {e} edges, density={density:.3f}")
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
