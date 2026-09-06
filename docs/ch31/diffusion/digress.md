# DiGress: 그래프 만들기를 위한 띄엄띄엄 잡소리 없애기 퍼짐
## 개요

DiGress(Vignac et al., 2023)은 갈래로 나뉜 마디와 변의 갈래에서 잡소리를 한꺼번에 없애 그래프를 만드는 띄엄띄엄 퍼짐 모델이다. 이어진 값으로 느슨히 하는 길과 달리 DiGress은 갈래 퍼짐 과정을 써서 띄엄띄엄한 그래프 얼개에서 그대로 돈다. 잡소리 없애는 그물로 그래프 트랜스포머를 쓰고 그래프에 맞게 설계한 잡소리 모형을 내놓아, 만들기 품질을 높이는 빨아들이는 상태와 주변 분포를 담는다. DiGress은 분자와 일반 그래프 만들기 잣대에서 가장 앞선 결과를 낸다.

## 그래프 위의 띄엄띄엄 퍼짐

DiGress에서 그래프는 $\mathcal{G} = (\mathbf{X}, \mathbf{E})$으로 나타내며, $\mathbf{X} \in \{0, \ldots, a\}^n$은 갈래 마디 갈래($a$개 갈래에 "없음" 갈래 하나)이고 $\mathbf{E} \in \{0, \ldots, b\}^{n \times n}$은 갈래 변 갈래($b$개 갈래에 "변 없음")이다.

### 앞으로 가는 과정

마디 갈래 $X_i$과 변 갈래 $E_{ij}$마다 갈래 마르코프 사슬을 따라 따로 옮겨 간다:

$$
q(X_i^{(t)} \mid X_i^{(t-1)}) = \text{Cat}(X_i^{(t)}; \mathbf{Q}_t^X \cdot \text{onehot}(X_i^{(t-1)}))
$$

$$
q(E_{ij}^{(t)} \mid E_{ij}^{(t-1)}) = \text{Cat}(E_{ij}^{(t)}; \mathbf{Q}_t^E \cdot \text{onehot}(E_{ij}^{(t-1)}))
$$

여기서 $\mathbf{Q}_t^X \in \mathbb{R}^{(a+1) \times (a+1)}$과 $\mathbf{Q}_t^E \in \mathbb{R}^{(b+1) \times (b+1)}$은 옮김 행렬이다.

$t$걸음 주변 분포는 다음과 같다:

$$
q(\mathcal{G}_t \mid \mathcal{G}_0) = \prod_{i} \text{Cat}(X_i^{(t)}; \bar{\mathbf{Q}}_t^X \cdot \text{onehot}(X_i^{(0)})) \prod_{i<j} \text{Cat}(E_{ij}^{(t)}; \bar{\mathbf{Q}}_t^E \cdot \text{onehot}(E_{ij}^{(0)}))
$$

여기서 $\bar{\mathbf{Q}}_t = \prod_{s=1}^{t} \mathbf{Q}_s$이다.

### 잡소리 모형

DiGress은 잡소리 모형 둘을 내놓는다:

**고른 잡소리.** 옮김이 갈래를 고른 분포 쪽으로 고르게 흐트러뜨린다:

$$
\mathbf{Q}_t = (1 - \beta_t) \mathbf{I} + \beta_t \frac{\mathbf{1}\mathbf{1}^\top}{K}
$$

여기서 $K$은 갈래 수다. $t = T$에서는 모든 갈래가 똑같이 나올 법하다.

**주변 잡소리.** 익히기 자료의 주변 분포 쪽으로 옮겨 간다:

$$
\mathbf{Q}_t = (1 - \beta_t) \mathbf{I} + \beta_t \mathbf{1} \mathbf{m}^\top
$$

여기서 $\mathbf{m}$은 갈래에 대해 겪어 얻은 주변 확률이다. 이는 $t = T$의 사전 분포를 고른 분포가 아니라 자료의 주변 분포에 모아, 큰 $t$의 잡소리 섞인 분포가 이미 올바른 갈래 잦음을 비추므로 잡소리 없애기를 쉽게 한다.

### 거꾸로 가는 과정

거꾸로 가는 걸음은 베이즈 규칙으로 사후 분포를 셈한다:

$$
q(\mathcal{G}_{t-1} \mid \mathcal{G}_t, \mathcal{G}_0) \propto q(\mathcal{G}_t \mid \mathcal{G}_{t-1}) \cdot q(\mathcal{G}_{t-1} \mid \mathcal{G}_0)
$$

잡소리 없애는 그물 $\phi_\theta(\mathcal{G}_t, t)$이 깨끗한 그래프에 대한 분포 $\hat{p}_\theta(\mathcal{G}_0 \mid \mathcal{G}_t)$을 헤아리고 거꿀 옮김은 다음과 같다:

$$
p_\theta(\mathcal{G}_{t-1} \mid \mathcal{G}_t) = \sum_{\hat{\mathcal{G}}_0} q(\mathcal{G}_{t-1} \mid \mathcal{G}_t, \hat{\mathcal{G}}_0) \cdot \hat{p}_\theta(\hat{\mathcal{G}}_0 \mid \mathcal{G}_t)
$$

## 그래프 트랜스포머 잡소리 없애개

잡소리 없애는 그물은 잡소리 섞인 그래프와 때 걸음을 다루어 깨끗한 그래프를 헤아리는 그래프 트랜스포머다. 층마다 마디와 변의 나타냄을 함께 고친다:

**마디 고치기:**

$$
\mathbf{h}_i^{(\ell+1)} = \mathbf{h}_i^{(\ell)} + \text{MHA}^{(\ell)}(\mathbf{h}_i^{(\ell)}, \{\mathbf{h}_j^{(\ell)}\}_{j \neq i}, \{\mathbf{e}_{ij}^{(\ell)}\}_{j \neq i})
$$

**변 고치기:**

$$
\mathbf{e}_{ij}^{(\ell+1)} = \mathbf{e}_{ij}^{(\ell)} + \text{MLP}^{(\ell)}([\mathbf{h}_i^{(\ell+1)} \| \mathbf{h}_j^{(\ell+1)} \| \mathbf{e}_{ij}^{(\ell)}])
$$

눈길 얼개는 변 특징을 눈길 무게에 담는다:

$$
\alpha_{ij}^{(\ell)} = \frac{(\mathbf{W}_Q \mathbf{h}_i)^\top (\mathbf{W}_K \mathbf{h}_j) + (\mathbf{W}_E \mathbf{e}_{ij})^\top \mathbf{w}_a}{\sqrt{d_k}}
$$

내놓기 머리가 마디와 변 갈래에 대한 로짓을 헤아린다:

$$
\hat{p}(X_i^{(0)} \mid \mathcal{G}_t) = \text{softmax}(\text{MLP}_X(\mathbf{h}_i^{(L)}))
$$

$$
\hat{p}(E_{ij}^{(0)} \mid \mathcal{G}_t) = \text{softmax}(\text{MLP}_E(\mathbf{e}_{ij}^{(L)}))
$$

## 익히기 손실

띄엄띄엄 퍼짐의 변분 아래 가둠은 걸음마다의 쿨백-라이블러 어긋남으로 쪼개진다:

$$
\mathcal{L}_{\text{VLB}} = \sum_{t=1}^{T} \mathbb{E}_{q(\mathcal{G}_t \mid \mathcal{G}_0)} \left[ D_{\text{KL}}(q(\mathcal{G}_{t-1} \mid \mathcal{G}_t, \mathcal{G}_0) \| p_\theta(\mathcal{G}_{t-1} \mid \mathcal{G}_t)) \right]
$$

실제로 DiGress은 헤아린 깨끗한 갈래와 참 갈래 사이의 더 단순한 교차 엔트로피 손실을 쓴다:

$$
\mathcal{L}_{\text{CE}} = \mathbb{E}_{t, \mathcal{G}_0, \mathcal{G}_t} \left[ -\sum_i \log \hat{p}_\theta(X_i^{(0)} \mid \mathcal{G}_t) - \sum_{i<j} \log \hat{p}_\theta(E_{ij}^{(0)} \mid \mathcal{G}_t) \right]
$$

## 짜기: DiGress의 핵심 조각

```python
"""
DiGress: 그래프 만들기의 띄엄띄엄 잡소리 없애기 퍼짐.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CategoricalDiffusion:
    """마디와 변의 갈래 퍼짐 과정."""

    def __init__(
        self,
        num_node_classes: int,
        num_edge_classes: int,
        num_timesteps: int = 500,
        noise_type: str = "marginal",
        node_marginals: torch.Tensor = None,
        edge_marginals: torch.Tensor = None,
    ):
        self.T = num_timesteps
        self.num_node_cls = num_node_classes
        self.num_edge_cls = num_edge_classes
        self.noise_type = noise_type

        # 잡소리 일정(코사인)
        s = 0.008
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999).float()

        # 옮김 행렬을 짓는다
        self.Q_bar_X = []  # 마디의 쌓인 옮김
        self.Q_bar_E = []  # 변의 쌓인 옮김

        Q_bar_x = torch.eye(num_node_classes)
        Q_bar_e = torch.eye(num_edge_classes)

        if node_marginals is None:
            node_marginals = torch.ones(num_node_classes) / num_node_classes
        if edge_marginals is None:
            edge_marginals = torch.ones(num_edge_classes) / num_edge_classes

        for t in range(num_timesteps):
            beta = betas[t].item()
            if noise_type == "uniform":
                Qt_x = (1 - beta) * torch.eye(num_node_classes) + beta / num_node_classes
                Qt_e = (1 - beta) * torch.eye(num_edge_classes) + beta / num_edge_classes
            else:  # 주변 분포
                Qt_x = (1 - beta) * torch.eye(num_node_classes) + beta * node_marginals.unsqueeze(0).expand(num_node_classes, -1)
                Qt_e = (1 - beta) * torch.eye(num_edge_classes) + beta * edge_marginals.unsqueeze(0).expand(num_edge_classes, -1)

            Q_bar_x = Q_bar_x @ Qt_x
            Q_bar_e = Q_bar_e @ Qt_e
            self.Q_bar_X.append(Q_bar_x.clone())
            self.Q_bar_E.append(Q_bar_e.clone())

    def q_sample(
        self,
        x_0: torch.Tensor,
        e_0: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        때 걸음 t의 잡소리 섞인 그래프를 뽑는다.
        
        인수:
            x_0: (B, n) 정수 마디 갈래
            e_0: (B, n, n) 정수 변 갈래
            t: (B,) 때 걸음
        """
        B, n = x_0.shape
        device = x_0.device

        x_t = torch.zeros_like(x_0)
        e_t = torch.zeros_like(e_0)

        for b in range(B):
            tb = t[b].item()
            Q_x = self.Q_bar_X[tb].to(device)
            Q_e = self.Q_bar_E[tb].to(device)

            # 잡소리 섞인 마디 갈래를 뽑는다
            x_onehot = F.one_hot(x_0[b], self.num_node_cls).float()
            x_probs = x_onehot @ Q_x  # (n, num_node_cls)
            x_t[b] = torch.multinomial(x_probs, 1).squeeze(-1)

            # 잡소리 섞인 변 갈래를 뽑는다
            e_onehot = F.one_hot(e_0[b], self.num_edge_cls).float()
            e_flat = e_onehot.view(-1, self.num_edge_cls)
            e_probs = e_flat @ Q_e
            e_t[b] = torch.multinomial(e_probs, 1).squeeze(-1).view(n, n)

        return x_t, e_t

    def posterior(
        self,
        x_t: torch.Tensor,
        x_0_pred: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Compute posterior q(x_{t-1} | x_t, x_0) for a single variable.
        
        인수:
            x_t: one-hot (K,)
            x_0_pred: predicted distribution (K,)
            t: 지금 때 걸음
        """
        if t == 0:
            return x_0_pred

        Q_t = self.Q_bar_X[t] if t < len(self.Q_bar_X) else self.Q_bar_X[-1]
        Q_tm1 = self.Q_bar_X[t - 1] if t > 0 else torch.eye(self.num_node_cls)

        # 가능한 x_0마다 q(x_t | x_0)과 q(x_{t-1} | x_0)
        prob_xt_given_x0 = x_0_pred @ Q_t  # (K,)
        prob_xtm1_given_x0 = x_0_pred @ Q_tm1  # (K,)

        # 베이즈: q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) q(x_{t-1} | x_0)
        posterior_unnorm = prob_xtm1_given_x0 * (Q_t[:, x_t.argmax()] + 1e-8)
        return posterior_unnorm / (posterior_unnorm.sum() + 1e-8)


class DiGressTransformerLayer(nn.Module):
    """마디와 변의 나타냄을 함께 고치는 트랜스포머 층."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_E = nn.Linear(hidden_dim, num_heads)

        self.node_out = nn.Linear(hidden_dim, hidden_dim)
        self.edge_out = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm_n = nn.LayerNorm(hidden_dim)
        self.norm_e = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(self, h, e, t_emb):
        """
        인수:
            h: (B, n, d) 마디 특징
            e: (B, n, n, d) 변 특징
            t_emb: (B, d) 때 박아 넣기
        """
        B, n, d = h.shape
        heads = self.num_heads
        hd = d // heads

        Q = self.W_Q(h).view(B, n, heads, hd)
        K = self.W_K(h).view(B, n, heads, hd)
        V = self.W_V(h).view(B, n, heads, hd)

        # 변 치우침을 더한 눈길
        attn = torch.einsum("bihd,bjhd->bijh", Q, K) / math.sqrt(hd)
        edge_bias = self.W_E(e)  # (B, n, n, heads)
        attn = attn + edge_bias
        attn = F.softmax(attn, dim=2)

        out = torch.einsum("bijh,bjhd->bihd", attn, V).reshape(B, n, d)
        h = self.norm_n(h + self.node_out(out) + t_emb.unsqueeze(1))
        h = self.norm_ffn(h + self.ffn(h))

        # 변 고치기
        e_input = torch.cat([
            h.unsqueeze(2).expand(-1, -1, n, -1),
            h.unsqueeze(1).expand(-1, n, -1, -1),
            e,
        ], dim=-1)
        e = self.norm_e(e + self.edge_out(e_input))

        return h, e


class DiGressDenoiser(nn.Module):
    """온전한 DiGress 잡소리 없애기 그물."""

    def __init__(
        self,
        num_node_classes: int,
        num_edge_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        num_timesteps: int = 500,
    ):
        super().__init__()
        self.node_embed = nn.Embedding(num_node_classes, hidden_dim)
        self.edge_embed = nn.Embedding(num_edge_classes, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList([
            DiGressTransformerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.node_out = nn.Linear(hidden_dim, num_node_classes)
        self.edge_out = nn.Linear(hidden_dim, num_edge_classes)
        self.T = num_timesteps

    def forward(self, x_t, e_t, t):
        """
        잡소리 섞인 그래프에서 깨끗한 그래프를 헤아린다.
        
        인수:
            x_t: (B, n) 잡소리 섞인 마디 갈래
            e_t: (B, n, n) 잡소리 섞인 변 갈래
            t: (B,) 때 걸음
            
        반환값:
            x_logits: (B, n, num_node_cls)
            e_logits: (B, n, n, num_edge_cls)
        """
        h = self.node_embed(x_t)  # (B, n, d)
        e = self.edge_embed(e_t)  # (B, n, n, d)
        t_emb = self.time_embed(t.float().unsqueeze(-1) / self.T)

        for layer in self.layers:
            h, e = layer(h, e, t_emb)

        x_logits = self.node_out(h)
        e_logits = self.edge_out(e)
        # 변 로짓을 맞섬으로 만든다
        e_logits = (e_logits + e_logits.transpose(1, 2)) / 2

        return x_logits, e_logits


if __name__ == "__main__":
    torch.manual_seed(42)

    n_node_cls = 3  # 보기로 분자의 C, N, O
    n_edge_cls = 3  # 변 없음, 홑, 겹
    max_n = 8
    T = 100

    print("=== DiGress Demo ===\n")

    # 지어낸 익히기 자료를 만든다
    B = 50
    x_0 = torch.randint(0, n_node_cls, (B, max_n))
    e_0 = torch.randint(0, n_edge_cls, (B, max_n, max_n))
    e_0 = torch.triu(e_0, diagonal=1)
    e_0 = e_0 + e_0.transpose(1, 2)

    # 갈래 퍼짐
    node_marg = torch.tensor([0.6, 0.3, 0.1])
    edge_marg = torch.tensor([0.7, 0.2, 0.1])
    diffusion = CategoricalDiffusion(
        n_node_cls, n_edge_cls, T,
        noise_type="marginal",
        node_marginals=node_marg,
        edge_marginals=edge_marg,
    )

    # 앞으로 가는 과정 시험
    t = torch.randint(0, T, (B,))
    x_t, e_t = diffusion.q_sample(x_0, e_0, t)
    print(f"Clean nodes unique: {x_0[0].unique().tolist()}")
    print(f"Noisy nodes (t={t[0].item()}): {x_t[0].unique().tolist()}")

    # 잡소리 없애개 시험
    denoiser = DiGressDenoiser(
        n_node_cls, n_edge_cls,
        hidden_dim=64, num_layers=2, num_heads=4, num_timesteps=T,
    )
    params = sum(p.numel() for p in denoiser.parameters())
    print(f"\nDenoiser parameters: {params:,}")

    x_logits, e_logits = denoiser(x_t, e_t, t)
    print(f"Node logits: {x_logits.shape}")
    print(f"Edge logits: {e_logits.shape}")

    # 학습 루프
    print("\n=== Training ===")
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)

    for epoch in range(40):
        denoiser.train()
        t = torch.randint(0, T, (B,))
        x_t, e_t = diffusion.q_sample(x_0, e_0, t)

        x_logits, e_logits = denoiser(x_t, e_t, t)

        # 교차 엔트로피 손실
        loss_x = F.cross_entropy(x_logits.view(-1, n_node_cls), x_0.view(-1))
        idx = torch.triu_indices(max_n, max_n, offset=1)
        loss_e = F.cross_entropy(
            e_logits[:, idx[0], idx[1]].reshape(-1, n_edge_cls),
            e_0[:, idx[0], idx[1]].reshape(-1),
        )
        loss = loss_x + loss_e

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f} (node={loss_x.item():.4f}, edge={loss_e.item():.4f})")

    print("\nDone. DiGress training framework operational.")
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
