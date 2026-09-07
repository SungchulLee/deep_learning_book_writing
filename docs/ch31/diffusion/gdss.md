# GDSS: 확률 미분 방정식으로 하는 점수 바탕 그래프 퍼짐
## 개요

GDSS(Jo et al., 2022)은 확률 미분 방정식으로 하는 점수 바탕 짓기 나타내기를 그래프 만들기에 쓴다. 퍼짐을 $T$걸음으로 띄엄띄엄하게 하는 대신 앞으로 가는 과정과 거꾸로 가는 과정을 이어진 때의 확률 미분 방정식으로 적어 너그럽게 뽑을 수 있게 한다. 이 모델은 얽힌 확률 미분 방정식으로 마디 특징과 이웃 행렬을 함께 만들며 점수 그물이 함께 점수 함수를 어림한다.

## 이어진 때의 틀

### 앞으로 가는 확률 미분 방정식

앞으로 가는 과정은 얽힌 VP-SDE으로 $\mathcal{G}_0 = (\mathbf{X}_0, \mathbf{A}_0)$을 흐트러뜨린다:

$$
d\mathbf{X} = -\tfrac{1}{2}\beta(t)\mathbf{X}\,dt + \sqrt{\beta(t)}\,d\mathbf{W}_X, \quad d\mathbf{A} = -\tfrac{1}{2}\beta(t)\mathbf{A}\,dt + \sqrt{\beta(t)}\,d\mathbf{W}_A
$$

때 $T$에서 주변 분포가 $\mathcal{N}(\mathbf{0}, \mathbf{I})$에 가까워진다.

### 거꾸로 가는 확률 미분 방정식

만들기는 점수 $\nabla \log p_t$을 써서 과정을 거꾸로 돌린다:

$$
d\mathbf{X} = \left[-\mathbf{f}_X + g_X^2 \nabla_{\mathbf{X}} \log p_t(\mathbf{X}, \mathbf{A})\right] dt + g_X\,d\bar{\mathbf{W}}_X
$$

$$
d\mathbf{A} = \left[-\mathbf{f}_A + g_A^2 \nabla_{\mathbf{A}} \log p_t(\mathbf{X}, \mathbf{A})\right] dt + g_A\,d\bar{\mathbf{W}}_A
$$

함께 점수 $\nabla_{\mathbf{X}} \log p_t(\mathbf{X}, \mathbf{A})$과 $\nabla_{\mathbf{A}} \log p_t(\mathbf{X}, \mathbf{A})$이 마디와 변을 얽는다. 마디 특징의 잡소리 없애기가 이웃 관계에 매이고 그 반대도 마찬가지다.

## 점수 맞추기 손실

익히기는 잡소리 없애기 점수 맞추기를 가장 작게 한다:

$$
\mathcal{L} = \mathbb{E}_{t, \mathcal{G}_0, \boldsymbol{\epsilon}} \left[ \lambda_X(t) \left\| \mathbf{s}_\theta^X(\mathcal{G}_t, t) + \frac{\boldsymbol{\epsilon}_X}{\sqrt{1-\bar{\alpha}_t}} \right\|^2 + \lambda_A(t) \left\| \mathbf{s}_\theta^A(\mathcal{G}_t, t) + \frac{\boldsymbol{\epsilon}_A}{\sqrt{1-\bar{\alpha}_t}} \right\|^2 \right]
$$

여기서 $\lambda(t) = g(t)^2$은 중요도 무게이고 과녁 점수는 VP-SDE 아래 닫힌 꼴 풀이를 가진다.

## 헤아리개-바로잡개 뽑기

GDSS은 헤아리개 걸음과 바로잡개 걸음을 번갈아 쓴다:

**헤아리개**(오일러-마루야마로 하는 거꿀 확률 미분 방정식 걸음):

$$
\mathbf{X}_{t-\Delta t} = \mathbf{X}_t + [-\mathbf{f}_X + g_X^2 \mathbf{s}_\theta^X]\Delta t + g_X\sqrt{\Delta t}\,\boldsymbol{\epsilon}
$$

**바로잡개**(랑주뱅 마르코프 사슬 몬테카를로 다듬기):

$$
\mathbf{X}_t \leftarrow \mathbf{X}_t + \tfrac{\eta}{2}\mathbf{s}_\theta^X(\mathbf{X}_t, \mathbf{A}_t, t) + \sqrt{\eta}\,\boldsymbol{\epsilon}
$$

걸음마다 $\mathbf{A}_t \leftarrow (\mathbf{A}_t + \mathbf{A}_t^\top)/2$으로 쏘아 맞섬을 지킨다.

## 점수 그물의 얼개

점수 그물은 마디 특징과 이웃 관계를 함께 다루는 그래프 신경망으로 차원이 맞는 점수 어림을 낸다. 이 얼개는 때 걸음 $t$의 잡소리 섞인 그래프를 다루어 마디 $\mathbf{s}_\theta^X \in \mathbb{R}^{n \times d}$과 변 $\mathbf{s}_\theta^A \in \mathbb{R}^{n \times n}$의 점수 어림을 따로 내놓는다.

## 띄엄띄엄 퍼짐보다 나은 점

GDSS은 이어진 자리에서 돌아 기울기 바탕 뽑기를 할 수 있고 띄엄띄엄한 거꿀 옮김의 얽음 어려움을 피한다. 확률 미분 방정식 틀은 정해진 뽑기와 가능도 셈을 위한 확률 흐름 상미분 방정식도 쓸 수 있게 한다:

$$
d\mathbf{X} = \left[-\mathbf{f}_X + \frac{1}{2}g_X^2 \nabla_{\mathbf{X}} \log p_t\right] dt
$$

## 금융 쓰임새: 이어진 그물의 바뀜

확률 미분 방정식 틀은 금융 그물이 이어져 바뀌는 모습을 자연스럽게 나타낸다. 노출을 나타내는 변 무게가 때에 따라 이어져 바뀌고, 흘러감과 퍼짐으로 쪼개면 그물 위상의 몸에 밴 흐름과 확률 흔들림이 갈라진다.

## 짜기: GDSS

```python
"""
GDSS: 확률 미분 방정식으로 하는 점수 바탕 그래프 만들기.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VPSDE:
    """그래프 퍼짐의 흩어짐 지키는 확률 미분 방정식."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        log_alpha = -0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2)
        return torch.exp(log_alpha)

    def marginal_params(self, t: torch.Tensor):
        """q(x_t | x_0)의 평균 계수와 표준편차를 돌려준다."""
        ab = self.alpha_bar(t)
        mean_coeff = torch.sqrt(ab)
        std = torch.sqrt(1 - ab)
        return mean_coeff, std

    def sample_marginal(self, x_0, t, noise=None):
        """Sample x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        mean_coeff, std = self.marginal_params(t)
        # 때 차원을 퍼뜨린다
        while mean_coeff.dim() < x_0.dim():
            mean_coeff = mean_coeff.unsqueeze(-1)
            std = std.unsqueeze(-1)
        return mean_coeff * x_0 + std * noise, noise


class ScoreNetworkGNN(nn.Module):
    """
    마디와 변의 점수를 함께 어림하는 그래프 신경망 바탕 점수 그물.
    """

    def __init__(
        self,
        max_nodes: int,
        node_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feat_dim = node_feat_dim

        # 때 박아 넣기
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 마디 특징 쏘기
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        # 이웃 행렬 쏘기(줄을 특징으로 본다)
        self.adj_proj = nn.Linear(max_nodes, hidden_dim)

        # 그래프 신경망 층
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(nn.ModuleDict({
                "node_msg": nn.Linear(hidden_dim, hidden_dim),
                "node_update": nn.Linear(hidden_dim * 3, hidden_dim),
                "edge_update": nn.Linear(hidden_dim * 3, hidden_dim),
                "norm_n": nn.LayerNorm(hidden_dim),
                "norm_e": nn.LayerNorm(hidden_dim),
            }))

        # 내놓는 머리
        self.score_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_feat_dim),
        )
        self.score_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        a_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        함께 점수를 어림한다.
        
        인수:
            x_t: (B, n, d) 잡소리 섞인 마디 특징
            a_t: (B, n, n) 잡소리 섞인 이웃 행렬
            t: (B,) time in [0, 1]
        반환값:
            score_x: (B, n, d) 마디 점수
            score_a: (B, n, n) 이웃 점수
        """
        B, n, d = x_t.shape

        t_emb = self.time_mlp(t.unsqueeze(-1))  # (B, hidden)
        h_node = self.node_proj(x_t)  # (B, n, hidden)
        h_edge = self.adj_proj(a_t)  # (B, n, hidden) -- 마디마다 제 이웃 줄을 받는다

        for layer in self.gnn_layers:
            # 잡소리 섞인 이웃 행렬을 무게로 쓰는 쪽지 건네기
            a_norm = a_t / (a_t.abs().sum(-1, keepdim=True).clamp(min=1))
            msg = torch.bmm(a_norm, layer["node_msg"](h_node))

            # 때 조건과 함께 마디 고치기
            h_node_new = layer["node_update"](
                torch.cat([h_node, msg, t_emb.unsqueeze(1).expand(-1, n, -1)], dim=-1)
            )
            h_node = layer["norm_n"](h_node + h_node_new)

            # 변 고치기
            h_i = h_node.unsqueeze(2).expand(-1, -1, n, -1)
            h_j = h_node.unsqueeze(1).expand(-1, n, -1, -1)
            h_edge_expanded = h_edge.unsqueeze(2).expand(-1, -1, n, -1)
            e_input = torch.cat([h_i, h_j, h_edge_expanded], dim=-1)
            h_edge_new = layer["edge_update"](e_input).mean(dim=2)
            h_edge = layer["norm_e"](h_edge + h_edge_new)

        # 내놓기 점수
        score_x = self.score_x(h_node)  # (B, n, d)

        # 짝별 마디 특징에서 얻는 변 점수
        h_i = h_node.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h_node.unsqueeze(1).expand(-1, n, -1, -1)
        h_pair = (h_i + h_j) / 2  # 맞섬 합침
        score_a = self.score_a(h_pair).squeeze(-1)  # (B, n, n)
        # 맞섬을 지킨다
        score_a = (score_a + score_a.transpose(1, 2)) / 2
        score_a.diagonal(dim1=1, dim2=2).zero_()

        return score_x, score_a


class GDSS(nn.Module):
    """
    온전한 GDSS 모델: 확률 미분 방정식 바탕 그래프 만들기.
    """

    def __init__(
        self,
        max_nodes: int,
        node_feat_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feat_dim = node_feat_dim
        self.sde = VPSDE(beta_min, beta_max)
        self.score_net = ScoreNetworkGNN(
            max_nodes, node_feat_dim, hidden_dim, num_layers
        )

    def forward(
        self,
        x_0: torch.Tensor,
        a_0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        익히기 걸음: 점수 맞추기 손실.
        
        인수:
            x_0: (B, n, d) 깨끗한 마디 특징
            a_0: (B, n, n) 깨끗한 이웃 행렬
        """
        B = x_0.size(0)
        device = x_0.device

        # 때를 고르게 뽑는다
        t = torch.rand(B, device=device) * 0.999 + 0.001  # t=0을 피한다

        # 잡소리 섞인 자료를 뽑는다
        noise_x = torch.randn_like(x_0)
        noise_a = torch.randn_like(a_0)
        noise_a = (noise_a + noise_a.transpose(1, 2)) / 2
        noise_a.diagonal(dim1=1, dim2=2).zero_()

        x_t, _ = self.sde.sample_marginal(x_0, t, noise_x)
        a_t, _ = self.sde.sample_marginal(a_0, t, noise_a)

        # 점수를 헤아린다
        score_x, score_a = self.score_net(x_t, a_t, t)

        # 과녁: -잡소리 / 표준 어긋남
        _, std = self.sde.marginal_params(t)
        while std.dim() < noise_x.dim():
            std = std.unsqueeze(-1)

        target_x = -noise_x / std
        std_2d = std.squeeze(-1) if std.dim() > noise_a.dim() else std
        while std_2d.dim() < noise_a.dim():
            std_2d = std_2d.unsqueeze(-1)
        target_a = -noise_a / std_2d

        # 무게 매긴 평균 제곱 어긋남 손실
        beta_t = self.sde.beta(t)
        weight = beta_t
        while weight.dim() < score_x.dim():
            weight = weight.unsqueeze(-1)

        loss_x = (weight * (score_x - target_x) ** 2).mean()

        weight_2d = beta_t
        while weight_2d.dim() < score_a.dim():
            weight_2d = weight_2d.unsqueeze(-1)
        n = self.max_nodes
        idx = torch.triu_indices(n, n, offset=1)
        loss_a = (weight_2d * (score_a - target_a) ** 2)[:, idx[0], idx[1]].mean()

        return {"loss": loss_x + loss_a, "loss_x": loss_x, "loss_a": loss_a}

    @torch.no_grad()
    def generate(
        self,
        num_graphs: int = 1,
        num_steps: int = 100,
        corrector_steps: int = 1,
        snr: float = 0.1,
        device: str = "cpu",
    ) -> list[torch.Tensor]:
        """헤아리개-바로잡개 뽑기."""
        self.eval()
        n = self.max_nodes

        # 잡음에서 시작한다
        x_t = torch.randn(num_graphs, n, self.node_feat_dim, device=device)
        a_t = torch.randn(num_graphs, n, n, device=device)
        a_t = (a_t + a_t.transpose(1, 2)) / 2
        a_t.diagonal(dim1=1, dim2=2).zero_()

        dt = 1.0 / num_steps
        times = torch.linspace(1.0, 0.001, num_steps, device=device)

        for i, t_val in enumerate(times):
            t = torch.full((num_graphs,), t_val, device=device)
            score_x, score_a = self.score_net(x_t, a_t, t)

            beta = self.sde.beta(t)

            # --- 바로잡개(랑주뱅) ---
            for _ in range(corrector_steps):
                noise_x = torch.randn_like(x_t) * snr
                noise_a = torch.randn_like(a_t) * snr
                noise_a = (noise_a + noise_a.transpose(1, 2)) / 2
                noise_a.diagonal(dim1=1, dim2=2).zero_()

                step_size = snr ** 2
                s_x, s_a = self.score_net(x_t, a_t, t)
                x_t = x_t + 0.5 * step_size * s_x + math.sqrt(step_size) * noise_x
                a_t = a_t + 0.5 * step_size * s_a + math.sqrt(step_size) * noise_a
                a_t = (a_t + a_t.transpose(1, 2)) / 2
                a_t.diagonal(dim1=1, dim2=2).zero_()

            # --- 헤아리개(거꿀 확률 미분 방정식) ---
            drift_x = 0.5 * beta.view(-1, 1, 1) * (x_t + score_x)
            drift_a = 0.5 * beta.view(-1, 1, 1) * (a_t + score_a)

            noise_x = torch.randn_like(x_t)
            noise_a = torch.randn_like(a_t)
            noise_a = (noise_a + noise_a.transpose(1, 2)) / 2
            noise_a.diagonal(dim1=1, dim2=2).zero_()

            diffusion = torch.sqrt(beta * dt).view(-1, 1, 1)
            x_t = x_t - drift_x * dt + diffusion * noise_x
            a_t = a_t - drift_a * dt + diffusion * noise_a
            a_t = (a_t + a_t.transpose(1, 2)) / 2
            a_t.diagonal(dim1=1, dim2=2).zero_()

        # 이웃 행렬에 문턱을 건다
        graphs = []
        for b in range(num_graphs):
            adj = (a_t[b] > 0).float()
            adj = torch.triu(adj, diagonal=1)
            adj = adj + adj.t()
            active = adj.sum(1) > 0
            if active.sum() < 2:
                active[:2] = True
            idx = torch.where(active)[0]
            graphs.append(adj[idx][:, idx].cpu())

        return graphs


if __name__ == "__main__":
    torch.manual_seed(42)
    max_n, feat_dim = 8, 2

    print("=== GDSS Demo ===\n")

    # 학습 데이터
    train_x, train_a = [], []
    for _ in range(100):
        n = torch.randint(4, max_n + 1, (1,)).item()
        x = torch.randn(max_n, feat_dim)
        x[n:] = 0
        adj = torch.zeros(max_n, max_n)
        for i in range(n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.3:
                    adj[i, j] = adj[j, i] = 1
        train_x.append(x)
        train_a.append(adj)

    train_x = torch.stack(train_x)
    train_a = torch.stack(train_a)

    # 학습
    model = GDSS(max_n, feat_dim, hidden_dim=64, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(50):
        model.train()
        idx = torch.randperm(100)[:32]
        result = model(train_x[idx], train_a[idx])
        optimizer.zero_grad()
        result["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={result['loss'].item():.4f} "
                  f"(x={result['loss_x'].item():.4f}, a={result['loss_a'].item():.4f})")

    # 생성
    print("\n=== Generation ===")
    generated = model.generate(num_graphs=8, num_steps=50, corrector_steps=1)
    for i, g in enumerate(generated):
        n, e = g.size(0), int(g.sum().item()) // 2
        print(f"Graph {i}: {n} nodes, {e} edges, density={2*e/(n*(n-1)) if n>1 else 0:.3f}")
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
