# 그래프 퍼짐 모델
## 개요

그래프의 퍼짐 모델은 잡소리 없애기 퍼짐 틀을 넓혀 그래프 얼개 자료를 만든다. 핵심 생각은 그래프를 차츰 아무 잡소리로 흐트러뜨리는 앞으로 가는 과정을 뜻매김하고 이를 걸음마다 되돌리도록 신경망을 익히는 것이다. 자기 되돌이나 한 번에 만들기와 달리 퍼짐 모델은 되풀이 다듬기로 그래프를 만든다. 순수한 잡소리에서 시작해 차츰 잡소리를 없애 올바른 그래프 얼개로 간다. 이 틀은 자리바꿈에 안 바뀜을 자연스럽게 다루고 품질 좋은 표본을 내며 분자와 일반 그래프 만들기 잣대에서 가장 앞선 결과를 세웠다.

## 그래프 퍼짐의 어려움

퍼짐을 그래프에 맞추면 이어진 마당(그림, 소리)에서 겪는 것 말고도 남다른 어려움이 생긴다:

**띄엄띄엄한 얼개.** 이웃 행렬은 두 값이다. 변이 있거나 없다. 여느 가우스 퍼짐은 이어진 자료를 여기므로 띄엄띄엄 퍼짐으로 적거나 그래프를 이어진 값으로 느슨히 해야 한다.

**함께 만들기.** 그래프는 마디 갈래 $\mathbf{X}$, 변 갈래 $\mathbf{E}$, 어쩌면 변 무게처럼 서로 얽힌 조각 여럿으로 이루어지며 이들을 아귀가 맞게 만들어야 한다. 퍼짐 과정은 이 조각들을 함께 흐트러뜨리고 함께 잡소리를 없애야 한다.

**자리바꿈 같이 바뀜.** 잡소리 없애는 그물 $\epsilon_\theta(\mathcal{G}_t, t)$은 마디 자리바꿈에 같이 바뀌어야 한다. 잡소리 섞인 들임의 자리를 바꾸면 헤아림도 그에 맞게 자리가 바뀌어야 한다. 그래프 신경망 바탕 잡소리 없애개는 이 요구를 자연스럽게 만족한다.

## 이어진 값으로 느슨히 하는 길

한 셈속은 띄엄띄엄한 이웃 행렬을 이어진 값으로 느슨히 하고 여느 가우스 퍼짐을 쓴다.

**앞으로 가는 과정.** 그래프 $\mathcal{G}_0 = (\mathbf{A}_0, \mathbf{X}_0)$에서 시작해 $T$걸음에 걸쳐 가우스 잡소리를 더한다:

$$
q(\mathcal{G}_t \mid \mathcal{G}_{t-1}) = \mathcal{N}(\mathcal{G}_t; \sqrt{1 - \beta_t} \mathcal{G}_{t-1}, \beta_t \mathbf{I})
$$

여기서 $\beta_t$은 잡소리 일정이다. 닫힌 꼴 주변 분포는 다음과 같다:

$$
q(\mathcal{G}_t \mid \mathcal{G}_0) = \mathcal{N}(\mathcal{G}_t; \sqrt{\bar{\alpha}_t} \mathcal{G}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

여기서 $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$이다.

**거꾸로 가는 과정.** 신경망이 잡소리 섞인 판에서 깨끗한 그래프를 헤아린다:

$$
p_\theta(\mathcal{G}_{t-1} \mid \mathcal{G}_t) = \mathcal{N}(\mathcal{G}_{t-1}; \boldsymbol{\mu}_\theta(\mathcal{G}_t, t), \sigma_t^2 \mathbf{I})
$$

**뒷손질.** $\mathcal{G}_0$까지 잡소리를 없앤 뒤 이어진 이웃 값에 문턱을 걸어 띄엄띄엄한 그래프를 얻는다: $A_{ij} = \mathbf{1}[\hat{A}_{ij} > 0.5]$.

## 띄엄띄엄 퍼짐의 길

다른 길은 갈래 퍼짐을 써서 띄엄띄엄한 그래프 얼개에서 곧바로 돈다.

**앞으로 가는 과정.** 띄엄띄엄한 변수(변이 있음, 마디 갈래)마다 마르코프 사슬을 따라 옮겨 간다. 두 값 변 변수에 대해:

$$
q(A_{ij}^{(t)} \mid A_{ij}^{(t-1)}) = \text{Cat}(A_{ij}^{(t)}; \mathbf{Q}_t \cdot \text{onehot}(A_{ij}^{(t-1)}))
$$

여기서 $\mathbf{Q}_t \in \mathbb{R}^{2 \times 2}$은 옮김 행렬이다. 흔한 고름:

$$
\mathbf{Q}_t = \begin{pmatrix} 1 - \beta_t & \beta_t \\ \beta_t & 1 - \beta_t \end{pmatrix}
$$

이는 변 상태를 $\{0, 1\}$의 고른 분포 쪽으로 고르게 흐트러뜨린다.

$t$걸음 주변 분포는 다음과 같다:

$$
q(A_{ij}^{(t)} \mid A_{ij}^{(0)}) = \text{Cat}(A_{ij}^{(t)}; \bar{\mathbf{Q}}_t \cdot \text{onehot}(A_{ij}^{(0)}))
$$

여기서 $\bar{\mathbf{Q}}_t = \prod_{s=1}^{t} \mathbf{Q}_s$이다.

**거꾸로 가는 과정.** 잡소리 없애는 그물이 잡소리 섞인 그래프 $\mathcal{G}_t$에서 깨끗한 그래프 $\hat{\mathcal{G}}_0$을 헤아린다:

$$
p_\theta(\mathcal{G}_{t-1} \mid \mathcal{G}_t) = \sum_{\hat{\mathcal{G}}_0} q(\mathcal{G}_{t-1} \mid \mathcal{G}_t, \hat{\mathcal{G}}_0) \cdot p_\theta(\hat{\mathcal{G}}_0 \mid \mathcal{G}_t)
$$

사후 분포 $q(\mathcal{G}_{t-1} \mid \mathcal{G}_t, \mathcal{G}_0)$은 베이즈 규칙으로 닫힌 꼴로 셈한다.

## 잡소리 없애는 그물의 얼개

잡소리 없애개는 잡소리 섞인 그래프 $\mathcal{G}_t$과 때 걸음 $t$을 다루어 잡소리 $\boldsymbol{\epsilon}$, 깨끗한 그래프 $\mathcal{G}_0$, 또는 점수 $\nabla_{\mathcal{G}_t} \log q(\mathcal{G}_t)$ 가운데 하나를 헤아려야 한다. 그래프 트랜스포머 얼개가 여느 고름이다:

$$
\mathbf{H}^{(\ell+1)} = \text{GraphTransformerBlock}^{(\ell)}(\mathbf{H}^{(\ell)}, \mathbf{E}^{(\ell)}, t)
$$

핵심 설계 선택은 다음과 같다.

- **마디 특징**: 잡소리 섞인 마디 갈래를 사인 꼴 때 박아 넣기와 이어 붙인다
- **변 특징**: 잡소리 섞인 변 갈래를 눈길에서 변 속성으로 다룬다
- **온 자리 조건 주기**: FiLM 조건 주기나 더하기 치우침으로 때 박아 넣기를 모든 층에 더한다
- **내놓기 머리**: 따로 둔 여러 층 신경망이 잡소리를 없앤 마디 갈래와 변 갈래를 헤아린다

## 그래프의 잡소리 일정

잡소리 일정 $\beta_1, \ldots, \beta_T$이 흐트러지는 빠르기를 다스린다. 그래프에서는:

- **선형 일정**: $\beta_t$이 $\beta_1 = 10^{-4}$에서 $\beta_T = 0.02$까지 선형으로 는다
- **코사인 일정**: 더 매끄러운 옮김을 위해 $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$
- **다항식 일정**: 자료 뭉치마다 지수 $s$을 손보는 $\bar{\alpha}_t = (1 - t/T)^s$

띄엄띄엄 퍼짐에서 일정은 갈래 변수가 얼마나 빨리 고른 분포에 가까워지는지 다스린다. 빠른 일정은 익히기 효율을 높이지만 잡소리 없애기를 너무 어렵게 할 수 있다.

## 금융 쓰임새: 퍼짐 바탕 그물 상황

퍼짐 모델은 다음으로 다채롭고 품질 좋은 금융 그물 상황을 만들 수 있다. (1) 지난 그물 스냅숏으로 익히기, (2) 배운 분포에서 뽑아 그럴듯한 다른 짜임 내놓기, (3) 퍼짐의 숨은 자리를 거쳐 기존 그물 사이를 사이 값으로 잇기. 되풀이 다듬기 과정은 자연스럽게 온 자리에서 아귀가 맞는 얼개를 내어, 자기 되돌이와 한 번에 만들기에 흔한 그 자리의 어긋남을 피한다.

## 짜기: 그래프 퍼짐 틀

```python
"""
그래프 퍼짐 모델: 이어진 판과 띄엄띄엄한 판의 앞으로/거꾸로 가는 과정.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ContinuousGraphDiffusion(nn.Module):
    """
    이어진 값으로 느슨히 한 이웃 행렬의 가우스 퍼짐.
    """

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_timesteps: int = 100,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.T = num_timesteps

        # 잡소리 일정
        if schedule == "cosine":
            s = 0.008
            steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
            alpha_bar = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = betas.clamp(max=0.999).float()
        else:
            betas = torch.linspace(1e-4, 0.02, num_timesteps)

        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1 - alpha_bar))

        # 잡소리 없애개: 단순한 그래프 신경망
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(max_nodes, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "msg": nn.Linear(hidden_dim, hidden_dim),
                "update": nn.Linear(hidden_dim * 2, hidden_dim),
                "norm": nn.LayerNorm(hidden_dim),
                "time_proj": nn.Linear(hidden_dim, hidden_dim),
            }))

        # 내놓기: 위쪽 삼각 이웃 행렬의 잡소리를 헤아린다
        n_edges = max_nodes * (max_nodes - 1) // 2
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_edges),
        )

    def q_sample(
        self,
        adj_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """앞으로 가는 과정: 깨끗한 이웃 행렬에 잡소리를 더한다."""
        if noise is None:
            noise = torch.randn_like(adj_0)

        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

        adj_t = sqrt_ab * adj_0 + sqrt_omab * noise
        return adj_t, noise

    def _denoise(
        self,
        adj_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """잡소리 섞인 이웃 행렬에서 잡소리를 헤아린다."""
        B = adj_t.size(0)

        # 때 박아 넣기
        t_emb = self.time_embed(t.float().unsqueeze(-1) / self.T)

        # 이웃 행렬의 줄을 마디 특징으로 쓴다
        h = self.input_proj(adj_t)  # (B, n, hidden)

        for layer in self.layers:
            # 이웃 행렬과 함께 쪽지 건네기
            adj_norm = adj_t / (adj_t.abs().sum(-1, keepdim=True) + 1)
            msg = torch.bmm(adj_norm, layer["msg"](h))

            # 때 조건과 함께 고치기
            t_bias = layer["time_proj"](t_emb).unsqueeze(1)
            h_new = layer["update"](torch.cat([h, msg], dim=-1)) + t_bias
            h = layer["norm"](h_new + h)

        # 위쪽 삼각 변의 잡소리를 헤아린다
        h_graph = h.sum(dim=1)  # (B, hidden)
        noise_pred = self.output_proj(h_graph)  # (B, n_edges)

        # 온 잡소리 행렬을 되짓는다
        n = self.max_nodes
        noise_matrix = torch.zeros(B, n, n, device=adj_t.device)
        idx = torch.triu_indices(n, n, offset=1)
        noise_matrix[:, idx[0], idx[1]] = noise_pred
        noise_matrix = noise_matrix + noise_matrix.transpose(1, 2)

        return noise_matrix

    def forward(self, adj_0: torch.Tensor) -> dict[str, torch.Tensor]:
        """익히기 걸음: t을 뽑고 잡소리를 더하고 잡소리를 헤아린다."""
        B = adj_0.size(0)
        device = adj_0.device

        # 아무 때 걸음을 뽑는다
        t = torch.randint(0, self.T, (B,), device=device)

        # 잡음 더하기
        noise = torch.randn_like(adj_0)
        adj_t, _ = self.q_sample(adj_0, t, noise)

        # 잡음을 헤아린다
        noise_pred = self._denoise(adj_t, t)

        # 위쪽 삼각의 평균 제곱 어긋남 손실
        n = self.max_nodes
        idx = torch.triu_indices(n, n, offset=1)
        loss = F.mse_loss(
            noise_pred[:, idx[0], idx[1]],
            noise[:, idx[0], idx[1]],
        )

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        num_graphs: int = 1,
        device: str = "cpu",
    ) -> list[torch.Tensor]:
        """거꿀 퍼짐으로 그래프를 만든다."""
        self.eval()
        n = self.max_nodes

        # 순수 잡음에서 시작한다
        adj_t = torch.randn(num_graphs, n, n, device=device)
        adj_t = (adj_t + adj_t.transpose(1, 2)) / 2  # 맞섬으로 만든다
        adj_t.diagonal(dim1=1, dim2=2).zero_()

        for t_idx in reversed(range(self.T)):
            t = torch.full((num_graphs,), t_idx, device=device, dtype=torch.long)

            # 잡음을 헤아린다
            noise_pred = self._denoise(adj_t, t)

            # DDPM 고치기
            alpha = self.alphas[t_idx]
            alpha_bar = self.alpha_bar[t_idx]
            beta = self.betas[t_idx]

            mean = (1 / alpha.sqrt()) * (
                adj_t - (beta / (1 - alpha_bar).sqrt()) * noise_pred
            )

            if t_idx > 0:
                noise = torch.randn_like(adj_t)
                noise = (noise + noise.transpose(1, 2)) / 2
                noise.diagonal(dim1=1, dim2=2).zero_()
                adj_t = mean + beta.sqrt() * noise
            else:
                adj_t = mean

        # 문턱을 걸어 두 값으로
        graphs = []
        for b in range(num_graphs):
            adj_b = (adj_t[b] > 0).float()
            adj_b = torch.triu(adj_b, diagonal=1)
            adj_b = adj_b + adj_b.t()
            # 외톨이 마디를 없앤다
            active = adj_b.sum(1) > 0
            if active.sum() < 2:
                active[:2] = True
            idx = torch.where(active)[0]
            graphs.append(adj_b[idx][:, idx].cpu())

        return graphs


if __name__ == "__main__":
    torch.manual_seed(42)
    max_n = 10
    T = 50

    # 학습 데이터
    print("=== Graph Diffusion Demo ===\n")
    train_adjs = []
    for _ in range(200):
        n = torch.randint(5, max_n + 1, (1,)).item()
        adj = torch.zeros(max_n, max_n)
        for i in range(n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.25:
                    adj[i, j] = adj[j, i] = 1
        train_adjs.append(adj)
    train_adjs = torch.stack(train_adjs)

    # 학습
    model = ContinuousGraphDiffusion(
        max_nodes=max_n, hidden_dim=64,
        num_layers=3, num_timesteps=T,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(60):
        model.train()
        idx = torch.randperm(len(train_adjs))[:32]
        result = model(train_adjs[idx])
        optimizer.zero_grad()
        result["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss={result['loss'].item():.4f}")

    # 생성
    print("\n=== Generation ===")
    generated = model.generate(num_graphs=8)
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
