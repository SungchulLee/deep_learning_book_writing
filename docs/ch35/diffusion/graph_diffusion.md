# 그래프 확산 모델

그래프 확산 모델은 디노이징 확산 틀을 넓혀 그래프 구조 데이터를 만든다. 핵심 생각은 그래프를 차츰 무작위 노이즈로 교란하는 순방향 과정을 정의하고, 이를 스텝마다 되돌리도록 뉴럴 네트워크를 트레인시키는 것이다. 자기회귀나 원샷 생성과 달리 확산 모델은 반복 정제로 그래프를 만든다. 순수한 노이즈에서 시작해 차츰 노이즈를 걷어 내며 올바른 그래프 구조로 다가간다. 이 틀은 순열 불변성을 자연스럽게 다루고 품질 좋은 표본을 내며, 분자와 일반 그래프 생성 지표에서 최고 수준의 결과를 세웠다.

---

## 1. 그래프 확산의 어려움

확산을 그래프에 적용하면 연속 도메인(이미지, 오디오)에서 겪는 것 말고도 고유한 어려움이 생긴다.

**이산적인 구조.** 인접 행렬은 이진값이다. 엣지가 있거나 없다. 일반적인 가우시안 확산은 연속 데이터를 전제하므로, 이산 확산으로 정식화하거나 그래프를 연속값으로 완화해야 한다.

**결합 생성.** 그래프는 노드 클래스 $\mathbf{X}$, 엣지 클래스 $\mathbf{E}$, 어쩌면 엣지 웨이트처럼 서로 의존하는 구성 요소 여럿으로 이루어지며 이들을 일관되게 만들어야 한다. 확산 과정은 이 조각들을 함께 교란하고 함께 노이즈를 제거해야 한다.

**순열 등변성.** 디노이징 네트워크 $\epsilon_\theta(\mathcal{G}_t, t)$는 노드 순열에 등변이어야 한다. 노이즈 섞인 입력의 자리를 바꾸면 예측도 그에 맞게 자리가 바뀌어야 한다. 그래프 뉴럴 네트워크 기반 디노이저는 이 요구를 자연스럽게 만족한다.

---

## 2. 연속값으로 완화하는 길

한 전략은 이산적인 인접 행렬을 연속값으로 완화하고 일반적인 가우시안 확산을 쓴다.

**순방향 과정.** 그래프 $\mathcal{G}_0 = (\mathbf{A}_0, \mathbf{X}_0)$에서 시작해 $T$ 스텝에 걸쳐 가우시안 노이즈를 더한다.

$$
q(\mathcal{G}_t \mid \mathcal{G}_{t-1}) = \mathcal{N}(\mathcal{G}_t; \sqrt{1 - \beta_t} \mathcal{G}_{t-1}, \beta_t \mathbf{I})
$$

여기서 $\beta_t$는 노이즈 스케줄이다. 주변 분포는 닫힌 형태로 다음과 같다.

$$
q(\mathcal{G}_t \mid \mathcal{G}_0) = \mathcal{N}(\mathcal{G}_t; \sqrt{\bar{\alpha}_t} \mathcal{G}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

여기서 $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$이다.

**역방향 과정.** 뉴럴 네트워크가 노이즈 섞인 그래프에서 한 스텝 덜 노이즈 섞인 그래프의 평균을 예측한다.

$$
p_\theta(\mathcal{G}_{t-1} \mid \mathcal{G}_t) = \mathcal{N}(\mathcal{G}_{t-1}; \boldsymbol{\mu}_\theta(\mathcal{G}_t, t), \sigma_t^2 \mathbf{I})
$$

**후처리.** $\mathcal{G}_0$까지 노이즈를 없앤 뒤 연속적인 인접 값에 임곗값을 걸어 이산적인 그래프를 얻는다: $A_{ij} = \mathbf{1}[\hat{A}_{ij} > 0.5]$.

---

## 3. 이산 확산의 길

다른 방법은 범주형(categorical) 확산을 써서 이산적인 그래프 구조 위에서 곧바로 동작한다.

**순방향 과정.** 이산 변수(엣지 존재, 노드 클래스)마다 마르코프 사슬을 따라 전이한다. 이진 엣지 변수에 대해서는 다음과 같다.

$$
q(A_{ij}^{(t)} \mid A_{ij}^{(t-1)}) = \text{Cat}(A_{ij}^{(t)}; \mathbf{Q}_t \cdot \text{onehot}(A_{ij}^{(t-1)}))
$$

여기서 $\mathbf{Q}_t \in \mathbb{R}^{2 \times 2}$는 전이 행렬이다. 흔한 선택은 다음과 같다.

$$
\mathbf{Q}_t = \begin{pmatrix} 1 - \beta_t & \beta_t \\ \beta_t & 1 - \beta_t \end{pmatrix}
$$

이는 엣지 상태를 $\{0, 1\}$ 위의 균일 분포 쪽으로 교란한다.

$t$ 스텝의 주변 분포는 다음과 같다.

$$
q(A_{ij}^{(t)} \mid A_{ij}^{(0)}) = \text{Cat}(A_{ij}^{(t)}; \bar{\mathbf{Q}}_t \cdot \text{onehot}(A_{ij}^{(0)}))
$$

여기서 $\bar{\mathbf{Q}}_t = \prod_{s=1}^{t} \mathbf{Q}_s$이다.

**역방향 과정.** 디노이징 네트워크가 노이즈 섞인 그래프 $\mathcal{G}_t$에서 깨끗한 그래프 $\hat{\mathcal{G}}_0$를 예측한다.

$$
p_\theta(\mathcal{G}_{t-1} \mid \mathcal{G}_t) = \sum_{\hat{\mathcal{G}}_0} q(\mathcal{G}_{t-1} \mid \mathcal{G}_t, \hat{\mathcal{G}}_0) \cdot p_\theta(\hat{\mathcal{G}}_0 \mid \mathcal{G}_t)
$$

사후 분포 $q(\mathcal{G}_{t-1} \mid \mathcal{G}_t, \mathcal{G}_0)$는 베이즈 규칙으로 닫힌 형태로 계산한다.

---

## 4. 디노이징 네트워크의 구조

디노이저는 노이즈 섞인 그래프 $\mathcal{G}_t$와 시간 스텝 $t$를 받아 노이즈 $\boldsymbol{\epsilon}$, 깨끗한 그래프 $\mathcal{G}_0$, 스코어 $\nabla_{\mathcal{G}_t} \log q(\mathcal{G}_t)$ 가운데 하나를 예측한다. 그래프 트랜스포머 구조가 일반적인 선택이다.

$$
\mathbf{H}^{(\ell+1)} = \text{GraphTransformerBlock}^{(\ell)}(\mathbf{H}^{(\ell)}, \mathbf{E}^{(\ell)}, t)
$$

핵심 설계 선택은 다음과 같다.

- **노드 피처**: 노이즈 섞인 노드 클래스를 사인파 시간 임베딩과 이어 붙인다
- **엣지 피처**: 노이즈 섞인 엣지 클래스를 어텐션에서 엣지 속성으로 다룬다
- **전역 조건 부여**: FiLM 조건 부여나 덧셈 바이어스로 시간 임베딩을 모든 레이어에 넣는다
- **출력 헤드**: 따로 둔 다층 뉴럴 네트워크가 노이즈를 없앤 노드 클래스와 엣지 클래스를 예측한다

---

## 5. 그래프의 노이즈 스케줄

노이즈 스케줄 $\beta_1, \ldots, \beta_T$가 교란 속도를 조절한다. 그래프에서 쓰는 스케줄은 다음과 같다.

- **선형 스케줄**: $\beta_t$가 $\beta_1 = 10^{-4}$에서 $\beta_T = 0.02$까지 선형으로 커진다
- **코사인 스케줄**: 더 매끄러운 전이를 위해 $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$
- **다항식 스케줄**: $\bar{\alpha}_t = (1 - t/T)^s$, 지수 $s$는 데이터셋마다 조정한다

이산 확산에서 스케줄은 범주형 변수가 얼마나 빨리 균일 분포에 가까워지는지 조절한다. 빠른 스케줄은 트레인 효율을 높이지만 디노이징을 너무 어렵게 만들 수 있다.

---

## 6. 금융 응용: 확산 기반 네트워크 시나리오

확산 모델은 다음 방법으로 다양하고 품질 좋은 금융 네트워크 시나리오를 만들 수 있다. (1) 과거 네트워크 스냅숏으로 트레인한다, (2) 트레인한 분포에서 샘플링해 그럴듯한 대안 구조를 만든다, (3) 확산의 잠재공간을 거쳐 기존 네트워크 사이를 보간한다. 반복 정제 과정은 자연스럽게 전역적으로 일관된 구조를 내어, 자기회귀와 원샷 생성에서 흔한 국소적 불일치를 피한다.

---

## 7. 구현: 그래프 확산 틀

```python
"""
그래프 확산 모델: 이어진 판과 띄엄띄엄한 판의 앞으로/거꾸로 가는 과정.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class ContinuousGraphDiffusion(nn.Module):
    """
    이어진 값으로 느슨히 한 이웃 행렬의 가우스 확산.
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

        # 잡소리 없애개: 단순한 그래프 뉴럴 네트워크
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
                # nn.ModuleDict에는 update()라는 메서드가 이미 있어서
                # "update"를 열쇠로 쓰면 KeyError가 난다.
                "update_proj": nn.Linear(hidden_dim * 2, hidden_dim),
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

        # 때 임베딩
        t_emb = self.time_embed(t.float().unsqueeze(-1) / self.T)

        # 이웃 행렬의 줄을 마디 피처로 쓴다
        h = self.input_proj(adj_t)  # (B, n, hidden)

        for layer in self.layers:
            # 이웃 행렬과 함께 쪽지 건네기
            adj_norm = adj_t / (adj_t.abs().sum(-1, keepdim=True) + 1)
            msg = torch.bmm(adj_norm, layer["msg"](h))

            # 때 조건과 함께 고치기
            t_bias = layer["time_proj"](t_emb).unsqueeze(1)
            h_new = layer["update_proj"](torch.cat([h, msg], dim=-1)) + t_bias
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
        """익히기 스텝: t을 뽑고 잡소리를 더하고 잡소리를 헤아린다."""
        B = adj_0.size(0)
        device = adj_0.device

        # 아무 때 스텝을 뽑는다
        t = torch.randint(0, self.T, (B,), device=device)

        # 잡음 더하기
        noise = torch.randn_like(adj_0)
        adj_t, _ = self.q_sample(adj_0, t, noise)

        # 잡음을 헤아린다
        noise_pred = self._denoise(adj_t, t)

        # 위쪽 삼각의 평균 제곱 어긋남 로스
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
        """거꿀 확산으로 그래프를 만든다."""
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

        # 임계를 걸어 두 값으로
        graphs = []
        for b in range(num_graphs):
            adj_b = (adj_t[b] > 0).float()
            adj_b = torch.triu(adj_b, diagonal=1)
            adj_b = adj_b + adj_b.t()
            # 외톨이 절를 없앤다
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

    # 트레인 데이터
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

    # 트레인
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

**출력:**

```
=== Graph Diffusion Demo ===

Parameters: 62,189
Epoch 20: loss=1.0021
Epoch 40: loss=0.9799
Epoch 60: loss=1.0280

=== Generation ===
Graph 0: 10 nodes, 20 edges, density=0.444
Graph 1: 10 nodes, 20 edges, density=0.444
Graph 2: 10 nodes, 20 edges, density=0.444
Graph 3: 10 nodes, 18 edges, density=0.400
Graph 4: 10 nodes, 24 edges, density=0.533
Graph 5: 10 nodes, 14 edges, density=0.311
Graph 6: 10 nodes, 21 edges, density=0.467
Graph 7: 10 nodes, 25 edges, density=0.556
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

이 절에서는 그래프 확산의 어려움, 연속값으로 완화하는 길, 이산 확산의 길, 디노이징 네트워크의 구조를 살펴보았다.
