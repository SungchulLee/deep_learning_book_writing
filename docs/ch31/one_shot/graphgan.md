# GraphGAN: 맞수 그래프 만들기

GraphGAN은 맞수 짓기 그물 틀을 그래프 만들기에 쓴다. 만들개는 그럴듯한 그래프 얼개를 내도록 익히고 가름개는 만든 그래프와 참 그래프를 가려내도록 배운다. 되짓기 목표를 가장 좋게 하는 GraphVAE과 달리 GraphGAN은 숨은 밀도 모형을 쓴다. 만들개는 $p_\theta(\mathcal{G})$을 드러나게 셈하지 않고 맞수 되먹임으로 잡소리를 그래프 같은 얼개로 바꾸는 법을 배운다. 이로써 더 또렷하고 그럴듯한 내놓기가 나오지만 GAN 익히기의 잘 알려진 어려움, 곧 갈래 무너짐, 익히기 불안정, 모임 판정의 어려움이 따라온다.

---

## 1. 구조

### 만들개

만들개는 숨은 벡터 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$을 그래프 $\hat{\mathcal{G}} = (\hat{\mathbf{A}}, \hat{\mathbf{X}})$으로 옮긴다:

$$
\hat{\mathbf{A}}, \hat{\mathbf{X}} = G_\theta(\mathbf{z})
$$

만들개는 보통 여러 층 신경망이나 인수로 나눈 얼개로 이어진 변 확률 $\hat{\mathbf{A}} \in [0,1]^{n \times n}$을 낸다. 가름개에 줄 띄엄띄엄한 그래프를 얻으려면 이어진 내놓기를 띄엄띄엄하게 해야 한다. 길 셋이 쓰인다:

**굳은 문턱 걸기.** $A_{ij} = \mathbf{1}[\hat{A}_{ij} > 0.5]$. 단순하지만 미분할 수 없다. 기울기가 문턱을 지나 흐르지 못한다.

**곧바로 지나는 어림개.** 앞으로 갈 때는 굳은 문턱을 쓰되 기울기는 그 연산이 항등인 것처럼 지나가게 한다:

$$
A_{ij}^{\text{forward}} = \mathbf{1}[\hat{A}_{ij} > 0.5], \quad \frac{\partial \mathcal{L}}{\partial \hat{A}_{ij}} \approx \frac{\partial \mathcal{L}}{\partial A_{ij}^{\text{forward}}}
$$

**검벨-소프트맥스로 느슨히 하기.** 검벨-소프트맥스 재주로 띄엄띄엄한 뽑기를 이어진 어림으로 바꾼다:

$$
A_{ij} = \sigma\left(\frac{\log \hat{A}_{ij} - \log(1 - \hat{A}_{ij}) + g_1 - g_0}{\tau}\right)
$$

여기서 $g_0, g_1 \sim \text{Gumbel}(0,1)$이고 $\tau > 0$은 온도 잡이다. $\tau \to 0$이면 느슨히 한 것이 띄엄띄엄한 뽑기에 가까워진다.

### 가름개

가름개 $D_\phi(\mathcal{G})$은 그래프를 받아 그것이 참일 낱값 확률을 내놓는다. 그래프 신경망 바탕 가름개는 쪽지 건네기 층으로 그래프를 다루어 그래프 켜의 점수로 모은다:

$$
D_\phi(\mathcal{G}) = \sigma\left(\text{MLP}\left(\text{READOUT}(\text{GNN}_\phi(\mathbf{A}, \mathbf{X}))\right)\right)
$$

가름개는 자리바꿈에 안 바뀌어야 한다. 마디 차례와 상관없이 같은 꼴 그래프에 같은 점수를 주어야 한다. 자리바꿈에 안 바뀌는 읽어내기(합, 평균, 눈길 바탕 모으기)를 쓰는 그래프 신경망이 이를 자연스럽게 보장한다.

---

## 2. 익히기 목표

그래프에 맞춘 여느 GAN 목표:

$$
\min_\theta \max_\phi \; \mathbb{E}_{\mathcal{G} \sim p_{\text{data}}} [\log D_\phi(\mathcal{G})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} [\log(1 - D_\phi(G_\theta(\mathbf{z})))]
$$

실제로는 안정을 위해 기울기 벌을 더한 바서슈타인 GAN 목표를 즐겨 쓴다:

$$
\min_\theta \max_\phi \; \mathbb{E}_{\mathcal{G} \sim p_{\text{data}}} [D_\phi(\mathcal{G})] - \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} [D_\phi(G_\theta(\mathbf{z}))] - \lambda \mathbb{E}_{\hat{\mathcal{G}}} \left[(\|\nabla_{\hat{\mathcal{G}}} D_\phi(\hat{\mathcal{G}})\|_2 - 1)^2\right]
$$

여기서 $\hat{\mathcal{G}}$은 참 그래프와 만든 그래프 사이의 사이 값이고 $\lambda$은 기울기 벌의 세기를 다스린다.

---

## 3. 조건 만들어 내기

GraphGAN은 바라는 그래프 성질 $\mathbf{c}$(보기로 마디 수, 과녁 차수 분포, 마당 특유의 매임)에 조건을 걸 수 있다:

$$
\hat{\mathcal{G}} = G_\theta(\mathbf{z}, \mathbf{c})
$$

가름개도 그 조건을 받는다:

$$
D_\phi(\mathcal{G}, \mathbf{c}) \to [0,1]
$$

이로써 겨냥한 만들기가 가능하다. 보기로 밝힌 빽빽함이나 차수 분포를 가진 금융 그물을 만든다.

---

## 4. 그래프 GAN 특유의 어려움

**띄엄띄엄한 얼개.** 근본 어려움은 그래프가 띄엄띄엄한 대상이라 만들개를 기울기로 가장 좋게 하기가 어렵다는 것이다. 검벨-소프트맥스로 느슨히 하면 도움이 되지만 온도에 매인 어림 어긋남이 생긴다.

**자리바꿈에 안 바뀜.** 가름개가 마디 차례에 안 바뀌어야 하므로 얼개 고름이 좁아진다. 그래프 신경망 바탕 가름개가 여느 풀이다.

**갈래 무너짐.** 그래프 GAN은 몇 가지 그래프 갈래만 만들기 쉽다. 과녁 분포에 서로 다른 그래프 무리(보기로 다른 무리 얼개)가 여럿 들어 있을 때 특히 말썽이다.

**따지기의 어려움.** 눈으로 보아 빠르게 되먹임을 얻는 그림 GAN과 달리 그래프 GAN의 품질을 따지려면 큰 표본 모임에 대해 통계 잣대(MMD, FGD)를 셈해야 한다.

---

## 5. GraphVAE과 견주기

| 갈래 | GraphVAE | GraphGAN |
|--------|----------|----------|
| 익히기 신호 | 되짓기 + 쿨백-라이블러 | 맞수 |
| 밀도 | 드러남(다룰 수 있는 증거 아래 가둠) | 숨음 |
| 내놓기 품질 | 흐릿함(평균 냄) | 더 또렷함 |
| 익히기 안정성 | 안정 | 조심스러운 손보기가 필요 |
| 갈래 덮음 | 좋음(쿨백-라이블러가 무너짐을 막음) | 갈래 무너짐이 잦음 |
| 숨은 자리 | 짜임 있음(사이 값을 낼 수 있음) | 짜임 없음 |

---

## 6. 금융 쓰임새: 맞수 버거움 시험

GraphGAN의 맞수 틀은 버거움 시험에 돌려 쓸 수 있다. 그럴듯하면서도(가름개를 지나면서도) 주어진 위험 모델에 가장 버거운 금융 그물 짜임을 내도록 만들개를 익힌다. 이 맞수 버거움 시험은 가장 나쁘면서도 그럴듯한 상황을 만들어 여느 몬테카를로 버거움 시험보다 튼튼함을 더 세게 보장한다.

---

## 7. 짜기: WGAN-GP을 쓴 GraphGAN

```python
"""
GraphGAN: WGAN-GP 익히기로 하는 맞수 그래프 만들기.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

class GraphGenerator(nn.Module):
    """만들개: 잡음 -> 그래프 이웃 행렬."""

    def __init__(
        self,
        latent_dim: int,
        max_nodes: int,
        hidden_dim: int = 256,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.temperature = temperature
        n_edges = max_nodes * (max_nodes - 1) // 2

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_edges),
        )

    def forward(self, z: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """
        인수:
            z: (B, latent_dim) 잡소리
            hard: True이면 hard=True인 검벨 소프트맥스를 쓴다
        반환값:
            adj: (B, n, n) 이웃 행렬(부드럽거나 딱딱한 것)
        """
        B = z.size(0)
        n = self.max_nodes
        logits = self.net(z)  # (B, n_edges)

        # 변마다 검벨-소프트맥스로 느슨히 하기
        if self.training:
            # 검벨 잡소리를 뽑는다
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            gumbel = -torch.log(-torch.log(u))
            edge_probs = torch.sigmoid(
                (logits + gumbel) / self.temperature
            )
        else:
            edge_probs = torch.sigmoid(logits)

        if hard:
            edge_hard = (edge_probs > 0.5).float()
            edge_probs = edge_hard - edge_probs.detach() + edge_probs

        # 맞섬 이웃 행렬을 짓는다
        adj = torch.zeros(B, n, n, device=z.device)
        idx = torch.triu_indices(n, n, offset=1)
        adj[:, idx[0], idx[1]] = edge_probs
        adj = adj + adj.transpose(1, 2)

        return adj

class GraphDiscriminator(nn.Module):
    """그래프를 위한 그래프 신경망 바탕 가름개."""

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.max_nodes = max_nodes

        # 들임: 이웃 행렬의 줄을 마디 특징으로
        self.input_proj = nn.Linear(max_nodes, hidden_dim)

        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                )
            )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        인수:
            adj: (B, n, n) 이웃 행렬
        반환값:
            score: (B,) 참/거짓 점수
        """
        B = adj.size(0)

        # 이웃 행렬의 줄을 첫 마디 특징으로 쓴다
        h = self.input_proj(adj)  # (B, n, hidden)

        # 쪽지 건네기
        for conv in self.conv_layers:
            h_msg = torch.bmm(adj, h) / (adj.sum(-1, keepdim=True) + 1)
            h = conv(h_msg) + h

        # 합 모으기(자리바꿈에 안 바뀜)
        h_graph = h.sum(dim=1)  # (B, hidden)
        score = self.readout(h_graph).squeeze(-1)  # (B,)

        return score

def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """WGAN 기울기 벌을 셈한다."""
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=real.device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(interpolated)
    gradients = torch_grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(B, -1)
    penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

class GraphGAN:
    """그래프 만들기를 위한 WGAN-GP 익히기 감싸개."""

    def __init__(
        self,
        max_nodes: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
    ):
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        self.generator = GraphGenerator(latent_dim, max_nodes, hidden_dim)
        self.discriminator = GraphDiscriminator(max_nodes, hidden_dim)

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=(0.5, 0.9)
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9)
        )

    def train_step(
        self, real_adj: torch.Tensor
    ) -> dict[str, float]:
        """익히기 걸음 하나(비평가 고침 여러 번 + 만들개 고침 한 번)."""
        B = real_adj.size(0)
        device = real_adj.device

        # --- 가름개 익히기 ---
        d_losses = []
        for _ in range(self.n_critic):
            z = torch.randn(B, self.latent_dim, device=device)
            fake_adj = self.generator(z).detach()

            d_real = self.discriminator(real_adj).mean()
            d_fake = self.discriminator(fake_adj).mean()
            gp = gradient_penalty(
                self.discriminator, real_adj, fake_adj, self.lambda_gp
            )

            d_loss = d_fake - d_real + gp

            self.opt_d.zero_grad()
            d_loss.backward()
            self.opt_d.step()
            d_losses.append(d_loss.item())

        # --- 만들개 익히기 ---
        z = torch.randn(B, self.latent_dim, device=device)
        fake_adj = self.generator(z)
        g_loss = -self.discriminator(fake_adj).mean()

        self.opt_g.zero_grad()
        g_loss.backward()
        self.opt_g.step()

        return {
            "d_loss": sum(d_losses) / len(d_losses),
            "g_loss": g_loss.item(),
            "wasserstein": (d_real - d_fake).item(),
        }

    @torch.no_grad()
    def generate(self, num_graphs: int = 1) -> list[torch.Tensor]:
        self.generator.eval()
        z = torch.randn(num_graphs, self.latent_dim)
        adj = self.generator(z, hard=True)

        graphs = []
        for b in range(num_graphs):
            # 외톨이 마디를 없앤다
            degrees = adj[b].sum(dim=1)
            active = degrees > 0
            if active.sum() < 2:
                active[:2] = True
            active_idx = torch.where(active)[0]
            graphs.append(adj[b][active_idx][:, active_idx])

        self.generator.train()
        return graphs

if __name__ == "__main__":
    torch.manual_seed(42)

    max_n = 12

    # 익히기 자료를 만든다
    print("=== GraphGAN Demo ===\n")
    train_adjs = []
    for _ in range(200):
        n = torch.randint(6, max_n + 1, (1,)).item()
        adj = torch.zeros(max_n, max_n)
        # 무리 그래프
        mid = n // 2
        for i in range(mid):
            for j in range(i + 1, mid):
                if torch.rand(1) < 0.5:
                    adj[i, j] = adj[j, i] = 1
        for i in range(mid, n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.5:
                    adj[i, j] = adj[j, i] = 1
        for i in range(mid):
            for j in range(mid, n):
                if torch.rand(1) < 0.1:
                    adj[i, j] = adj[j, i] = 1
        train_adjs.append(adj)

    train_adjs = torch.stack(train_adjs)

    # 학습
    print("Training GraphGAN (WGAN-GP)...")
    gan = GraphGAN(
        max_nodes=max_n,
        latent_dim=32,
        hidden_dim=64,
        n_critic=3,
    )
    g_params = sum(p.numel() for p in gan.generator.parameters())
    d_params = sum(p.numel() for p in gan.discriminator.parameters())
    print(f"Generator params: {g_params:,}, Discriminator params: {d_params:,}")

    batch_size = 32
    for epoch in range(100):
        idx = torch.randperm(len(train_adjs))[:batch_size]
        metrics = gan.train_step(train_adjs[idx])

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}: d_loss={metrics['d_loss']:.4f}, "
                  f"g_loss={metrics['g_loss']:.4f}, "
                  f"wasserstein={metrics['wasserstein']:.4f}")

    # 생성
    print("\n=== Generation ===")
    generated = gan.generate(num_graphs=10)
    for i, g in enumerate(generated):
        n = g.size(0)
        e = int(g.sum().item()) // 2
        density = 2 * e / (n * (n - 1)) if n > 1 else 0
        print(f"Graph {i}: {n} nodes, {e} edges, density={density:.3f}")

    # 통계를 견준다
    ref_densities = []
    for a in train_adjs:
        active = a.sum(1) > 0
        n = active.sum().item()
        e = a.sum().item() / 2
        ref_densities.append(2 * e / (n * (n - 1)) if n > 1 else 0)
    gen_densities = [2 * int(g.sum().item()) // 2 / (g.size(0) * (g.size(0) - 1))
                     if g.size(0) > 1 else 0 for g in generated]
    print(f"\nRef avg density: {sum(ref_densities)/len(ref_densities):.3f}")
    print(f"Gen avg density: {sum(gen_densities)/len(gen_densities):.3f}")
```

---

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

## 정리하며

이 마당은 구조、익히기 목표、조건 만들어 내기、그래프 GAN 특유의 어려움을 차례로 짚었다.
