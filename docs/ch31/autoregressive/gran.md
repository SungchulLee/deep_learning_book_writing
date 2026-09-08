# GRAN: 그래프 되돌이 눈길 신경망

GRAN(Liao et al., 2019)은 마디를 하나씩이 아니라 **덩이**로 만들어 GraphRNN의 커지기 한계를 다룬다. 걸음마다 마디 $B$개의 덩이를 한꺼번에 더하며 그래프 신경망 바탕 눈길 얼개로 새 마디와 기존 마디의 주고받음을 나타낸다. 이 덩이 단위 만들기는 그래프 신경망 쪽지 건네기의 비용을 여러 마디에 고루 나누어 만들기 걸음을 $O(n)$ 대신 $O(n/B)$으로 줄이면서, 눈길 바탕 변 헤아리기로 나타냄 힘을 지킨다.

---

## 1. 덩이 단위 만들기

GRAN은 그래프 만들기를 $\lceil n/B \rceil$걸음으로 가른다. 걸음 $t$에서 모델은:

1. 부분으로 지어진 그래프 $\mathcal{G}_{<t}$에 새 후보 마디 $B$개를 더한다
2. 늘린 그래프에 그래프 신경망을 돌려 마디 박아 넣기를 셈한다
3. 새 마디마다 기존 마디와 새 마디 모두와의 변을 헤아린다
4. 뽑은 변으로 그래프를 고친다

그래프의 함께 확률은 덩이마다로 인수 분해된다:

$$
p_\theta(\mathcal{G}) = \prod_{t=1}^{\lceil n/B \rceil} p_\theta(\mathbf{A}_t \mid \mathcal{G}_{<t})
$$

여기서 $\mathbf{A}_t$은 걸음 $t$의 새 마디 $B$개와 얽힌 모든 변 결정을 담는다.

---

## 2. 구조

### 그래프 신경망 등뼈

만들기 걸음마다 GRAN은 늘린 그래프(기존 마디 + 새 후보 $B$개)에 눈길이 있는 그래프 신경망을 쓴다. 그래프 신경망은 쪽지 건네기를 $L$바퀴 돈다:

$$
\mathbf{h}_v^{(\ell+1)} = \mathbf{h}_v^{(\ell)} + \text{MLP}^{(\ell)}\left(\sum_{u \in \tilde{\mathcal{N}}(v)} \alpha_{vu}^{(\ell)} \cdot \mathbf{W}^{(\ell)} \mathbf{h}_u^{(\ell)}\right)
$$

여기서 $\tilde{\mathcal{N}}(v)$은 기존 이웃과 새 마디로의 후보 변을 함께 담는다. 눈길 무게 $\alpha_{vu}^{(\ell)}$은 다음과 같이 셈한다:

$$
\alpha_{vu}^{(\ell)} = \frac{\exp(e_{vu}^{(\ell)})}{\sum_{w \in \tilde{\mathcal{N}}(v)} \exp(e_{vw}^{(\ell)})}
$$

$$
e_{vu}^{(\ell)} = \text{LeakyReLU}\left(\mathbf{a}^{(\ell)\top} [\mathbf{W}^{(\ell)} \mathbf{h}_v^{(\ell)} \| \mathbf{W}^{(\ell)} \mathbf{h}_u^{(\ell)}]\right)
$$

### 베르누이 섞음으로 하는 변 헤아리기

변을 저마다 따로 헤아리는 대신 GRAN은 변 확률을 성분 $K$개의 **베르누이 섞음**으로 나타낸다:

$$
p(A_{uv} = 1) = \sum_{k=1}^{K} w_k \cdot \sigma\left(\mathbf{h}_u^{(L)\top} \mathbf{W}_k \mathbf{h}_v^{(L)} + b_k\right)
$$

여기서 $w_k$은 $\sum_k w_k = 1$인 섞음 무게이고 $\sigma$은 시그모이드 함수다. 섞음 모형은 봉우리가 여럿인 변 분포를 담는다. 보기로 무리 그래프에서 두 마디가 같은 무리에 있으면 변이 있을 확률이 높고 아니면 낮다.

---

## 3. 익히기 목표

GRAN은 덩이 걸음마다 변 헤아림에 대한 두 값 교차 엔트로피로 익힌다:

$$
\mathcal{L} = -\sum_{t=1}^{\lceil n/B \rceil} \sum_{(u,v) \in \mathcal{C}_t} \left[ A_{uv}^* \log p_\theta(A_{uv} = 1 \mid \mathcal{G}_{<t}) + (1 - A_{uv}^*) \log(1 - p_\theta(A_{uv} = 1 \mid \mathcal{G}_{<t})) \right]
$$

여기서 $\mathcal{C}_t$은 걸음 $t$의 후보 변 자리의 모임, 곧 $u, v$ 가운데 적어도 하나가 새 마디인 모든 짝 $(u, v)$이다.

---

## 4. GraphRNN과 견주기

| 갈래 | GraphRNN | GRAN |
|--------|----------|------|
| 만들기 낱덩이 | 마디 하나 | 마디 $B$개의 덩이 |
| 상태 나타냄 | 되돌이 신경망의 숨은 상태 | 온 그래프의 그래프 신경망 |
| 변 나타내기 | 차례(변 되돌이 신경망) | 나란히(베르누이 섞음) |
| 그래프마다 걸음 | $O(n)$ | $O(n/B)$ |
| 걸음마다 비용 | $O(M)$(너비 우선 띠너비) | $O((n_t + B)^2)$(그래프 신경망) |
| 변의 매임 | 걸음 안에서 차례 | 그래프 신경망 상태가 주어지면 독립 |

GRAN의 그래프 신경망 등뼈는 눌러 담은 숨은 상태에 기대는 대신 지금 그래프의 어느 마디든 곧바로 살필 수 있어 GraphRNN의 되돌이 신경망보다 엄밀히 나타냄 힘이 세다. 덩이 단위 만들기는 실제 차례 길이도 줄여 멀리 떨어진 매임 말썽을 누그러뜨린다.

---

## 5. 커지기에서 살필 것

GRAN의 걸음마다 비용은 커지는 그래프에 대한 그래프 신경망 앞으로 가기가 대부분이다. 기존 마디가 $n_t = t \cdot B$개인 걸음 $t$의 그래프에서 쪽지 건네기는 $|\mathcal{E}_t|$이 지금 변 개수일 때 $O(L \cdot |\mathcal{E}_t|)$이 들고 변 헤아리기는 후보 짝에 $O(B \cdot n_t)$이 든다. 모든 걸음을 통틀은 온 비용은 빽빽한 그래프에서 $O(n^2 \cdot L / B)$, 평균 차수가 $\bar{d}$인 성긴 그래프에서 $O(n \cdot \bar{d} \cdot L / B)$이다. $B = \Theta(\sqrt{n})$으로 고르면 걸음 수와 걸음마다 비용의 균형이 잡힌다.

---

## 6. 금융 쓰임새: 꾸러미 그물 짓기

GRAN의 덩이 단위 만들기는 금융 낱것 무리가 한꺼번에 시장에 들어오는 상황을 자연스럽게 나타낸다. 보기로 묶음 상장, 새 시장 참여자를 낳는 규제 판의 바뀜, 파생 계약의 한꺼번에 맺음이 그렇다. 베르누이 섞음은 금융 이음의 봉우리가 여럿인 성질을 담는다. 거래 상대 관계가 업종, 지역, 규제 관할로 뭉칠 수 있다.

---

## 7. 짜기: 눈길을 쓴 GRAN

```python
"""
GRAN: 덩이 단위 그래프 만들기를 위한 그래프 되돌이 눈길 신경망.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class GRANAttentionLayer(nn.Module):
    """GRAN의 눈길 바탕 그래프 신경망 층 하나."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        인수:
            h: (n_total, hidden_dim) 마디 박아 넣기
            adj: (n_total, n_total) 이웃 행렬(이미 있는 것 + 후보)
            candidate_mask: (n_total, n_total) 후보 변 가리개
        """
        n = h.size(0)

        Q = self.W_q(h).view(n, self.num_heads, self.head_dim)
        K = self.W_k(h).view(n, self.num_heads, self.head_dim)
        V = self.W_v(h).view(n, self.num_heads, self.head_dim)

        scores = torch.einsum("ihd,jhd->ijh", Q, K) / math.sqrt(self.head_dim)

        # 기존 변과 후보 자리를 살핀다
        attn_mask = (adj + candidate_mask).clamp(max=1.0)
        # 스스로 이음을 더한다
        attn_mask = attn_mask + torch.eye(n, device=h.device)
        attn_mask = attn_mask.unsqueeze(-1).expand_as(scores)
        scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn = F.softmax(scores, dim=1)
        out = torch.einsum("ijh,jhd->ihd", attn, V)
        out = out.reshape(n, self.hidden_dim)
        out = self.W_o(out)

        h = self.norm(h + out)
        h = self.norm2(h + self.mlp(h))
        return h

class MixtureBernoulliDecoder(nn.Module):
    """베르누이 섞음 변 헤아리개."""

    def __init__(self, hidden_dim: int, num_components: int = 4):
        super().__init__()
        self.K = num_components
        self.edge_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_components)
        ])
        self.mix_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_components),
        )

    def forward(self, h_u: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        """마디 짝의 변 확률을 헤아린다."""
        pair_feat = torch.cat([h_u, h_v], dim=-1)
        mix_weights = F.softmax(self.mix_predictor(pair_feat), dim=-1)

        component_probs = torch.stack([
            torch.sigmoid(pred(pair_feat).squeeze(-1))
            for pred in self.edge_predictors
        ], dim=-1)

        return (mix_weights * component_probs).sum(dim=-1)

class GRAN(nn.Module):
    """
    그래프 되돌이 눈길 신경망.
    
    그래프 신경망 눈길과 베르누이 섞음 변 헤아리기로
    그래프를 덩이 단위로 만든다.
    """

    def __init__(
        self,
        max_nodes: int,
        block_size: int = 1,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_mix_components: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.block_size = block_size
        self.hidden_dim = hidden_dim

        self.node_embed = nn.Embedding(2, hidden_dim)  # 0=기존, 1=새것
        self.gnn_layers = nn.ModuleList([
            GRANAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_gnn_layers)
        ])
        self.edge_decoder = MixtureBernoulliDecoder(hidden_dim, num_mix_components)

    def _predict_block_edges(
        self,
        adj_padded: torch.Tensor,
        n_existing: int,
        n_new: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """새 마디 덩이의 변을 헤아린다."""
        n_total = n_existing + n_new
        device = adj_padded.device

        node_types = torch.zeros(n_total, dtype=torch.long, device=device)
        node_types[n_existing:] = 1
        h = self.node_embed(node_types)

        # 후보 가리개: 새것에서 모두로, 모두에서 새것으로
        candidate_mask = torch.zeros(n_total, n_total, device=device)
        candidate_mask[n_existing:, :n_total] = 1.0
        candidate_mask[:n_total, n_existing:] = 1.0
        candidate_mask.fill_diagonal_(0)

        for layer in self.gnn_layers:
            h = layer(h, adj_padded[:n_total, :n_total], candidate_mask)

        # 후보 짝을 얻는다(위쪽 삼각)
        pairs_i, pairs_j = torch.where(
            torch.triu(candidate_mask, diagonal=1) > 0
        )
        candidate_pairs = torch.stack([pairs_i, pairs_j], dim=1)

        if len(candidate_pairs) == 0:
            return torch.tensor([], device=device), candidate_pairs

        h_u = h[candidate_pairs[:, 0]]
        h_v = h[candidate_pairs[:, 1]]
        edge_probs = self.edge_decoder(h_u, h_v)

        return edge_probs, candidate_pairs

    def forward(self, adj_list: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """그래프 목록에 대한 익히기 앞으로 가기."""
        total_loss = torch.tensor(0.0)
        num_preds = 0

        for adj in adj_list:
            n = adj.size(0)
            adj_padded = torch.zeros(self.max_nodes, self.max_nodes)
            adj_padded[:n, :n] = adj

            for step_start in range(self.block_size, n, self.block_size):
                n_existing = step_start
                n_new = min(self.block_size, n - step_start)

                edge_probs, candidate_pairs = self._predict_block_edges(
                    adj_padded, n_existing, n_new
                )
                if len(candidate_pairs) == 0:
                    continue

                targets = adj_padded[candidate_pairs[:, 0], candidate_pairs[:, 1]]
                loss = F.binary_cross_entropy(
                    edge_probs.clamp(1e-6, 1 - 1e-6), targets, reduction="sum"
                )
                total_loss = total_loss + loss
                num_preds += len(candidate_pairs)

        return {"total_loss": total_loss / max(num_preds, 1)}

    @torch.no_grad()
    def generate(
        self,
        num_graphs: int = 1,
        num_nodes: int = 10,
        temperature: float = 1.0,
    ) -> list[torch.Tensor]:
        """그래프를 덩이 단위로 만든다."""
        self.eval()
        graphs = []

        for _ in range(num_graphs):
            adj = torch.zeros(self.max_nodes, self.max_nodes)

            for step_start in range(self.block_size, num_nodes, self.block_size):
                n_existing = step_start
                n_new = min(self.block_size, num_nodes - step_start)

                edge_probs, candidate_pairs = self._predict_block_edges(
                    adj, n_existing, n_new
                )
                if len(candidate_pairs) == 0:
                    continue

                # 변을 뽑는다
                adjusted_probs = torch.sigmoid(
                    torch.log(edge_probs / (1 - edge_probs + 1e-8)) / temperature
                )
                sampled = torch.bernoulli(adjusted_probs)

                # 이웃 행렬을 고친다(맞섬)
                for idx in range(len(candidate_pairs)):
                    if sampled[idx] > 0:
                        i, j = candidate_pairs[idx]
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0

            graphs.append(adj[:num_nodes, :num_nodes])

        return graphs

if __name__ == "__main__":
    torch.manual_seed(42)

    max_n = 16
    block_size = 2

    # 익히기 자료를 만든다
    print("=== Preparing Training Data ===")
    train_graphs = []
    for _ in range(100):
        n = torch.randint(6, max_n, (1,)).item()
        # 단순하게 하려 n이 block_size으로 나누어떨어지게 한다
        n = (n // block_size) * block_size
        if n < 4:
            n = 4
        adj = torch.zeros(n, n)
        # 무리 얼개
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
        train_graphs.append(adj)

    print(f"Training graphs: {len(train_graphs)}")
    print(f"Avg nodes: {sum(g.size(0) for g in train_graphs)/len(train_graphs):.1f}")

    # GRAN 익히기
    print("\n=== Training GRAN ===")
    model = GRAN(
        max_nodes=max_n,
        block_size=block_size,
        hidden_dim=64,
        num_gnn_layers=2,
        num_mix_components=3,
        num_heads=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    for epoch in range(40):
        model.train()
        # 작은 묶음
        batch = train_graphs[: 32]
        result = model(batch)
        loss = result["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    # 생성
    print("\n=== Generation ===")
    generated = model.generate(num_graphs=10, num_nodes=10)
    for i, g in enumerate(generated):
        n = g.size(0)
        e = int(g.sum().item()) // 2
        density = 2 * e / (n * (n - 1)) if n > 1 else 0
        print(f"Graph {i}: {n} nodes, {e} edges, density={density:.3f}")
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

이 마당은 덩이 단위 만들기、구조、익히기 목표、GraphRNN과 견주기을 차례로 짚었다.
