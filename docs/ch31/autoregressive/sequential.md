# 차례로 하는 그래프 만들기
## 개요

자기 되돌이 그래프 만들기는 함께 분포 $p_\theta(\mathcal{G})$을 매인 분포의 차례로 쪼개어 원소를 하나씩 만든다. 이 틀은 크기가 제각각인 그래프를 자연스럽게 다루고 이웃 행렬 전체를 한꺼번에 내놓는 얽음의 어려움을 피한다. 핵심 생각은 그래프 짓기 움직임에 차례를 매기고 확률의 사슬 규칙으로 가능도를 인수 분해하는 것이다.

## 자기 되돌이 인수 분해

마디가 $n$개인 그래프 $\mathcal{G} = (\mathbf{A}, \mathbf{X})$이 주어질 때 움직임 $a_t$마다 마디를 더하거나 변을 더하거나 만들기를 끝내는 만들기 차례 $\sigma = (a_1, a_2, \ldots, a_T)$을 뜻매김한다. 함께 확률은 다음과 같이 인수 분해된다:

$$
p_\theta(\mathcal{G}) = \sum_{\sigma \in \Sigma(\mathcal{G})} \prod_{t=1}^{T} p_\theta(a_t \mid a_{1:t-1})
$$

여기서 $\Sigma(\mathcal{G})$은 $\mathcal{G}$을 내는 올바른 만들기 차례의 모임이다. 차례에 대해 주변화하기는 다룰 수 없으므로($|\Sigma(\mathcal{G})|$이 지수일 수 있다) 실제 방법은 표준 차례 $\sigma^*$을 붙박고 다음을 가장 좋게 한다:

$$
\log p_\theta(\mathcal{G}) \geq \log \prod_{t=1}^{T} p_\theta(a_t^* \mid a_{1:t-1}^*)
$$

그래프가 주어졌을 때 차례가 정해져 있으면 이 아래 가둠이 빈틈없어진다.

## 마디 켜의 쪼개기

가장 흔한 쪼개기는 마디를 하나씩 더한다. 걸음 $t$에서 모델은:

1. **새 마디를 더할지** 끝낼지 정한다: $p_\theta(\text{stop} \mid \mathcal{G}_{<t})$
2. **마디 특징을 만든다**: $p_\theta(\mathbf{x}_t \mid \mathcal{G}_{<t})$
3. 기존 마디 모두와의 **변을 만든다**: $p_\theta(\mathbf{a}_t \mid \mathbf{x}_t, \mathcal{G}_{<t})$

여기서 $\mathcal{G}_{<t}$은 부분으로 지어진 그래프를, $\mathbf{a}_t \in \{0,1\}^{t-1}$은 마디 $t$을 마디 $1, \ldots, t-1$에 잇는 변 벡터를 뜻한다.

마디 차례 $\pi$ 아래 마디가 $n$개인 그래프의 로그 가능도는 다음과 같다:

$$
\log p_\theta(\mathcal{G} \mid \pi) = \sum_{t=1}^{n} \left[ \log p_\theta(\mathbf{x}_{\pi(t)} \mid \mathcal{G}_{<t}) + \sum_{s=1}^{t-1} \log p_\theta(A_{\pi(t),\pi(s)} \mid \mathbf{x}_{\pi(t)}, \mathcal{G}_{<t}) \right] + \log p_\theta(\text{stop} \mid \mathcal{G})
$$

## 변 켜의 쪼개기

다른 길은 만들기를 변 켜에서 쪼개어 가능한 모든 마디 짝을 훑는다:

$$
p_\theta(\mathbf{A} \mid \mathbf{X}) = \prod_{i=1}^{n} \prod_{j=1}^{i-1} p_\theta(A_{ij} \mid A_{<(i,j)}, \mathbf{X})
$$

여기서 곱은 위쪽 삼각 이웃 행렬을 줄줄이 훑는 차례를 따른다. 변 결정마다 앞서 정해진 모든 변에 매인 베르누이 변수다.

## 차례 셈속

마디 차례 $\pi$을 어떻게 고르느냐가 만들기 품질과 셈 비용에 근본에서 영향을 준다.

**너비 우선 차례**는 차수가 가장 높은 마디에서 그래프를 너비 우선으로 밟는다. 이는 변 결정을 다시 늘어놓은 이웃 행렬의 대각 언저리에 모아 맥락 창을 잘라 쓸 수 있게 한다. 핵심 성질은 너비 우선 차례에서 $i < j$인 변 $(i, j)$이 $B$이 너비 우선 띠너비일 때 $j - i \leq B$을 만족한다는 것이다. 곧 마디 $j$은 앞선 $j-1$개 마디 모두가 아니라 가장 가까운 앞선 $B$개와의 이음만 살피면 된다.

**깊이 우선 차례**도 비슷한 가까움을 내지만 얼개 성질이 다르다. 깊이 우선 밟기는 갈라지기 전에 긴 사슬을 만들어 나무 같은 그래프에 이로울 수 있다.

자료 늘리기를 곁들인 **아무 차례**는 그래프마다 여러 아무 차례로 모델을 익혀 몬테카를로 뽑기로 $\Sigma(\mathcal{G})$에 대한 주변화를 어림한다:

$$
\log p_\theta(\mathcal{G}) \approx \log \frac{1}{K} \sum_{k=1}^{K} \prod_{t=1}^{T} p_\theta(a_t^{(k)} \mid a_{1:t-1}^{(k)})
$$

## 상태 표현

만들기 걸음 $t$마다 모델은 부분으로 지어진 그래프 $\mathcal{G}_{<t}$을 차원이 붙박인 상태 벡터로 담아야 한다. 두 길이 주로 쓰인다:

**되돌이 담기.** GRU나 장단기 기억망이 걸음마다 고쳐지는 숨은 상태 $\mathbf{h}_t$을 지닌다:

$$
\mathbf{h}_t = \text{GRU}(\mathbf{h}_{t-1}, \mathbf{e}_t)
$$

여기서 $\mathbf{e}_t$은 걸음 $t$에서 한 움직임(마디 특징과 변 결정)을 담는다. 숨은 상태가 만들기의 온 지난 일을 숨은 채로 간추린다.

**그래프 신경망 담기.** 걸음마다 $\mathcal{G}_{<t}$에 그래프 신경망을 써서 마디 박아 넣기를 낸 뒤 그래프 켜의 나타냄으로 모은다:

$$
\mathbf{h}_t = \text{READOUT}(\text{GNN}(\mathcal{G}_{<t}))
$$

커지는 그래프에 걸음마다 그래프 신경망을 다시 써야 하므로 나타냄 힘은 세지만 더 비싸다.

## 학습 절차

익히기는 스승 밀어 넣기를 쓴다. 걸음 $t$마다 모델은 자기 헤아림이 아니라 참 부분 그래프 $\mathcal{G}_{<t}^*$을 받는다. 손실은 모든 걸음에 걸쳐 더한 음의 로그 가능도다:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log p_\theta(a_t^* \mid \mathcal{G}_{<t}^*)
$$

변 헤아리기에서는 이것이 변 결정마다의 두 값 교차 엔트로피로 줄어든다:

$$
\mathcal{L}_{\text{edge}} = -\sum_{t=2}^{n} \sum_{s=1}^{t-1} \left[ A_{ts}^* \log \hat{p}_{ts} + (1 - A_{ts}^*) \log (1 - \hat{p}_{ts}) \right]
$$

여기서 $\hat{p}_{ts} = p_\theta(A_{ts} = 1 \mid \mathcal{G}_{<t}^*)$이다.

## 드러남 치우침과 누그러뜨리기

스승 밀어 넣기는 익히기와 시험의 어긋남을 낳는다. 만들 때 모델은 참값이 아니라 (틀렸을 수도 있는) 제 헤아림에 매인다. 이 **드러남 치우침** 때문에 긴 차례에서 어긋남이 쌓일 수 있다.

덜어 내는 전략은 다음과 같다:

- **일정 잡은 뽑기**: (익히는 동안 식혀 가는) 확률 $\epsilon_t$으로 걸음 $t$에서 참값 대신 모델의 헤아림을 쓴다
- **차례 켜의 목표**: 그래프 켜 잣대를 보상으로 삼아 REINFORCE으로 익힌다
- **차례 배우기**: 작은 그래프에서 시작해 익히는 동안 크기를 차츰 키운다

## 금융 쓰임새: 차례로 그물 짓기

금융 그물 만들기에서 차례로 짓기는 실제 그물이 생기는 모습을 그대로 비춘다. 은행이 때에 따라 은행 사이 시장에 들어와 빌려주기 관계를 차츰 맺는다. 자기 되돌이 틀이 이 때 흐름을 나타낼 수 있다:

$$
p(\mathcal{G}_T) = \prod_{t=1}^{T} p(\text{bank}_t \text{ joins}) \cdot \prod_{s < t} p(\text{lends}(t, s) \mid \text{attributes}, \mathcal{G}_{<t})
$$

이는 우선 붙기(새 은행이 잘 이어진 기존 은행에 먼저 붙는다)와 끼리끼리(속성이 닮은 은행끼리 이어진다)를 담으며 둘 다 금융 그물에서 실제로 관찰된다.

## 짜기: 차례로 만드는 그래프 만들개

```python
"""
차례로 하는(자기 되돌이) 그래프 만들기 틀.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import deque


class SequentialGraphGenerator(nn.Module):
    """
    마디 하나씩 만드는 자기 되돌이 그래프 만들개.
    
    At each step t:
    1. Graph-level RNN state summarizes G_{<t}
    2. 변 여러 층 신경망이 앞선 마디와의 이음을 헤아린다
    3. 멈춤 헤아리개가 마디를 더 더할지 정한다
    """

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 128,
        node_feature_dim: int = 0,
        rnn_type: str = "gru",
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim

        # 그래프 켜 되돌이 신경망: 만들기의 지난 일을 간추린다
        rnn_input_dim = max_nodes  # max_nodes까지 채운 변 벡터
        if rnn_type == "gru":
            self.graph_rnn = nn.GRU(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
        else:
            self.graph_rnn = nn.LSTM(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )

        # 변 헤아리개: 그래프 상태가 주어지면 앞선 마디와의 변을 헤아린다
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes),
        )

        # 멈춤 헤아리개
        self.stop_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 처음 숨은 상태
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def _get_init_hidden(self, batch_size: int) -> torch.Tensor:
        return self.h0.expand(1, batch_size, self.hidden_dim).contiguous()

    def forward(
        self,
        adj_sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        스승 밀어 넣기와 함께 하는 익히기 앞으로 가기.
        
        인수:
            adj_sequences: (B, max_nodes, max_nodes) 채운 변 벡터
                adj_sequences[b, t, :t] = edges from node t to nodes 0..t-1
            lengths: (B,) 그래프마다 마디 개수
            
        반환값:
            Dictionary with 'edge_loss' and 'stop_loss'
        """
        batch_size = adj_sequences.size(0)
        device = adj_sequences.device
        h = self._get_init_hidden(batch_size).to(device)

        edge_loss = torch.tensor(0.0, device=device)
        stop_loss = torch.tensor(0.0, device=device)
        num_edge_preds = 0
        num_stop_preds = 0

        for t in range(1, self.max_nodes):
            # 들임: 앞 걸음의 변 벡터
            if t == 1:
                x_t = torch.zeros(batch_size, 1, self.max_nodes, device=device)
            else:
                x_t = adj_sequences[:, t - 1, :].unsqueeze(1)  # (B, 1, max_nodes)

            # 되돌이 신경망 걸음
            out, h = self.graph_rnn(x_t, h)
            graph_state = out.squeeze(1)  # (B, hidden_dim)

            # 걸음 t의 변 헤아림
            edge_logits = self.edge_mlp(graph_state)  # (B, max_nodes)
            edge_targets = adj_sequences[:, t, :]  # (B, max_nodes)

            # 올바른 자리에서만 손실을 셈한다(마디 t은 0..t-1에 이어진다)
            # 그리고 마디가 적어도 t+1개인 그래프에서만
            active = (lengths > t).float()  # (B,)
            if active.sum() > 0:
                # 가리개: 자리 0..t-1만 올바른 변 과녁이다
                mask = torch.zeros(batch_size, self.max_nodes, device=device)
                mask[:, :t] = 1.0
                mask = mask * active.unsqueeze(1)

                edge_bce = F.binary_cross_entropy_with_logits(
                    edge_logits, edge_targets, reduction="none"
                )
                edge_loss = edge_loss + (edge_bce * mask).sum()
                num_edge_preds += mask.sum().item()

            # 멈춤 헤아리기
            stop_logits = self.stop_mlp(graph_state).squeeze(-1)  # (B,)
            # 과녁: 마지막 마디면 stop=1
            stop_target = (lengths == t + 1).float()
            # 아직 만드는 중인 그래프에서만 셈한다
            stop_active = (lengths >= t + 1).float()
            if stop_active.sum() > 0:
                stop_bce = F.binary_cross_entropy_with_logits(
                    stop_logits, stop_target, reduction="none"
                )
                stop_loss = stop_loss + (stop_bce * stop_active).sum()
                num_stop_preds += stop_active.sum().item()

        # 정규화
        edge_loss = edge_loss / max(num_edge_preds, 1)
        stop_loss = stop_loss / max(num_stop_preds, 1)

        return {
            "edge_loss": edge_loss,
            "stop_loss": stop_loss,
            "total_loss": edge_loss + stop_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        num_graphs: int = 1,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> list[torch.Tensor]:
        """
        자기 되돌이로 그래프를 만든다.
        
        반환값:
            List of adjacency matrices (variable sizes)
        """
        self.eval()
        graphs = []

        for _ in range(num_graphs):
            h = self._get_init_hidden(1).to(device)
            edges = []
            x_t = torch.zeros(1, 1, self.max_nodes, device=device)

            for t in range(1, self.max_nodes):
                out, h = self.graph_rnn(x_t, h)
                graph_state = out.squeeze(1)

                # 앞선 마디와의 변을 뽑는다
                edge_logits = self.edge_mlp(graph_state)[0, :t]  # (t,)
                edge_probs = torch.sigmoid(edge_logits / temperature)
                edge_sample = torch.bernoulli(edge_probs)
                edges.append(edge_sample)

                # 멈춤을 살핀다
                stop_logit = self.stop_mlp(graph_state).squeeze()
                stop_prob = torch.sigmoid(stop_logit / temperature)
                if torch.bernoulli(stop_prob).item() > 0.5 and t >= 2:
                    break

                # 다음 들임을 채비한다
                x_t = torch.zeros(1, 1, self.max_nodes, device=device)
                x_t[0, 0, :t] = edge_sample

            # 이웃 행렬을 되짓는다
            n = len(edges) + 1
            adj = torch.zeros(n, n, device=device)
            for t, e in enumerate(edges):
                adj[t + 1, : t + 1] = e
                adj[: t + 1, t + 1] = e
            graphs.append(adj.cpu())

        return graphs


def bfs_node_ordering(adj: torch.Tensor) -> list[int]:
    """차수가 가장 높은 마디에서 시작하는 너비 우선 차례를 셈한다."""
    n = adj.size(0)
    start = adj.sum(dim=1).argmax().item()
    visited = {start}
    order = [start]
    queue = deque([start])

    while queue:
        node = queue.popleft()
        neighbors = torch.where(adj[node] > 0)[0].tolist()
        neighbors.sort(key=lambda x: -adj[x].sum().item())
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                order.append(nb)
                queue.append(nb)

    for i in range(n):
        if i not in visited:
            order.append(i)
    return order


def prepare_training_sequence(
    adj: torch.Tensor,
    max_nodes: int,
) -> tuple[torch.Tensor, int]:
    """
    너비 우선 차례 아래에서 이웃 행렬을 익히기 차례로 바꾼다.
    
    반환값:
        adj_sequence: (max_nodes, max_nodes) 채운 변 벡터
        num_nodes: 실제 마디 개수
    """
    order = bfs_node_ordering(adj)
    n = adj.size(0)

    # 이웃 행렬을 다시 늘어놓는다
    perm = torch.tensor(order)
    adj_ordered = adj[perm][:, perm]

    # 차례 만들기: 줄 t은 마디 0..t-1과의 변을 담는다
    seq = torch.zeros(max_nodes, max_nodes)
    for t in range(min(n, max_nodes)):
        for s in range(t):
            seq[t, s] = adj_ordered[t, s]

    return seq, n


if __name__ == "__main__":
    torch.manual_seed(42)
    max_n = 15

    # 지어낸 익히기 자료를 만든다
    print("=== Preparing Training Data ===")
    graphs = []
    for _ in range(100):
        n = torch.randint(5, max_n, (1,)).item()
        adj = (torch.rand(n, n) < 0.2).float()
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.t()
        graphs.append(adj)

    sequences = []
    lengths = []
    for adj in graphs:
        seq, n = prepare_training_sequence(adj, max_n)
        sequences.append(seq)
        lengths.append(n)

    adj_seq = torch.stack(sequences)  # (100, max_n, max_n)
    lens = torch.tensor(lengths)
    print(f"Training data: {adj_seq.shape}, lengths: {lens.float().mean():.1f} avg")

    # 학습
    print("\n=== Training ===")
    model = SequentialGraphGenerator(max_nodes=max_n, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        losses = model(adj_seq, lens)
        loss = losses["total_loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f} "
                  f"(edge={losses['edge_loss'].item():.4f}, "
                  f"stop={losses['stop_loss'].item():.4f})")

    # 생성
    print("\n=== Generation ===")
    generated = model.generate(num_graphs=10)
    for i, g in enumerate(generated):
        n = g.size(0)
        e = int(g.sum().item()) // 2
        density = 2 * e / (n * (n - 1)) if n > 1 else 0
        print(f"Graph {i}: {n} nodes, {e} edges, density={density:.3f}")

    # 통계를 견준다
    ref_sizes = [adj.size(0) for adj in graphs]
    gen_sizes = [g.size(0) for g in generated]
    print(f"\nRef avg size: {sum(ref_sizes)/len(ref_sizes):.1f}")
    print(f"Gen avg size: {sum(gen_sizes)/len(gen_sizes):.1f}")
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
