# GraphRNN
## 개요

GraphRNN(You et al., 2018)은 그래프 만들기의 바탕이 되는 자기 되돌이 모델이다. 그래프 만들기를 두 켜로 쪼개는 켜진 되돌이 신경망 얼개를 내놓는다. 온 자리 만들기 상태를 지니며 걸음마다 마디 하나를 만드는 **그래프 켜 되돌이 신경망**과, 새 마디를 기존 마디에 잇는 변을 만드는 **변 켜 되돌이 신경망**이다. 이 두 켜 쪼개기에 그래프의 가까움을 살리는 너비 우선 차례를 더해 GraphRNN은 나타냄 힘도 세고 셈도 다룰 만하다.

## 구조

GraphRNN은 되돌이 신경망 둘을 켜지게 늘어놓아 쓴다:

**그래프 켜 되돌이 신경망($f_{\text{graph}}$).** 걸음 $t$까지 지은 그래프를 간추린 숨은 상태 $\mathbf{h}_t^G$을 지닌다. 걸음마다 앞 걸음의 변 벡터를 받아 변 켜 되돌이 신경망의 첫 숨은 상태를 내놓는다:

$$
\mathbf{h}_t^G = f_{\text{graph}}(\mathbf{h}_{t-1}^G, \mathbf{a}_{t-1})
$$

여기서 $\mathbf{a}_{t-1} \in \{0,1\}^{M}$은 걸음 $t-1$의 (잘렸을 수도 있는) 변 벡터이고 $M$은 너비 우선 띠너비다.

**변 켜 되돌이 신경망($f_{\text{edge}}$).** $\mathbf{h}_t^G$에서 얻은 상태로 시작해 벡터 $\mathbf{a}_t$의 변을 차례로 헤아린다:

$$
\mathbf{h}_{t,s}^E = f_{\text{edge}}(\mathbf{h}_{t,s-1}^E, a_{t,s-1})
$$

$$
p(a_{t,s} = 1) = \sigma(\text{MLP}(\mathbf{h}_{t,s}^E))
$$

여기서 $a_{t,s}$은 (너비 우선 차례에서) 마디 $t$과 마디 $t - s$ 사이의 변을 가리키고 $\sigma$은 시그모이드 함수다.

## 너비 우선 잘라내기

셈에서의 핵심 통찰은 너비 우선 차례에서 새 마디의 변이 너비 우선 띠너비 $M$ 안으로만 되돌아간다는 것이다. 앞선 $t-1$개 마디 모두와의 이음을 헤아리는 대신 GraphRNN은 걸음마다 변 결정을 $\min(t-1, M)$개만 헤아린다. 이는 온 변 헤아림 수를 $O(n^2)$에서 $O(n \cdot M)$으로 줄이며, 성긴 그래프에서 $M$은 보통 $n$보다 훨씬 작다.

그래프의 너비 우선 띠너비는 다음과 같다:

$$
M = \max_{(u,v) \in \mathcal{E}} |\pi^{-1}(u) - \pi^{-1}(v)|
$$

여기서 $\pi$은 너비 우선 차례다. 많은 그래프 무리(무리 그래프, 분자 그래프, 사회 그물)에서 $M \ll n$이어서 잘라내기가 앎을 거의 잃지 않는 쓸 만한 어림이 된다.

## 학습

GraphRNN은 두 켜 모두에서 스승 밀어 넣기와 함께 최대 가능도로 익힌다. 너비 우선 차례가 $\pi$이고 띠너비가 $M$인 그래프가 주어질 때:

$$
\mathcal{L} = -\sum_{t=2}^{n} \sum_{s=1}^{\min(t-1, M)} \left[ a_{t,s}^* \log p_\theta(a_{t,s} = 1 \mid \mathbf{h}_{t,s-1}^E) + (1 - a_{t,s}^*) \log p_\theta(a_{t,s} = 0 \mid \mathbf{h}_{t,s-1}^E) \right]
$$

익히는 동안 참 변 값 $a_{t,s}^*$을 변 켜 되돌이 신경망의 들임으로 넣는다(스승 밀어 넣기). 변 켜 되돌이 신경망은 걸음 $t$마다 그래프 켜 상태를 배운 바꿈으로 옮겨 첫 값을 잡는다:

$$
\mathbf{h}_{t,0}^E = \text{MLP}_{\text{init}}(\mathbf{h}_t^G)
$$

## 간단히 한 변형: GraphRNN-S

셈 효율을 위해 GraphRNN-S은 변 켜 되돌이 신경망을 변 벡터 전체를 한꺼번에 헤아리는 여러 층 신경망 하나로 바꾼다:

$$
\hat{\mathbf{a}}_t = \sigma(\text{MLP}(\mathbf{h}_t^G)) \in [0,1]^{M}
$$

이는 한 걸음 안의 변 결정 사이의 차례 매임을 없애 나타냄 힘과 나란함을 맞바꾼다. GraphRNN-S은 익히기가 훨씬 빠르지만 걸음 안의 변 얽힘이 중요한 자료 뭉치에서는 품질이 떨어지는 그래프를 낼 수 있다.

## 만들기 절차

모델이 끝냄 신호를 낼 때까지 마디를 하나씩 만들어 간다:

1. $\mathbf{h}_0^G$을 배운 벡터나 0 벡터로 둔다
2. $t = 1, 2, \ldots, n_{\max}$에 대해:
    - $\mathbf{h}_t^G = f_{\text{graph}}(\mathbf{h}_{t-1}^G, \mathbf{a}_{t-1})$을 셈한다
    - $\mathbf{h}_{t,0}^E = \text{MLP}_{\text{init}}(\mathbf{h}_t^G)$으로 첫 값을 잡는다
    - $s = 1, \ldots, M$에 대해:
        - $p_{t,s} = \sigma(\text{MLP}(\mathbf{h}_{t,s}^E))$을 셈한다
        - $a_{t,s} \sim \text{Bernoulli}(p_{t,s})$을 뽑는다
        - 모든 $s$에 대해 $a_{t,s} = 0$이면(끝 무늬) 만들기를 멈춘다
        - $\mathbf{h}_{t,s}^E = f_{\text{edge}}(\mathbf{h}_{t,s-1}^E, a_{t,s})$으로 고친다
3. 변 차례에서 이웃 행렬을 되짓는다

끝냄 조건은 특별한 차례 끝 무늬를 쓴다. 변 벡터 $\mathbf{a}_t$이 통째로 0이면(앞선 어느 마디와도 잇지 않으면) 만들기를 멈춘다. 이는 자연스럽게 크기가 제각각인 그래프를 낸다.

## 한계

**차례 민감도.** 같은 그래프를 너비 우선으로 다르게 밟으면 익히기 차례가 달라진다. 여러 차례로 자료를 늘리면 도움이 되지만 모델은 여전히 차례에 매인 분포를 숨은 채로 배운다. $p_\theta(\mathcal{G} \mid \pi)$과 참 $p(\mathcal{G})$의 틈은 고른 차례 갈래가 얼개의 규칙성을 얼마나 잘 담느냐에 매인다.

**멀리 떨어진 매임.** 그래프 켜 되돌이 신경망은 걸음이 많을 수 있는데도 앎을 죽 퍼뜨려야 한다. 큰 그래프에서는 뒤의 만들기 고름에 영향을 주는 이른 얼개 결정을 숨은 상태가 담지 못할 수 있다.

**커지기.** 익히기와 만들기가 마디를 가로질러 본디 차례여서 나란함이 가둬진다. 마디가 수백 개인 그래프에서는 너비 우선 잘라내기를 해도 익히기가 느릴 수 있다.

## 금융 쓰임새: 은행 사이 그물 만들기

GraphRNN은 차례로 짓는 방식이 때에 따른 그물 생김을 그대로 비추므로 지어낸 은행 사이 빌려주기 그물을 만드는 데 잘 맞는다. 은행은 은행 사이 시장에 들어와 빌려주기 관계를 차츰 맺는다. 너비 우선 차례는 금융 그물의 켜진 얼개를 자연스럽게 담는다. 속 은행(차수가 높음)이 먼저 나오고 바깥 기관이 주로 가까운 속 마디에 이어진다.

## 짜기: 너비 우선 잘라내기를 쓴 GraphRNN

```python
"""
GraphRNN: 너비 우선 잘라내기를 쓰는 그래프 만들기의 켜진 되돌이 신경망.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional


class EdgeRNN(nn.Module):
    """변 켜 되돌이 신경망: 변 벡터를 차례로 만든다."""

    def __init__(self, hidden_dim: int, edge_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(
            input_size=edge_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, edge_dim),
        )

    def forward(
        self,
        h_init: torch.Tensor,
        edge_targets: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        스승 밀어 넣기 앞으로 가기.
        
        인수:
            h_init: (B, hidden_dim) 그래프 되돌이 신경망에서 온 첫 숨은 상태
            edge_targets: (B, M) 참 변 벡터
            lengths: (B,) 표본마다 올바른 길이
            
        반환값:
            logits: (B, M) 변 로짓
        """
        batch_size, M = edge_targets.shape
        device = edge_targets.device

        # 들임 채비: 시작 토막(0)과 함께 밀린 과녁
        sos = torch.zeros(batch_size, 1, 1, device=device)
        inputs = edge_targets[:, :-1].unsqueeze(-1)  # (B, M-1, 1)
        inputs = torch.cat([sos, inputs], dim=1)  # (B, M, 1)

        # 변 되돌이 신경망을 돌린다
        h = h_init.unsqueeze(0)  # (1, B, hidden_dim)
        output, _ = self.rnn(inputs, h)  # (B, M, hidden_dim)
        logits = self.output(output).squeeze(-1)  # (B, M)

        return logits

    @torch.no_grad()
    def generate(
        self,
        h_init: torch.Tensor,
        max_edges: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        변 벡터를 자기 되돌이로 만든다.
        
        인수:
            h_init: (1, hidden_dim) 첫 숨은 상태
            max_edges: 만들 변의 최대 개수
            temperature: 뽑기 온도
            
        반환값:
            edges: (max_edges,) 뽑은 두 값 변 벡터
        """
        device = h_init.device
        h = h_init.unsqueeze(0)  # (1, 1, hidden_dim)
        edges = []
        x = torch.zeros(1, 1, 1, device=device)

        for s in range(max_edges):
            out, h = self.rnn(x, h)
            logit = self.output(out).squeeze()
            prob = torch.sigmoid(logit / temperature)
            edge = torch.bernoulli(prob)
            edges.append(edge.item())
            x = edge.view(1, 1, 1)

        return torch.tensor(edges, device=device)


class GraphRNN(nn.Module):
    """
    GraphRNN: 그래프 만들기의 두 켜 켜진 되돌이 신경망.
    
    Graph-level RNN tracks global state; edge-level RNN generates
    새 마디마다의 이음.
    """

    def __init__(
        self,
        max_nodes: int,
        bfs_bandwidth: int,
        graph_hidden_dim: int = 128,
        edge_hidden_dim: int = 64,
        use_edge_rnn: bool = True,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.M = bfs_bandwidth  # 너비 우선 띠너비(잘라내기 창)
        self.graph_hidden_dim = graph_hidden_dim
        self.use_edge_rnn = use_edge_rnn

        # 그래프 켜 되돌이 신경망
        self.graph_rnn = nn.GRU(
            input_size=bfs_bandwidth,
            hidden_size=graph_hidden_dim,
            batch_first=True,
        )
        self.h0 = nn.Parameter(torch.zeros(1, 1, graph_hidden_dim))

        if use_edge_rnn:
            # 변 켜 되돌이 신경망을 갖춘 온전한 GraphRNN
            self.edge_rnn = EdgeRNN(edge_hidden_dim)
            self.edge_init = nn.Linear(graph_hidden_dim, edge_hidden_dim)
        else:
            # GraphRNN-S: 변 벡터 전체를 위한 여러 층 신경망
            self.edge_mlp = nn.Sequential(
                nn.Linear(graph_hidden_dim, graph_hidden_dim),
                nn.ReLU(),
                nn.Linear(graph_hidden_dim, bfs_bandwidth),
            )

    def _get_init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.h0.expand(1, batch_size, self.graph_hidden_dim).contiguous().to(device)

    def forward(
        self,
        edge_sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        익히기 앞으로 가기.
        
        인수:
            edge_sequences: (B, max_nodes, M) 너비 우선으로 자른 변 벡터
            lengths: (B,) 그래프마다 마디 개수
        """
        B, T, M = edge_sequences.shape
        device = edge_sequences.device
        h_graph = self._get_init_hidden(B, device)

        total_loss = torch.tensor(0.0, device=device)
        num_preds = 0

        for t in range(1, T):
            # 그래프 되돌이 신경망 들임: 앞 걸음의 변 벡터
            if t == 1:
                x_t = torch.zeros(B, 1, M, device=device)
            else:
                x_t = edge_sequences[:, t - 1, :].unsqueeze(1)

            out, h_graph = self.graph_rnn(x_t, h_graph)
            graph_state = out.squeeze(1)  # (B, graph_hidden_dim)

            # 어느 표본이 살아 있는가(마디 t을 가짐)
            active = (lengths > t).float()
            if active.sum() == 0:
                break

            # 변 헤아림
            targets = edge_sequences[:, t, :]  # (B, M)
            valid_len = min(t, M)

            if self.use_edge_rnn:
                h_edge_init = self.edge_init(graph_state)  # (B, edge_hidden_dim)
                edge_lengths = torch.full((B,), valid_len, device=device)
                logits = self.edge_rnn(h_edge_init, targets, edge_lengths)
            else:
                logits = self.edge_mlp(graph_state)

            # 올바른 변 자리의 가리개
            mask = torch.zeros(B, M, device=device)
            mask[:, :valid_len] = 1.0
            mask = mask * active.unsqueeze(1)

            loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            total_loss = total_loss + (loss * mask).sum()
            num_preds += mask.sum().item()

        total_loss = total_loss / max(num_preds, 1)

        return {"total_loss": total_loss}

    @torch.no_grad()
    def generate(
        self,
        num_graphs: int = 1,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> list[torch.Tensor]:
        """조상 뽑기로 그래프를 만든다."""
        self.eval()
        graphs = []

        for _ in range(num_graphs):
            h_graph = self._get_init_hidden(1, torch.device(device))
            edge_vectors = []
            x_t = torch.zeros(1, 1, self.M, device=device)

            for t in range(1, self.max_nodes):
                out, h_graph = self.graph_rnn(x_t, h_graph)
                graph_state = out.squeeze(1)

                valid_len = min(t, self.M)

                if self.use_edge_rnn:
                    h_edge = self.edge_init(graph_state[0])
                    edges = self.edge_rnn.generate(
                        h_edge.unsqueeze(0), valid_len, temperature
                    )
                    # M까지 채운다
                    edge_vec = torch.zeros(self.M, device=device)
                    edge_vec[:valid_len] = edges[:valid_len]
                else:
                    logits = self.edge_mlp(graph_state)[0]
                    probs = torch.sigmoid(logits / temperature)
                    edge_vec = torch.bernoulli(probs)
                    edge_vec[valid_len:] = 0  # 올바르지 않은 자리를 가린다

                # 끝 살피기: 올바른 자리가 모두 0
                if edge_vec[:valid_len].sum() == 0 and t > 1:
                    break

                edge_vectors.append(edge_vec[:valid_len].clone())
                x_t = edge_vec.unsqueeze(0).unsqueeze(0)

            # 이웃 행렬을 되짓는다
            n = len(edge_vectors) + 1
            adj = torch.zeros(n, n, device=device)
            for t, ev in enumerate(edge_vectors):
                step = t + 1
                for s in range(len(ev)):
                    target_node = step - s - 1
                    if target_node >= 0 and ev[s] > 0:
                        adj[step, target_node] = 1.0
                        adj[target_node, step] = 1.0

            graphs.append(adj.cpu())

        return graphs


def bfs_edge_sequences(
    adj: torch.Tensor,
    max_nodes: int,
    bandwidth: int,
) -> tuple[torch.Tensor, int]:
    """
    이웃 행렬을 너비 우선으로 자른 변 차례로 바꾼다.
    
    반환값:
        sequences: (max_nodes, bandwidth) 변 벡터
        n: 마디 개수
    """
    n = adj.size(0)

    # 차수가 가장 높은 마디에서 너비 우선 차례
    start = adj.sum(1).argmax().item()
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

    # 이웃 행렬을 다시 늘어놓는다
    perm = torch.tensor(order)
    adj_bfs = adj[perm][:, perm]

    # 너비 우선 잘라내기로 변 차례를 짓는다
    seq = torch.zeros(max_nodes, bandwidth)
    for t in range(1, min(n, max_nodes)):
        for s in range(min(t, bandwidth)):
            seq[t, s] = adj_bfs[t, t - s - 1]

    return seq, n


if __name__ == "__main__":
    torch.manual_seed(42)

    max_n = 20
    bw = 8  # 너비 우선 띠너비

    # 익히기 자료 만들기: 무리 얼개 그래프
    print("=== Preparing Training Data ===")
    graphs = []
    for _ in range(200):
        n = torch.randint(8, max_n, (1,)).item()
        # 무리 둘의 얼개
        n1 = n // 2
        n2 = n - n1
        adj = torch.zeros(n, n)
        # 무리 안의 변(빽빽함)
        for i in range(n1):
            for j in range(i + 1, n1):
                if torch.rand(1) < 0.4:
                    adj[i, j] = adj[j, i] = 1
        for i in range(n1, n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.4:
                    adj[i, j] = adj[j, i] = 1
        # 무리 사이의 변(성김)
        for i in range(n1):
            for j in range(n1, n):
                if torch.rand(1) < 0.05:
                    adj[i, j] = adj[j, i] = 1
        graphs.append(adj)

    # 차례를 채비한다
    sequences = []
    lengths = []
    for adj in graphs:
        seq, n = bfs_edge_sequences(adj, max_n, bw)
        sequences.append(seq)
        lengths.append(n)

    all_seqs = torch.stack(sequences)
    all_lens = torch.tensor(lengths)
    print(f"Data shape: {all_seqs.shape}, avg nodes: {all_lens.float().mean():.1f}")

    # GraphRNN 익히기
    print("\n=== Training GraphRNN (full) ===")
    model = GraphRNN(
        max_nodes=max_n,
        bfs_bandwidth=bw,
        graph_hidden_dim=128,
        edge_hidden_dim=64,
        use_edge_rnn=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(50):
        model.train()
        result = model(all_seqs, all_lens)
        loss = result["total_loss"]
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    # GraphRNN-S 익히기
    print("\n=== Training GraphRNN-S (simplified) ===")
    model_s = GraphRNN(
        max_nodes=max_n,
        bfs_bandwidth=bw,
        graph_hidden_dim=128,
        use_edge_rnn=False,
    )
    optimizer_s = torch.optim.Adam(model_s.parameters(), lr=0.001)
    print(f"Parameters: {sum(p.numel() for p in model_s.parameters()):,}")

    for epoch in range(50):
        model_s.train()
        result = model_s(all_seqs, all_lens)
        loss = result["total_loss"]
        optimizer_s.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
        optimizer_s.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    # 만들어 견준다
    print("\n=== Generation Results ===")
    for name, m in [("GraphRNN", model), ("GraphRNN-S", model_s)]:
        gen = m.generate(num_graphs=10)
        sizes = [g.size(0) for g in gen]
        edges = [int(g.sum().item()) // 2 for g in gen]
        densities = [2 * e / (n * (n - 1)) if n > 1 else 0
                     for n, e in zip(sizes, edges)]
        print(f"\n{name}:")
        print(f"  Avg nodes: {sum(sizes)/len(sizes):.1f} (ref: {all_lens.float().mean():.1f})")
        print(f"  Avg edges: {sum(edges)/len(edges):.1f}")
        print(f"  Avg density: {sum(densities)/len(densities):.3f}")
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
