# GraphRNN

GraphRNN(You et al., 2018)은 그래프 생성의 기초가 되는 자기회귀 모델이다. 그래프 생성을 두 수준으로 분해하는 계층적 순환 뉴럴 네트워크 구조를 내놓는다. 두 수준은 전역 생성 상태를 지니며 스텝마다 노드 하나를 만드는 **그래프 수준 순환 뉴럴 네트워크**와, 새 노드를 기존 노드에 잇는 엣지를 만드는 **엣지 수준 순환 뉴럴 네트워크**다. 이 두 수준 분해에 그래프의 국소성을 살리는 너비 우선 탐색(BFS) 순서를 더해 GraphRNN은 표현력이 높으면서도 계산량을 감당할 만하다.

---

## 1. 구조

GraphRNN은 순환 뉴럴 네트워크 둘을 계층적으로 쌓아 쓴다.

**그래프 수준 순환 뉴럴 네트워크($f_{\text{graph}}$).** 스텝 $t$까지 지은 그래프를 간추린 은닉 상태 $\mathbf{h}_t^G$를 지닌다. 스텝마다 앞 스텝의 엣지 벡터를 받아 엣지 수준 순환 뉴럴 네트워크의 첫 은닉 상태를 내놓는다:

$$
\mathbf{h}_t^G = f_{\text{graph}}(\mathbf{h}_{t-1}^G, \mathbf{a}_{t-1})
$$

여기서 $\mathbf{a}_{t-1} \in \{0,1\}^{M}$은 스텝 $t-1$의 (잘렸을 수도 있는) 엣지 벡터이고 $M$은 너비 우선 대역폭이다.

**엣지 수준 순환 뉴럴 네트워크($f_{\text{edge}}$).** $\mathbf{h}_t^G$에서 얻은 상태로 시작해 벡터 $\mathbf{a}_t$의 엣지를 차례로 예측한다:

$$
\mathbf{h}_{t,s}^E = f_{\text{edge}}(\mathbf{h}_{t,s-1}^E, a_{t,s-1})
$$

$$
p(a_{t,s} = 1) = \sigma(\text{MLP}(\mathbf{h}_{t,s}^E))
$$

여기서 $a_{t,s}$는 (너비 우선 순서에서) 노드 $t$와 노드 $t - s$ 사이의 엣지를 가리키고 $\sigma$는 시그모이드 함수다.

---

## 2. 너비 우선 잘라내기

계산에서의 핵심 통찰은 너비 우선 순서에서 새 노드의 엣지가 너비 우선 대역폭 $M$ 안으로만 되돌아간다는 것이다. 앞선 $t-1$개 노드 모두와의 연결을 예측하는 대신 GraphRNN은 스텝마다 엣지 결정을 $\min(t-1, M)$개만 예측한다. 이는 전체 엣지 예측 수를 $O(n^2)$에서 $O(n \cdot M)$으로 줄이며, 희소 그래프에서 $M$은 보통 $n$보다 훨씬 작다.

그래프의 너비 우선 대역폭은 다음과 같다:

$$
M = \max_{(u,v) \in \mathcal{E}} |\pi^{-1}(u) - \pi^{-1}(v)|
$$

여기서 $\pi$는 너비 우선 순서다. 많은 그래프 계열(커뮤니티 그래프, 분자 그래프, 소셜 네트워크)에서 $M \ll n$이어서 잘라내기는 정보를 거의 잃지 않는 쓸 만한 근사가 된다.

---

## 3. 트레인

GraphRNN은 두 수준 모두에서 Teacher Forcing과 함께 최대 우도로 트레인된다. 너비 우선 순서가 $\pi$이고 대역폭이 $M$인 그래프가 주어질 때:

$$
\mathcal{L} = -\sum_{t=2}^{n} \sum_{s=1}^{\min(t-1, M)} \left[ a_{t,s}^* \log p_\theta(a_{t,s} = 1 \mid \mathbf{h}_{t,s-1}^E) + (1 - a_{t,s}^*) \log p_\theta(a_{t,s} = 0 \mid \mathbf{h}_{t,s-1}^E) \right]
$$

트레인하는 동안 참 엣지 값 $a_{t,s}^*$를 엣지 수준 순환 뉴럴 네트워크의 입력으로 넣는다(Teacher Forcing). 엣지 수준 순환 뉴럴 네트워크는 스텝 $t$마다 그래프 수준 상태를 학습된 변환으로 옮겨 초깃값을 잡는다:

$$
\mathbf{h}_{t,0}^E = \text{MLP}_{\text{init}}(\mathbf{h}_t^G)
$$

---

## 4. 간단히 한 변형: GraphRNN-S

계산 효율을 위해 GraphRNN-S는 엣지 수준 순환 뉴럴 네트워크를 엣지 벡터 전체를 한꺼번에 예측하는 다층 뉴럴 네트워크 하나로 바꾼다:

$$
\hat{\mathbf{a}}_t = \sigma(\text{MLP}(\mathbf{h}_t^G)) \in [0,1]^{M}
$$

이는 한 스텝 안의 엣지 결정 사이의 차례 제약을 없애 표현력과 병렬성을 맞바꾼다. GraphRNN-S는 트레인이 훨씬 빠르지만 스텝 안의 엣지 의존성이 중요한 데이터셋에서는 품질이 떨어지는 그래프를 낼 수 있다.

---

## 5. 생성 절차

모델이 종료 신호를 낼 때까지 노드를 하나씩 만들어 간다:

1. $\mathbf{h}_0^G$를 배운 벡터나 0 벡터로 둔다
2. $t = 1, 2, \ldots, n_{\max}$에 대해:
    - $\mathbf{h}_t^G = f_{\text{graph}}(\mathbf{h}_{t-1}^G, \mathbf{a}_{t-1})$을 계산한다
    - $\mathbf{h}_{t,0}^E = \text{MLP}_{\text{init}}(\mathbf{h}_t^G)$으로 첫 값을 잡는다
    - $s = 1, \ldots, M$에 대해:
        - $p_{t,s} = \sigma(\text{MLP}(\mathbf{h}_{t,s}^E))$를 계산한다
        - $a_{t,s} \sim \text{Bernoulli}(p_{t,s})$를 뽑는다
        - $\mathbf{h}_{t,s}^E = f_{\text{edge}}(\mathbf{h}_{t,s-1}^E, a_{t,s})$로 갱신한다
    - 모든 $s$에 대해 $a_{t,s} = 0$이면(끝 패턴) 생성을 멈춘다
3. 엣지 시퀀스에서 인접 행렬을 복원한다

종료 조건은 특별한 시퀀스 끝 패턴을 쓴다. 엣지 벡터 $\mathbf{a}_t$가 통째로 0이면(앞선 어느 노드와도 잇지 않으면) 생성을 멈춘다. 이는 자연스럽게 크기가 제각각인 그래프를 낸다.

---

## 6. 한계

**순서 민감도.** 같은 그래프를 너비 우선으로 다르게 밟으면 트레인 순서가 달라진다. 여러 순서로 데이터를 늘리면 도움이 되지만 모델은 여전히 순서에 의존하는 분포를 암묵적으로 배운다. $p_\theta(\mathcal{G} \mid \pi)$와 참 $p(\mathcal{G})$의 차이는 고른 순서 방식이 구조의 규칙성을 얼마나 잘 담느냐에 달려 있다.

**장거리 의존성.** 그래프 수준 순환 뉴럴 네트워크는 스텝이 많을 수 있는데도 정보를 계속 전파해야 한다. 큰 그래프에서는 뒤의 생성 선택에 영향을 주는 이른 구조 결정을 은닉 상태가 담지 못할 수 있다.

**확장성.** 트레인과 생성이 노드를 가로질러 본질적으로 순차적이어서 병렬화가 제한된다. 노드가 수백 개인 그래프에서는 너비 우선 잘라내기를 해도 트레인이 느릴 수 있다.

---

## 7. 금융 응용: 은행 간 네트워크 생성

GraphRNN은 순차 생성 방식이 시간에 따른 네트워크 형성을 그대로 비추므로 합성 은행 간 대출 네트워크를 만드는 데 잘 맞는다. 은행은 은행 간 시장에 들어와 대출 관계를 차츰 맺는다. 너비 우선 순서는 금융 네트워크의 계층 구조를 자연스럽게 담는다. 핵심 은행(차수가 높음)이 먼저 나오고 바깥 기관이 주로 가까운 핵심 노드에 이어진다.

---

## 8. 구현: 너비 우선 잘라내기를 쓴 GraphRNN

```python
"""
GraphRNN: 너비 우선 잘라내기를 쓰는 그래프 만들기의 켜진 회귀 뉴럴 네트워크.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional

class EdgeRNN(nn.Module):
    """변 켜 회귀 뉴럴 네트워크: 변 벡터를 차례로 만든다."""

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
            h_init: (B, hidden_dim) 그래프 회귀 뉴럴 네트워크에서 온 첫 숨은 상태
            edge_targets: (B, M) 참 변 벡터
            lengths: (B,) 표본마다 올바른 길이
            
        반환값:
            logits: (B, M) 변 로짓
        """
        batch_size, M = edge_targets.shape
        device = edge_targets.device

        # 들임 채비: 시작 토막(0)과 함께 밀린 타깃
        sos = torch.zeros(batch_size, 1, 1, device=device)
        inputs = edge_targets[:, :-1].unsqueeze(-1)  # (B, M-1, 1)
        inputs = torch.cat([sos, inputs], dim=1)  # (B, M, 1)

        # 변 회귀 뉴럴 네트워크를 돌린다
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
        변 벡터를 자기 회귀로 만든다.
        
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
    GraphRNN: 그래프 만들기의 두 켜 켜진 회귀 뉴럴 네트워크.
    
    그래프 레이어 되도는 네트워크가 온 상태를 좇고, 변 레이어 되도는 네트워크가
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

        # 그래프 켜 회귀 뉴럴 네트워크
        self.graph_rnn = nn.GRU(
            input_size=bfs_bandwidth,
            hidden_size=graph_hidden_dim,
            batch_first=True,
        )
        self.h0 = nn.Parameter(torch.zeros(1, 1, graph_hidden_dim))

        if use_edge_rnn:
            # 변 켜 회귀 뉴럴 네트워크를 갖춘 온전한 GraphRNN
            self.edge_rnn = EdgeRNN(edge_hidden_dim)
            self.edge_init = nn.Linear(graph_hidden_dim, edge_hidden_dim)
        else:
            # GraphRNN-S: 변 벡터 전체를 위한 여러 레이어 뉴럴 네트워크
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
            # 그래프 회귀 뉴럴 네트워크 들임: 앞 스텝의 변 벡터
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

            # 올바른 변 자리의 마스크
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

    # 익히기 자료 만들기: 무리 구조 그래프
    print("=== Preparing Training Data ===")
    graphs = []
    for _ in range(200):
        n = torch.randint(8, max_n, (1,)).item()
        # 무리 둘의 구조
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

**출력:**

```
=== Preparing Training Data ===
Data shape: torch.Size([200, 20, 8]), avg nodes: 13.4

=== Training GraphRNN (full) ===
Parameters: 76,353
Epoch 10: loss=0.6773
Epoch 20: loss=0.5673
Epoch 30: loss=0.5488
Epoch 40: loss=0.5427
Epoch 50: loss=0.5398

=== Training GraphRNN-S (simplified) ===
Parameters: 70,664
Epoch 10: loss=0.6473
Epoch 20: loss=0.5659
Epoch 30: loss=0.5451
Epoch 40: loss=0.5395
Epoch 50: loss=0.5352

=== Generation Results ===

GraphRNN:
  Avg nodes: 5.5 (ref: 13.4)
  Avg edges: 7.1
  Avg density: 0.606

GraphRNN-S:
  Avg nodes: 8.8 (ref: 13.4)
  Avg edges: 13.9
  Avg density: 0.389
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

이 절에서는 구조, 너비 우선 잘라내기, 트레인, 간단히 한 변형인 GraphRNN-S를 살펴보았다.
