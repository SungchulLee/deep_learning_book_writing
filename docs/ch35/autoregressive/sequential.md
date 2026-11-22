# 순차 그래프 생성

자기회귀 그래프 생성은 결합 분포 $p_\theta(\mathcal{G})$를 조건부 분포의 곱으로 쪼개어 원소를 하나씩 만든다. 이 틀은 크기가 제각각인 그래프를 자연스럽게 다루고 인접 행렬 전체를 한꺼번에 내놓는 조합적 어려움을 피한다. 핵심 생각은 그래프 생성 액션에 순서를 매기고 확률의 사슬 규칙으로 우도를 인수 분해하는 것이다.

---

## 1. 자기회귀 인수 분해

노드가 $n$개인 그래프 $\mathcal{G} = (\mathbf{A}, \mathbf{X})$가 주어질 때, 각 액션 $a_t$가 노드를 더하거나 엣지를 더하거나 생성을 끝내는 생성 순서열 $\sigma = (a_1, a_2, \ldots, a_T)$를 정의한다. 결합 확률은 다음과 같이 인수 분해된다:

$$
p_\theta(\mathcal{G}) = \sum_{\sigma \in \Sigma(\mathcal{G})} \prod_{t=1}^{T} p_\theta(a_t \mid a_{1:t-1})
$$

여기서 $\Sigma(\mathcal{G})$는 $\mathcal{G}$를 내는 올바른 생성 순서열의 집합이다. 순서열에 대한 주변화는 계산할 수 없으므로($|\Sigma(\mathcal{G})|$가 지수적으로 클 수 있다) 실제 방법은 표준 순서 $\sigma^*$를 고정하고 다음을 최대화한다.

$$
\log p_\theta(\mathcal{G}) \geq \log \prod_{t=1}^{T} p_\theta(a_t^* \mid a_{1:t-1}^*)
$$

그래프마다 생성 순서열이 하나로 정해져 있으면 이 하한은 등식이 된다.

---

## 2. 노드 수준의 분해

가장 흔한 분해는 노드를 하나씩 더한다. 스텝 $t$에서 모델은 다음을 한다.

1. **새 노드를 더할지** 끝낼지 정한다: $p_\theta(\text{stop} \mid \mathcal{G}_{<t})$
2. **노드 피처를 만든다**: $p_\theta(\mathbf{x}_t \mid \mathcal{G}_{<t})$
3. 기존 노드 모두와의 **엣지를 만든다**: $p_\theta(\mathbf{a}_t \mid \mathbf{x}_t, \mathcal{G}_{<t})$

여기서 $\mathcal{G}_{<t}$는 지금까지 만든 부분 그래프를, $\mathbf{a}_t \in \{0,1\}^{t-1}$은 노드 $t$를 노드 $1, \ldots, t-1$에 잇는 엣지 벡터를 뜻한다.

노드 순서 $\pi$ 아래 노드가 $n$개인 그래프의 로그 우도는 다음과 같다:

$$
\log p_\theta(\mathcal{G} \mid \pi) = \sum_{t=1}^{n} \left[ \log p_\theta(\mathbf{x}_{\pi(t)} \mid \mathcal{G}_{<t}) + \sum_{s=1}^{t-1} \log p_\theta(A_{\pi(t),\pi(s)} \mid \mathbf{x}_{\pi(t)}, \mathcal{G}_{<t}) \right] + \log p_\theta(\text{stop} \mid \mathcal{G})
$$

---

## 3. 엣지 수준의 분해

다른 길은 생성을 엣지 수준에서 분해해 가능한 모든 노드 짝을 훑는다:

$$
p_\theta(\mathbf{A} \mid \mathbf{X}) = \prod_{i=1}^{n} \prod_{j=1}^{i-1} p_\theta(A_{ij} \mid A_{<(i,j)}, \mathbf{X})
$$

여기서 곱은 인접 행렬의 하삼각 부분($j < i$)을 행 순서로 훑는 차례를 따른다. 엣지 결정 하나하나가 앞서 정해진 모든 엣지에 조건부인 베르누이 변수다.

---

## 4. 순서 전략

노드 순서 $\pi$를 어떻게 고르느냐가 생성 품질과 계산 비용을 근본적으로 좌우한다.

**너비 우선 순서**는 차수가 가장 높은 노드에서 시작해 그래프를 너비 우선으로 탐색한다. 이렇게 하면 엣지가 재배열한 인접 행렬의 대각선 근처에 모여 맥락 창을 잘라 쓸 수 있다. 핵심 성질은, $B$가 너비 우선 대역폭일 때 너비 우선 순서에서 $i < j$인 엣지 $(i, j)$가 $j - i \leq B$를 만족한다는 것이다. 곧 노드 $j$는 앞선 $j-1$개 노드 모두가 아니라 가장 가까운 앞선 $B$개와의 연결만 살피면 된다.

**깊이 우선 순서**도 비슷한 국소성을 내지만 구조 성질이 다르다. 깊이 우선 탐색은 갈라지기 전에 긴 사슬을 만들어 트리 같은 그래프에 유리할 수 있다.

데이터 증강을 곁들인 **무작위 순서**는 그래프마다 여러 무작위 순서로 모델을 트레인시켜 몬테카를로 샘플링으로 $\Sigma(\mathcal{G})$에 대한 주변화를 근사한다:

$$
\log p_\theta(\mathcal{G}) \approx \log \frac{1}{K} \sum_{k=1}^{K} \prod_{t=1}^{T} p_\theta(a_t^{(k)} \mid a_{1:t-1}^{(k)})
$$

---

## 5. 상태 표현

생성 스텝 $t$마다 모델은 지금까지 만든 부분 그래프 $\mathcal{G}_{<t}$를 차원이 고정된 상태 벡터로 인코딩해야 한다. 두 방법이 주로 쓰인다.

**순환 인코딩.** GRU나 LSTM이 스텝마다 갱신되는 은닉 상태 $\mathbf{h}_t$를 지닌다.

$$
\mathbf{h}_t = \text{GRU}(\mathbf{h}_{t-1}, \mathbf{e}_t)
$$

여기서 $\mathbf{e}_t$는 스텝 $t$에서 한 액션(노드 피처와 엣지 결정)을 인코딩한다. 은닉 상태가 생성의 전체 이력을 암묵적으로 요약한다.

**그래프 뉴럴 네트워크 인코딩.** 스텝마다 $\mathcal{G}_{<t}$에 그래프 뉴럴 네트워크를 써서 노드 임베딩을 낸 뒤 그래프 수준의 표현으로 모은다:

$$
\mathbf{h}_t = \text{READOUT}(\text{GNN}(\mathcal{G}_{<t}))
$$

표현력은 더 높지만, 커지는 그래프에 스텝마다 그래프 뉴럴 네트워크를 다시 적용해야 하므로 비용이 더 크다.

---

## 6. 트레인 절차

트레인은 Teacher Forcing을 쓴다. 스텝 $t$마다 모델은 자기 예측이 아니라 참 부분 그래프 $\mathcal{G}_{<t}^*$를 받는다. 로스는 모든 스텝에 걸쳐 더한 음의 로그 우도다:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log p_\theta(a_t^* \mid \mathcal{G}_{<t}^*)
$$

엣지 예측에서는 이것이 엣지 결정마다의 이진 교차 엔트로피로 줄어든다:

$$
\mathcal{L}_{\text{edge}} = -\sum_{t=2}^{n} \sum_{s=1}^{t-1} \left[ A_{ts}^* \log \hat{p}_{ts} + (1 - A_{ts}^*) \log (1 - \hat{p}_{ts}) \right]
$$

여기서 $\hat{p}_{ts} = p_\theta(A_{ts} = 1 \mid \mathcal{G}_{<t}^*)$이다.

---

## 7. 노출 바이어스와 완화

Teacher Forcing은 트레인과 테스트의 불일치를 낳는다. 만들 때 모델은 참값이 아니라 (틀렸을 수도 있는) 제 예측에 달려 있다. 이 **노출 바이어스** 때문에 긴 시퀀스에서 오차가 쌓일 수 있다.

완화 전략은 다음과 같다.

- **스케줄드 샘플링**: 스텝 $t$에서 확률 $\epsilon_t$로 참값 대신 모델의 예측을 쓰고, 트레인이 진행될수록 $\epsilon_t$를 키운다
- **시퀀스 수준 목표**: 그래프 수준 지표를 리워드로 삼아 REINFORCE로 트레인한다
- **커리큘럼 학습**: 작은 그래프에서 시작해 트레인하는 동안 크기를 차츰 키운다

---

## 8. 금융 응용: 차례로 네트워크 짓기

금융 네트워크 생성에서 순차 생성은 실제 네트워크가 생기는 모습을 그대로 비춘다. 은행이 시간에 따라 은행 간 시장에 들어와 대출 관계를 차츰 맺는다. 자기회귀 틀은 이 시간적 흐름을 나타낼 수 있다.

$$
p(\mathcal{G}_T) = \prod_{t=1}^{T} p(\text{bank}_t \text{ joins}) \cdot \prod_{s < t} p(\text{lends}(t, s) \mid \text{attributes}, \mathcal{G}_{<t})
$$

이는 선호적 연결(새 은행이 연결이 많은 기존 은행에 먼저 붙는다)과 동종 선호(homophily, 속성이 닮은 은행끼리 이어진다)를 담으며 둘 다 금융 네트워크에서 실제로 관찰된다.

---

## 9. 구현: 차례로 만드는 그래프 생성기

```python
"""
차례로 하는(자기 회귀) 그래프 만들기 틀.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import deque

class SequentialGraphGenerator(nn.Module):
    """
    마디 하나씩 만드는 자기 회귀 그래프 생성기.
    
    At each step t:
    1. 그래프 레이어 되도는 네트워크의 상태가 G_{<t}를 간추린다
    2. 변 여러 레이어 뉴럴 네트워크가 앞선 마디와의 이음을 헤아린다
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

        # 그래프 켜 회귀 뉴럴 네트워크: 만들기의 지난 일을 간추린다
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
                adj_sequences[b, t, :t]는 마디 t에서 마디 0..t-1로 가는 변이다
            lengths: (B,) 그래프마다 마디 개수
            
        반환값:
            'edge_loss'와 'stop_loss'를 담은 사전
        """
        batch_size = adj_sequences.size(0)
        device = adj_sequences.device
        h = self._get_init_hidden(batch_size).to(device)

        edge_loss = torch.tensor(0.0, device=device)
        stop_loss = torch.tensor(0.0, device=device)
        num_edge_preds = 0
        num_stop_preds = 0

        for t in range(1, self.max_nodes):
            # 들임: 앞 스텝의 변 벡터
            if t == 1:
                x_t = torch.zeros(batch_size, 1, self.max_nodes, device=device)
            else:
                x_t = adj_sequences[:, t - 1, :].unsqueeze(1)  # (B, 1, max_nodes)

            # 회귀 뉴럴 네트워크 스텝
            out, h = self.graph_rnn(x_t, h)
            graph_state = out.squeeze(1)  # (B, hidden_dim)

            # 스텝 t의 변 헤아림
            edge_logits = self.edge_mlp(graph_state)  # (B, max_nodes)
            edge_targets = adj_sequences[:, t, :]  # (B, max_nodes)

            # 올바른 자리에서만 로스를 셈한다(마디 t은 0..t-1에 이어진다)
            # 그리고 마디가 적어도 t+1개인 그래프에서만
            active = (lengths > t).float()  # (B,)
            if active.sum() > 0:
                # 마스크: 자리 0..t-1만 올바른 변 타깃이다
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
            # 타깃: 마지막 마디면 stop=1
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
        자기 회귀로 그래프를 만든다.
        
        반환값:
            이웃 행렬의 목록(크기가 저마다 다르다)
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

    # 트레인
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

**출력:**

```
=== Preparing Training Data ===
Training data: torch.Size([100, 15, 15]), lengths: 9.8 avg

=== Training ===
Epoch 10: loss=1.2960 (edge=0.6738, stop=0.6223)
Epoch 20: loss=1.1268 (edge=0.6294, stop=0.4975)
Epoch 30: loss=0.9484 (edge=0.4856, stop=0.4629)

=== Generation ===
Graph 0: 5 nodes, 3 edges, density=0.300
Graph 1: 3 nodes, 1 edges, density=0.333
Graph 2: 15 nodes, 28 edges, density=0.267
Graph 3: 3 nodes, 1 edges, density=0.333
Graph 4: 3 nodes, 0 edges, density=0.000
Graph 5: 15 nodes, 24 edges, density=0.229
Graph 6: 4 nodes, 1 edges, density=0.167
Graph 7: 3 nodes, 2 edges, density=0.667
Graph 8: 3 nodes, 0 edges, density=0.000
Graph 9: 12 nodes, 14 edges, density=0.212

Ref avg size: 9.8
Gen avg size: 6.6
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

이 절에서는 자기회귀 인수 분해, 노드 수준의 분해, 엣지 수준의 분해, 순서 전략을 살펴보았다.
