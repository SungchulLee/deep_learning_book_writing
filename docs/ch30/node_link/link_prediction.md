# 이음 헤아리기

실제 그래프는 온전하지 않은 것이 많다. 사회 그물에는 아직 찾지 못한
벗 관계가 빠져 있고, 앎 그래프에는 관계가 빠져 있으며, 금융 그물에는
보지 못한 거래가 있다. 이음 헤아리기는 본 그래프 얼개와 마디 특징을 바탕으로
두 마디 사이에 변이 있을 법함에 점수를 매겨
이 문제를 다룬다.

---

## 1. 문제 정식화

마디 특징이 $X$인 그래프 $G = (V, E)$이 주어질 때 목표는 점수 함수
하나를 배우는 것이다

$$
s(u, v) = f(\mathbf{h}_u, \mathbf{h}_v) \in \mathbb{R}
$$

여기서 $\mathbf{h}_u, \mathbf{h}_v$은 배운 마디 박아 넣기(보통 그래프 신경망에서 나온다)이고
$f$은 변 $(u, v)$이 있을 법할 때 높은 점수를
내놓는다.

---

## 2. 점수 함수

### 안쪽 곱

가장 단순한 길:

$$
s(u, v) = \mathbf{h}_u^{\top} \mathbf{h}_v
$$

셈이 효율 좋지만 관계가 맞섬이라고 여긴다.

### 겹선형(DistMult)

배울 수 있는 대각 행렬을 들인다:

$$
s(u, v) = \mathbf{h}_u^{\top} \text{diag}(\mathbf{r}) \, \mathbf{h}_v
$$

여기서 $\mathbf{r}$은 관계마다 다른 잡 벡터이다. 앎 그래프 채우기에
널리 쓰인다.

### 여러 층 신경망 풀개

박아 넣기를 이어 붙여 신경망에 넣는다:

$$
s(u, v) = \text{MLP}([\mathbf{h}_u \| \mathbf{h}_v])
$$

안쪽 곱보다 나타냄 힘이 세지만 헤아릴 때 더 느리다.

### TransE(옮김 바탕)

변을 박아 넣기 자리에서의 옮김으로 나타낸다:

$$
s(u, v) = -\|\mathbf{h}_u + \mathbf{r} - \mathbf{h}_v\|
$$

여기서 $\mathbf{r}$은 관계마다 다른 옮김 벡터이다. 점수가 높을수록
(0에 가까울수록) 변이 있을 법함을 가리킨다.

---

## 3. 학습

### 음의 뽑기

본 변 $(u, v) \in E$(양의 표본)마다 $v'$을 아무렇게나 골라
$(u, v') \notin E$인 음의 표본 $(u, v')$을 $k$개 만든다.
손실은 양의 변이 음의 변보다 높은 점수를 받도록 이끈다.

### 손실 함수

**두 값 교차 엔트로피:**

$$
\mathcal{L} = -\sum_{(u,v) \in E} \log \sigma(s(u,v)) - \sum_{(u,v') \notin E} \log(1 - \sigma(s(u,v')))
$$

**여백 바탕(경첩) 손실:**

$$
\mathcal{L} = \sum_{(u,v) \in E} \sum_{(u,v') \notin E} \max\bigl(0, \, \gamma - s(u,v) + s(u,v')\bigr)
$$

여기서 $\gamma > 0$은 여백 웃잡이다.

---

## 4. 평가 지표

| 잣대 | 밝힘 |
|---|---|
| AUC-ROC | ROC 굽은 줄 아래 넓이. 매김 품질을 잰다 |
| 평균 정밀도 | 정밀도-되불러옴 굽은 줄 아래 넓이. 갈래 치우침에 튼튼하다 |
| Hits@K | 참인 변 가운데 위 $K$개 안에 든 몫 |
| MRR | 참인 변의 평균 거꿀 매김 |

!!! note "자료 나누기"
    이음 헤아리기는 보통 때 차례나 아무 나누기를 쓴다. 익히기 변이
    본 그래프를 이루고 모델은 남겨 둔 변을 헤아린다. 쪽지용 변(그래프 신경망 셈에 쓰는 것)과
    가르침용 변(손실에 쓰는 것)은 자료가 새지 않도록
    조심해서 다루어야 한다.

---

## 5. 구현

```python
"""
그래프 신경망 박아 넣기와 안쪽 곱 점수로 하는 이음 헤아리기.

음의 뽑기와 두 값 교차 엔트로피 손실로 익히기를 보여 준다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# === 단순 그래프 신경망 담개 ===
class GCNEncoder(nn.Module):
    """마디 박아 넣기를 내놓는 두 층 GCN."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # 간단히 한 GCN: A * X * W
        h = F.relu(self.w1(adj @ x))
        h = self.w2(adj @ h)
        return h

# === 이음 헤아리개 ===
class LinkPredictor(nn.Module):
    """안쪽 곱 이음 헤아리기 모델."""

    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, emb_dim)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, adj)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor,
        pos_edges: torch.Tensor, neg_edges: torch.Tensor
    ) -> torch.Tensor:
        z = self.encode(x, adj)
        pos_scores = self.decode(z, pos_edges)
        neg_scores = self.decode(z, neg_edges)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores),
        ])
        return F.binary_cross_entropy_with_logits(scores, labels)

# === 보기 ===
if __name__ == "__main__":
    torch.manual_seed(42)
    n, d = 6, 4
    x = torch.randn(n, d)
    adj = torch.eye(n)
    adj[0, 1] = adj[1, 0] = 1
    adj[1, 2] = adj[2, 1] = 1
    adj[2, 3] = adj[3, 2] = 1

    model = LinkPredictor(in_dim=d, hidden_dim=8, emb_dim=4)
    pos_edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    neg_edges = torch.tensor([[0, 3, 4], [4, 5, 5]])

    loss = model(x, adj, pos_edges, neg_edges)
    print(f"Loss: {loss.item():.4f}")
```

**출력:**

```
Loss: 0.4241
```

---

## 연습문제

**연습문제 1.**
이음 헤아리기 일을 뜻매김하고 여느 따지기 짜임을 밝혀라.

??? success "연습문제 1 풀이"
    이음 헤아리기: 변을 얼마간 덜어 낸 그래프가 주어질 때 빠진 변 가운데 어느 것이 있을 법한지 헤아린다. 따지기: 변을 익히기/살피기/시험 모임으로 나눈다. 모델은 익히기 그래프를 보고 뽑힌 변에 점수를 매긴다. 양의(실제) 변마다 음의(없는) 변을 뽑는다. 잣대: AUC-ROC(ROC 굽은 줄 아래 넓이), 평균 정밀도, Hits@K(양의 변 가운데 위 K개에 든 몫). 흔한 나누기: 익히기 85%, 살피기 5%, 시험 10%이며 음의 표본도 같은 수만큼 둔다.

---

**연습문제 2.**
이음 헤아리기의 단순한 어림짐작 밑그림 셋을 밝혀라: 공통 이웃, 자카드, 애더믹-애더.

??? success "연습문제 2 풀이"
    공통 이웃: $\text{CN}(u,v) = |N(u) \cap N(v)|$. 함께 가진 이웃이 많을수록 이음 확률이 높다. 자카드: $\text{JC}(u,v) = |N(u) \cap N(v)| / |N(u) \cup N(v)|$. 온 이웃 수로 고르게 맞추어 차수가 높은 마디에 벌을 준다. 애더믹-애더: $\text{AA}(u,v) = \sum_{w \in N(u) \cap N(v)} 1/\log|N(w)|$. 차수가 낮은 공통 이웃이 더 크게 이바지한다(앎이 더 많기 때문이다). 이 어림짐작들은 뜻밖에 힘이 세고 $d$이 평균 차수일 때 마디 짝마다 $O(d^2)$에 돈다.

---

**연습문제 3.**
그래프 신경망 바탕 이음 헤아리기는 어떻게 도는가? 담개-풀개 틀을 밝혀라.

??? success "연습문제 3 풀이"
    담개: 그래프 신경망이 얼개와 특징의 앎을 담은 마디 박아 넣기 $z_u = \text{GNN}(G, X)_u$을 셈한다. 풀개: 점수 함수가 변이 있는지 헤아린다. $\text{score}(u,v) = \sigma(z_u^T z_v)$(안쪽 곱)이거나 $\text{score}(u,v) = \text{MLP}(z_u \| z_v)$(이어 붙이기 + 여러 층 신경망)이다. 익히기: 헤아린 점수와 참 이름표(양의 변은 1, 음의 표본은 0) 사이의 두 값 교차 엔트로피를 가장 작게 한다. 그래프 신경망 담개는 어림짐작을 넓힌다. 공통 이웃, 애더믹-애더, 더 복잡한 무늬까지 어림하도록 배울 수 있다.

---

**연습문제 4.**
이음 헤아리기에서 쪽지 건네기의 한계와 SEAL 같은 방법이 이를 어떻게 다루는지 밝혀라.

??? success "연습문제 4 풀이"
    여느 그래프 신경망은 마디 박아 넣기를 따로 셈한 뒤 합쳐 변 점수를 매긴다. 이러면 우연히 같은 박아 넣기를 받은, 얼개가 다른 마디 짝을 가려내지 못한다(보기로 그 자리 이웃은 같지만 겹침 무늬가 다른 두 짝). SEAL은 과녁 변 $(u,v)$ 둘레의 그 자리 부분 그래프를 뽑아내고 $u$과 $v$까지의 거리로 마디에 이름표를 매긴 뒤(두 반지름 마디 이름표 매기기) 이 이름표 붙은 부분 그래프에 그래프 신경망을 돌린다. 이는 이음 얼개를 드러나게 담아 모델이 자료에서 어떤 이음 어림짐작이든 배울 수 있게 한다.

## 정리하며

이 마당은 문제 정식화、점수 함수、학습、평가 지표을 차례로 짚었다.

**참고 문헌**

- Zhang, M. & Chen, Y. "Link Prediction Based on Graph Neural Networks."
  NeurIPS 2018.
- Bordes, A. et al. "Translating Embeddings for Modeling Multi-relational
  Data." NeurIPS 2013.
