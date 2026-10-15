# 노드 임베딩

노드 임베딩은 노드 ID를 저차원 벡터에 대응시키는 방법이다. 이 예제는 그래프에서 랜덤 워크를 만들고, 워크 안에서 가까이 등장하는 노드 쌍에 높은 내적을 주도록 두 임베딩 테이블을 학습한다. 노드 특징을 이웃과 함께 처리하는 GNN과 달리, 여기서는 노드마다 직접 학습할 벡터를 저장한다.

## 1. 코드와 출력

```python
"""29.6.3: 마디 임베딩 - DeepWalk과 Node2Vec."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx
from collections import defaultdict

# ========================================================================
# 메인
# ========================================================================

def random_walk(G, start, length):
    walk = [start]
    for _ in range(length - 1):
        neighbors = list(G.neighbors(walk[-1]))
        if not neighbors: break
        walk.append(np.random.choice(neighbors))
    return walk

def node2vec_walk(G, start, length, p=1.0, q=1.0):
    walk = [start]
    if len(list(G.neighbors(start))) == 0: return walk
    walk.append(np.random.choice(list(G.neighbors(start))))
    for _ in range(length - 2):
        cur = walk[-1]; prev = walk[-2]
        neighbors = list(G.neighbors(cur))
        if not neighbors: break
        probs = []
        for n in neighbors:
            if n == prev: probs.append(1.0/p)
            elif G.has_edge(n, prev): probs.append(1.0)
            else: probs.append(1.0/q)
        probs = np.array(probs); probs /= probs.sum()
        walk.append(np.random.choice(neighbors, p=probs))
    return walk

class SkipGramEmbedding(nn.Module):
    def __init__(self, n_nodes, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(n_nodes, embed_dim)
        self.context = nn.Embedding(n_nodes, embed_dim)
    def forward(self, center, context, neg):
        c = self.embed(center); ctx = self.context(context); n = self.context(neg)
        pos_score = (c * ctx).sum(dim=-1)
        neg_score = (c.unsqueeze(1) * n).sum(dim=-1)
        return -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()

def demo():
    print("=" * 60); print("Node Embedding"); print("=" * 60)
    np.random.seed(42); torch.manual_seed(42)
    G = nx.karate_club_graph(); n = G.number_of_nodes()
    # 걸음 만들기
    walks = []
    for _ in range(10):
        for node in range(n):
            walks.append(node2vec_walk(G, node, length=10, p=1, q=0.5))
    # 익히기 짝 짓기
    window = 3; centers, contexts = [], []
    for walk in walks:
        for i, c in enumerate(walk):
            for j in range(max(0,i-window), min(len(walk),i+window+1)):
                if i != j: centers.append(c); contexts.append(walk[j])
    model = SkipGramEmbedding(n, 16)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    centers_t = torch.tensor(centers); contexts_t = torch.tensor(contexts)
    for epoch in range(50):
        idx = torch.randperm(len(centers))[:512]
        c, ctx = centers_t[idx], contexts_t[idx]
        neg = torch.randint(0, n, (512, 5))
        loss = model(c, ctx, neg)
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 10 == 0: print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
    # 임베딩 살피기
    emb = model.embed.weight.detach()
    y = [0 if G.nodes[i].get('club','')=='Mr. Hi' else 1 for i in range(n)]
    c0 = emb[torch.tensor([i for i,l in enumerate(y) if l==0])].mean(0)
    c1 = emb[torch.tensor([i for i,l in enumerate(y) if l==1])].mean(0)
    print(f"  Cluster distance: {(c0-c1).norm():.4f}")

if __name__ == "__main__":
    demo()
```

```
============================================================
Node Embedding
============================================================
  Epoch 10: Loss = 3.1935
  Epoch 20: Loss = 2.5519
  Epoch 30: Loss = 2.2658
  Epoch 40: Loss = 2.0507
  Epoch 50: Loss = 1.8794
  Cluster distance: 1.4172
```

기존 출력은 손실과 두 그룹 중심의 거리다. 이번 편집에서는 재실행하지 않았다. 이 값만으로 분류 정확도나 새로운 노드에 대한 성능을 확인할 수는 없다.

## 2. node2vec 방식의 랜덤 워크

현재 노드 v, 직전에 방문한 노드 t에서 다음 이웃 x를 고를 때, 코드의 가중치는 다음과 같다. 데모는 무방향 그래프이며 엣지 가중치는 사용하지 않는다.

$$a(t,x)=\begin{cases}
1/p & x=t\\
1 & x\ne t\text{이고 }(t,x)\in E\\
1/q & \text{그 밖의 경우}
\end{cases}$$

가중치를 현재 이웃 전체의 합으로 나누어 전이 확률을 만든다. p가 작으면 바로 되돌아갈 상대적 가중치가 커지고, q가 작으면 직전 노드와 연결되지 않은 이웃으로 이동할 상대적 가중치가 커진다. 이는 탐색의 편향이며, 특정 커뮤니티나 구조적 역할을 반드시 복원한다는 보장은 아니다. [node2vec 원 논문](https://arxiv.org/abs/1607.00653)은 이러한 2차 랜덤 워크와 임베딩 학습을 제안한다.

Karate Club의 34개 노드마다 10번, 길이 10의 워크를 생성하므로 이 그래프에서는 총 340개 워크다. 기본 random_walk 함수는 정의만 하고 데모에서는 node2vec_walk를 p=1, q=0.5로 사용한다. 고립 노드는 시작 노드만 반환한다. 일반 함수의 p·q는 양수여야 하며, 길이 1을 요청해도 node2vec_walk가 이웃을 하나 덧붙이는 경계 동작에 주의한다.

## 3. 문맥 쌍과 네거티브 샘플링

각 위치의 앞뒤 최대 세 위치를 문맥으로 사용한다. 같은 위치만 제외하며, 랜덤 워크가 같은 노드를 재방문하면 중심 ID와 문맥 ID가 같을 수도 있다. 길이 10의 워크 하나는 방향이 있는 문맥 쌍 48개를 만들므로 전체 양성 쌍은 16,320개다.

모델은 중심 테이블 e와 문맥 테이블 c를 별도로 가진다. 둘 다 34×16이므로 파라미터는 총 1,088개다. 양성 쌍 B개와 쌍당 K개 음성 표본에 대해 구현 손실은 다음과 같다.

$$\mathcal L=-\frac1B\sum_i\log\sigma(e_{u_i}^{\top}c_{v_i})
-\frac1{BK}\sum_{i,k}\log\sigma(-e_{u_i}^{\top}c_{n_{ik}})$$

음성 항을 K개에 대해 **합산하지 않고 평균**하므로, K개의 음성 손실 합을 사용하는 전형적인 목적함수와 양성·음성 항의 상대 가중치가 다르다. K를 늘려도 음성 항의 기대 가중치가 K배가 되는 구현은 아니다.

음성 노드는 전체 ID에서 균등하게 뽑는다. 중심·실제 문맥·연결된 노드를 제외하지 않아 양성과 같은 쌍이 음성에 포함될 수 있다. 여기의 음성은 그래프에서 확실히 존재하지 않는 엣지라는 뜻이 아니라 대조 학습용 노이즈 표본이다.

## 4. 실제 학습량

50번의 반복마다 전체 쌍에서 512개를 선택해 한 번 업데이트한다. 따라서 표시된 epoch는 전체 16,320개 쌍을 순회하는 에포크가 아니라 **미니배치 업데이트 한 번**이다. 총 50회 업데이트에서 25,600개 양성 쌍을 중복 가능하게 사용한다. 각 업데이트 내부에서는 randperm으로 골라 같은 쌍 인덱스를 중복 선택하지 않는다.

손실 출력도 해당 미니배치의 값이다. 전체 데이터 손실의 단조 감소나 임베딩의 품질 향상을 직접 보장하지 않는다. 워크 생성과 학습 샘플링의 시드, 차원과 업데이트 예산을 구분해 비교해야 한다.

## 5. 임베딩을 평가하는 방법

학습 후에는 중심 테이블만 꺼낸다. 클럽 레이블은 학습 손실에 쓰지 않고, 두 레이블 그룹의 벡터 평균 사이 유클리드 거리를 계산하는 데 사용한다. 큰 중심 거리만으로 그룹이 잘 분리되었다고 판단할 수는 없다. 그룹 내부의 분산·중첩이나 전체 벡터 스케일을 함께 보아야 한다.

특히 중심 벡터를 a배, 문맥 벡터를 1/a배 하면 학습에 사용하는 모든 내적은 그대로지만 중심 테이블의 그룹 간 거리는 a배가 된다. 따라서 이 거리 하나는 목적함수가 정하는 고유한 품질 척도가 아니다.

분류에 사용할 때는 임베딩 위의 분류기와 분리된 학습·검증·테스트 레이블로 평가한다. 링크 예측에서는 워크를 만들기 전에 평가 엣지를 제거해야 한다. 테이블에 없는 신규 노드는 즉시 표현할 수 없으므로 재학습이나 별도 특징 기반 인코더가 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
길이 10, 문맥 반경 3의 워크 하나가 만드는 방향 있는 문맥 쌍 수와 데모의 전체 쌍 수를 구하라.

</div>

??? success "연습문제 1 풀이"

    거리 1·2·3인 위치 쌍이 각각 9·8·7개이고 양방향으로 쓰므로 2×(9+8+7)=48개다. 고립 노드가 없는 이 데모의 340개 워크에서는 16,320개다. 반복 노드 ID도 위치가 다르면 별도 쌍으로 센다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
p=1, q=0.5일 때 되돌아갈 이웃, 직전 노드와 연결된 다른 이웃, 그 밖의 이웃의 가중치를 비교하라.

</div>

??? success "연습문제 2 풀이"

    각각 1, 1, 2다. 실제 확률은 가능한 모든 이웃의 가중치 합으로 나눈 값이다. 그 밖의 이웃 하나의 상대 가중치가 두 배라는 뜻이며 그 방향으로 항상 이동하는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
학습 내적을 바꾸지 않고 출력된 그룹 중심 거리를 바꾸는 변환을 설명하라.

</div>

??? success "연습문제 3 풀이"

    0이 아닌 양수 a에 대해 중심 테이블을 a배, 문맥 테이블을 1/a배 하면 모든 중심·문맥 내적은 유지된다. 중심 테이블의 유클리드 거리는 a배가 되므로 중심 거리만으로 임베딩 품질을 비교할 수 없다. 분리된 레이블의 분류 지표나 내부 분산 등과 함께 평가한다.



## 정리하며

랜덤 워크의 방문 문맥을 두 임베딩 테이블의 내적으로 학습한다. 전이 편향·문맥 범위·음성 표본 분포와 손실 가중치를 구분하고, 50번의 미니배치 업데이트와 중심 거리만을 전체 학습·분류 성능으로 해석하지 않는다.
