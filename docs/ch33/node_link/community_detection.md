# GNN을 이용한 커뮤니티 탐지

커뮤니티 탐지는 연결 구조에서 밀접한 노드 집단을 찾는 문제다. 이 예제는 GNN으로 노드 임베딩을 학습한 뒤 K-means를 적용하고, 라플라시안 고유벡터를 사용하는 방법과 비교한다. 커뮤니티의 정의·개수·평가 기준은 데이터에 따라 달라지며, 연결이 많은 집단과 외부에서 정한 레이블이 항상 일치하는 것은 아니다.

## 1. 코드와 출력

```python
"""29.6.4: 그래프 뉴럴 넷트웍으로 하는 무리 찾기."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx
from sklearn.cluster import KMeans

# ========================================================================
# 메인
# ========================================================================

class GNNCommunity(nn.Module):
    """무리 찾기를 위한 그래프 뉴럴 넷트웍 인코더."""
    def __init__(self, in_ch, hidden_ch, embed_ch):
        super().__init__()
        self.lin1 = nn.Linear(in_ch, hidden_ch)
        self.lin2 = nn.Linear(hidden_ch, embed_ch)
    def forward(self, x, ei):
        n = x.shape[0]; src, dst = ei[0], ei[1]
        loop = torch.arange(n, device=x.device)
        sa = torch.cat([src, loop]); da = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, da, torch.ones(da.shape[0], device=x.device))
        norm = (deg[sa]*deg[da]).pow(-0.5); norm[norm==float('inf')]=0
        for lin in [self.lin1, self.lin2]:
            h = lin(x); msg = h[sa]*norm.unsqueeze(1)
            out = torch.zeros(n, h.shape[1], device=x.device)
            out.scatter_add_(0, da.unsqueeze(1).expand_as(msg), msg)
            x = F.relu(out)
        return x

def demo():
    print("=" * 60); print("Community Detection"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    sizes = [25, 25, 25]; probs = [[0.3,0.02,0.02],[0.02,0.3,0.02],[0.02,0.02,0.3]]
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    n = G.number_of_nodes(); true_labels = [0]*25+[1]*25+[2]*25
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.eye(n)

    # 이음 헤아리기 로스로 익히기
    model = GNNCommunity(n, 32, 16)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(100):
        opt.zero_grad(); z = model(x, ei)
        pos_ei = torch.tensor([[e[0] for e in edges[:50]],[e[1] for e in edges[:50]]], dtype=torch.long)
        pos_s = (z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=-1)
        neg_ei = torch.stack([torch.randint(0,n,(50,)), torch.randint(0,n,(50,))])
        neg_s = (z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=-1)
        loss = -F.logsigmoid(pos_s).mean() - F.logsigmoid(-neg_s).mean()
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        z = model(x, ei).numpy()
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(z)
    pred = km.labels_
    # NMI 비슷한 맞음(가장 좋은 자리바꿈)
    from itertools import permutations
    best_acc = 0
    for perm in permutations(range(3)):
        mapped = [perm[p] for p in pred]
        acc = sum(1 for a,b in zip(mapped, true_labels) if a==b) / n
        best_acc = max(best_acc, acc)
    print(f"  GNN + KMeans accuracy: {best_acc:.4f}")

    # 스펙트럼 뭉치기와 견주기
    A = nx.adjacency_matrix(G).toarray().astype(float)
    D = np.diag(A.sum(1)); L = D - A
    _, U = np.linalg.eigh(L)
    km2 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(U[:, 1:4])
    best_acc2 = 0
    for perm in permutations(range(3)):
        mapped = [perm[p] for p in km2.labels_]
        acc = sum(1 for a,b in zip(mapped, true_labels) if a==b) / n
        best_acc2 = max(best_acc2, acc)
    print(f"  Spectral clustering accuracy: {best_acc2:.4f}")

if __name__ == "__main__":
    demo()
```

```
============================================================
Community Detection
============================================================
  GNN + KMeans accuracy: 0.4933
  Spectral clustering accuracy: 0.6800
```

출력은 기존 합성 실험의 기록이다. 이번 편집에서는 코드를 실행하거나 두 방법의 성능을 재측정하지 않았다.

## 2. 합성 그래프와 입력

각각 25개 노드인 세 블록을 만들고, 블록 안의 연결 확률은 0.3, 블록 사이에는 0.02로 둔다. 노드 수는 75이며 무방향 엣지는 양방향으로 저장한다. 학습 입력은 75×75 단위 행렬이다. 각 노드 ID를 구별할 수 있지만 새로운 노드를 곧바로 같은 입력 차원에 추가하는 피처 인코더는 아니다.

생성에 사용한 블록 번호는 학습 로스에 넣지 않는다. 다만 K-means의 클러스터 수를 정답과 같은 3으로 미리 정하므로 커뮤니티 개수를 자동으로 찾는 실험도 아니다. 블록이 잘 분리되도록 만든 단일 그래프에 대한 결과라는 범위를 유지한다.

## 3. 임베딩과 학습 목적

두 GCN 레이어가 75→32→16으로 변환한다. 각 레이어에서 Linear 후에 자기 루프를 포함한 대칭 정규화 집계를 적용하고 ReLU를 사용한다. 파라미터는 2,432+528=2,960개다. 마지막 ReLU 때문에 모든 임베딩 좌표가 비음수이고, 두 임베딩의 내적도 항상 0 이상이다.

학습은 엣지 존재를 복원하는 대조 목적이다. 양성·음성 점수에 대해 다음 로스를 100번 업데이트한다.

$$\mathcal L=-\operatorname{mean}\log\sigma(z_u^\top z_v)
-\operatorname{mean}\log\sigma(-z_a^\top z_b)$$

그러나 양성은 전체 엣지에서 매번 무작위로 고르지 않고 **목록의 처음 50개**만 사용한다. 같은 집합을 반복하므로 노드·블록별 감독이 편중될 수 있다. 전체 그래프는 메시지 패싱에 사용하지만 모든 엣지가 양성 로스에 직접 들어가는 것은 아니다.

음성은 두 노드 ID를 독립적으로 50쌍 뽑는다. 자기 쌍·실제 엣지·중복을 제외하지 않아 양성으로도 쓰이는 관계가 음성에 포함될 수 있다. 또한 비음수 임베딩 내적은 음수가 될 수 없으므로 음성 로스 항은 적어도 log 2이고, 내적을 줄여도 시그모이드 점수가 0.5 아래로 내려가지 못한다. 모델과 로스의 이 제약을 알고 해석해야 한다.

## 4. K-means와 레이블 대응

학습된 75×16 임베딩을 K-means로 세 그룹으로 나눈다. 클러스터 번호 자체에는 의미가 없어, 예측의 0·1·2가 정답 블록 번호와 일치할 필요가 없다. 코드는 3!=6개의 번호 대응을 모두 조사해 가장 높은 정확도를 출력한다.

이 대응은 레이블 순서의 임의성을 제거하는 평가 절차다. 정답 레이블로 GNN을 학습하는 것은 아니지만, 정답에 맞춘 최적 대응의 결과이지 고정 클래스 이름을 예측하는 분류 정확도와 같지는 않다. 큰 클러스터 수에서는 순열 전수 조사보다 할당 문제 풀이가 적합하다. ARI·NMI 등 번호 순열에 불변인 지표와 클러스터 크기도 함께 볼 수 있다.

## 5. 스펙트럴 비교의 실제 설정

비교 모델은 조합 라플라시안 L=D−A를 고유분해한다. 정규화 라플라시안은 사용하지 않는다. 정렬된 고유벡터 중 첫 벡터를 제외한 열 1~3을 골라 3차원 표현을 만들고, 여기에 K-means를 적용한다.

연결 그래프의 조합 라플라시안에서 첫 벡터는 상수 방향이다. 그래프가 연결되지 않으면 0 고유값이 여러 개일 수 있어 첫 벡터 하나를 제외하는 의미가 달라진다. 또한 다른 스펙트럴 클러스터링 구현이 사용하는 첫 세 고유벡터·행 정규화 등과 이 설정을 혼동하지 않는다.

두 방법은 같은 합성 그래프를 보지만 표현 차원·목적함수·계산량이 다르다. 한 번의 출력만으로 GNN이 스펙트럴 방법보다 우수하거나 반대라고 일반화할 수 없다. K-means 초기화, 그래프 생성 시드, 블록 간 연결 확률, 알려진 클러스터 수의 영향을 나누어 비교해야 한다.

## 6. 평가 범위

여기서는 전체 그래프의 구조로 표현을 만든 뒤 같은 노드의 커뮤니티를 복원한다. 엣지를 숨긴 링크 예측이나 새로운 그래프의 노드를 분류하는 테스트가 아니다. 외부 레이블과의 일치율뿐 아니라 커뮤니티를 찾는 목적에 맞는 연결 특성도 검토해야 한다. 모듈러리티를 쓰더라도 높은 값이 유일한 정답 커뮤니티를 보장하지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
마지막 ReLU가 음성 내적 점수에 주는 제약과 음성 로스의 하한을 구하라.

</div>

??? success "연습문제 1 풀이"

    임베딩의 각 좌표가 0 이상이므로 내적 s도 0 이상이다. 음성 로스는 −log σ(−s)=log(1+exp(s))이고 최솟값은 s=0에서 log 2다. 이 구조에서는 내적 로짓으로 음성 확률을 0.5 아래로 낮출 수 없다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
학습 양성·음성 쌍을 고르는 방식과 그 한계를 설명하라.

</div>

??? success "연습문제 2 풀이"

    양성은 엣지 목록의 처음 50개로 고정되어 무작위 전체 엣지 표본이 아니다. 음성은 두 ID를 독립적으로 뽑아 실제 엣지·자기 쌍·중복과 겹칠 수 있다. 전체 그래프로 집계하더라도 감독 쌍의 편중과 모순이 사라지는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
최적 레이블 대응 정확도의 목적과 스펙트럴 비교가 사용하는 고유벡터를 설명하라.

</div>

??? success "연습문제 3 풀이"

    클러스터 번호가 임의이므로 6개 번호 순열 중 정답 블록과 가장 잘 대응하는 것을 골라 일치율을 계산한다. 스펙트럴 모델은 조합 라플라시안의 첫 벡터를 제외한 세 고유벡터로 K-means를 수행한다. 클러스터 수 3을 미리 알고 같은 그래프를 복원하는 실험으로, 신규 데이터의 분류 성능은 아니다.



## 정리하며

GNN 임베딩과 K-means로 합성 블록 구조를 복원한다. 고정된 처음 50개 양성, 무필터 음성, 마지막 ReLU의 내적 제약과 알려진 클러스터 수를 확인하고, 레이블 순열을 보정한 정확도와 스펙트럴 비교의 설정을 구분한다.
