# 링크 예측

링크 예측은 관측한 그래프와 노드 피처를 바탕으로 노드 쌍 사이에 관계가 있는지 점수를 매기는 문제다. 정적 그래프의 누락된 엣지를 복원하는 과제와 앞으로 생길 엣지를 예측하는 과제는 관측 가능한 정보와 평가 분할이 다르다. 관측되지 않은 쌍이 모두 실제 음성이라는 가정도 데이터 수집 방식에 따라 달라진다.

## 1. 인코더와 디코더

인코더는 메시지 패싱용 그래프에서 노드 표현을 만들고, 디코더는 후보 노드 쌍에 점수를 매긴다.

$$z_u=\operatorname{GNN}(G_{\mathrm{obs}},X)_u,\qquad s(u,v)=f(z_u,z_v)$$

점수는 일반적으로 실수 로짓이다. 시그모이드를 적용하면 0~1 값이 되지만, 음성 샘플링으로 만든 학습 분포의 확률을 실제 전체 후보의 엣지 확률로 곧바로 해석할 수는 없다. 음성 비율·분포와 확률 보정을 확인해야 한다.

## 2. 점수 함수의 대칭성

| 방식 | 점수 | 표현의 특성 |
|---|---|---|
| 내적 | $z_u^\top z_v$ | 두 노드의 순서를 바꿔도 같다 |
| DistMult | $z_u^\top\operatorname{diag}(r)z_v$ | 관계별 웨이트를 두지만 같은 관계에서 대칭이다 |
| MLP | $\operatorname{MLP}([z_u\Vert z_v])$ | 순서에 따라 달라질 수 있다 |
| TransE | $-\lVert z_u+r-z_v\rVert$ | 관계를 벡터 이동으로 표현한다 |

방향 있는 관계를 내적이나 DistMult 하나로 구분할 수는 없다. 반대로 무방향 과제에서 순서 있는 MLP를 쓰면 양방향 점수를 평균하거나 대칭적인 쌍 피처를 사용하는 등 대칭성을 설계해야 한다. 모델의 표현력뿐 아니라 파라미터 수와 후보별 계산비용을 함께 비교한다.

## 3. 로스와 음성 샘플링

관측된 학습 양성 집합 E⁺와 샘플링한 음성 집합 E⁻에 대해 로지스틱 로스는 다음과 같다.

$$\mathcal L=-\sum_{(u,v)\in E^+}\log\sigma(s(u,v))
-\sum_{(u,v)\in E^-}\log\sigma(-s(u,v))$$

실제 구현에서는 합·평균 방식과 양성·음성 웨이트를 명시한다. 순위 학습에는 양성 쌍과 대응하는 음성 쌍에 대해 max(0, γ−s⁺+s⁻)를 최소화하는 마진 로스도 사용할 수 있다. γ>0은 목표 점수 차이다.

음성 샘플링은 자기 연결·중복·관측 양성과 겹치는 쌍의 처리 규칙이 필요하다. 무방향 그래프에서는 (u,v)와 (v,u)를 같은 관계로 다룬다. 미관측 쌍 중에는 아직 발견하지 못한 실제 양성도 있을 수 있다. 시간 예측에서는 미래를 보고 학습용 음성을 고르는 방식 자체가 정보를 누출할 수 있으므로 관측 시점과 벤치마크 규약을 따른다.

## 4. 분할과 평가

학습·검증·테스트 엣지를 나눌 때는 같은 무방향 엣지의 반대 방향이 다른 분할에 남지 않게 한다. 테스트 양성을 메시지 패싱용 인접 행렬에 넣은 채 복원 성능을 평가하면 정답 연결을 이미 본 셈이다. 미래 예측은 시간순으로 분할하고 해당 시점 이전의 피처·연결만 사용한다.

메시지 패싱 엣지와 로스를 계산할 감독 엣지는 개념적으로 구분한다. 학습 양성을 학습 그래프에도 사용하는 재구성 학습은 가능하지만, 평가 양성에 같은 처리를 해서는 안 된다. 모델 선택에는 별도 검증 집합을 사용한다.

| 지표 | 의미와 비교 조건 |
|---|---|
| ROC-AUC | 무작위 양성이 음성보다 높은 점수를 받을 확률에 해당하며 동점은 절반으로 센다 |
| 평균 정밀도(AP) | 재현율 증가량으로 정밀도를 가중한 평균으로, 양성 비율·음성 표본에 영향을 받는다 |
| Hits@K | 정의된 후보 집합에서 정답의 순위가 K 이내인 질의의 비율이다 |
| MRR | 질의별 정답 순위의 역수 평균이다 |

AP는 단순한 사다리꼴 PR 곡선 면적과 다를 수 있다. [scikit-learn 문서](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.average_precision_score.html)는 사용하는 가중 평균 정의를 명시한다. 모든 순위 지표는 후보 수·음성 난도·동점 처리·다른 알려진 정답을 제외하는 filtered 평가 여부가 같아야 비교할 수 있다. 특정 분할 비율이나 양성·음성 1:1이 모든 과제의 표준인 것은 아니다.

## 5. 보존된 구현과 출력

```python
"""
그래프 뉴럴 넷트웍 임베딩과 안쪽 곱 점수로 하는 이음 헤아리기.

음의 뽑기와 두 값 교차 엔트로피 로스로 익히기를 보여 준다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# === 단순 그래프 뉴럴 넷트웍 인코더 ===
class GCNEncoder(nn.Module):
    """마디 임베딩을 내놓는 두 레이어 GCN."""

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

```
Loss: 0.4241
```

위 코드는 6개 노드에 대해 **무작위 초기 모델의 로스 한 번**을 계산한다. optimizer·backward·학습 반복·데이터 분할·순위 지표는 없다. 기존 출력은 학습 완료나 테스트 예측 성능을 보여 주지 않는다.

GCNEncoder는 입력 adj를 그대로 두 번 곱한다. adj의 차수 정규화나 자기 루프 추가를 내부에서 수행하지 않는다. 데모가 직접 단위 행렬에 세 경로 엣지를 넣어 자기 루프를 제공하지만 정규화는 하지 않는다. 따라서 앞 절의 대칭 정규화 GCN과 동일한 연산이 아니다. adj 행이 정보를 받을 노드, 열이 메시지를 보낼 노드에 대응하며 방향 그래프에서는 이 규약도 맞춰야 한다.

인코더는 4→8→4로 변환하며 파라미터는 76개다. 내적 디코더의 양성·음성 로짓 각각 3개를 이어 붙여 여섯 항의 평균 BCE를 구한다. 양성 세 엣지가 인접 행렬에도 있으므로 이는 관측 그래프의 재구성 로스 예제다. 노드 4·5는 다른 노드와 연결되지 않아도 자기 루프 때문에 자기 피처를 처리한다.

## 6. 구조 기반 기준모델과 확장

단순 무방향 그래프에서는 공통 이웃 수, Jaccard 계수, Adamic–Adar 점수가 유용한 기준모델이다. 이들은 그래프 구조만 사용하므로 피처 기반 모델과 정보 조건을 맞춰 비교한다. 이웃 집합을 해시 집합으로 저장하면 교집합 계산은 작은 쪽 차수에 비례하도록 구현할 수 있으며, 모든 방법의 비용이 항상 평균 차수 제곱인 것은 아니다.

개별 노드 임베딩을 결합하는 디코더는 서로 다른 노드 쌍의 주변 구조를 구분하지 못할 수 있다. [SEAL](https://arxiv.org/abs/1802.09691)은 대상 쌍 주위의 부분 그래프와 두 끝점에 대한 구조적 레이블을 사용한다. 대상 엣지를 제거한 뒤 피처를 계산해 정답의 존재를 직접 읽지 않도록 해야 한다. 부분 그래프 범위와 피처·모델에 따른 표현 한계가 있으므로 모든 링크 규칙을 보장해서 학습한다고 설명하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
무방향 엣지 (u,v)를 테스트로 남길 때 어떤 연결을 메시지 패싱 그래프에서 제거해야 하는가?

</div>

??? success "풀이"

    같은 관계를 나타내는 (u,v)와 (v,u)를 모두 제거한다. 중복 레코드가 있다면 그것도 함께 처리한다. 정규화·구조 피처도 허용된 관측 그래프에서 다시 계산해야 한다. 시간 예측에서는 미래의 다른 연결·피처도 사용하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
공통 이웃·Jaccard·Adamic–Adar 점수를 정의하고 빈 이웃의 처리와 계산비용을 설명하라.

</div>

??? success "풀이"

    공통 이웃은 |N(u)∩N(v)|, Jaccard는 이를 |N(u)∪N(v)|로 나눈 값이다. 합집합이 비면 별도 규약으로 보통 0을 준다. Adamic–Adar는 공통 이웃 w에 대해 1/log d(w)를 합한다. 서로 다른 u,v의 공통 이웃은 단순 무방향 그래프에서 차수가 적어도 2다. 해시 집합의 교집합은 작은 쪽 차수에 비례하도록 구현할 수 있으며 저장 형식에 따라 비용이 달라진다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
데모 출력이 링크 예측 성능을 입증하지 못하는 이유와 adj의 역할을 설명하라.

</div>

??? success "풀이"

    학습 없이 초기 모델에서 양성·음성 세 쌍씩의 BCE만 계산한다. adj에는 양성 엣지가 이미 있고 정규화도 하지 않는다. 따라서 재구성 로스 계산 예제이며, 일반화 성능에는 분할된 엣지와 누출 없는 그래프, 학습·검증 절차와 별도 평가 지표가 필요하다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
내적 디코더의 대칭성과 SEAL의 접근을 비교하라.

</div>

??? success "풀이"

    내적은 z_uᵀz_v=z_vᵀz_u라 방향을 구분하지 못하고, 서로 같은 임베딩 쌍에 같은 점수를 준다. SEAL은 대상 두 노드를 기준으로 레이블을 붙인 주변 부분 그래프를 직접 인코딩해 쌍별 구조를 사용할 수 있다. 대상 엣지 제거가 필요하며 모든 구조를 구별한다는 보장은 아니다.



## 정리하며

링크 예측은 점수 함수뿐 아니라 관측 그래프·음성 분포·분할·후보 집합의 정의가 중요하다. 보존된 코드는 정규화하지 않은 인접 행렬과 내적 디코더로 초기 재구성 로스를 계산하며, 학습된 평가 성능을 보고하는 예제는 아니다.

**참고 문헌**

- Zhang, M. & Chen, Y. "Link Prediction Based on Graph Neural Networks."
  NeurIPS 2018.
- Bordes, A. et al. "Translating Embeddings for Modeling Multi-relational
  Data." NeurIPS 2013.
