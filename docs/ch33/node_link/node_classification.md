# 노드 분류

노드 분류는 그래프의 연결 구조와 노드 피처를 이용해 각 노드의 클래스를 예측하는 문제다. 그래프 전체에 하나의 레이블을 붙이는 그래프 분류와 달리, 출력은 노드 수 N과 클래스 수 C에 대한 N×C 로짓이다. 이 예제에서는 일부 노드의 레이블만으로 GCN을 학습한다.

## 1. 코드와 출력

```python
"""29.6.1: 마디 가름 - GCN으로 하는 반쯤 스승 있는 러닝."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx

# ========================================================================
# 메인
# ========================================================================

class GCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_ch, hidden_ch)
        self.lin2 = nn.Linear(hidden_ch, out_ch)
        self.dropout = dropout
    def gcn_layer(self, x, ei):
        n = x.shape[0]; src, dst = ei[0], ei[1]
        loop = torch.arange(n, device=x.device)
        sa = torch.cat([src, loop]); da = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, da, torch.ones(da.shape[0], device=x.device))
        norm = (deg[sa]*deg[da]).pow(-0.5); norm[norm==float('inf')]=0
        msg = x[sa]*norm.unsqueeze(1)
        out = torch.zeros_like(x)
        out.scatter_add_(0, da.unsqueeze(1).expand_as(msg), msg)
        return out
    def forward(self, x, ei):
        x = self.gcn_layer(self.lin1(x), ei)
        x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        return self.gcn_layer(self.lin2(x), ei)

def demo():
    print("=" * 60); print("Node Classification (Semi-supervised)"); print("=" * 60)
    torch.manual_seed(42)
    # 확률 덩이 모형
    sizes = [30, 30, 30]; probs = [[0.3,0.02,0.02],[0.02,0.3,0.02],[0.02,0.02,0.3]]
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.randn(n, 16)
    y = torch.tensor([0]*30+[1]*30+[2]*30)
    # 피처에 무리 신호를 더한다
    for i in range(n): x[i, :3] += y[i].float() * 0.5
    # 반쯤 스승 있는 러닝: 이름표 5%
    train_mask = torch.zeros(n, dtype=torch.bool)
    for c in range(3): train_mask[c*30:c*30+3] = True
    model = GCN(16, 32, 3); opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(200):
        opt.zero_grad()
        F.cross_entropy(model(x, ei)[train_mask], y[train_mask]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(x, ei).argmax(1)
        print(f"  Train acc: {(pred[train_mask]==y[train_mask]).float().mean():.4f}")
        print(f"  Test acc:  {(pred[~train_mask]==y[~train_mask]).float().mean():.4f}")
        print(f"  Overall:   {(pred==y).float().mean():.4f}")

if __name__ == "__main__":
    demo()
```

```
============================================================
Node Classification (Semi-supervised)
============================================================
  Train acc: 1.0000
  Test acc:  1.0000
  Overall:   1.0000
```

출력은 기존 실행 기록이다. 이번 본문 편집에서는 코드를 실행하거나 정확도를 다시 측정하지 않았다.

## 2. 합성 데이터와 레이블 정보

데이터는 각각 30개 노드로 이루어진 세 그룹의 확률적 블록 모델(SBM)이다. 같은 그룹의 연결 확률은 0.3, 서로 다른 그룹은 0.02로 설정한다. 따라서 구조 자체에 그룹 레이블을 예측할 단서가 있다. 이는 실제 벤치마크 데이터가 아니라 생성 규칙이 알려진 90개 노드의 합성 그래프다.

노드 피처는 16차원 표준정규 벡터로 시작한다. 처음 세 좌표 모두에 클래스 번호의 0.5배를 더한다. 클래스 0·1·2의 평균 이동은 각각 0·0.5·1이다. 세 좌표에 클래스별 원핫 벡터를 더하는 방식은 아니다.

전체 레이블로 피처를 만드는 것은 이 합성 문제의 생성 규칙이다. 실제 미관측 노드를 예측할 때 정답 레이블을 읽어 피처를 만들 수는 없다. 현실 데이터에 적용하려면 예측 시점에 확보할 수 있는 피처로 바꾸고, 구조만·피처만 사용하는 기준모델과 비교해야 한다.

## 3. GCN의 계산

무방향 엣지를 양쪽 방향으로 저장하고, 각 노드에 자기 루프를 추가한다. 루프를 포함한 차수로 메시지를 정규화해 이웃 정보를 합산한다. 대칭 인접 행렬 A에 대해 다음과 같이 쓸 수 있다.

$$\widetilde A=A+I,\qquad
S=\widetilde D^{-1/2}\widetilde A\widetilde D^{-1/2}$$

이 구현은 먼저 Linear를 적용한 뒤 S로 집계하므로 한 레이어는 S(XW+b)에 해당한다. bias를 집계한 뒤 더하는 SXW+b와 일반적으로 다르다. 첫 레이어는 16→32로 변환하고 ReLU·dropout을 거친다. 둘째 레이어는 32→3 로짓을 반환하며 마지막 softmax는 별도로 적용하지 않는다. 교차 엔트로피 로스가 로짓을 직접 받기 때문이다.

자기 루프는 forward마다 추가되며 이미 입력에 있는 루프·중복 엣지는 제거하지 않는다. 임의 방향 그래프에서는 양 끝의 진입 차수를 사용하는 이 계산을 무방향 GCN 정규화와 동일하게 해석할 수 없다. 입력 피처·인덱스·모델의 장치와 dtype도 일치해야 한다.

## 4. 준지도 학습과 전이적 평가

각 클래스의 처음 세 노드만 학습 대상으로 정해 총 9개 레이블을 사용한다. 나머지 81개가 테스트 대상이다. 교차 엔트로피는 학습 마스크의 로짓에만 적용하지만 메시지 패싱에는 **90개 노드의 전체 구조와 피처**을 사용한다.

이는 테스트 레이블을 로스에 넣지 않는 전이적(transductive) 준지도 설정이다. 테스트 노드 자체를 학습 때 보지 않는 귀납적(inductive) 평가와 구분한다. 그래프·피처를 미리 사용할 수 있다는 과제 조건에서는 이 설정이 가능하다. 미래 예측에서는 연결과 피처를 관측할 수 있는 시점까지 따로 제한해야 한다.

Adam으로 200번 전체 그래프 업데이트를 수행한다. 학습 중 dropout은 확률 0.5로 활성화되고 eval에서는 꺼진다. 파라미터는 첫 Linear 544개와 둘째 Linear 99개를 합한 643개다. 숨은 차원을 늘리면 모델 크기도 함께 늘므로 차원 변경을 성능 원인 하나로 해석하지 않는다.

## 5. 정확도가 말해 주는 범위

학습·테스트·전체 정확도는 학습을 마친 동일 모델에서 no_grad로 계산한다. 전체 정확도는 9개 학습 노드와 81개 테스트 노드의 정확도를 표본 수로 가중한 평균이다. 학습 정확도와 테스트 정확도를 절반씩 평균한 값이 아니다.

단일 합성 그래프와 고정 마스크의 결과만으로 실제 네트워크의 성능이나 새로운 그래프에 대한 일반화를 판단할 수는 없다. 여러 그래프·마스크 시드, 클래스별 지표, 피처만 쓰는 MLP와 구조 기반 기준모델을 비교하면 학습한 정보의 출처를 더 분명히 알 수 있다. 하이퍼파라미터를 선택할 때는 테스트와 별도의 검증 레이블이 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
90개 노드에서 클래스당 3개를 학습에 사용할 때 학습·테스트 노드 수와 전체 정확도의 계산식을 구하라.

</div>

??? success "연습문제 1 풀이"

    학습 노드는 3×3=9개, 테스트 노드는 81개다. 전체 정확도는 (9×학습 정확도+81×테스트 정확도)/90이다. 모든 정확도는 학습 후 같은 eval 모델로 계산한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
테스트 노드의 피처·엣지를 메시지 패싱에 사용하면서 테스트 레이블을 로스에 넣지 않는 설정을 설명하라.

</div>

??? success "연습문제 2 풀이"

    전이적 준지도 노드 분류다. 학습 마스크는 레이블 감독의 범위를 제한하며 그래프 계산 범위를 제한하지 않는다. 과제에서 전체 구조·피처를 미리 제공한다는 조건이 필요하고, 학습에 없던 노드를 평가하는 귀납적 설정과 구분한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
피처 생성 코드가 처음 세 좌표에 어떤 신호를 넣는지 설명하고, 실제 데이터에서 같은 규칙을 사용할 수 있는지 판단하라.

</div>

??? success "연습문제 3 풀이"

    클래스 c의 처음 세 좌표 모두에 0.5c를 더하므로 원핫 인코딩이 아니라 공통 방향의 평균 이동이다. 합성 분포를 정의하는 데는 가능하지만 예측할 정답을 읽어 실제 입력을 만들면 레이블 누출이다. 예측 시점에 관측 가능한 피처와 피처 전용 기준모델로 검증해야 한다.



## 정리하며

노드 분류에서는 노드별 로짓과 레이블 마스크를 구분한다. 이 예제는 구조·피처에 클래스 신호를 심은 합성 그래프의 전이적 준지도 학습이며, 9개 감독 레이블과 81개 테스트 노드를 사용한다. 생성 규칙과 평가 범위를 알아야 정확도를 해석할 수 있다.
