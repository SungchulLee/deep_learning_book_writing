# 심층 그래프 뉴럴 넷트웍

메시지 패싱 레이어를 늘리면 더 먼 노드의 정보를 전달할 수 있지만, 표현의 과도한 평활화(over-smoothing), 많은 경로를 좁은 벡터에 압축하는 병목(over-squashing), 최적화와 과적합 문제가 생길 수 있다. 이들은 서로 다른 현상이다. 깊이가 늘면 반드시 성능이 떨어지거나 잔차 연결 하나로 모두 해결된다고 볼 수는 없다.

## 1. 코드와 출력

```python
"""
29.4.1: 깊은 그래프 뉴럴 넷트웍 - 나머지 이음, 변 떨구기, 고르게 맞추기.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx

# ========================================================================
# 메인
# ========================================================================

class GCNLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)
    def forward(self, x, edge_index):
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n, device=x.device)
        src_all = torch.cat([src, loop]); dst_all = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst_all, torch.ones(dst_all.shape[0], device=x.device))
        norm = (deg[src_all] * deg[dst_all]).pow(-0.5)
        norm[norm == float('inf')] = 0
        h = self.lin(x)
        msg = h[src_all] * norm.unsqueeze(1)
        out = torch.zeros(n, h.shape[1], device=x.device)
        out.scatter_add_(0, dst_all.unsqueeze(1).expand_as(msg), msg)
        return out

class ResGCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=8, dropout=0.5):
        super().__init__()
        self.input_lin = nn.Linear(in_ch, hidden_ch)
        self.convs = nn.ModuleList([GCNLayer(hidden_ch, hidden_ch) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_ch) for _ in range(num_layers)])
        self.output_lin = nn.Linear(hidden_ch, out_ch)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.input_lin(x))
        for conv, norm in zip(self.convs, self.norms):
            res = x
            x = conv(x, edge_index); x = norm(x); x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + res
        return self.output_lin(x)

def drop_edge(edge_index, p=0.2):
    mask = torch.rand(edge_index.shape[1]) > p
    return edge_index[:, mask]

def demo_deep_gcn():
    print("=" * 60); print("Deep GCN with Residual Connections"); print("=" * 60)
    torch.manual_seed(42)
    G = nx.karate_club_graph(); n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.eye(n)
    y = torch.tensor([0 if G.nodes[i].get('club','')=='Mr. Hi' else 1 for i in range(n)])
    tm = torch.zeros(n, dtype=torch.bool); tm[::2] = True
    for nl in [2, 4, 8, 16]:
        torch.manual_seed(42)
        model = ResGCN(n, 16, 2, num_layers=nl); opt = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(200):
            opt.zero_grad(); F.cross_entropy(model(x, ei)[tm], y[tm]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(x, ei).argmax(1)[~tm] == y[~tm]).float().mean()
        print(f"  {nl:2d} layers: Test Acc = {acc:.4f}")

if __name__ == "__main__":
    demo_deep_gcn()
```

```
============================================================
Deep GCN with Residual Connections
============================================================
   2 layers: Test Acc = 0.8824
   4 layers: Test Acc = 0.9412
   8 layers: Test Acc = 0.9412
  16 layers: Test Acc = 0.8824
```

기존 출력은 Karate Club 한 그래프에서 깊이를 바꾼 실험 기록이다. 이번 편집에서는 재실행하지 않았다. 잔차 없는 모델이나 DropEdge와 비교한 결과는 이 출력에 포함되지 않는다.

## 2. 잔차 블록의 구조

입력 Linear와 ReLU로 34차원 노드 ID 피처를 16차원으로 바꾼다. 각 블록은 GCN→LayerNorm→ReLU→dropout을 적용한 결과를 블록 입력에 더한다.

$$H^{(l+1)}=H^{(l)}+
\operatorname{Dropout}\bigl(\operatorname{ReLU}(\operatorname{LN}(\operatorname{GCN}_l(H^{(l)})))\bigr)$$

LayerNorm은 **각 노드의 16개 피처 좌표**에 걸쳐 정규화한다. 노드 전체를 같은 값으로 만드는 연산도, 그래프별 평균을 빼는 연산도 아니다. GCN은 Linear의 bias를 포함한 투영 뒤 자기 루프를 더한 대칭 정규화 집계를 한다.

잔차 경로는 이전 표현과 그래디언트가 다음 블록에 직접 전달될 경로를 제공한다. 일반 잔차 함수의 야코비안은 I+J_F이며 항상 좋은 조건수를 보장하지는 않는다. 이 특정 블록은 평가 시 ReLU 결과를 더하지만, 최종 선형 레이어에서 정보를 잃을 수 있고 전체 표현의 크기 증가도 제어해야 한다. LayerNorm·dropout 역시 정보 보존이나 일반화의 보증은 아니다.

## 3. 선형 전파로 이해하는 평활화

대칭 비음수 A에 루프를 더하고 그 차수로 정규화한 S는 다음과 같다.

$$S=\widetilde D^{-1/2}(A+I)\widetilde D^{-1/2},\qquad
P=\widetilde D^{-1}(A+I)$$

S는 행 확률 행렬 P와 닮음 관계이므로 고유값의 절댓값이 1 이하이고, S 자체가 이중 확률 행렬인 것은 아니다. 연결 그래프의 각 노드에 양의 자기 루프가 있으면 고유값 1을 제외한 절댓값은 1보다 작다. 단위 벡터 q를 q_i∝√d̃_i로 두면 다음 극한을 얻는다.

$$S^kH\longrightarrow qq^\top H$$

따라서 원래 좌표의 모든 노드가 같은 벡터가 되는 것이 아니라 √d̃_i에 비례한 행으로 남을 수 있다. 연결 성분이 여러 개면 성분별 극한이 남는다. 실제 ResGCN에는 학습 웨이트·정규화·비선형·잔차가 있어 이 단순 극한을 그대로 적용할 수 없다.

## 4. 깊이 비교의 조건

34개 노드 중 짝수 인덱스 17개 레이블로 학습하고 나머지 17개에서 평가한다. 전체 그래프·단위 행렬 피처는 학습부터 사용하므로 전이적 평가다. 깊이는 메시지 패싱 블록 수이며 앞뒤 두 Linear는 별도다.

각 모델을 200번 업데이트한다. 폭 16에서 블록당 GCN 272개와 LayerNorm 32개 파라미터가 추가된다. 전체는 594+304L개이므로 2·4·8·16블록 모델의 크기도 함께 달라진다. 깊이 변경은 같은 파라미터 예산의 비교가 아니다. 잔차의 효과를 보려면 같은 깊이·폭·분할·학습 예산에서 잔차만 제거하고 여러 시드로 비교해야 한다. 정확도가 50%가 된다고 미리 단정할 근거는 없다.

## 5. DropEdge의 역할과 실제 구현

[DropEdge 논문](https://arxiv.org/abs/1907.10903)은 학습 중 연결을 일부 제거해 과적합과 평활화 문제를 완화하는 방법을 제안한다. 그러나 여기의 기본 데모는 drop_edge를 **호출하지 않는다**. 뒤의 연습문제 코드가 별도 적용 예다.

보존된 함수는 edge_index의 각 열을 독립적으로 남긴다. 무방향 엣지를 양방향 열로 저장했다면 한 방향만 남을 수 있어 그래프가 비대칭이 된다. 무방향성을 유지하려면 원래 엣지 쌍에 같은 마스크를 적용해야 한다. 자기 루프는 GCN 내부에서 나중에 추가되므로 이 함수로 제거되지 않는다.

마스크는 기본 CPU에서 생성되므로 장치 호환성도 확인해야 한다. 연결을 줄이면 유용한 다리 엣지가 사라지거나 고립 노드가 생길 수 있다. p=0.3이 항상 좋거나 더 깊은 모델이 항상 유리한 것은 아니며 검증 데이터로 선택해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
16블록 모델에서 잔차 효과를 분리해 비교하려면 어떤 조건을 고정해야 하는가? 결과를 미리 특정 정확도로 단정할 수 있는가?

</div>

??? success "연습문제 1 풀이"

    깊이·폭·데이터·마스크·optimizer·업데이트 수를 고정하고 잔차 덧셈만 바꾼다. 여러 초기화 시드에서 검증과 최종 테스트를 구분한다. 잔차 제거가 최적화에 영향을 줄 수 있지만 단일 이론만으로 테스트가 50%가 된다고 예측할 수는 없다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
S의 스펙트럼 반지름이 1인 이유와 연결 그래프에서 S의 반복 곱이 남기는 방향을 설명하라.

</div>

??? success "연습문제 2 풀이"

    S는 행 확률 행렬 P와 닮음이므로 같은 고유값을 갖고 스펙트럼 반지름은 1이다. 연결성과 양의 자기 루프 아래 나머지 고유값의 절댓값은 1보다 작다. 정규화한 q_i∝√d̃_i에 대해 S^k→qqᵀ다. S의 행 합이 모두 1이거나 극한 행이 모두 같은 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 DropEdge 적용 예가 기본 데모와 다른 점과 무방향성·변수 범위를 점검하라.

</div>

??? success "연습문제 3 풀이"

    아래는 별도 학습 반복이며 n·x·ei·tm·y를 같은 범위에 준비해야 한다. 기본 데모 내부의 지역 변수는 함수 밖에서 자동으로 사용할 수 없다. 함수는 양방향 열을 독립 제거해 대칭성을 깨뜨릴 수 있고 평가 코드도 포함하지 않는다. 원래 그래프에서 eval 정확도를 별도로 계산하고 같은 조건의 기준모델과 비교해야 한다.

    ```python
    model = ResGCN(n, 16, 2, num_layers=16)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(200):
        opt.zero_grad()
        ei_dropped = drop_edge(ei, p=0.3)  # 변의 30%를 떨군다
        F.cross_entropy(model(x, ei_dropped)[tm], y[tm]).backward()
        opt.step()
    ```


## 정리하며

깊은 GNN은 잔차·정규화·dropout으로 정보와 그래디언트 전달을 도울 수 있다. 선형 평활화의 극한과 실제 잔차 모델을 구분하고, 깊이에 따른 파라미터 증가와 DropEdge의 방향·장치·평가 조건을 확인한다.
