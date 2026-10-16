# Jumping Knowledge 네트워크

L레이어 메시지 패싱의 최종 표현은 최대 L홉 안의 정보를 사용할 수 있지만, 모든 노드에 정확히 L홉의 정보가 동일한 강도로 반영되지는 않는다. 그래프의 연결 구조·정규화·학습 웨이트에 따라 유효 범위가 달라진다. Jumping Knowledge(JK)는 여러 깊이의 표현을 마지막에 결합해 이런 서로 다른 범위의 정보를 활용하는 방법이다.

## 1. 코드와 출력

```python
"""
29.4.3: 뛰어넘는 앎 뉴럴 넷트웍
"""
import torch, torch.nn as nn, torch.nn.functional as F
import networkx as nx

# ========================================================================
# 메인
# ========================================================================

class JKNet(nn.Module):
    """모으기를 고를 수 있는 뛰어넘는 앎 뉴럴 넷트웍."""
    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=4, jk_mode='cat'):
        super().__init__()
        self.jk_mode = jk_mode
        self.input_lin = nn.Linear(in_ch, hidden_ch)
        self.convs = nn.ModuleList([nn.Linear(hidden_ch, hidden_ch) for _ in range(num_layers)])
        if jk_mode == 'cat':
            self.out_lin = nn.Linear(hidden_ch * num_layers, out_ch)
        elif jk_mode == 'lstm':
            self.lstm = nn.LSTM(hidden_ch, hidden_ch, batch_first=True)
            self.att_lin = nn.Linear(hidden_ch, 1)
            self.out_lin = nn.Linear(hidden_ch, out_ch)
        else:
            self.out_lin = nn.Linear(hidden_ch, out_ch)

    def forward(self, x, edge_index):
        n = x.shape[0]; x = F.relu(self.input_lin(x))
        src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n, device=x.device)
        src_a = torch.cat([src, loop]); dst_a = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst_a, torch.ones(dst_a.shape[0], device=x.device))
        norm = (deg[src_a]*deg[dst_a]).pow(-0.5); norm[norm==float('inf')]=0

        layer_outputs = []
        for lin in self.convs:
            h = lin(x); msg = h[src_a]*norm.unsqueeze(1)
            out = torch.zeros(n, h.shape[1], device=x.device)
            out.scatter_add_(0, dst_a.unsqueeze(1).expand_as(msg), msg)
            x = F.relu(out)
            layer_outputs.append(x)

        if self.jk_mode == 'cat':
            h = torch.cat(layer_outputs, dim=-1)
        elif self.jk_mode == 'max':
            h = torch.stack(layer_outputs, dim=0).max(dim=0)[0]
        elif self.jk_mode == 'lstm':
            stacked = torch.stack(layer_outputs, dim=1)  # [n, L, d]
            lstm_out, _ = self.lstm(stacked)
            att = torch.softmax(self.att_lin(lstm_out).squeeze(-1), dim=-1)
            h = (lstm_out * att.unsqueeze(-1)).sum(dim=1)
        return self.out_lin(h)

def demo_jk():
    print("=" * 60); print("Jumping Knowledge Networks"); print("=" * 60)
    torch.manual_seed(42)
    G = nx.karate_club_graph(); n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.eye(n)
    y = torch.tensor([0 if G.nodes[i].get('club','')=='Mr. Hi' else 1 for i in range(n)])
    tm = torch.zeros(n, dtype=torch.bool); tm[::2] = True

    for mode in ['cat', 'max', 'lstm']:
        torch.manual_seed(42)
        model = JKNet(n, 16, 2, num_layers=4, jk_mode=mode)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(200):
            opt.zero_grad(); F.cross_entropy(model(x, ei)[tm], y[tm]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(x, ei).argmax(1)[~tm] == y[~tm]).float().mean()
        print(f"  JK-{mode:5s}: Test Acc = {acc:.4f}")

if __name__ == "__main__":
    demo_jk()
```

```
============================================================
Jumping Knowledge Networks
============================================================
  JK-cat  : Test Acc = 0.9412
  JK-max  : Test Acc = 0.9412
  JK-lstm : Test Acc = 0.9412
```

기존 출력은 Karate Club 한 그래프에서 세 결합 방식을 비교한 기록이다. 이번 편집에서는 재학습하지 않았다. 특정 방식이 항상 더 우수하다는 순위로 읽지 않는다.

## 2. 레이어별 표현 수집

입력 Linear·ReLU로 34차원 단위 행렬 피처를 16차원으로 변환한다. 이어 네 번의 정규화 메시지 집계·Linear·ReLU를 거쳐 각 단계의 N×16 표현을 저장한다. 입력 투영 직후의 표현은 수집 목록에 포함하지 않는다.

각 레이어는 자기 루프를 포함하므로 k번째 표현의 구조적 범위는 **최대 k홉**이다. 비선형과 압축 때문에 그 범위의 모든 정보를 보존한다는 뜻은 아니다. JK는 이미 각 레이어에서 잃어버린 정보를 자동으로 복구하지도 않는다. 결합 후 노드별 선형 분류기가 두 클래스 로짓을 만든다.

## 3. 세 결합 방식

**cat**은 레이어별 표현을 정해진 순서로 이어 붙여 N×(Ld)를 만든다. 수집한 벡터를 결합 단계에서 모두 유지하지만 뒤의 분류기까지 정보 보존을 보장하지는 않는다. 깊이에 비례해 분류기의 입력 폭과 파라미터가 늘어난다.

**max**는 레이어 축에서 좌표별 최댓값을 골라 N×d를 만든다. 결합 연산 자체는 학습 파라미터가 없고, 레이어 순서가 바뀌어도 같다. 좌표마다 서로 다른 레이어를 고를 수 있어 한 노드에 단일 최적 깊이를 지정하는 방식은 아니다. 어떤 깊이에서 값이 나왔는지는 잃는다.

**lstm**은 노드마다 깊이 순서의 L개 표현을 시퀀스로 보고 단방향 LSTM에 넣는다. LSTM 출력에 선형 점수와 softmax를 적용해 레이어별 웨이트를 얻는다. 보존된 구현의 결합은 다음과 같다.

$$o_v^{(1:L)}=\operatorname{LSTM}(h_v^{(1:L)}),\qquad
\alpha_{vl}=\operatorname{softmax}_l(w^\top o_v^{(l)}+b),\qquad
z_v=\sum_l\alpha_{vl}o_v^{(l)}$$

즉 **원래 레이어 표현 h가 아니라 LSTM 출력 o를 가중 합산**한다. [JK 원 논문](https://arxiv.org/abs/1806.03536)의 LSTM 기반 가중 결합을 설명할 때 이 단방향 구현과 논문의 구체적 설계를 구분한다. 레이어 순서에는 민감하지만, 모든 노드에 같은 연산을 적용하므로 올바르게 재인덱싱한 노드 순서에는 등변적이다.

## 4. 파라미터 수와 비교 조건

입력 34·숨은 차원 16·깊이 4·출력 2에서 공통 부분은 입력 Linear 560개와 네 Linear 1,088개로 총 1,648개다.

| 방식 | 결합·출력 부분 | 전체 파라미터 |
|---|---|---:|
| cat | 64→2 Linear: 130개 | 1,778 |
| max | 16→2 Linear: 34개 | 1,682 |
| lstm | LSTM 2,176개 + 점수 17개 + 출력 34개 | 3,875 |

PyTorch LSTM은 입력·은닉 웨이트 외에 **두 bias 벡터**를 가지므로 파라미터는 4d²+4d²+8d다. d=16이면 2,176개이며 bias 하나만 계산한 2,112개가 아니다. 세 방식은 같은 모델 크기의 비교가 아니다.

데모는 짝수 인덱스 17개 레이블로 200번 업데이트하고 홀수 인덱스 17개에서 평가한다. 전체 그래프와 피처는 학습 때부터 사용한다. 별도의 검증 분할·반복 시드 통계 없이 테스트 정확도 하나만으로 방법을 선택하지 않는다.

## 5. 전역 가중 합과 구현 계약

연습문제의 weighted 방식은 L개의 학습 스칼라를 softmax로 정규화해 원래 레이어 표현을 가중 합산한다. 웨이트는 **모든 노드가 공유**하므로 노드별로 다른 웨이트를 내는 LSTM과 다르다. 단순한 대안이지만 max보다 언제나 표현력이 높거나 두 방식 사이의 성능을 보장하는 것은 아니다.

현재 클래스는 cat·max·lstm만 완성되어 있다. 다른 문자열을 주면 forward에서 h가 정의되지 않는다. num_layers=0도 빈 목록을 결합할 수 없어 유효하지 않다. 뒤의 보존된 코드는 두 메서드의 해당 분기에 삽입하는 조각이며 독립 실행 파일이 아니다.

모든 레이어의 표현을 저장하므로 결합용 메모리는 O(LNd)다. 이 구현은 레이어마다 전체 엣지를 처리하고 dropout·잔차·이웃 샘플링은 사용하지 않는다. JK의 레이어 선택 효과와 기본 메시지 패싱 구조의 변경 효과를 구분해 비교해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 34·숨은 차원 16·깊이 4·출력 2일 때 cat·max·lstm의 파라미터 수를 비교하라.

</div>

??? success "연습문제 1 풀이"

    공통 1,648개에 cat 출력 130개를 더해 1,778개, max 출력 34개를 더해 1,682개다. LSTM은 두 웨이트와 두 bias로 2,176개이며 점수레이어 17개·출력 레이어 34개를 더하면 전체 3,875개다. 모델 크기가 달라 정확도의 차이를 결합 규칙 하나만의 효과로 단정할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
max 결합의 레이어 순서 불변성과 노드 순서 등변성을 구분하라.

</div>

??? success "연습문제 2 풀이"

    max는 레이어 축의 좌표별 최댓값이라 레이어 표현을 어떤 순서로 나열해도 같다. 노드 순서를 바꾸면 각 레이어의 행 순서가 같이 바뀌고 결과 행도 그 순서로 바뀌므로 등변적이다. LSTM은 레이어 순서에는 민감하지만 같은 노드별 연산을 공유하므로 노드 재인덱싱에는 여전히 등변적이다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 weighted 조각의 웨이트 공유 범위와 현재 lstm 방식이 합산하는 값을 비교하라.

</div>

??? success "연습문제 3 풀이"

    weighted는 L개의 전역 스칼라를 학습해 모든 노드에 같은 레이어 웨이트를 사용하고 원래 layer_outputs를 합산한다. 현재 lstm은 노드별 LSTM 출력으로 웨이트를 만들며 합산 대상도 원래 표현이 아닌 LSTM 출력이다. 보존된 조각은 __init__과 forward에 각각 삽입해야 하며 성능 순위는 실험 없이 정할 수 없다.

    ```python
    # JKNet.__init__에 더한다:
    elif jk_mode == 'weighted':
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        self.out_lin = nn.Linear(hidden_ch, out_ch)

    # JKNet.forward에 더한다:
    elif self.jk_mode == 'weighted':
        weights = torch.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(layer_outputs, dim=0)  # [L, n, d]
        h = (stacked * weights.view(-1, 1, 1)).sum(dim=0)  # [n, d]
    ```


## 정리하며

JK는 여러 메시지 패싱 깊이의 표현을 결합한다. cat·max·LSTM의 정보 로스·순서 민감성·파라미터 수가 다르며, 이 구현의 LSTM 출력 가중 합과 모든 노드가 공유하는 전역 가중 합을 구분해야 한다.
