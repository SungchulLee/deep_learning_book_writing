# Set2Set readout

한 번의 합·평균 대신 질의를 갱신하며 노드 집합을 여러 번 읽을 수 있다. [Set2Set의 집합 처리 아이디어](https://arxiv.org/abs/1511.06391)는 recurrent 질의와 순서 불변 어텐션 합을 결합한다. 이 페이지의 구현은 마지막 질의와 맥락을 반환하며, 처리 횟수만큼 모든 상태를 이어 붙이지 않는다.

## 1. 코드와 보존된 출력

```python
"""29.5.5: Set2Set 읽어내기"""
import torch, torch.nn as nn, torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

class Set2Set(nn.Module):
    def __init__(self, in_ch, processing_steps=3):
        super().__init__()
        self.steps = processing_steps
        self.lstm = nn.LSTMCell(2 * in_ch, in_ch)
        self.in_ch = in_ch
    def forward(self, x, batch):
        ng = batch.max().item() + 1
        q = torch.zeros(ng, self.in_ch, device=x.device)
        c = torch.zeros(ng, self.in_ch, device=x.device)
        r = torch.zeros(ng, self.in_ch, device=x.device)
        for _ in range(self.steps):
            q, c = self.lstm(torch.cat([q, r], dim=-1), (q, c))
            e = (x * q[batch]).sum(dim=-1)
            e_max = torch.full((ng,), float('-inf'), device=x.device)
            e_max.scatter_reduce_(0, batch, e, reduce='amax')
            a = torch.exp(e - e_max[batch])
            a_sum = torch.zeros(ng, device=x.device)
            a_sum.scatter_add_(0, batch, a)
            a = a / a_sum[batch].clamp(min=1e-10)
            r = torch.zeros(ng, self.in_ch, device=x.device)
            r.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x * a.unsqueeze(1))
        return torch.cat([q, r], dim=-1)

def demo():
    print("=" * 60); print("Set2Set Readout"); print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(10, 8)
    batch = torch.tensor([0,0,0,0,1,1,1,2,2,2])
    s2s = Set2Set(8, processing_steps=3)
    out = s2s(x, batch)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Output dim = 2 * input_dim = {2*8}")
    # 단순 모으기와 견주기
    pool = torch.zeros(3, 8)
    pool.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
    print(f"Sum pool output: {pool.shape}")
    print(f"Set2Set output (richer): {out.shape}")

if __name__ == "__main__":
    demo()
```

```
============================================================
Set2Set Readout
============================================================
Input: torch.Size([10, 8]), Output: torch.Size([3, 16])
Output dim = 2 * input_dim = 16
Sum pool output: torch.Size([3, 8])
Set2Set output (richer): torch.Size([3, 16])
```

데모는 임의의10×8 노드 특징을4개·3개·3개의 세 그래프로 나눈다. processing_steps=3에서 출력은3×16이며 sum pooling의3×8과 모양을 비교한다. 학습이나 그래프 예측 평가는 없으므로 출력 문구의 “richer”를 성능 개선의 증거로 읽지 않는다.

## 2. 질의와 맥락의 반복

그래프별 질의 q, cell c와 가중 합 r를0으로 초기화한다. 각 처리 단계에서 [q_prev∥r_prev]를 LSTMCell 입력으로, (q_prev,c_prev)를 hidden·cell로 사용한다. 새 q를 각 노드에 broadcast해 점수를 계산한다.

$$e_v=x_v^Tq_{g(v)},\qquad
\alpha_v=\operatorname{softmax}_{u\in V_{g(v)}}(e_u),
\qquad r_g=\sum_{v\in V_g}\alpha_vx_v$$

g(v)는 batch 벡터가 가리키는 그래프 번호다. 여러 그래프의 노드가 같은 softmax 분모를 공유하지 않도록 batch별 최대·합을 scatter로 계산한다. 최종 출력은 **[q_last∥r_last]∈R^(2d)**다.

q·c를0으로 시작해도 첫 LSTMCell에는 학습 가능한 가중치·bias가 있으므로 첫 질의가 항상0이거나 첫 어텐션이 반드시 균등하다고 가정하지 않는다. 처리 단계 수는 입력 그래프의 홉 수와 같지 않으며, 매 단계 같은 그래프의 모든 노드를 다시 읽는다.

## 3. 순열 불변성의 이유

노드를 순열하고 batch도 같은 순서로 바꾸면 노드별 점수와 α는 같은 순열로 바뀐다. 그래프별 가중 합 r는 같고, 따라서 다음 LSTM 상태 q·c도 같다. 이를 단계마다 적용하면 최종 그래프 표현이 순서에 불변임을 확인할 수 있다.

LSTM을 사용한다는 사실만으로 이웃을 임의 순서로 읽는 LSTM 집계와 같은 것은 아니다. 여기서는 LSTM이 노드 순서가 아니라 **처리 반복의 질의**를 갱신한다. 노드들을 합하는 부분은 순서 불변이다.

다만 모든 노드 다중집합을 단사적으로 표현하지는 않는다. 같은 노드 상태 집합 전체를 두 번 복제하면 각 동일 점수의 softmax 질량이 나뉘고 합 r는 같을 수 있다. 같은 초기 LSTM 상태에서 이후 질의도 같아 전체 출력이 그대로일 수 있다. 그래프 크기가 필요하면 별도 개수·합 특징을 고려한다.

## 4. batch·모양·수치 계약

입력은 N×d의 특징과 길이 N의 비음수 정수 batch다. 그래프 수를 max(batch)+1로 정하므로 빈 입력은 실패하고, 마지막 빈 그래프는 알 수 없다. 중간 번호가 비면 해당 행의 LSTM 상태는 생성되어도 의미 있는 관측 그래프를 읽은 결과는 아니다.

LSTM 입력은2d, hidden·cell 폭은d다. 새0 텐서들은 장치를 따르지만 dtype을 명시적으로 맞추지 않는 부분이 있어 x·모델과 함께 정밀도를 확인해야 한다. 처리 횟수0이면 루프를 돌지 않고0의2d 벡터를 반환하며 노드 정보를 사용하지 않는다. 의도한 입력에서는 양의 정수 횟수를 정한다.

점수 최대를 −∞에서 시작해 그래프별 실제 최댓값을 빼므로 유한 점수의 softmax를 안정화한다. 하지만 비정상 입력의 NaN·Inf를 검사하지 않으며 batch 인덱스·특징 모양의 검증도 별도다.

## 5. 비용과 학습 범위

그래프 수 G, 노드 수 N, 폭 d와 처리 횟수 P에서 비용은 대략 P×(Gd²+Nd)로 나뉜다. LSTM 상태 갱신과 노드 점수·가중 합을 모두 계산하기 때문이다. 단순 합 O(Nd)보다 복잡하며, P를 늘려도 파라미터를 공유하므로 수가 반복마다 늘지는 않는다.

LSTMCell(2d,d)의 파라미터는 가중치12d²와 두 bias 벡터의 합8d로, d=8에서는832개다. output 차원은 언제나2d이며 P에 비례하지 않는다. 더 넓은 출력이나 학습 가능한 질의가 항상 더 좋은 예측을 보장하지 않는다.

이 readout은 인접 행렬을 받지 않는다. 앞 GNN이 만든 임베딩을 읽으면 구조 정보가 들어올 수 있지만, 같은 원시 노드 특징만 주고 다른 엣지 구조를 직접 구분할 수는 없다. 실제 작업에는 node encoder·예측 head·손실과 별도의 일반화 평가가 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
N=10, d=8, 그래프3개와 processing_steps=3에서 입력·상태·출력 모양을 구하라.

</div>

??? success "연습문제 1 풀이"

    x는10×8, q·c·r는각각3×8이다. LSTM 입력은3×16이고 최종 [q∥r]는3×16이다. 세 번 처리했다고3×48이 되는 것은 아니다. 모든 중간 상태를 concat하는 구현이 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
LSTM을 사용하면서 노드 순서에 불변일 수 있는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    LSTM은 노드를 차례로 읽지 않고 그래프별 질의를 갱신한다. 노드를 순열하면 점수·softmax 항도 순열되지만 그래프별 가중 합은 같다. 따라서 다음 질의와 cell도 같다는 귀납으로 최종 출력을 보존한다. batch를 노드와 같은 순열로 바꿔야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
Set2Set이 노드 수를 항상 구분하는지와 처리 횟수의 비용을 설명하라.

</div>

??? success "연습문제 3 풀이"

    같은 노드 다중집합 전체를 두 번 복제하면 점수도 중복되어 가중 평균 r가 같고 이후 질의도 같을 수 있다. 따라서 항상 개수를 구분하지 않는다. 처리 횟수를 늘리면 공유 LSTM·노드 점수 계산이 반복되어 비용이 늘지만 파라미터 수나 출력2d가 함께 늘지는 않는다. 데모의 모양 비교만으로 성능 우위를 주장할 수 없다.



## 정리하며

Set2Set은 그래프별 반복 질의와 순서 불변의 가중 합을 결합한다. 마지막2d 출력, batch 정규화와 입력·횟수 계약을 확인하고, 반복 어텐션의 유연성과 단사성·성능 보장을 구분한다.
