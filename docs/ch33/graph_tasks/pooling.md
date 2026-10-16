# 그래프 풀링과 readout

노드별 표현에서 그래프 전체의 출력을 만들려면 노드 수와 순서에 맞는 집계가 필요하다. global readout은 노드 집합을 벡터로 줄이고, 계층적 pooling은 더 작은 그래프를 만들어 후속 연산을 이어 간다. 둘 모두 정보를 요약하므로 보존하는 양과 버리는 양을 명시해야 한다.

## 1. 합·평균·최대 readout

노드 상태 hᵥ∈Rᵈ에 대해 합, 평균과 좌표별 최대를 사용할 수 있다. 세 연산은 순열 불변이지만 서로 다른 정보를 남긴다. 합은 중복·총량에 민감하고 평균은 크기로 정규화하며 최대는 좌표별 극값을 남긴다.

합도 노드 수를 항상 정확히 알려 주지는 않는다. 모든 상태가0이거나 상쇄하면 다른 크기의 그래프도 같은 합이 된다. 정확한 개수를 함께 보존하려면 상수1 피처가나 별도 노드 수를 넣을 수 있다. 여러 readout을 이어 붙이면 서로 다른 요약을 제공하지만 전체 다중집합을 항상 복원하지는 못한다.

readout이 인접 행렬을 직접 받지 않더라도 입력 노드 임베딩에는 앞선 메시지 패싱의 구조 정보가 들어 있을 수 있다. 따라서 global pooling을 썼다는 이유로 모든 그래프 구조를 사용하지 않았다고 설명하지 않는다.

## 2. 어텐션 readout

각 노드의 점수 sᵥ=fθ(hᵥ)를 같은 그래프 안에서 정규화한다.

$$\alpha_v=\frac{e^{s_v}}{\sum_{u\in V}e^{s_u}},
\qquad h_G=\sum_v\alpha_vh_v$$

웨이트가 비음수·합1이면 출력은 노드 상태의 가중 평균이다. 더 큰 점수가 학습된 목적에 기여할 수 있지만, 인과적으로 중요한 노드를 찾아냈다고 자동으로 해석하지 않는다. 같은 노드 다중집합 전체를 두 번 반복하면 분모·분자가 함께 늘어 이 가중 평균은 같을 수 있다.

## 3. 구현과 보존된 출력

```python
"""
그래프 뉴럴 넷트웍의 그래프 모으기 연산.

합, 평균, 어텐션, 위 k개 모으기를 보여 준다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# === 납작한 모으기 ===
def global_sum_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """배치 번호로 묶어 마디에 대해 합 모으기."""
    num_graphs = batch.max().item() + 1
    out = torch.zeros(num_graphs, x.size(1), device=x.device)
    out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
    return out

def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """배치 번호로 묶어 마디에 대해 평균 모으기."""
    sums = global_sum_pool(x, batch)
    counts = torch.zeros(sums.size(0), device=x.device)
    counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
    return sums / counts.unsqueeze(1).clamp(min=1)

# === 어텐션 모으기 ===
class AttentionPool(nn.Module):
    """어텐션 무게를 준 온 자리 모으기."""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        scores = self.score_fn(x)
        # 그래프마다 소프트맥스
        max_scores = torch.zeros(batch.max() + 1, 1, device=x.device)
        max_scores.scatter_reduce_(0, batch.unsqueeze(1), scores, reduce="amax")
        scores = scores - max_scores[batch]
        exp_scores = scores.exp()
        sum_exp = torch.zeros(batch.max() + 1, 1, device=x.device)
        sum_exp.scatter_add_(0, batch.unsqueeze(1), exp_scores)
        alpha = exp_scores / sum_exp[batch].clamp(min=1e-8)
        weighted = x * alpha
        return global_sum_pool(weighted, batch)

# === 보기 ===
if __name__ == "__main__":
    torch.manual_seed(42)
    # 그래프 둘을 배치: 0번 그래프는 마디 3개, 1번 그래프는 마디 2개
    x = torch.randn(5, 8)
    batch = torch.tensor([0, 0, 0, 1, 1])

    print("Sum pool shape:", global_sum_pool(x, batch).shape)    # [2, 8]
    print("Mean pool shape:", global_mean_pool(x, batch).shape)  # [2, 8]

    attn_pool = AttentionPool(in_dim=8)
    print("Attn pool shape:", attn_pool(x, batch).shape)         # [2, 8]
```

```
Sum pool shape: torch.Size([2, 8])
Mean pool shape: torch.Size([2, 8])
Attn pool shape: torch.Size([2, 8])
```

x는 전체 배치 노드를 모은 N×d, batch는 각 행의 그래프 번호를 담는 길이 N 정수 벡터다. 예제는5노드를3개·2개로 나눈 두 그래프를 사용해 모두2×8 출력을 확인한다. 학습·그래프 예측 정확도는 측정하지 않는다.

sum은 그래프별 scatter_add, mean은 sum을 노드 수로 나눈다. 출력 그래프 수를 max(batch)+1로 추정하므로 빈 batch는 실패하고, 마지막에 노드가 없는 그래프는 알 수 없다. 중간 번호가 비면 해당 sum·mean은0행이 될 수 있지만 의미 있는 빈 그래프 처리를 설계한 것은 아니다. batch와 x의 장치·행 수·인덱스와 새 텐서의 dtype을 맞춰야 한다.

## 4. 어텐션 softmax의 수치 제한

AttentionPool의 max_scores는 −∞가 아니라0에서 시작한다. 따라서 각 그래프의 모든 점수가 음수이면 실제 최댓값 대신0을 뺀다. 정확한 산술에서 임의 상수 이동은 softmax를 바꾸지 않지만, 여기서는 exp의 underflow와 **분모 clamp** 때문에 결과가 달라질 수 있다.

예를 들어 점수 두 개가 모두−20이면 exp 합은 약4.12×10⁻⁹로 clamp 하한1e−8보다 작다. 그러면 계수 합이 약0.412가 되어 의도한 합1 가중 평균이 아니다. 모든 점수가 큰 음수일 때 출력이0에 가깝게 줄어드는 현상도 가능하다. 기본 shape 데모를 통과했다고 이 극단값 처리가 검증된 것은 아니다.

## 5. 계층적 pooling의 두 방식

[DiffPool](https://arxiv.org/abs/1806.08804)은 노드 임베딩 Z와 행별 softmax 배정 S∈R^(n×k)를 학습해 X′=SᵀZ, A′=SᵀAS를 만든다. X′는 배정 가중 **합**이며 자동으로 클러스터 평균은 아니다. A′는 가중·밀집이 될 수 있고 대각도 생긴다.

S 저장은 O(nk), 출력 인접 저장은 O(k²)다. k가 n에 비례하거나 원본 A를 밀집 저장할 때 이차 비용이 생기며, 모든 설정을 무조건 같은 O(n²)로 요약하지 않는다. 임베딩·배정 GNN과 실제 곱셈 비용도 추가된다.

TopK 계열은 점수로 일부 노드를 고르고 그 노드 사이의 유도 엣지를 남긴다. [SAGPool](https://arxiv.org/abs/1904.08082)은 그래프 구조를 사용하는 점수 GNN을 결합한다. 선택 개수와 보존 비율 r을 구분해 k=ceil(rn) 또는 floor 규칙을 명시한다. 중요한 bridge를 버려 연결성이 깨질 수 있으며 선택된 부분 그래프가 원래 구조를 전부 보존하지는 않는다.

hard top-k는 선택 경계에서 매끄럽지 않다. 선택된 값·게이트에 gradient가 흐르는 것과 이산 인덱스 선택 자체가 모든 곳에서 미분 가능한 것은 다르다. 비용도 점수 GNN·top-k·엣지 필터링을 포함하므로 일괄 O(n)이라 할 수 없다.

## 6. 반복 어텐션과 선택 기준

[Set2Set](set2set.md)은 LSTM 질의를 여러 번 갱신해 그래프 안에서 어텐션 가중 합을 반복한다. 이 페이지와 연결된 구현은 **마지막 질의와 마지막 가중 합을 붙인2d 출력**이며 모든 중간 맥락 벡터를 concat하지 않는다.

어떤 readout이 적절한지는 총량·평균·극값 등 목표의 의미와 그래프 크기에 달렸다. 노드 선택·soft assignment를 통해 후속 구조 연산을 이어 갈 필요가 있는지도 판단한다. 출력 차원이나 학습 가능한 파라미터 수만으로 더 높은 성능을 보장하지 않는다. 세부 구현과 제한은 [계층적 풀링](hierarchical_pooling.md)에서 다룬다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
두 노드 상태(1,2),(3,0)의 합·평균·좌표별 최대를 구하라. 노드를 모두 두 번 복제하면 어떻게 되는가?

</div>

??? success "연습문제 1 풀이"

    합은(4,2), 평균은(2,1), 최대는(3,2)다. 두 벡터를 각각 두 번 반복하면 합은(8,4)로 두 배가 되고 평균·최대는 같다. 합도 일반적으로 노드 수를 정확히 식별하지는 않으며 별도 상수 피처가 필요할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
AttentionPool의 두 점수가 모두−20일 때 clamp 전후 정규화의 차이를 설명하라.

</div>

??? success "연습문제 2 풀이"

    정확한 softmax라면 두 계수는 각각1/2다. 코드는 최대 초기값0을 빼서 exp 합이 약4.12×10⁻⁹이고 이를1e−8로 올린다. 두 계수는 각각 약0.206, 합은약0.412가 된다. 안정적인 실제 최댓값 이동과 분모 처리가 필요하며 shape 출력만으로 이 성질을 검증할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
n개 노드·k개 클러스터의 DiffPool 저장과 결과의 의미를 설명하라.

</div>

??? success "연습문제 3 풀이"

    S는 n×k, X′는 k×d, A′는 k×k다. SᵀZ는 배정 가중 합이지 평균이 아니고, A′는 가중·밀집·루프가 있는 그래프일 수 있다. S 저장 O(nk)와 출력 O(k²), 원본 행렬·GNN 비용을 구분해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
global readout과 계층적 pooling이 구조 정보를 사용하는 방식을 비교하고 Set2Set 출력 차원을 설명하라.

</div>

??? success "연습문제 4 풀이"

    global readout은 앞 GNN이 구조를 반영한 노드 상태를 요약할 수 있지만 후속 노드·엣지 그래프를 만들지는 않는다. 계층적 pooling은 축소된 그래프를 만들어 추가 연산을 이어 간다. 둘 다 정보 로스 가능성이 있다. 연결된 Set2Set 구현은 마지막 q와 r를 붙인2d 출력이며 처리 반복 수를 곱한 차원이 아니다.



## 정리하며

readout과 계층적 pooling은 다른 종류의 요약이다. 그래프별 batch 구분·빈 입력·수치 정규화를 확인하고, 가중 합·평균·노드 선택의 의미와 정보 로스를 구분해 선택한다.

**참고 문헌**

- Ying, R. et al. "Hierarchical Graph Representation Learning with
  Differentiable Pooling." NeurIPS 2018.
- Lee, J. et al. "Self-Attention Graph Pooling." ICML 2019.
- Xu, K. et al. "How Powerful are Graph Neural Networks?" ICLR 2019.
