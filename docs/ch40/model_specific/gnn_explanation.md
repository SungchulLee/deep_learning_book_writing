# GNN 해석 방법

**그래프 뉴럴 네트워크(GNN)**은 마디의 피처와 그래프 구조가 함께 추론을 가르는 그래프 형태 데이터를 다룬다. GNN의 추론을 풀이하려면 입력 피처뿐 아니라 **변**과 **밑그래프 구조**에도 중요함을 돌려야 한다. 이 절은 GNNExplainer, PGExplainer, SubgraphX을 다룬다.

---

## 1. GNN 설명 문제

$G = (V, E)$이 그래프이고 $X$이 절 피처일 때 GNN 추론 $\hat{y} = f(G, X)$에 대해 다음을 찾는다.

1. **마디 중요함**: 이 추론에 어느 마디가 중요한가?
2. **변 중요함**: 어느 이음이 중요한이?
3. **피처 중요함**: 어느 마디 피처가 추론을 이끄는가?
4. **밑그래프 설명**: 어떤 가장 작은 밑그래프가 이 추론을 해석하는가?

---

## 2. GNNExplainer

GNNExplainer(잉 외, 2019)은 변과 피처에 취약한 마스크를 배운다.

$$
\max_{M_E, M_F} MI(Y, (G_s, X_s)) = H(Y) - H(Y | G = G_s, X = X_s)
$$

여기서 $G_s = G \odot M_E$은 가린 그래프이고 $X_s = X \odot M_F$은 가린 피처가다.

### 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNExplainer:
    """마디 가름과 그래프 가름을 위한 GNNExplainer."""

    def __init__(self, model, num_hops=3, lr=0.01, epochs=100):
        self.model = model
        self.num_hops = num_hops
        self.lr = lr
        self.epochs = epochs

    def explain_node(self, node_idx, x, edge_index, target=None):
        self.model.eval()

        if target is None:
            with torch.no_grad():
                out = self.model(x, edge_index)
                target = out[node_idx].argmax().item()

        num_edges = edge_index.shape[1]
        edge_mask = nn.Parameter(torch.randn(num_edges) * 0.1)

        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            mask = torch.sigmoid(edge_mask)
            masked_edge_weight = mask

            out = self.model(x, edge_index, edge_weight=masked_edge_weight)
            log_prob = F.log_softmax(out[node_idx], dim=0)

            # 미루어 봄의 확률을 가장 크게 한다
            pred_loss = -log_prob[target]

            # 성기게 하는 정칙화
            size_loss = mask.sum() * 0.01

            # 마스크를 또렷이 가르는 엔트로피
            entropy = -mask * torch.log(mask + 1e-10) - (1-mask) * torch.log(1-mask + 1e-10)
            entropy_loss = entropy.mean() * 0.1

            loss = pred_loss + size_loss + entropy_loss
            loss.backward()
            optimizer.step()

        return torch.sigmoid(edge_mask).detach()
```

---

## 3. 계량 금융 응용

GNN 설명은 금융 네트워크에 잘 맞는다.

- **맞거래 위험**: 은행 네트워크의 어느 이음이 구조 전체의 위험을 키우는가?
- **대는 사슬**: 어느 대는 이와의 상관이 회사의 위험 피처를 가장 크게 가르는가?
- **사람 네트워크**: 거래하는 이들의 네트워크에서 어느 정보 흐름이 저자의 액션을 미리 알리는가?

```python
def explain_financial_network(model, node_idx, x, edge_index):
    """금융 네트워크의 미루어 봄을 풀이한다."""
    explainer = GNNExplainer(model)
    edge_importance = explainer.explain_node(node_idx, x, edge_index)

    # 앞선 이음
    top_edges = edge_importance.argsort(descending=True)[:10]
    print(f"마디 {node_idx}에 가장 중요한 이음:")
    for idx in top_edges:
        src, dst = edge_index[:, idx]
        print(f"  {src.item()} -> {dst.item()}: {edge_importance[idx]:.3f}")

    return edge_importance
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 절에서 다룬 설명 방법을, XOR 입력을 분류하는 ReLU 활성화의 두 레이어 뉴럴 네트워크에 적용하라. 입력 $x = [1, 1]$에 대한 설명을 계산하여라.

</div>

??? success "풀이"
    웨이트가 $W_1, b_1, W_2, b_2$인 학습된 XOR 네트워크에서 출력은 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 설명 방법은 입력 피처마다 기여도를 내놓는다. $x = [1, 1]$(클래스 0)이면 두 피처 모두 음수 쪽 분류에 기여한다. 기여도 값은 방법마다 다르다. 그래디언트 기반 방법은 $\partial f / \partial x_i$을 계산하고, 섭동 기반 방법은 피처를 가렸을 때 출력이 얼마나 바뀌는지 잰다. XOR 문제는 결정 경계가 선형이 아니므로 선형 설명 방법이 틀릴 수 있음을 보여 준다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 방법이 내놓는 설명이 얼마나 믿을 만한지 평가하는 실험을 설계하라. 방법이 중요하다고 짚은 피처가 실제로 모델에 중요한지를 삽입 곡선과 삭제 곡선으로 재어라.

</div>

??? success "풀이"
    절차는 이렇다. (1) 시험 이미지마다 피처 기여도를 계산한다. (2) 삭제: 기여도가 큰 차례로 피처를 하나씩 가리며 모델의 확신도가 떨어지는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 떨어진다. (3) 삽입: 빈 베이스라인에서 시작해 기여도가 큰 차례로 피처를 하나씩 드러내며 확신도가 오르는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 오른다. (4) 두 곡선 아래 넓이를 계산한다. (5) 무작위 순서(베이스라인)와 다른 방법에 견준다. 믿을 만한 방법이면 삭제 곡선의 넓이가 작고 삽입 곡선의 넓이가 커야 한다. 통계적으로 믿을 만하려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 설명 방법을 신용 부도를 예측하는 금융 모델에 어떻게 적용할 수 있는지 논하라. 설명이 채워야 할 규제 요건은 무엇인가?

</div>

??? success "풀이"
    신용 모델에는 규제(ECOA, GDPR 22조)가 불리한 결정마다 그 사람에게 맞춘 설명을 요구한다. 방법은 다음을 내놓아야 한다. (1) 거절에 가장 크게 기여한 요인(불리한 처분 사유). (2) 일관된 설명(비슷한 신청자는 비슷한 설명을 받는다). (3) 실행 가능한 설명(신청자가 무엇을 바꾸어야 하는지 안다). 이 절의 설명 방법으로 피처 중요도를 짚을 수 있으나, 강건성(입력이 조금 바뀌었다고 설명이 확 달라지면 안 된다)과 충실성(중요한 피처를 없애면 예측이 바뀌어야 한다)을 평가해야 한다. 보호 속성은 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

GNN 설명 방법은 마디, 변, 밑구조에 중요함을 돌려 그래프 형태 데이터로 해석하기를 넓힌다. 상관 구조가 위험과 수익률을 이끄는 금융 네트워크 살피기에 특히 잘 맞는다.

**살펴볼 거리**

1. Ying, R., et al. (2019). "GNNExplainer: Generating Explanations for Graph Neural Networks." *NeurIPS*.

2. Luo, D., et al. (2020). "Parameterized Explainer for Graph Neural Network." *NeurIPS*.

3. Yuan, H., et al. (2021). "On Explainability of Graph Neural Networks via Subgraph Explorations." *ICML*.
