# GNN 해석 방법

**그래프 뉴럴 네트워크(GNN)**는 노드 피처와 그래프 구조가 함께 예측을 결정하는 그래프 데이터를 다룬다. GNN의 예측을 설명하려면 입력 피처뿐 아니라 **엣지**와 **서브그래프 구조**에도 중요도를 부여해야 한다. 이 절은 GNNExplainer를 자세히 다루고, PGExplainer와 SubgraphX는 끝의 참고 문헌으로 남긴다.

---

## 1. GNN 설명 문제

$G = (V, E)$가 그래프이고 $X$가 노드 피처일 때 GNN 예측 $\hat{y} = f(G, X)$에 대해 다음을 찾는다.

1. **노드 중요도**: 이 예측에 어느 노드가 중요한가?
2. **엣지 중요도**: 어느 엣지가 중요한가?
3. **피처 중요도**: 어느 노드 피처가 예측을 이끄는가?
4. **서브그래프 설명**: 이 예측을 설명하는 가장 작은 서브그래프는 무엇인가?

---

## 2. GNNExplainer

GNNExplainer(Ying 등, 2019)는 엣지와 피처에 씌울 부드러운(soft) 마스크, 곧 0과 1 사이 값을 갖는 마스크를 학습한다. 가린 그래프와 예측 사이의 상호정보량(MI)을 최대화하는 것이 목표다.

$$
\max_{M_E, M_F} MI(Y, (G_s, X_s)) = H(Y) - H(Y | G = G_s, X = X_s)
$$

여기서 $G_s = G \odot M_E$는 가린 그래프이고 $X_s = X \odot M_F$는 가린 피처다. $H(Y)$는 마스크와 상관없는 상수이므로, 상호정보량을 최대화하는 것은 조건부 엔트로피를 최소화하는 것과 같다. 실제로는 원래 예측한 클래스의 로그 확률을 최대화하고, 마스크가 작고 0이나 1에 가깝도록 크기 정칙화와 엔트로피 정칙화를 더한다. 아래 구현은 간추려 엣지 마스크 $M_E$만 학습한다.

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

- **거래상대방 위험**: 은행 네트워크의 어느 연결이 시스템 전체의 위험을 키우는가?
- **공급망**: 어느 공급업체와의 상관이 회사의 위험 노출을 가장 크게 결정하는가?
- **소셜 네트워크**: 트레이더 네트워크에서 어느 정보 흐름이 시장의 움직임을 미리 알리는가?

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
노드 $v$에 이웃 $a$, $b$가 엣지로 이어져 있고 두 이웃의 피처가 $x_a = x_b = 1$이다. GNN은 엣지 가중치 $m_a, m_b \in [0, 1]$을 받아 $s_v = \text{ReLU}(m_a x_a - m_b x_b) + \text{ReLU}(m_b x_b - m_a x_a)$를 계산하고, $s_v > 0.5$이면 클래스 1, 아니면 클래스 0으로 분류한다(마스크가 0 또는 1이면 두 이웃 피처의 XOR이다). 원래 그래프의 예측은 무엇인가? 예측을 지키면서 마스크의 합을 가장 작게 하는 GNNExplainer의 해는 무엇이며, 그 설명은 무엇을 놓치는가?

</div>

??? success "풀이"
    $m_a = m_b = 1$이면 $s_v = 0$이라 클래스 0이다. $x_a = x_b = 1$이면 $s_v = |m_a - m_b|$이므로, 예측을 지키는 마스크는 $|m_a - m_b| \le 0.5$인 모든 쌍이다. 크기 정칙화는 마스크의 합을 줄이므로 해는 $m_a = m_b = 0$, 곧 엣지가 하나도 없는 빈 서브그래프다. GNNExplainer는 「이 예측에는 아무 엣지도 필요 없다」고 답한다. 그런데 엣지 하나만 지우면($m_a = 1$, $m_b = 0$) $s_v = 1$이 되어 예측이 클래스 1로 뒤집힌다. 두 엣지는 함께 있어서 서로를 상쇄하고 있었고, 빈 그래프도 우연히 같은 예측을 낼 뿐이다. 「예측을 지키는 가장 작은 서브그래프」라는 목표는 이런 상호작용을 설명으로 내놓지 못한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. GNNExplainer의 마스크 값은 「예측을 지키는 데 그 엣지를 얼마나 남겨야 하는가」를 나타낼 뿐 출력의 차이를 나눈 것이 아니므로 완전성을 만족하지 않는다. 연습문제 1이 반례다. 빈 그래프와 원래 그래프의 예측이 같아 나눌 차이가 없는데도, 엣지 하나만 지우면 예측이 바뀐다. $\square$

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

GNN 설명 방법은 노드, 엣지, 서브구조에 중요도를 부여해 해석 방법을 그래프 데이터로 넓힌다. 상관 구조가 위험과 수익률을 좌우하는 금융 네트워크 살피기에 특히 잘 맞는다.

**살펴볼 거리**

1. Ying, R., et al. (2019). "GNNExplainer: Generating Explanations for Graph Neural Networks." *NeurIPS*.

2. Luo, D., et al. (2020). "Parameterized Explainer for Graph Neural Network." *NeurIPS*.

3. Yuan, H., et al. (2021). "On Explainability of Graph Neural Networks via Subgraph Explorations." *ICML*.
