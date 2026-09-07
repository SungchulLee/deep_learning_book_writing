# GNN 풀이 방법
## 들머리

**그래프 신경 그물(GNN)**은 마디의 결과 그래프 얼개가 함께 미루어 봄을 가르는 그래프 꼴 자료를 다룬다. GNN의 미루어 봄을 풀이하려면 들임 결뿐 아니라 **변**과 **밑그래프 얼개**에도 중요함을 돌려야 한다. 이 마디는 GNNExplainer, PGExplainer, SubgraphX을 다룬다.

## GNN 풀이 문제

$G = (V, E)$이 그래프이고 $X$이 마디 결일 때 GNN 미루어 봄 $\hat{y} = f(G, X)$에 대해 다음을 찾는다.

1. **마디 중요함**: 이 미루어 봄에 어느 마디가 중요한가?
2. **변 중요함**: 어느 이음이 종요로운가?
3. **결 중요함**: 어느 마디 결이 미루어 봄을 이끄는가?
4. **밑그래프 풀이**: 어떤 가장 작은 밑그래프가 이 미루어 봄을 풀이하는가?

## GNNExplainer

GNNExplainer(잉 외, 2019)은 변과 결에 무른 가리개를 배운다.

$$
\max_{M_E, M_F} MI(Y, (G_s, X_s)) = H(Y) - H(Y | G = G_s, X = X_s)
$$

여기서 $G_s = G \odot M_E$은 가린 그래프이고 $X_s = X \odot M_F$은 가린 결이다.

### 짜보기

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

            # 미루어 봄의 낌새를 가장 크게 한다
            pred_loss = -log_prob[target]

            # 성기게 하는 정칙화
            size_loss = mask.sum() * 0.01

            # 가리개를 또렷이 가르는 엔트로피
            entropy = -mask * torch.log(mask + 1e-10) - (1-mask) * torch.log(1-mask + 1e-10)
            entropy_loss = entropy.mean() * 0.1

            loss = pred_loss + size_loss + entropy_loss
            loss.backward()
            optimizer.step()

        return torch.sigmoid(edge_mask).detach()
```

## 계량 금융에 쓰기

GNN 풀이는 금융 그물에 잘 맞는다.

- **맞거래 무릅씀**: 은행 그물의 어느 이음이 얼개 전체의 무릅씀을 키우는가?
- **대는 사슬**: 어느 대는 이와의 얽힘이 회사의 무릅씀 결을 가장 크게 가르는가?
- **사람 그물**: 거래하는 이들의 그물에서 어느 소식 흐름이 저자의 움직임을 미리 알리는가?

```python
def explain_financial_network(model, node_idx, x, edge_index):
    """금융 그물의 미루어 봄을 풀이한다."""
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

## 간추림

GNN 풀이 방법은 마디, 변, 밑얼개에 중요함을 돌려 그래프 꼴 자료로 풀이하기를 넓힌다. 얽힘 얼개가 무릅씀과 돌아옴을 이끄는 금융 그물 살피기에 특히 잘 맞는다.

## 살펴볼 거리

1. Ying, R., et al. (2019). "GNNExplainer: Generating Explanations for Graph Neural Networks." *NeurIPS*.

2. Luo, D., et al. (2020). "Parameterized Explainer for Graph Neural Network." *NeurIPS*.

3. Yuan, H., et al. (2021). "On Explainability of Graph Neural Networks via Subgraph Explorations." *ICML*.

## 익힘 문제

**익힘 1.**
이 마디에서 밝힌 풀이 방법을, XOR 들임을 가르는 ReLU 살림의 두 켜 신경 그물에 걸어라. 들임 $x = [1, 1]$에 대한 풀이를 셈하여라.

??? success "익힘 1 풀이"
    짐이 $W_1, b_1, W_2, b_2$인 익힌 XOR 그물에서 내놓기는 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 풀이 방법은 들임 결마다 몫을 내놓는다. $x = [1, 1]$(갈래 0)이면 두 결 모두 음수 가름에 이바지한다. 몫 값은 방법마다 다르다. 기울기 바탕 방법은 $\partial f / \partial x_i$을 셈하고, 흔들어 보는 방법은 결을 가렸을 때 내놓기가 얼마나 바뀌는지 잰다. XOR 문제는 판단 금이 선형이 아니므로 선형 풀이 방법이 그르칠 수 있음을 보여 준다. $\square$

---

**익힘 2.**
이 마디의 풀이 방법이 온전함 공리를 채우는지, 곧 어떤 밑금 $x_0$에 대해 모든 결 몫의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 뒤집어라.

??? success "익힘 2 풀이"
    온전함 공리(섀플리 값 이론에서는 효율이라고도 한다)는 몫의 합이 들임에서의 모형 내놓기와 밑금에서의 내놓기의 차이와 같다는 것이다. 이 방법이 온전함을 채우는지는 그 세움새에 달렸다. 기울기 방법은 온전함을 채우지 못한다(기울기는 그 자리의 것이고 길을 따라 쌓은 것이 아니다). 쌓은 기울기는 세움새 자체로 온전함을 채운다(길을 따라 미적분의 밑정리를 쓴다). SHAP 값은 섀플리 공리로 효율을 채운다. 온전함을 어기는 방법은 몫을 너무 많거나 적게 매길 수 있어, 온 몫을 온 세상 풀이로 믿기 어렵게 만든다. $\square$

---

**익힘 3.**
이 방법이 내놓는 풀이가 얼마나 미더운지 따지는 시험을 꾸며라. 짚어 준 결이 참으로 모형에 중요한지를 넣기와 빼기 곡선으로 재어라.

??? success "익힘 3 풀이"
    절차는 이렇다. (1) 시험 그림마다 결 몫을 셈한다. (2) 빼기: 몫이 큰 차례로 결을 하나씩 가리며 모형의 자신함이 떨어지는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 떨어진다. (3) 넣기: 빈 밑금에서 시작해 몫이 큰 차례로 결을 하나씩 드러내며 자신함이 오르는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 오른다. (4) 두 곡선의 아래 넓이를 셈한다. (5) 아무렇게나 매긴 차례(밑금)와 다른 방법에 견준다. 미더운 방법이면 빼기 넓이가 작고 넣기 넓이가 커야 한다. 통계로 미더우려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

**익힘 4.**
이 풀이 방법을 신용 부도를 미루어 보는 금융 모형에 어떻게 걸 수 있는지 다루어라. 풀이가 채워야 할 규정 요건은 무엇인가?

??? success "익힘 4 풀이"
    신용 모형에는 규정(ECOA, GDPR 22조)이 불리한 판단마다 그 사람에게 맞춘 풀이를 바란다. 방법은 다음을 내놓아야 한다. (1) 물리침에 가장 크게 이바지한 인자(불리한 처분 까닭). (2) 한결같은 풀이(비슷한 신청자는 비슷한 풀이를 받는다). (3) 손에 잡히는 풀이(신청자가 무엇을 바꾸어야 하는지 안다). 이 마디의 풀이 방법으로 결의 중요함을 짚을 수 있으나, 든든함(들임이 조금 바뀌었다고 풀이가 확 달라지면 안 된다)과 옳음(중요한 결을 없애면 미루어 봄이 바뀌어야 한다)을 따져 보아야 한다. 지켜야 할 됨됨이는 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$
