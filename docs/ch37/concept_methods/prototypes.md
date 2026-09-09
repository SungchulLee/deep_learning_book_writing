# 풀이를 위한 본보기 그물

**본보기 바탕 풀이** 방법은 들임을 배운 본보기와 견주어 미루어 본다. 손에 잡히지 않는 결 몫 매기기 대신, 이 방법은 이렇게 풀이한다. "이 들임은 **이** 본보기 자리와 닮았으므로 X로 가른다." 밭 밝은 이가 자연스레 알아듣는, 보기에 바탕을 둔 풀이를 준다.

---

## 1. 수학 밑바탕

### 풀이를 위한 본보기 그물

배운 쏘아 넣기 밭에 본보기 $P$개 $\{\mathbf{p}_1, \ldots, \mathbf{p}_P\}$이 있을 때 모형은 이렇게 셈한다.

$$
f(\mathbf{x}) = h\left(d(g(\mathbf{x}), \mathbf{p}_1), \ldots, d(g(\mathbf{x}), \mathbf{p}_P)\right)
$$

여기서

- $g(\mathbf{x})$은 들임 $\mathbf{x}$의 쏘아 넣기
- $d(\cdot, \cdot)$은 거리나 닮음 함수
- $h$은 본보기와의 닮음을 아울러 미루어 봄을 낸다

### ProtoPNet 얼개(첸 외, 2019)

ProtoPNet은 갈래마다의 본보기 조각을 배운다.

1. **겹치는 등뼈**가 결 그림을 뽑아낸다
2. **본보기 켜**가 배운 본보기와의 닮음을 셈한다
3. **온통 이은 켜**가 본보기 살아남에 짐을 실어 가른다

미루어 봄은 이렇다.

$$
\hat{y}_c = \sum_{p \in P_c} w_{cp} \max_{(i,j)} \log\left(\frac{\|z_{(i,j)} - \mathbf{p}_p\|^2 + 1}{\|z_{(i,j)} - \mathbf{p}_p\|^2 + \epsilon}\right)
$$

---

## 2. PyTorch 짜보기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeNetwork(nn.Module):
    """
    풀이되는 본보기 바탕 가름개.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        n_prototypes: int,
        n_classes: int,
        prototype_dim: int = 128
    ):
        super().__init__()
        self.backbone = backbone
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes

        # 본보기 밭으로 쏜다
        self.projection = nn.Linear(feature_dim, prototype_dim)

        # 배울 수 있는 본보기
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, prototype_dim)
        )

        # 본보기 닮음으로 가른다
        self.classifier = nn.Linear(n_prototypes, n_classes, bias=False)

    def compute_similarities(self, x):
        features = self.backbone(x)
        projected = self.projection(features)

        # 본보기마다의 L2 거리
        distances = torch.cdist(projected.unsqueeze(1), 
                               self.prototypes.unsqueeze(0))
        distances = distances.squeeze(1)

        # 닮음으로 바꾼다
        similarities = torch.log((distances + 1) / (distances + 1e-4))
        return similarities, distances

    def forward(self, x):
        similarities, _ = self.compute_similarities(x)
        logits = self.classifier(similarities)
        return logits

    def explain(self, x, prototype_images=None):
        """
        가장 가까운 본보기를 보여 미루어 봄을 풀이한다.
        """
        similarities, distances = self.compute_similarities(x)
        prediction = self.classifier(similarities).argmax(dim=1).item()

        # 본보기에 대한 갈래 짐
        class_weights = self.classifier.weight[prediction].detach().cpu().numpy()

        sim_values = similarities[0].detach().cpu().numpy()
        contributions = sim_values * class_weights

        sorted_idx = np.argsort(contributions)[::-1]

        explanation = []
        for idx in sorted_idx[:5]:
            explanation.append({
                'prototype_idx': idx,
                'similarity': sim_values[idx],
                'contribution': contributions[idx],
                'distance': distances[0, idx].item()
            })

        return prediction, explanation
```

---

## 3. 계량 금융에 쓰기

본보기 그물은 금융에서 다음에 특히 쓸모 있다.

- **신용 무릅씀**: "이 신청자의 결은 지난날 부도난/부도나지 않은 이런 자리와 닮았다"
- **속임수 찾기**: "이 거래 결은 알려진 속임수 본보기 3번과 맞물린다"
- **판 가르기**: "지금 저자 형편은 2018년 출렁임이 치솟은 때와 가장 닮았다"

---

## 연습문제

**연습문제 1.**
이 마디에서 밝힌 풀이 방법을, XOR 들임을 가르는 ReLU 살림의 두 켜 신경 그물에 걸어라. 들임 $x = [1, 1]$에 대한 풀이를 셈하여라.

??? success "연습문제 1 풀이"
    짐이 $W_1, b_1, W_2, b_2$인 익힌 XOR 그물에서 내놓기는 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 풀이 방법은 들임 결마다 몫을 내놓는다. $x = [1, 1]$(갈래 0)이면 두 결 모두 음수 가름에 이바지한다. 몫 값은 방법마다 다르다. 기울기 바탕 방법은 $\partial f / \partial x_i$을 셈하고, 흔들어 보는 방법은 결을 가렸을 때 내놓기가 얼마나 바뀌는지 잰다. XOR 문제는 판단 금이 선형이 아니므로 선형 풀이 방법이 그르칠 수 있음을 보여 준다. $\square$

---

**연습문제 2.**
이 마디의 풀이 방법이 온전함 공리를 채우는지, 곧 어떤 밑금 $x_0$에 대해 모든 결 몫의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 뒤집어라.

??? success "연습문제 2 풀이"
    온전함 공리(섀플리 값 이론에서는 효율이라고도 한다)는 몫의 합이 들임에서의 모형 내놓기와 밑금에서의 내놓기의 차이와 같다는 것이다. 이 방법이 온전함을 채우는지는 그 세움새에 달렸다. 기울기 방법은 온전함을 채우지 못한다(기울기는 그 자리의 것이고 길을 따라 쌓은 것이 아니다). 쌓은 기울기는 세움새 자체로 온전함을 채운다(길을 따라 미적분의 밑정리를 쓴다). SHAP 값은 섀플리 공리로 효율을 채운다. 온전함을 어기는 방법은 몫을 너무 많거나 적게 매길 수 있어, 온 몫을 온 세상 풀이로 믿기 어렵게 만든다. $\square$

---

**연습문제 3.**
이 방법이 내놓는 풀이가 얼마나 미더운지 따지는 시험을 꾸며라. 짚어 준 결이 참으로 모형에 중요한지를 넣기와 빼기 곡선으로 재어라.

??? success "연습문제 3 풀이"
    절차는 이렇다. (1) 시험 그림마다 결 몫을 셈한다. (2) 빼기: 몫이 큰 차례로 결을 하나씩 가리며 모형의 자신함이 떨어지는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 떨어진다. (3) 넣기: 빈 밑금에서 시작해 몫이 큰 차례로 결을 하나씩 드러내며 자신함이 오르는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 오른다. (4) 두 곡선의 아래 넓이를 셈한다. (5) 아무렇게나 매긴 차례(밑금)와 다른 방법에 견준다. 미더운 방법이면 빼기 넓이가 작고 넣기 넓이가 커야 한다. 통계로 미더우려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

**연습문제 4.**
이 풀이 방법을 신용 부도를 미루어 보는 금융 모형에 어떻게 걸 수 있는지 다루어라. 풀이가 채워야 할 규정 요건은 무엇인가?

??? success "연습문제 4 풀이"
    신용 모형에는 규정(ECOA, GDPR 22조)이 불리한 판단마다 그 사람에게 맞춘 풀이를 바란다. 방법은 다음을 내놓아야 한다. (1) 물리침에 가장 크게 이바지한 인자(불리한 처분 까닭). (2) 한결같은 풀이(비슷한 신청자는 비슷한 풀이를 받는다). (3) 손에 잡히는 풀이(신청자가 무엇을 바꾸어야 하는지 안다). 이 마디의 풀이 방법으로 결의 중요함을 짚을 수 있으나, 든든함(들임이 조금 바뀌었다고 풀이가 확 달라지면 안 된다)과 옳음(중요한 결을 없애면 미루어 봄이 바뀌어야 한다)을 따져 보아야 한다. 지켜야 할 됨됨이는 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

본보기 그물은 밭 밝은 이가 자연스레 헤아리는 결과 맞는, 자리에 바탕을 둔 헤아림을 준다. 모형이 담을 수 있는 것을 얼마쯤 내주고 풀이할 수 있음을 얻으며, 손에 잡히지 않는 결 몫이 아니라 손에 잡히는 보기를 들어 풀이한다.

**살펴볼 거리**

1. Chen, C., et al. (2019). "This Looks Like That: Deep Learning for Interpretable Image Recognition." *NeurIPS*.

2. Snell, J., et al. (2017). "Prototypical Networks for Few-shot Learning." *NeurIPS*.

3. Li, O., et al. (2018). "Deep Learning for Case-Based Reasoning through Prototypes." *AAAI*.
