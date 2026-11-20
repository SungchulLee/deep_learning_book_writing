# 해석을 위한 프로토타입 네트워크

**프로토타입 기초 설명** 방법은 입력을 배운 프로토타입과 비교하여 예측한이다. 손에 잡히지 않는 피처 기여도 분석 대신, 이 방법은 이렇게 해석한다. "이 입력은 **이** 프로토타입 자리와 닮았으므로 X로 가른다." 그 분야 전문가가 자연스레 알아듣는, 보기에 기초를 둔 설명을 준다.

---

## 1. 수학 기초

### 해석을 위한 프로토타입 네트워크

배운 쏘아 넣기 분야에 프로토타입 $P$개 $\{\mathbf{p}_1, \ldots, \mathbf{p}_P\}$이 있을 때 모델은 이렇게 계산한다.

$$
f(\mathbf{x}) = h\left(d(g(\mathbf{x}), \mathbf{p}_1), \ldots, d(g(\mathbf{x}), \mathbf{p}_P)\right)
$$

여기서

- $g(\mathbf{x})$은 입력 $\mathbf{x}$의 쏘아 넣기
- $d(\cdot, \cdot)$은 거리나 닮음 함수
- $h$은 프로토타입와의 닮음을 아울러 추론을 낸다

### ProtoPNet 구조(첸 외, 2019)

ProtoPNet은 클래스마다의 프로토타입 조각을 배운다.

1. **겹치는 등뼈**가 피처 그림을 뽑아낸다
2. **프로토타입 레이어**가 배운 프로토타입와의 닮음을 계산한다
3. **온통 이은 레이어**가 프로토타입 활성화에 웨이트를 실어 가른다

추론은 이렇다.

$$
\hat{y}_c = \sum_{p \in P_c} w_{cp} \max_{(i,j)} \log\left(\frac{\|z_{(i,j)} - \mathbf{p}_p\|^2 + 1}{\|z_{(i,j)} - \mathbf{p}_p\|^2 + \epsilon}\right)
$$

---

## 2. PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeNetwork(nn.Module):
    """
    풀이되는 본보기 바탕 분류기.
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

        # 본보기 분야로 쏜다
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

        # 본보기에 대한 클래스 짐
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

## 3. 계량 금융 응용

프로토타입 네트워크는 금융에서 다음에 특히 쓸모 있다.

- **신용 위험**: "이 신청자의 피처는 지난날 부도난/부도나지 않은 이런 자리와 닮았다"
- **속임수 찾기**: "이 거래 피처는 알려진 속임수 프로토타입 3번과 맞물린다"
- **판 가르기**: "지금 시장 상황은 2018년 변동성이 치솟은 때와 가장 닮았다"

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

프로토타입 네트워크는 그 분야 전문가가 자연스레 헤아리는 피처와 맞는, 자리에 기초를 둔 헤아림을 준다. 모델이 담을 수 있는 것을 얼마쯤 내주고 해석할 수 있음을 얻으며, 손에 잡히지 않는 피처 기여도가 아니라 손에 잡히는 보기를 들어 해석한다.

**살펴볼 거리**

1. Chen, C., et al. (2019). "This Looks Like That: Deep Learning for Interpretable Image Recognition." *NeurIPS*.

2. Snell, J., et al. (2017). "Prototypical Networks for Few-shot Learning." *NeurIPS*.

3. Li, O., et al. (2018). "Deep Learning for Case-Based Reasoning through Prototypes." *AAAI*.
