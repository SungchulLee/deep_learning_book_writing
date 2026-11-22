# 개념 병목 모델

**개념 병목 모델(Concept Bottleneck Model, CBM)**은 설계 단계부터 해석할 수 있게 만든 모델이다. 네트워크가 먼저 사람이 해석할 수 있는 개념들을 예측하게 하고, 그 개념만으로 최종 예측을 하게 한다. 이렇게 하면 해석 가능한 중간 표현이라는 "병목"이 생긴다.

트레인이 끝난 뒤 해석하는 사후(post-hoc) 방법(SHAP, Grad-CAM)과 달리 CBM은 **본질적으로 해석 가능하다**. 예측마다 명시적인 개념 활성화를 따라 근거를 추적할 수 있다.

---

## 1. 구조

### 일반 CBM

```
입력 x → 개념 예측기 → [c₁, c₂, ..., cₖ] → 태스크 예측기 → y
```

입력 $\mathbf{x}$가 주어지면 모델이 먼저 개념 값을 예측한다.

$$
\hat{c}_i = g_i(\mathbf{x}), \quad i = 1, \ldots, k
$$

그러고 나서 개념만으로 최종 예측을 한다.

$$
\hat{y} = h(\hat{c}_1, \hat{c}_2, \ldots, \hat{c}_k)
$$

### 트레인 목표

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{task}}(\hat{y}, y)}_{\text{태스크 로스}} + \lambda \underbrace{\sum_{i=1}^{k} \mathcal{L}_{\text{concept}}(\hat{c}_i, c_i)}_{\text{개념 로스}}
$$

여기서 $c_i$는 정답 개념 레이블이다.

---

## 2. PyTorch 구현

```python
import torch
import torch.nn as nn

class ConceptBottleneckModel(nn.Module):
    """
    개념 헤드와 태스크 헤드를 따로 둔 개념 병목 모델.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_dim: int,
        n_concepts: int,
        n_classes: int,
        concept_names: list = None
    ):
        super().__init__()
        self.backbone = backbone
        self.concept_names = concept_names or [f'c_{i}' for i in range(n_concepts)]

        # 개념 예측기
        self.concept_head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_concepts),
            nn.Sigmoid()
        )

        # 태스크 예측기(개념만 쓴다)
        self.task_head = nn.Sequential(
            nn.Linear(n_concepts, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x, return_concepts=False):
        features = self.backbone(x)
        concepts = self.concept_head(features)
        output = self.task_head(concepts)

        if return_concepts:
            return output, concepts
        return output

    def explain(self, x):
        """사람이 읽을 수 있는 설명을 만든다."""
        output, concepts = self.forward(x, return_concepts=True)

        concept_values = concepts[0].detach().cpu().numpy()
        prediction = output.argmax(dim=1).item()

        # 태스크 헤드의 웨이트가 개념 → 예측의 관계를 보인다
        task_weights = self.task_head[0].weight.data[prediction].cpu().numpy()
        contributions = concept_values * task_weights

        explanation = []
        sorted_idx = np.argsort(np.abs(contributions))[::-1]
        for idx in sorted_idx:
            explanation.append({
                'concept': self.concept_names[idx],
                'value': concept_values[idx],
                'contribution': contributions[idx]
            })

        return prediction, explanation

    def intervene(self, x, concept_idx, new_value):
        """
        반사실적 개입: 개념 값이 달랐다면 어땠을까?
        이는 CBM만이 지닌 장점이다.
        """
        _, concepts = self.forward(x, return_concepts=True)
        concepts_modified = concepts.clone()
        concepts_modified[0, concept_idx] = new_value
        return self.task_head(concepts_modified)
```

---

## 3. 계량 금융 응용

### 신용 점수 매기기 CBM

```python
# 개념: 높은 부채 비율, 안정적인 소득, 긴 신용 이력,
#       낮은 신용 사용률, 최근 연체 없음, 다양한 신용 구성
concept_names = [
    '높은 부채 비율', '안정적인 소득', '긴 신용 이력',
    '낮은 신용 사용률', '최근 연체 없음', '다양한 신용 구성'
]

model = ConceptBottleneckModel(
    backbone=feature_extractor,
    backbone_dim=512,
    n_concepts=6,
    n_classes=2,
    concept_names=concept_names
)

# 결정을 설명한다
pred, explanation = model.explain(applicant_features)
print(f"결정: {'승인' if pred == 0 else '거절'}")
for item in explanation[:4]:
    print(f"  {item['concept']}: {item['value']:.2f} "
          f"(기여: {item['contribution']:+.3f})")
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 $x \in \{0,1\}^2$의 XOR를 맞히는 CBM을 생각하자. (가) 개념을 $c_1 = x_1$, $c_2 = x_2$ 두 개로 두고 과제 머리 $h$를 선형 함수 하나로 두면 XOR를 맞힐 수 있는가? (나) 개념 $c_3 = [x_1 = x_2]$를 더하면 어떤가? 입력 $x = [1, 1]$에서 각 개념의 기여와, $c_3$을 0으로 개입했을 때의 예측을 구하여라.

</div>

??? success "풀이"
    (가) 맞힐 수 없다. 선형 머리는 $(c_1, c_2)$ 평면을 직선 하나로 가르는데, XOR의 두 클래스 $\{(0,0), (1,1)\}$과 $\{(0,1), (1,0)\}$은 직선 하나로 갈리지 않는다. 개념이 정확히 예측되어도 정확도는 75%를 넘지 못한다. (나) $h(c) = 1 - c_3$이면 네 입력을 모두 맞힌다. $x = [1, 1]$이면 $c = (1, 1, 1)$이고 $\hat{y} = 0$이다. 웨이트가 $(0, 0, -1)$이므로 기여는 $c_1, c_2$가 0, $c_3$이 $-1$이다. 곧 「두 입력이 같다」는 개념 하나가 판단을 다 설명한다. $c_3$을 0으로 개입하면 $\hat{y} = 1$로 뒤집힌다. 이 문제는 정리하며에서 말한 대가를 그대로 보여 준다. 개념 집합에 필요한 개념이 빠지면 병목이 정보를 막아 정확도가 떨어지고, 알맞은 개념을 더하면 정확도와 해석이 함께 돌아온다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. CBM에서는 과제 머리가 선형이면 개념값 × 웨이트의 합에 편향을 더한 것이 로짓과 정확히 같아, 개념을 0으로 둔 베이스라인에 대해 완전성이 성립한다. 과제 머리가 이 쪽의 구현처럼 은닉 레이어를 가진 MLP이면 그 곱은 완전성을 만족하지 않는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 방법이 내놓는 설명이 얼마나 믿을 만한지 평가하는 실험을 설계하라. 방법이 중요하다고 짚은 피처가 실제로 모델에 중요한지를 삽입 곡선과 삭제 곡선으로 재어라.

</div>

??? success "풀이"
    절차는 이렇다. (1) 시험 이미지마다 피처 기여도를 계산한다. (2) 삭제: 기여도가 큰 차례로 피처를 하나씩 가리며 모델의 확신도가 떨어지는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 떨어진다. (3) 삽입: 빈 베이스라인에서 시작해 기여도가 큰 차례로 피처를 하나씩 드러내며 확신도가 오르는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 오른다. (4) 두 곡선 아래 넓이를 계산한다. (5) 무작위 순서(베이스라인)와 다른 방법과 비교한다. 믿을 만한 방법이면 삭제 곡선의 넓이가 작고 삽입 곡선의 넓이가 커야 한다. 통계적으로 믿을 만하려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 설명 방법을 신용 부도를 예측하는 금융 모델에 어떻게 적용할 수 있는지 논하라. 설명이 채워야 할 규제 요건은 무엇인가?

</div>

??? success "풀이"
    신용 모델에는 규제(ECOA, GDPR 22조)가 불리한 결정마다 그 사람에게 맞춘 설명을 요구한다. 방법은 다음을 내놓아야 한다. (1) 거절에 가장 크게 기여한 요인(불리한 처분 사유). (2) 일관된 설명(비슷한 신청자는 비슷한 설명을 받는다). (3) 실행 가능한 설명(신청자가 무엇을 바꾸어야 하는지 안다). 이 절의 설명 방법으로 피처 중요도를 짚을 수 있으나, 강건성(입력이 조금 바뀌었다고 설명이 확 달라지면 안 된다)과 충실성(중요한 피처를 없애면 예측이 바뀌어야 한다)을 평가해야 한다. 보호 속성은 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

개념 병목 모델은 구성 자체로 해석할 수 있게 하며, 설명과 반사실적 개입(intervention)을 함께 준다. 대가는 트레인할 때 개념 레이블이 있어야 하고, 개념 집합이 불완전하면 정확도가 떨어질 수 있다는 것이다.

**살펴볼 거리**

1. Koh, P. W., et al. (2020). "Concept Bottleneck Models." *ICML*.

2. Yuksekgonul, M., et al. (2022). "Post-hoc Concept Bottleneck Models." *ICLR*.
