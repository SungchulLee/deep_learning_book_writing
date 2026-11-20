# 개념 활성화 벡터(CAV)

**개념 활성화 벡터(CAV)**는 뉴럴 네트워크가 사람이 이해할 수 있는 어떤 개념을 배웠는지 평가하는 길을 준다. 예측을 원시 입력 피처(픽셀, 단어 임베딩)로 해석하는 대신, CAV는 "줄무늬 있음", "털 있음", "높은 변동성", "평균 회귀" 같은 고수준 개념으로 해석한다.

Kim 등(2018)이 제안한 CAV는 뉴럴 네트워크가 계산하는 것(고차원 공간의 활성화)과 사람이 알아듣는 것(개념) 사이의 간극을 메운다.

---

## 1. 수학 기초

### 활성화 공간에서의 개념 방향

학습된 뉴럴 네트워크에서 입력 집합에 대한 레이어 $l$의 활성화를 생각하자. 레이어 $l$에서 개념 $k$의 CAV는 활성화 공간에서 그 개념의 방향을 가리키는 벡터 $v_l^k$이다.

이 방향을 찾으려면

1. 개념 $k$를 드러내는 **양성 예시** $P_k$를 모은다
2. **음성 예시** $N_k$(무작위 입력 또는 그 개념이 없는 것)를 모은다
3. 레이어 $l$의 활성화 $h_l(x)$에 선형 분류기를 학습한다

$$
v_l^k = \arg\min_v \sum_{x \in P_k} \ell(\sigma(v^\top h_l(x)), 1) + \sum_{x \in N_k} \ell(\sigma(v^\top h_l(x)), 0)
$$

이 선형 분류기의 결정 경계에 수직인 법선 벡터가 곧 CAV $v_l^k$이다. 구현에서는 길이를 1로 맞추어 방향만 남긴다.

### 개념에 대한 민감도

레이어 $l$에서 클래스 $c$가 개념 $k$에 대해 지니는 **개념 민감도**는 이렇다.

$$
S_{c,k,l}(x) = \nabla_{h_l(x)} f_c(x) \cdot v_l^k
$$

곧 활성화를 개념 방향으로 조금 움직일 때 클래스 점수가 얼마나 바뀌는지를 재는 방향 미분이다. 양수면 개념이 강해질수록 클래스 점수가 오르고, 음수면 내려간다.

---

## 2. PyTorch 구현

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression

class ConceptActivationVector:
    """개념 활성화 벡터(CAV)를 계산하고 쓴다."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def get_activations(self, inputs: torch.Tensor) -> np.ndarray:
        """대상 레이어의 활성값을 뽑아낸다."""
        self.model.eval()
        with torch.no_grad():
            self.model(inputs)

        act = self.activations
        if act.dim() > 2:
            act = act.mean(dim=tuple(range(2, act.dim())))
        return act.cpu().numpy()

    def train_cav(
        self,
        concept_examples: torch.Tensor,
        random_examples: torch.Tensor
    ) -> np.ndarray:
        """
        선형 분류기를 학습해 CAV를 얻는다.

        Returns:
            cav_vector: 결정 경계의 법선 벡터
        """
        pos_act = self.get_activations(concept_examples)
        neg_act = self.get_activations(random_examples)

        X = np.vstack([pos_act, neg_act])
        y = np.array([1] * len(pos_act) + [0] * len(neg_act))

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)

        cav = clf.coef_[0]
        cav = cav / np.linalg.norm(cav)

        accuracy = clf.score(X, y)
        print(f"CAV 분류기 정확도: {accuracy:.3f}")

        return cav

    def conceptual_sensitivity(
        self,
        input_tensor: torch.Tensor,
        cav: np.ndarray,
        target_class: int
    ) -> float:
        """대상 클래스가 개념 방향에 얼마나 민감한지 계산한다."""
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(False)

        self.model(input_tensor)
        act = self.activations
        act.requires_grad_(True)

        output = self.model.fc(act.mean(dim=(2, 3)) if act.dim() > 2 else act)
        output[0, target_class].backward()

        grad = act.grad
        if grad.dim() > 2:
            grad = grad.mean(dim=tuple(range(2, grad.dim())))

        grad_np = grad[0].cpu().numpy()
        cav_tensor = cav

        sensitivity = np.dot(grad_np, cav_tensor)
        return sensitivity
```

---

## 3. 계량 금융 응용

CAV로 금융 모델이 의미 있는 경제 개념을 배웠는지 평가할 수 있다.

| 개념 | 양성 예시 | 용도 |
|---------|-------------------|----------|
| "고변동성 국면" | VIX > 25인 시기 | 위험 모델 평가 |
| "평균 회귀" | 이동 평균으로 돌아오는 자산 | 전략 해석 |
| "모멘텀" | 최근 크게 오른 자산 | 인자 모델 살피기 |
| "신용 경색" | 신용 스프레드가 크게 벌어진 시기 | 신용 위험 모델 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
레이어 $l$의 활성화가 2차원 $h = (h_1, h_2)$이고, 개념 예시의 활성화는 $(1, 1)$ 근처에, 무작위 예시의 활성화는 $(0, 0)$ 근처에 모인다고 하자. 그 위의 클래스 점수가 $f_c(h) = 2h_1 - h_2$일 때 CAV $v_l^k$와 개념 민감도 $S_{c,k,l}$을 계산하여라. 점수가 $f_c(h) = h_1 - 2h_2$이면 어떻게 되는가?

</div>

??? success "풀이"
    두 집단이 각자의 중심 둘레에 같은 모양으로 고르게 퍼져 있으면, 둘을 가르는 선형 분류기의 경계는 두 중심을 잇는 선분에 수직이 되므로, 길이를 1로 맞춘 법선은 $v_l^k = (1, 1)/\sqrt{2}$이다. $\nabla_h f_c = (2, -1)$이므로 $S_{c,k,l} = (2 - 1)/\sqrt{2} \approx 0.707 > 0$, 곧 개념을 더할수록 클래스 점수가 오른다. 점수가 $h_1 - 2h_2$이면 $\nabla_h f_c = (1, -2)$이고 $S_{c,k,l} = -1/\sqrt{2} < 0$이다. 두 모델 모두 $h_1$을 양의 웨이트로 쓰지만, 개념 방향은 두 축을 섞은 것이라 민감도의 부호는 그래디언트 전체와 CAV의 내적이 정한다. 점수가 선형이 아니면 그래디언트가 입력마다 달라지므로, 민감도의 부호도 입력마다 달라질 수 있다. 다음 쪽의 TCAV가 그 부호를 여러 입력에 걸쳐 센다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. CAV의 개념 민감도는 입력 피처마다 기여도를 나누는 방법이 아니라 한 점에서의 방향 미분이므로 완전성을 만족하지 않으며, 애초에 그 공리가 적용되는 대상이 아니다. $\square$

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

CAV는 사람이 이해할 수 있는 개념에 대응하는 방향을 활성화 공간에서 찾아 개념 수준의 설명을 준다. 뉴럴 네트워크의 내부 표현과 도메인 전문가의 앎 사이를 잇는다.

**살펴볼 거리**

1. Kim, B., et al. (2018). "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors." *ICML*.

2. Ghorbani, A., et al. (2019). "Towards Automatic Concept-based Explanations." *NeurIPS*.
