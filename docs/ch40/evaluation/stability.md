# 안정성 평가

비슷한 입력이 비슷한 설명을 내놓으면 그 설명 방법은 **강건하다**. 눈치채기 어려운 작은 잡음을 더했을 뿐인데 설명이 완전히 달라진다면 사용자가 믿을 수 없다. 설명의 재현성이 요구되는 규제 환경에서는 강건성이 필수다.

---

## 1. 지표

### 상대 입력 안정성(RIS)

$$
\text{RIS}(x, x') = \frac{\|E(x) - E(x')\|_2}{\|E(x)\|_2 \cdot \|x - x'\|_2}
$$

여기서 $E(x)$는 입력 $x$에 대한 설명(기여도 벡터)이다. RIS는 입력이 바뀐 거리에 견주어 설명이 상대적으로 얼마나 바뀌었는지를 재며, 작을수록 더 강건하다. 아래 구현은 이 비를 무작위 섭동 여럿에 걸쳐 평균 내고 최댓값도 함께 보고한다.

### 최대 민감도

$$
\text{MaxSens}(x) = \max_{\|\epsilon\| \leq r} \|E(x) - E(x + \epsilon)\|_2
$$

반지름 $r$ 안의 섭동 가운데 설명을 가장 크게 바꾸는 것을 찾아 그 변화량을 잰다. 평균이 아니라 최악의 경우를 보므로, 무작위 잡음으로는 잘 드러나지 않는 불안정성까지 잡는다.

### 구현

```python
import torch
import numpy as np

def compute_stability(
    explanation_fn, input_tensor, n_perturbations=50, noise_level=0.01
):
    """들임을 흔들 때 풀이가 얼마나 강건한지 잰다."""
    base_explanation = explanation_fn(input_tensor)
    base_norm = np.linalg.norm(base_explanation)

    relative_changes = []
    for _ in range(n_perturbations):
        noise = torch.randn_like(input_tensor) * noise_level
        perturbed_explanation = explanation_fn(input_tensor + noise)

        explanation_change = np.linalg.norm(base_explanation - perturbed_explanation)
        input_change = noise.norm().item()

        if base_norm > 0 and input_change > 0:
            relative_changes.append(explanation_change / (base_norm * input_change))

    return {
        'mean_ris': np.mean(relative_changes),
        'max_ris': np.max(relative_changes),
        'std_ris': np.std(relative_changes)
    }
```

---

## 2. 방법마다의 강건성 비교

| 방법 | 일반적인 강건성 | 까닭 |
|--------|------------------|-----|
| 바닐라 그래디언트 | 낮음 | 구성부터 잡음이 많다 |
| SmoothGrad | 높음 | 평균을 내어 민감도가 줄어든다 |
| Integrated Gradients | 중간 | 경로에 의존하고 베이스라인이 중요하다 |
| SHAP | 중간~높음 | 표본 추출이 어느 정도 불안정성을 더한다 |
| LIME | 낮음~중간 | 무작위로 표본을 뽑고 커널 너비에 민감하다 |
| Grad-CAM | 높음 | 공간적으로 평균을 내어 출력이 매끄럽다 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
XOR를 계산하는 ReLU 네트워크 $f(x) = \text{ReLU}(x_1 - x_2) + \text{ReLU}(x_2 - x_1) = |x_1 - x_2|$의 설명을 바닐라 그래디언트 $E(x) = \nabla f(x)$로 준다. $x = [1, 1]$ 바로 옆의 점 $x = [1, 1 - \delta]$($\delta > 0$은 작다)에서 MaxSens와 RIS를 구하고, $\delta \to 0$일 때 어떻게 되는지 보여라.

</div>

??? success "풀이"
    $x_1 > x_2$이면 $\nabla f = (1, -1)$이고 $x_1 < x_2$이면 $(-1, 1)$이다. $x = [1, 1 - \delta]$에서 $E(x) = (1, -1)$인데, 길이가 $\delta/\sqrt{2}$보다 조금 큰 섭동이면 $x_1 < x_2$ 쪽으로 넘어가 $E = (-1, 1)$이 된다. 그래서 $r > \delta/\sqrt{2}$인 한 $\text{MaxSens}(x) = \|(2, -2)\|_2 = 2\sqrt{2}$이다. 그 섭동에서 RIS는 $2\sqrt{2} / (\sqrt{2} \cdot \|x - x'\|_2) \approx 2\sqrt{2}/\delta$이므로, $\delta \to 0$이면 끝없이 커진다. 출력 $f$는 연속이고 $x = [1, 1]$ 근처에서 0에 가깝게 변하는데, 설명은 부호가 통째로 뒤집힌다. 그래디언트가 ReLU의 꺾인 곳에서 불연속이기 때문이다. 같은 자리에서 SmoothGrad는 잡음 위의 그래디언트를 평균 내므로 두 부호가 섞여 $(0, 0)$ 근처의 값을 주고, 입력이 조금 바뀌어도 그 평균은 조금만 움직인다. 위 표에서 바닐라 그래디언트를 「낮음」, SmoothGrad를 「높음」에 둔 까닭이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. 이 쪽의 안정성 지표는 설명 방법이 아니라 설명을 재는 자이므로 완전성과는 따로 본다. 완전성을 만족하는 방법도 불안정할 수 있고(Integrated Gradients는 베이스라인에 민감하다), 완전성을 어기는 방법도 안정적일 수 있다(Grad-CAM). $\square$

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

강건성 평가는 입력이 조금 달라져도 설명이 흔들리지 않는지 살핀다. SmoothGrad와 Grad-CAM 같은 방법은 평균을 내는 구조 덕에 본래부터 강건하다.

**살펴볼 거리**

1. Alvarez-Melis, D., & Jaakkola, T. S. (2018). "On the Robustness of Interpretability Methods." *ICML Workshop*.
2. Yeh, C. K., et al. (2019). "On the (In)fidelity and Sensitivity of Explanations." *NeurIPS*.
