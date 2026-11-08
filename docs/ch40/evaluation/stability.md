# 안정성 평가

비슷한 입력이 비슷한 풀이를 내놓으면 그 풀이 방법은 **강건하다**. 알아채지도 못할 잡음을 더했을 뿐인데 풀이가 통째로 달라진다면 쓰는 이가 믿을 수 없다. 풀이를 다시 낼 수 있어야 하는 규정된 자리에서는 강건성이 꼭 있어야 한다.

---

## 1. 자

### 입력에 견준 강건성(RIS)

$$
\text{RIS}(x, x') = \frac{\|E(x) - E(x')\|_2}{\|E(x)\|_2 \cdot \|x - x'\|_2}
$$

RIS이 작을수록 더 강건하다.

### 가장 큰 예민함

$$
\text{MaxSens}(x) = \max_{\|\epsilon\| \leq r} \|E(x) - E(x + \epsilon)\|_2
$$

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

| 방법 | 흔한 강건성 | 까닭 |
|--------|------------------|-----|
| 맨 그래디언트 | 낮음 | 구성부터 잡음이 많다 |
| SmoothGrad | 높음 | 고르게 해서 예민함이 준다 |
| Integrated Gradients | 가운데 | 길에 매이고 베이스라인이 중요하다 |
| SHAP | 가운데~높음 | 뽑아 쓰기가 얼마쯤 들쭉날쭉함을 들인다 |
| LIME | 낮음~가운데 | 아무렇게나 뽑고 커널 너비에 예민하다 |
| Grad-CAM | 높음 | 자리를 고르게 해 출력이 매끄럽다 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 절에서 밝힌 풀이 방법을, XOR 입력을 가르는 ReLU 활성화의 두 레이어 뉴럴 네트워크에 걸어라. 입력 $x = [1, 1]$에 대한 풀이를 계산하여라.

</div>

??? success "풀이"
    웨이트가 $W_1, b_1, W_2, b_2$인 학습된 XOR 네트워크에서 출력은 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 풀이 방법은 입력 피처마다 기여도를 내놓는다. $x = [1, 1]$(클래스 0)이면 두 피처 모두 음수 분류에 기여한다. 기여도 값은 방법마다 다르다. 그래디언트 기반 방법은 $\partial f / \partial x_i$을 계산하고, 흔들어 보는 방법은 피처를 가렸을 때 출력이 얼마나 바뀌는지 잰다. XOR 문제는 판단 금이 선형이 아니므로 선형 풀이 방법이 그르칠 수 있음을 보여 준다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 풀이 방법이 완전성 공리를 채우는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 뒤집어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율이라고도 한다)는 기여도의 합이 입력에서의 모델 배포와 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 채우는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 채우지 못한다(그래디언트는 그 자리의 것이고 길을 따라 쌓은 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 채운다(길을 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율을 채운다. 완전성을 어기는 방법은 기여도를 너무 많거나 적게 매길 수 있어, 온 기여도를 온 세상 풀이로 믿기 어렵게 만든다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 방법이 내놓는 풀이가 얼마나 믿을 만한지 평가하는 시험을 꾸며라. 짚어 준 피처가 참으로 모델에 중요한지를 넣기와 빼기 곡선으로 재어라.

</div>

??? success "풀이"
    절차는 이렇다. (1) 시험 그림마다 피처 기여도를 계산한다. (2) 빼기: 기여도가 큰 차례로 피처를 하나씩 가리며 모델의 확신도가 떨어지는 모습을 적는다. 믿을 만한 풀이면 확신도가 빠르게 떨어진다. (3) 넣기: 빈 베이스라인에서 시작해 기여도가 큰 차례로 피처를 하나씩 드러내며 확신도가 오르는 모습을 적는다. 믿을 만한 풀이면 확신도가 빠르게 오른다. (4) 두 곡선의 아래 넓이를 계산한다. (5) 아무렇게나 매긴 차례(베이스라인)와 다른 방법에 견준다. 믿을 만한 방법이면 빼기 넓이가 작고 넣기 넓이가 커야 한다. 통계로 믿을 만하려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 풀이 방법을 신용 부도를 예측하는 금융 모델에 어떻게 걸 수 있는지 다루어라. 풀이가 채워야 할 규정 요건은 무엇인가?

</div>

??? success "풀이"
    신용 모델에는 규정(ECOA, GDPR 22조)이 불리한 판단마다 그 사람에게 맞춘 풀이를 바란다. 방법은 다음을 내놓아야 한다. (1) 물리침에 가장 크게 기여한 인자(불리한 처분 까닭). (2) 한결같은 풀이(비슷한 신청자는 비슷한 풀이를 받는다). (3) 손에 잡히는 풀이(신청자가 무엇을 바꾸어야 하는지 안다). 이 절의 풀이 방법으로 피처의 중요함을 짚을 수 있으나, 강건성(입력이 조금 바뀌었다고 풀이가 확 달라지면 안 된다)과 충실성(중요한 피처를 없애면 추론이 바뀌어야 한다)을 따져 보아야 한다. 지켜야 할 성능은 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

강건성 평가는 입력이 조금 달라져도 풀이가 흔들리지 않는지 살핀다. SmoothGrad과 Grad-CAM 같은 방법은 고르게 하는 피처 덕에 본디부터 강건하다.

**살펴볼 거리**

1. Alvarez-Melis, D., & Jaakkola, T. S. (2018). "On the Robustness of Interpretability Methods." *ICML Workshop*.
2. Yeh, C. K., et al. (2019). "On the (In)fidelity and Sensitivity of Explanations." *NeurIPS*.
