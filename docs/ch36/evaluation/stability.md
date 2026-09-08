# 든든함 따지기

비슷한 들임이 비슷한 풀이를 내놓으면 그 풀이 방법은 **든든하다**. 알아채지도 못할 잡음을 더했을 뿐인데 풀이가 통째로 달라진다면 쓰는 이가 믿을 수 없다. 풀이를 다시 낼 수 있어야 하는 규정된 자리에서는 든든함이 꼭 있어야 한다.

---

## 1. 자

### 들임에 견준 든든함(RIS)

$$
\text{RIS}(x, x') = \frac{\|E(x) - E(x')\|_2}{\|E(x)\|_2 \cdot \|x - x'\|_2}
$$

RIS이 작을수록 더 든든하다.

### 가장 큰 예민함

$$
\text{MaxSens}(x) = \max_{\|\epsilon\| \leq r} \|E(x) - E(x + \epsilon)\|_2
$$

### 짜보기

```python
import torch
import numpy as np

def compute_stability(
    explanation_fn, input_tensor, n_perturbations=50, noise_level=0.01
):
    """들임을 흔들 때 풀이가 얼마나 든든한지 잰다."""
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

## 2. 방법마다의 든든함 견주기

| 방법 | 흔한 든든함 | 까닭 |
|--------|------------------|-----|
| 맨 기울기 | 낮음 | 세움새부터 잡음이 많다 |
| SmoothGrad | 높음 | 고르게 해서 예민함이 준다 |
| 쌓은 기울기 | 가운데 | 길에 매이고 밑금이 중요하다 |
| SHAP | 가운데~높음 | 뽑아 쓰기가 얼마쯤 들쭉날쭉함을 들인다 |
| LIME | 낮음~가운데 | 아무렇게나 뽑고 커널 너비에 예민하다 |
| Grad-CAM | 높음 | 자리를 고르게 해 내놓기가 매끄럽다 |

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

든든함 따지기는 들임이 조금 달라져도 풀이가 흔들리지 않는지 살핀다. SmoothGrad과 Grad-CAM 같은 방법은 고르게 하는 결 덕에 본디부터 든든하다.

**살펴볼 거리**

1. Alvarez-Melis, D., & Jaakkola, T. S. (2018). "On the Robustness of Interpretability Methods." *ICML Workshop*.
2. Yeh, C. K., et al. (2019). "On the (In)fidelity and Sensitivity of Explanations." *NeurIPS*.
