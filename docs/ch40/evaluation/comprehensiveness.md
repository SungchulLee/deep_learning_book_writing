# 포괄성 평가

**포괄성**은 설명이 중요한 피처를 몇몇이 아니라 모두 담았는지를 잰다. 그 짝인 **충분성**은 방법이 짚은 피처만으로 그 예측을 재현할 수 있는지 따진다.

---

## 1. 지표

### 포괄성 점수

기여도가 큰 상위 $k$개의 피처를 없애고 예측이 얼마나 바뀌는지 잰다. 아래 코드에서 「없앤다」는 그 피처 값을 0으로 바꾸는 것이고, $k$는 전체 피처 가운데 비율로 주며, $f$는 목표 클래스의 소프트맥스 확률이다. $x_{\setminus E}$는 상위 피처를 없앤 입력, $x_E$는 상위 피처만 남긴 입력이다.

$$
\text{두루 갖춤}(E, x) = f(x) - f(x_{\setminus E})
$$

클수록 그 설명이 중요한 피처를 담았다는 뜻이다.

### 충분성 점수

상위 $k$개의 피처만 남기고 예측이 얼마나 유지되는지 잰다.

$$
\text{넉넉함}(E, x) = f(x) - f(x_E)
$$

작을수록 방법이 짚은 피처만으로 충분하다는 뜻이다.

### 구현

```python
import torch
import numpy as np

def comprehensiveness_sufficiency(
    model, input_tensor, attribution, target_class,
    k_values=[0.1, 0.2, 0.3, 0.5]
):
    """임계를 달리하며 두루 갖춤과 넉넉함을 셈한다."""
    model.eval()

    with torch.no_grad():
        base_score = torch.softmax(model(input_tensor), dim=1)[0, target_class].item()

    attr_flat = attribution.flatten()
    sorted_idx = np.argsort(np.abs(attr_flat))[::-1]
    n_features = len(attr_flat)

    results = {}
    for k in k_values:
        n_top = int(k * n_features)
        top_indices = sorted_idx[:n_top]

        # 두루 갖춤: 앞선 k개를 없앤다
        removed = input_tensor.clone().flatten()
        removed[top_indices] = 0
        with torch.no_grad():
            removed_score = torch.softmax(model(removed.reshape(input_tensor.shape)), dim=1)[0, target_class].item()

        # 넉넉함: 앞선 k개만 남긴다
        kept = torch.zeros_like(input_tensor).flatten()
        kept[top_indices] = input_tensor.flatten()[top_indices]
        with torch.no_grad():
            kept_score = torch.softmax(model(kept.reshape(input_tensor.shape)), dim=1)[0, target_class].item()

        results[k] = {
            'comprehensiveness': base_score - removed_score,
            'sufficiency': base_score - kept_score
        }

    return results
```

---

## 2. 읽는 법

좋은 설명이라면 **포괄적이면서**(포괄성 점수가 크고) **충분해야**(충분성 점수가 작아야) 한다. 이 두 지표는 서로 보완한다. 포괄성만 보면 늘 모든 피처를 고르는 꼼수에 넘어갈 수 있다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
XOR 문제에서 클래스 0의 점수가 $f(x) = 1 - |x_1 - x_2|$인 ReLU 네트워크가 있다($|x_1 - x_2| = \text{ReLU}(x_1 - x_2) + \text{ReLU}(x_2 - x_1)$). 입력 $x = [1, 1]$에 대해 어떤 설명이 $x_1$을 가장 중요한 피처로 짚었다. 피처를 없애는 것은 0으로 바꾸는 것이라 할 때, 상위 1개($k = 50\%$)와 상위 2개($k = 100\%$)에서 포괄성 점수와 충분성 점수를 구하여라.

</div>

??? success "풀이"
    $f([1, 1]) = 1$이다. 상위 1개이면 $x_{\setminus E} = [0, 1]$이고 $f = 0$이므로 포괄성은 $1 - 0 = 1$이다. $x_E = [1, 0]$이고 $f = 0$이므로 충분성은 $1 - 0 = 1$로 나쁘다. 상위 2개이면 $x_{\setminus E} = [0, 0]$이고 $f = 1$이므로 포괄성이 0으로 떨어지고, $x_E = [1, 1]$이라 충분성은 0이다. 피처를 더 많이 없앴는데 포괄성이 1에서 0으로 줄었다. 두 피처가 함께 있어야 의미가 있는 상호작용이 있고, 0으로 채우는 것이 우연히 같은 클래스의 입력 $[0, 0]$을 만들기 때문이다. 그래서 포괄성은 $k$에 대해 단조롭지 않을 수 있고, 한 $k$에서 잰 값만으로 설명을 판정하면 안 된다. 「없앤다」를 무엇으로 채우느냐에 따라 결론이 바뀐다는 점도 함께 보고해야 한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. 포괄성과 충분성은 설명 방법이 아니라 설명을 재는 지표이므로 그 자체가 완전성을 만족하는지 물을 대상은 아니다. 다만 모든 피처를 없앤 입력을 베이스라인 $x_0$으로 보면 $k = 100\%$의 포괄성은 $f(x) - f(x_0)$이고, 완전성을 만족하는 방법이면 기여도의 합이 정확히 이 값과 같다. $\square$

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

포괄성과 충분성은 설명이 관련된 피처를 모두 담았는지, 그리고 그 피처만으로 예측을 재현할 수 있는지를 정량적으로 잰다.

**살펴볼 거리**

1. DeYoung, J., et al. (2020). "ERASER: A Benchmark to Evaluate Rationalized NLP Models." *ACL*.
2. Carton, S., Rathore, A., & Tan, C. (2020). "Evaluating and Characterizing Human Rationales." *EMNLP*.
