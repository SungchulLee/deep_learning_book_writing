# 포괄성 평가

**포괄성**은 설명이 중요한 피처를 몇몇이 아니라 모두 담았는지를 잰다. 그 짝인 **충분성**은 방법이 짚은 피처만으로 그 예측을 재현할 수 있는지 따진다.

---

## 1. 자

### 포괄성 점수

앞선 $k$개의 피처를 없애고 추론이 얼마나 바뀌는지 잰다.

$$
\text{두루 갖춤}(E, x) = f(x) - f(x_{\setminus E})
$$

클수록 그 설명이 중요한 피처를 담았다는 뜻이다.

### 충분성 점수

앞선 $k$개의 피처만 남기고 추론이 얼마나 지켜지는지 잰다.

$$
\text{넉넉함}(E, x) = f(x) - f(x_E)
$$

작을수록 방법이 짚은 피처만으로 넉넉하다는 뜻이다.

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

좋은 설명라면 **두루 갖추면서**(포괄성 점수가 크고) **넉넉해야**(충분성 점수가 작아야) 한다. 이 두 자는 서로 채워 준다. 포괄성만 보면 늘 모든 피처를 고르는 꼼수에 넘어갈 수 있다.

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

포괄성과 충분성은 설명이 걸린 피처를 모두 담았는지, 그리고 그 피처만으로 추론을 되살릴 수 있는지를 수로 잰다.

**살펴볼 거리**

1. DeYoung, J., et al. (2020). "ERASER: A Benchmark to Evaluate Rationalized NLP Models." *ACL*.
2. Carton, S., Rathore, A., & Tan, C. (2020). "Evaluating and Characterizing Human Rationales." *EMNLP*.
