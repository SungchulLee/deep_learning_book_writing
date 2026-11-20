# Deep SHAP

**Deep SHAP**은 DeepLIFT의 효율적인 역전파 기반 기여도 분석과 SHAP의 게임 이론 틀을 합쳐 뉴럴 네트워크의 섀플리 값을 근사한다. 배경 분포에서 뽑은 여러 비교점(reference)에 걸쳐 DeepLIFT 기여도를 평균하므로, 계산이 빠르면서도 섀플리 값에 가까운 값을 얻는다.

---

## 1. 이론 기초

### 토대가 되는 DeepLIFT

DeepLIFT(슈리쿠마르 외, 2017)는 뉴런의 활성화를 비교점(reference)의 활성화와 비교해 뉴럴 네트워크의 예측을 해석한다. 입력 $\mathbf{x}$와 비교점 $\mathbf{x}^0$에 대해

$$
\Delta y = f(\mathbf{x}) - f(\mathbf{x}^0)
$$

DeepLIFT는 **합산-델타(summation-to-delta) 성질**을 만족하는 기여 점수 $C_{\Delta x_i \Delta y}$를 계산한다.

$$
\sum_i C_{\Delta x_i \Delta y} = \Delta y
$$

### 재조정 규칙

선형이 아닌 활성화에서 DeepLIFT는 재조정(Rescale) 규칙으로 기여를 전파한다. 그래디언트 대신 출력 차이와 입력 차이의 비를 배수(multiplier)로 쓰는 것이다.

$$
m_{\Delta x \Delta y} = \frac{\Delta y}{\Delta x}
$$

이로써 입력이 분명히 중요한데도 포화된 활성화(양 끝의 시그모이드, 음수 입력의 ReLU)에서 기울기가 0이 되는 **포화 문제**를 피한다.

### DeepLIFT에서 Deep SHAP으로

Deep SHAP은 배경 분포 $D$에서 뽑은 여러 비교점에 걸쳐 평균해 DeepLIFT를 확장한다.

$$
\phi_i(\mathbf{x}) = \mathbb{E}_{\mathbf{x}^0 \sim D}\left[C_{\Delta x_i \Delta y}\right]
$$

이렇게 평균하면 가능한 모든 연합을 살피는 섀플리 값 계산에 다가간다. 핵심은 이렇다. 피처들이 서로 독립이라는 가정 아래에서, 비교점을 주변화하는 일은 피처 연합을 주변화하는 일과 같다.

### 섀플리 값과의 관계

활성화가 조각마다 선형인 네트워크(ReLU)에서는, 비교점 하나를 쓰는 DeepLIFT가 비교점에서 입력으로 가는 경로 근처에서 모델을 **선형화한** 것에 대한 섀플리 값을 계산한다. Deep SHAP은 이를 이렇게 개선한다.

1. 여러 비교점에 걸쳐 평균한다 → 섀플리 값의 기댓값에 더 가까워진다
2. 선형이 아닌 활성화를 다룬다 → 재조정 규칙이 그래디언트가 놓치는 비선형 효과를 담는다

---

## 2. 구현

### PyTorch Deep SHAP

```python
import torch
import torch.nn as nn
import numpy as np

class DeepSHAP:
    """
    DeepLIFT 방식의 역전파를 여러 배경 표본에 걸쳐 평균한
    Deep SHAP.
    """

    def __init__(self, model: nn.Module, background: torch.Tensor):
        """
        Args:
            model: PyTorch 뉴럴 네트워크
            background: 비교 분포로 쓸 배경 표본 [N, 피처 수]
        """
        self.model = model
        self.background = background

        with torch.no_grad():
            self.base_output = model(background).mean(dim=0)

    def _deep_lift_gradient(self, x, baseline, target_class):
        """베이스라인 하나에 대한 DeepLIFT 방식의 기여도를 계산한다."""
        x = x.clone().requires_grad_(True)
        baseline = baseline.clone().requires_grad_(True)

        output_x = self.model(x)
        output_baseline = self.model(baseline)

        diff = output_x[:, target_class] - output_baseline[:, target_class]

        grads_x = torch.autograd.grad(
            diff.sum(), x, create_graph=False
        )[0]

        # 기여도 = 그래디언트 * (입력 - 베이스라인)
        attr = grads_x * (x - baseline)

        return attr

    def explain(
        self,
        instance: torch.Tensor,
        target_class: int = None,
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        Deep SHAP 값을 계산한다.

        Args:
            instance: 입력 텐서 [1, 피처 수]
            target_class: 설명할 대상 클래스
            n_samples: 평균할 배경 표본의 수

        Returns:
            SHAP 값 텐서 [1, 피처 수]
        """
        self.model.eval()

        if target_class is None:
            with torch.no_grad():
                output = self.model(instance)
                target_class = output.argmax(dim=1).item()

        # 배경에서 뽑는다
        idx = torch.randperm(len(self.background))[:n_samples]
        baselines = self.background[idx]

        # 베이스라인에 걸쳐 기여도를 평균한다
        shap_values = torch.zeros_like(instance)

        for baseline in baselines:
            baseline = baseline.unsqueeze(0)
            attr = self._deep_lift_gradient(
                instance, baseline, target_class
            )
            shap_values += attr

        shap_values /= n_samples

        return shap_values

    def verify_completeness(
        self,
        instance: torch.Tensor,
        shap_values: torch.Tensor,
        target_class: int
    ) -> float:
        """
        SHAP 값의 합이 예측 - 기준값이 되는지 확인한다.
        """
        with torch.no_grad():
            prediction = self.model(instance)[0, target_class].item()

        base = self.base_output[target_class].item()
        shap_sum = shap_values.sum().item()
        expected_sum = prediction - base

        error = abs(shap_sum - expected_sum)
        print(f"예측:     {prediction:.4f}")
        print(f"기준값:   {base:.4f}")
        print(f"SHAP 합:  {shap_sum:.4f}")
        print(f"기대 합:  {expected_sum:.4f}")
        print(f"오차:     {error:.6f}")

        return error
```

짚어 둘 점이 있다. 위 `_deep_lift_gradient` 는 이름과 달리 DeepLIFT의 재조정 규칙을 쓰지 않는다. 입력 점의 그래디언트에 (입력 − 베이스라인)을 곱한 「그래디언트 × 입력 차이」라서, 모델이 선형이 아니면 기여도의 합이 출력 차이와 맞지 않고 아래 3절의 포화 문제도 풀지 못한다. `verify_completeness` 가 큰 어긋남을 보인다면 이 때문이다. 재조정 규칙까지 쓰려면 아래의 `shap.DeepExplainer` 나 Captum의 `DeepLiftShap` 을 쓴다.

### SHAP 라이브러리 쓰기

```python
import shap
import torch

def deep_shap_with_library(model, background, test_samples):
    """
    shap 라이브러리의 잘 최적화된 Deep SHAP 구현을 쓴다.
    """
    # DeepExplainer가 Deep SHAP을 구현한 것이다
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)

    return shap_values

def gradient_shap_alternative(model, test_samples, baselines):
    """
    Captum의 GradientSHAP - 뉴럴 네트워크 SHAP의 또 다른 방법.
    """
    from captum.attr import GradientShap

    gradient_shap = GradientShap(model)
    attribution = gradient_shap.attribute(
        test_samples,
        baselines=baselines,
        target=0  # 대상 클래스
    )

    return attribution
```

---

## 3. 포화 문제 다루기

Deep SHAP이 바닐라 그래디언트 방법보다 나은 핵심은 포화된 활성화를 다룬다는 것이다.

입력이 $x = 10$이고 비교점이 $x^0 = 0$인 시그모이드 뉴런을 보자. 아래 표의 값은 배수이고, 기여도는 여기에 입력 차이 10을 곱한 것이다.

| 방법 | 배수 | 읽는 법 |
|--------|------------|-------|
| 그래디언트 | $\sigma'(10) \approx 0.000045$ | 기여도가 약 0.00045로, 출력 차이가 0.5에 가까운데도 거의 0이다 |
| DeepLIFT/Deep SHAP | $(\sigma(10) - \sigma(0)) / (10 - 0) \approx 0.05$ | 기여도가 약 0.5로, 출력 차이를 거의 다 설명한다 |

$x = -10$, $x^0 = 0$인 ReLU에서는

| 방법 | 기여도 | 읽는 법 |
|--------|------------|-------|
| 그래디언트 | $0$(꺼진 ReLU) | 결과는 맞지만, 비교점을 보지 않고 한 점의 기울기만 본 값이다 |
| DeepLIFT/Deep SHAP | $0$ | 비교점과 출력이 같으므로 0을 옳게 매긴다 |

---

## 4. 다른 뉴럴 네트워크 기여도 분석 방법과 비교

| 방법 | 완전성 | 포화 다루기 | 속도 | 베이스라인 여럿 |
|--------|-------------|-------------------|-------|-------------------|
| 바닐라 그래디언트 | 아니오 | 나쁨 | 아주 빠름 | 해당 없음 |
| Integrated Gradients | 예 | 좋음 | 느림(스텝이 많다) | 경로 하나 |
| DeepLIFT | 예 | 좋음 | 빠름 | 비교점 하나 |
| **Deep SHAP** | **예** | **좋음** | **중간** | **예(평균)** |
| Gradient SHAP | 근사 | 중간 | 빠름 | 예 |

---

## 5. 배경 데이터 고르기

배경 분포를 어떻게 고르느냐가 Deep SHAP의 결과를 크게 가른다.

| 데이터 종류 | 권하는 배경 |
|-----------|----------------------|
| 이미지 | 학습 데이터의 부분 표본, 검은 이미지, 또는 가우스 잡음 |
| 표 데이터 | 학습 데이터를 대표하는 부분 표본(100~1000개) |
| 시계열 | 과거의 기준 구간 |
| 텍스트 | 패딩 토큰의 임베딩 |
| 금융 | 시장 중립 상태나 과거 평균 |

**가이드:**

- 추정이 강건하려면 배경 표본을 적어도 100개 쓴다
- 배경은 "정보가 없는" 기준 상태를 나타내야 한다
- 시험 입력의 이웃을 배경으로 쓰지 않는다(정보가 샌다)

---

## 6. 계량 금융 응용

```python
def explain_neural_risk_model(
    risk_model: nn.Module,
    portfolio_features: torch.Tensor,
    background: torch.Tensor,
    feature_names: list
):
    """
    Deep SHAP으로 뉴럴 네트워크 리스크 예측을 설명한다.
    """
    explainer = DeepSHAP(risk_model, background)

    shap_values = explainer.explain(
        portfolio_features.unsqueeze(0),
        target_class=0,  # 리스크 점수
        n_samples=200
    )

    values = shap_values.squeeze().detach().cpu().numpy()

    sorted_idx = np.argsort(np.abs(values))[::-1]

    print("리스크 요인 기여도 분석(Deep SHAP):")
    print("-" * 50)
    for idx in sorted_idx[:10]:
        direction = "↑ 리스크" if values[idx] > 0 else "↓ 리스크"
        print(f"{feature_names[idx]:30s}: {values[idx]:+.6f} ({direction})")

    return shap_values
```

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
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. Deep SHAP은 비교점마다 DeepLIFT의 합산-델타 성질 $\sum_i C_{\Delta x_i \Delta y} = f(\mathbf{x}) - f(\mathbf{x}^0)$ 이 성립하므로, 평균하면 $\sum_i \phi_i = f(\mathbf{x}) - \mathbb{E}[f(\mathbf{x}^0)]$ 가 되어 배경의 평균 출력을 기준으로 완전성을 만족한다. 다만 이 쪽의 구현처럼 재조정 규칙 대신 그래디언트 × 입력 차이를 쓰면 이 성질이 깨진다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. $\square$

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

Deep SHAP은 DeepLIFT의 효율적인 역전파와 SHAP의 섀플리 값 틀을 합쳐 이론에 근거한 뉴럴 네트워크 기여도 분석을 준다. 여러 비교점에 걸쳐 평균해 섀플리 값에 다가가면서, 바닐라 그래디언트 방법을 무너뜨리는 활성화 포화도 다룬다.

**핵심 식:**

$$
\phi_i(\mathbf{x}) = \mathbb{E}_{\mathbf{x}^0 \sim D}\left[C_{\Delta x_i \Delta y}\right]
$$

**참고 문헌**

1. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.

2. Shrikumar, A., et al. (2017). "Learning Important Features Through Propagating Activation Differences." *ICML*.

3. Ancona, M., et al. (2018). "Towards Better Understanding of Gradient-based Attribution Methods for Deep Neural Networks." *ICLR*.

4. Erion, G., et al. (2021). "Improving Performance of Deep Learning Models with Axiomatic Attribution Priors and Expected Gradients." *Nature Machine Intelligence*.
