# 깊은 SHAP

**깊은 SHAP**은 DeepLIFT의 잘 드는 되짚기 바탕 몫 매기기와 SHAP의 놀이 이론 틀을 아울러, 신경 그물의 섀플리 값을 어림해 셈한다. 바탕 분포에서 뽑은 여러 견줌 자리에 걸쳐 DeepLIFT 몫을 고르게 하므로, 셈이 잘 들면서도 섀플리 값의 이론 보장을 물려받는다.

---

## 1. 이론 밑바탕

### 벽돌이 되는 DeepLIFT

DeepLIFT(슈리쿠마르 외, 2017)은 신경 세포의 살아남을 그 "견줌" 살아남과 맞대어 신경 그물의 미루어 봄을 풀이한다. 들임 $\mathbf{x}$과 견줌 $\mathbf{x}^0$에 대해

$$
\Delta y = f(\mathbf{x}) - f(\mathbf{x}^0)
$$

DeepLIFT은 **델타로 더해짐 됨됨이**를 채우는 이바지 점수 $C_{\Delta x_i \Delta y}$을 셈한다.

$$
\sum_i C_{\Delta x_i \Delta y} = \Delta y
$$

### 잣대 다시 잡기 규칙

선형이 아닌 살림에는 DeepLIFT이 잣대 다시 잡기 규칙으로 이바지를 퍼뜨린다.

$$
m_{\Delta x \Delta y} = \frac{\Delta y}{\Delta x}
$$

이로써 들임이 또렷이 중요한데도 잦아든 살아남(끝자락의 시그모이드, 음수 들임의 ReLU)에서 기울기가 0이 되는 **잦아듦 문제**를 피한다.

### DeepLIFT에서 깊은 SHAP으로

깊은 SHAP은 바탕 분포 $D$에서 뽑은 여러 견줌에 걸쳐 고르게 해 DeepLIFT을 넓힌다.

$$
\phi_i(\mathbf{x}) = \mathbb{E}_{\mathbf{x}^0 \sim D}\left[C_{\Delta x_i \Delta y}\right]
$$

이렇게 고르게 하면 있을 수 있는 모든 뭉치를 살피는 섀플리 값 셈에 다가간다. 고갱이 깨침은 이렇다. 서로 매이지 않는다는 여김 아래에서, 견줌을 가장자리로 밀어내는 일은 결 뭉치를 가장자리로 밀어내는 일과 같다.

### 섀플리 값과의 이어짐

살림이 조각마다 선형인 그물(ReLU)에서는, 견줌 하나를 쓰는 DeepLIFT이 견줌에서 들임으로 가는 길 언저리에서 모형을 **선형으로 편** 것에 대한 꼭 맞는 섀플리 값을 셈한다. 깊은 SHAP은 이를 이렇게 낫게 한다.

1. 여러 견줌에 걸쳐 고르게 한다 → 온전한 섀플리 기댓값에 더 가까워진다
2. 선형이 아닌 살림을 다룬다 → 잣대 다시 잡기 규칙이 기울기가 놓치는 선형 아닌 힘을 담는다

---

## 2. 짜보기

### PyTorch 깊은 SHAP

```python
import torch
import torch.nn as nn
import numpy as np

class DeepSHAP:
    """
    DeepLIFT 결의 되짚기를 여러 바탕 표본에 걸쳐 고르게 한
    깊은 SHAP.
    """

    def __init__(self, model: nn.Module, background: torch.Tensor):
        """
        Args:
            model: PyTorch 신경 그물
            background: 견줌 분포로 쓸 바탕 표본 [N, 결]
        """
        self.model = model
        self.background = background

        with torch.no_grad():
            self.base_output = model(background).mean(dim=0)

    def _deep_lift_gradient(self, x, baseline, target_class):
        """밑금 하나에 대한 DeepLIFT 결의 몫을 셈한다."""
        x = x.clone().requires_grad_(True)
        baseline = baseline.clone().requires_grad_(True)

        output_x = self.model(x)
        output_baseline = self.model(baseline)

        diff = output_x[:, target_class] - output_baseline[:, target_class]

        grads_x = torch.autograd.grad(
            diff.sum(), x, create_graph=False
        )[0]

        # 몫 = 기울기 * (들임 - 밑금)
        attr = grads_x * (x - baseline)

        return attr

    def explain(
        self,
        instance: torch.Tensor,
        target_class: int = None,
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        깊은 SHAP 값을 셈한다.

        Args:
            instance: 들임 텐서 [1, 결]
            target_class: 풀이할 겨눈 갈래
            n_samples: 고르게 할 바탕 표본의 수

        Returns:
            SHAP 값 텐서 [1, 결]
        """
        self.model.eval()

        if target_class is None:
            with torch.no_grad():
                output = self.model(instance)
                target_class = output.argmax(dim=1).item()

        # 바탕에서 뽑는다
        idx = torch.randperm(len(self.background))[:n_samples]
        baselines = self.background[idx]

        # 밑금에 걸쳐 몫을 고르게 한다
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
        SHAP 값의 합이 미루어 봄 - 밑값이 되는지 살핀다.
        """
        with torch.no_grad():
            prediction = self.model(instance)[0, target_class].item()

        base = self.base_output[target_class].item()
        shap_sum = shap_values.sum().item()
        expected_sum = prediction - base

        error = abs(shap_sum - expected_sum)
        print(f"미루어 봄: {prediction:.4f}")
        print(f"밑값:     {base:.4f}")
        print(f"SHAP 합:  {shap_sum:.4f}")
        print(f"바란 값:  {expected_sum:.4f}")
        print(f"어긋남:   {error:.6f}")

        return error
```

### SHAP 곳집 쓰기

```python
import shap
import torch

def deep_shap_with_library(model, background, test_samples):
    """
    shap 곳집의 잘 다듬은 깊은 SHAP 짜보기를 쓴다.
    """
    # DeepExplainer이 깊은 SHAP을 짜 넣은 것이다
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)

    return shap_values

def gradient_shap_alternative(model, test_samples, baselines):
    """
    Captum의 GradientSHAP - 신경 그물 SHAP의 또 다른 길.
    """
    from captum.attr import GradientShap

    gradient_shap = GradientShap(model)
    attribution = gradient_shap.attribute(
        test_samples,
        baselines=baselines,
        target=0  # 겨눈 갈래
    )

    return attribution
```

---

## 3. 잦아듦 문제 다루기

깊은 SHAP이 맨 기울기 방법보다 나은 고갱이는 잦아든 살아남을 다룬다는 것이다.

들임이 $x = 10$이고 견줌이 $x^0 = 0$인 시그모이드 신경 세포를 보자.

| 방법 | 몫 | 탈 |
|--------|------------|-------|
| 기울기 | $\sigma'(10) \approx 0.000045$ | 살아남 차이가 큰데도 거의 0이다 |
| DeepLIFT/깊은 SHAP | $(\sigma(10) - \sigma(0)) / (10 - 0) \approx 0.05$ | 뜻있는 이바지를 담는다 |

$x = -10$, $x^0 = 0$인 ReLU에서는

| 방법 | 몫 | 탈 |
|--------|------------|-------|
| 기울기 | $0$(깨어 있지 않은 ReLU) | 깨어 있지 않다는 것도 알려 주는 바가 있음을 놓친다 |
| DeepLIFT/깊은 SHAP | $0$ | 0을 옳게 매긴다(견줌과 내놓기가 같다) |

---

## 4. 다른 신경 그물 몫 매기기 방법과 견주기

| 방법 | 온전함 | 잦아듦 다루기 | 빠르기 | 밑금 여럿 |
|--------|-------------|-------------------|-------|-------------------|
| 맨 기울기 | 아니오 | 나쁨 | 아주 빠름 | 해당 없음 |
| 쌓은 기울기 | 예 | 좋음 | 느림(걸음이 많다) | 길 하나 |
| DeepLIFT | 예 | 좋음 | 빠름 | 견줌 하나 |
| **깊은 SHAP** | **예** | **좋음** | **가운데** | **예(고르게 함)** |
| 기울기 SHAP | 어림 | 가운데 | 빠름 | 예 |

---

## 5. 바탕 고르기

바탕 분포가 깊은 SHAP의 열매를 크게 가른다.

| 자료 갈래 | 권하는 바탕 |
|-----------|----------------------|
| 그림 | 익힘 자료를 솎아 낸 것, 0 그림, 또는 가우스 잡음 |
| 표 자료 | 익힘 자료를 대표하게 솎아 낸 것(100~1000개) |
| 때 열 | 지난날의 밑금이 되는 때 |
| 글 | 채움 낱말의 쏘아 넣기 |
| 금융 | 저자에 치우치지 않은 상태나 지난날의 평균 |

**길잡이:**

- 어림이 든든하려면 바탕 표본을 적어도 100개 쓴다
- 바탕은 "아무것도 모르는" 밑금 상태를 나타내야 한다
- 시험 들임의 이웃을 바탕으로 쓰지 않는다(소식이 샌다)

---

## 6. 계량 금융에 쓰기

```python
def explain_neural_risk_model(
    risk_model: nn.Module,
    portfolio_features: torch.Tensor,
    background: torch.Tensor,
    feature_names: list
):
    """
    깊은 SHAP으로 신경 그물 무릅씀 미루어 봄을 풀이한다.
    """
    explainer = DeepSHAP(risk_model, background)

    shap_values = explainer.explain(
        portfolio_features.unsqueeze(0),
        target_class=0,  # 무릅씀 점수
        n_samples=200
    )

    values = shap_values.squeeze().detach().cpu().numpy()

    sorted_idx = np.argsort(np.abs(values))[::-1]

    print("무릅씀 인자 몫 매기기(깊은 SHAP):")
    print("-" * 50)
    for idx in sorted_idx[:10]:
        direction = "↑ 무릅씀" if values[idx] > 0 else "↓ 무릅씀"
        print(f"{feature_names[idx]:30s}: {values[idx]:+.6f} ({direction})")

    return shap_values
```

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

깊은 SHAP은 DeepLIFT의 잘 드는 되짚기와 SHAP의 섀플리 값 틀을 아울러 이론에 뿌리내린 신경 그물 몫 매기기를 준다. 여러 바탕 견줌에 걸쳐 고르게 함으로써 참 섀플리 값에 다가가면서, 맨 기울기 방법을 무너뜨리는 살아남 잦아듦도 다룬다.

**고갱이 식:**

$$
\phi_i(\mathbf{x}) = \mathbb{E}_{\mathbf{x}^0 \sim D}\left[C_{\Delta x_i \Delta y}\right]
$$

**살펴볼 거리**

1. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.

2. Shrikumar, A., et al. (2017). "Learning Important Features Through Propagating Activation Differences." *ICML*.

3. Ancona, M., et al. (2018). "Towards Better Understanding of Gradient-based Attribution Methods for Deep Neural Networks." *ICLR*.

4. Erion, G., et al. (2021). "Improving Performance of Deep Learning Models with Axiomatic Attribution Priors and Expected Gradients." *Nature Machine Intelligence*.
