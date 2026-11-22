# Kernel SHAP

**커널 SHAP**은 가중 선형 회귀로 SHAP 값을 근사하는, 모델에 구애받지 않는(model-agnostic) 방법이다. 섀플리 값 계산을 특별한 웨이트를 준 회귀 문제로 바꾸어, LIME의 국소 대리 모델 접근과 SHAP의 게임 이론 기반을 잇는다. 그래서 어떤 블랙박스 모델에도 쓰면서 섀플리 값의 이론 보장을 함께 얻는다.

---

## 1. 이론 기초

### SHAP에서 회귀로

SHAP 값은 국소 정확도(local accuracy), 결측성(missingness), 일관성(consistency)을 만족하는 유일한 가산적(additive) 피처 기여도로 정의된다. 룬드베리와 리(2017)는 이것을 가중 최소 제곱 문제를 풀어 계산할 수 있음을 밝혔다.

모델 $f$와 피처가 $d$개인 입력 $\mathbf{x}$에 대해, 단순화한 입력 $z' \in \{0, 1\}^d$를 세운다. $z'_i = 1$이면 피처 $i$가 원래 값을 지니고, $z'_i = 0$이면 "없다"(주변화했다)는 뜻이다.

SHAP 값 $\phi_1, \ldots, \phi_d$는 다음을 최소화한다.

$$
\sum_{z' \in \{0,1\}^d} \pi_x(z') \left[ f_x(z') - \left(\phi_0 + \sum_{i=1}^{d} \phi_i z'_i \right) \right]^2
$$

### SHAP 커널

핵심은 특별한 웨이트를 주는 커널에 있다.

$$
\pi_x(z') = \frac{d - 1}{\binom{d}{|z'|} \cdot |z'| \cdot (d - |z'|)}
$$

여기서 $|z'| = \sum_i z'_i$는 연합의 크기다. 연합(coalition)은 "있다"고 둔 피처의 모임이다. 이 커널은 빈 연합과 전체 연합에 무한대의 웨이트를 주어 양 끝에서 식이 정확히 맞게 하고, 그다음으로 아주 작은 연합과 아주 큰 연합에 큰 웨이트를 준다. 그런 연합이 피처 하나의 한계 기여(marginal contribution)를 가장 많이 알려 주기 때문이다.

| 연합 크기 $|z'|$ | 웨이트 (d=10) | 읽는 법 |
|----------------------|---------------|----------------|
| 0이나 $d$ | $\infty$ | 정확히 맞춰야 하는 경계 조건 |
| 1이나 $d-1$ | 큼 | 피처 하나의 한계 기여 |
| $d/2$ | 작음 | 연합이 가장 많고 하나하나로는 알려 주는 것이 적다 |

### 연합 값 계산

연합 $S$($z'_i = 1$인 피처의 모임)에 대해 연합 값은 이렇다.

$$
f_x(S) = \mathbb{E}[f(X) \mid X_S = x_S]
$$

실제로는 이 기댓값을 배경(background) 데이터로 주변화해 추정한다. 아래 근사는 조건부 기댓값을 주변 기댓값으로 바꾸는 것이라 연합 밖의 피처가 연합 안의 피처와 독립이라고 가정한다.

$$
f_x(S) \approx \frac{1}{N} \sum_{j=1}^{N} f(x_S, x^{(j)}_{\bar{S}})
$$

여기서 $x^{(j)}_{\bar{S}}$는 연합에 없는 피처에 넣는 배경 데이터의 값이다.

---

## 2. 구현

### 완전한 커널 SHAP

```python
import numpy as np
from math import comb
from sklearn.linear_model import LinearRegression

class KernelSHAP:
    """
    커널 SHAP - 모델에 구애받지 않는(model-agnostic) SHAP 근사.

    SHAP 커널로 가중 선형 회귀를 풀어 섀플리 값을 근사한다.
    """

    def __init__(self, model, background_data):
        """
        Args:
            model: 예측 함수 (numpy 배열 -> 예측값)
            background_data: 기댓값을 계산할 배경 데이터
        """
        self.model = model
        self.background_data = background_data
        self.base_value = model(background_data).mean()

    def _shap_kernel(self, n_features, coalition_size):
        """주어진 크기의 연합에 주는 SHAP 커널 웨이트."""
        if coalition_size == 0 or coalition_size == n_features:
            return 1e6  # 경계의 연합(빈 연합·전체 연합)에는 큰 웨이트

        return (n_features - 1) / (
            comb(n_features, coalition_size) * 
            coalition_size * (n_features - coalition_size)
        )

    def _compute_coalition_value(self, instance, coalition, background):
        """
        연합에 대한 모델 출력의 기댓값을 계산한다.

        연합에 든 피처는 이 입력의 값을 쓰고,
        들지 않은 피처는 배경 데이터로 주변화한다.
        """
        n_samples = len(background)
        samples = np.tile(instance, (n_samples, 1))

        # 연합에 들지 않은 피처를 배경 값으로 바꾼다
        mask = np.ones(len(instance), dtype=bool)
        mask[list(coalition)] = False
        samples[:, mask] = background[:, mask]

        return self.model(samples).mean()

    def explain(
        self,
        instance: np.ndarray,
        num_samples: int = 2048
    ) -> np.ndarray:
        """
        입력 하나의 SHAP 값을 계산한다.

        Args:
            instance: 설명할 입력(1차원 배열)
            num_samples: 뽑을 연합의 수

        Returns:
            피처마다의 SHAP 값
        """
        n_features = len(instance)

        # 연합을 뽑고 값을 계산한다
        coalitions = []
        coalition_values = []
        weights = []

        for _ in range(num_samples):
            size = np.random.randint(0, n_features + 1)
            coalition = tuple(sorted(
                np.random.choice(n_features, size, replace=False)
            ))

            if coalition not in coalitions:
                coalitions.append(coalition)
                value = self._compute_coalition_value(
                    instance, coalition, self.background_data
                )
                coalition_values.append(value)
                weights.append(self._shap_kernel(n_features, len(coalition)))

        # 이진 연합 행렬을 만든다
        Z = np.zeros((len(coalitions), n_features))
        for i, coalition in enumerate(coalitions):
            Z[i, list(coalition)] = 1

        # 가중 선형 회귀
        W = np.diag(np.clip(weights, 0, 1e10))
        y = np.array(coalition_values) - self.base_value
        ZtWZ = Z.T @ W @ Z + 1e-6 * np.eye(n_features)
        ZtWy = Z.T @ W @ y

        shap_values = np.linalg.solve(ZtWZ, ZtWy)

        return shap_values
```

### SHAP 라이브러리 쓰기

```python
import shap

def kernel_shap_explain(model, X_train, X_test, feature_names):
    """
    shap 라이브러리로 하는 실무용 커널 SHAP.
    """
    # 배경 데이터로 설명기(explainer)를 만든다
    explainer = shap.KernelExplainer(
        model.predict, 
        shap.sample(X_train, 100)  # 효율을 위해 배경 데이터를 샘플링한다
    )

    # SHAP 값을 계산한다
    shap_values = explainer.shap_values(X_test[:100])

    # 요약 플롯(summary plot)
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)

    # 입력 하나의 포스 플롯(force plot)
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        X_test[0],
        feature_names=feature_names
    )

    return shap_values
```

---

## 3. 계산상 고려할 점

### 복잡도

정확한 섀플리 값을 얻으려면 $2^d$개의 연합을 모두 따져야 한다. 커널 SHAP은 연합을 표본으로 뽑아 이를 추정하며 복잡도는 이렇다.

$$
O(N_{\text{표본}} \cdot N_{\text{배경}} \cdot C_{\text{모델}})
$$

여기서 $C_{\text{모델}}$은 모델을 한 번 돌리는 비용이다.

### 분산 줄이기

커널 SHAP은 연합을 표본으로 뽑으므로 추정값에 분산이 생긴다. 줄이는 전략은 이렇다.

| 전략 | 효과 | 비용 |
|----------|--------|------|
| 연합을 더 샘플링 | 분산이 줄어든다 | 모델을 더 돌린다 |
| 짝지어 샘플링(연합과 그 여집합을 함께 뽑기) | 분산이 크게 준다 | 돌리는 횟수는 같다 |
| 층화 샘플링 | 연합 크기를 고루 덮는다 | 웨이트를 다시 맞춰야 한다 |
| 배경 데이터 늘리기 | 주변화 추정이 나아진다 | 계산과 메모리가 는다 |

### 커널 SHAP 대 다른 방법

| 상황 | 권장 |
|----------|---------------|
| 트리 기반 모델 | Tree SHAP을 쓴다(정확하고 빠르다) |
| 뉴럴 네트워크 | Deep SHAP이나 Gradient SHAP을 쓴다 |
| 아무 블랙박스 모델 | 커널 SHAP |
| 피처가 적음($d < 15$) | 커널 SHAP이 잘 작동한다 |
| 피처가 많음($d > 100$) | 다른 근사 방법을 고려한다 |

---

## 4. 금융 응용

```python
def explain_portfolio_allocation(
    allocation_model,
    market_features,
    feature_names,
    background_data
):
    """
    커널 SHAP으로 포트폴리오 배분 결정을 설명한다.
    """
    explainer = shap.KernelExplainer(
        allocation_model.predict, 
        background_data
    )

    shap_values = explainer.shap_values(market_features)

    print("포트폴리오 배분 설명:")
    print("-" * 50)

    sorted_idx = np.argsort(np.abs(shap_values[0]))[::-1]
    for idx in sorted_idx[:10]:
        direction = "↑ 비중 확대" if shap_values[0][idx] > 0 else "↓ 비중 축소"
        print(f"{feature_names[idx]:30s}: {shap_values[0][idx]:+.4f} ({direction})")

    return shap_values
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 절에서 다룬 설명 방법을, XOR 입력을 분류하는 ReLU 활성화의 두 레이어 뉴럴 네트워크에 적용하라. 입력 $x = [1, 1]$에 대한 설명을 계산하여라.

</div>

??? success "풀이"
    웨이트가 $W_1, b_1, W_2, b_2$인 트레인된 XOR 네트워크에서 출력은 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 설명 방법은 입력 피처마다 기여도를 내놓는다. $x = [1, 1]$(클래스 0)이면 두 피처 모두 음수 쪽 분류에 기여한다. 기여도 값은 방법마다 다르다. 그래디언트 기반 방법은 $\partial f / \partial x_i$을 계산하고, 섭동 기반 방법은 피처를 가렸을 때 출력이 얼마나 바뀌는지 잰다. XOR 문제는 결정 경계가 선형이 아니므로 선형 설명 방법이 틀릴 수 있음을 보여 준다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 커널 SHAP은 빈 연합과 전체 연합에 아주 큰 웨이트를 주어(구현에서는 `1e6`) $\phi_0 + \sum_i \phi_i \approx f(x)$ 를 사실상 강제하므로, 배경 데이터의 평균 출력 $\phi_0$ 을 기준으로 보면 완전성을 근사적으로 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. $\square$

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

커널 SHAP은 특별히 설계한 커널로 가중 선형 회귀를 풀어, 모델을 가리지 않고 섀플리 값을 추정한다. 어떤 블랙박스 모델에도 쓸 수 있고, 연합을 충분히 뽑으면 섀플리 값으로 수렴하므로 그 이론적 성질을 물려받는다.

**핵심 식:**

$$
\pi_x(z') = \frac{d - 1}{\binom{d}{|z'|} \cdot |z'| \cdot (d - |z'|)}
$$

**참고 문헌**

1. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.

2. Covert, I., & Lee, S. I. (2021). "Improving KernelSHAP: Practical Shapley Value Estimation Using Linear Regression." *AISTATS*.

3. Shapley, L. S. (1953). "A Value for n-Person Games." *Contributions to the Theory of Games*.
