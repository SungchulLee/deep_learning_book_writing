# 요인 기여도 분석

인자(요인, factor) 기여도 분석은 포트폴리오의 수익률과 위험을 시장·가치·규모·모멘텀 같은 체계적 인자(systematic factor)의 기여로 분해한다. 설명 방법, 그 가운데서도 SHAP은 인자 노출(factor exposure), 곧 포트폴리오가 각 인자에 얼마나 민감한지가 모델의 예측을 어떻게 이끄는지 알아보는 엄밀한 틀을 주어, 포트폴리오 매니저가 경제적 의미를 평가하고 숨은 상관을 탐지하게 한다.

---

## 1. 인자 노출 이해하기

### 선형 인자 모델 읽기

선형 인자 모델에서는 계수를 그대로 읽을 수 있다.

$$
r_p = \alpha + \sum_{k=1}^{K} \beta_k f_k + \epsilon
$$

여기서 $\beta_k$는 인자 $k$에 대한 노출이고 $f_k$는 인자의 수익률이다.

### 선형이 아닌 인자 모델

수익률을 예측하는 데 뉴럴 네트워크나 트리 기반 모델을 쓰면 인자의 기여가 더는 단순한 계수가 아니다. 이때는 SHAP 값이 그에 해당하는 분해를 준다.

$$
\hat{r}_p = \phi_0 + \sum_{k=1}^{K} \phi_k
$$

여기서 $\phi_k$는 인자 $k$의 SHAP 값이고, $\phi_0$은 배경 데이터에 대한 예측의 평균이다. $\phi_0$은 선형 모델의 절편 $\alpha$와 같은 양이 아니다. 선형 모델에 SHAP을 쓰면 피처가 독립이라는 가정 아래 $\phi_k = \beta_k (x_k - \mathbb{E}[x_k])$가 되어, 계수 × 값을 그대로 읽은 것과 평균만큼 차이가 난다. 아래 구현은 선형 모델이면 계수 × 입력을, 아니면 SHAP 값을 쓰므로 두 갈래의 기여도는 기준점이 다르다는 점을 염두에 두고 읽는다. 또 구현에서 모델의 입력은 자산마다의 인자 노출이고, 학습된 계수는 위 식의 $\beta_k$가 아니라 노출 한 단위당 수익률, 곧 $f_k$ 쪽에 해당한다.

---

## 2. 구현

### 인자 모델 설명기

```python
import numpy as np
import shap
import matplotlib.pyplot as plt

class FactorModelExplainer:
    """인자 모델의 예측을 설명한다."""

    def __init__(self, model, factor_names):
        self.model = model
        self.factor_names = factor_names

    def explain_return_forecast(self, factor_exposures):
        """예측 수익률의 분해를 설명한다."""
        predicted_return = self.model.predict(
            factor_exposures.reshape(1, -1)
        )[0]

        if hasattr(self.model, 'coef_'):
            # 선형 모델: 계수를 그대로 읽는다
            factor_contributions = self.model.coef_ * factor_exposures
            intercept = self.model.intercept_
        else:
            # 비선형 모델: SHAP을 쓴다
            explainer = shap.Explainer(self.model)
            shap_values = explainer(factor_exposures.reshape(1, -1))
            factor_contributions = shap_values.values[0]
            intercept = shap_values.base_values[0]

        return {
            'predicted_return': predicted_return,
            'alpha': intercept,
            'factor_contributions': dict(
                zip(self.factor_names, factor_contributions)
            )
        }

    def visualize_decomposition(self, explanation):
        """수익률 분해를 워터폴 차트로 만든다."""
        factors = list(explanation['factor_contributions'].keys())
        contributions = list(explanation['factor_contributions'].values())

        sorted_idx = np.argsort(np.abs(contributions))[::-1]

        fig, ax = plt.subplots(figsize=(12, 6))

        cumsum = explanation['alpha']
        positions = []

        for i, idx in enumerate(sorted_idx):
            contrib = contributions[idx]
            left = cumsum if contrib > 0 else cumsum + contrib
            width = abs(contrib)
            color = 'green' if contrib > 0 else 'red'

            ax.barh(i, width, left=left, color=color, alpha=0.7)
            ax.text(left + width/2, i, f'{contrib:.2%}', 
                   ha='center', va='center')

            cumsum += contrib
            positions.append(factors[idx])

        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels(positions)
        ax.set_xlabel('수익률에 대한 기여')
        ax.set_title(f'수익률 분해 (합계: {explanation["predicted_return"]:.2%})')
        ax.axvline(x=0, color='black', linestyle='-')

        return fig

    def factor_interaction_analysis(self, factor_data):
        """인자 사이의 상호작용을 분석한다."""
        explainer = shap.TreeExplainer(self.model)
        interactions = explainer.shap_interaction_values(factor_data[:100])

        mean_interactions = np.abs(interactions).mean(axis=0)

        # 대각선 밖: 상호작용, 대각선: 주효과
        main_effects = np.diag(mean_interactions)

        print("인자의 주효과 대 상호작용:")
        print("-" * 60)
        for i, name in enumerate(self.factor_names):
            interaction_total = mean_interactions[i].sum() - main_effects[i]
            ratio = interaction_total / (main_effects[i] + 1e-10)
            print(f"{name:20s}: 주효과={main_effects[i]:.4f}, "
                  f"상호작용={interaction_total:.4f}, 비={ratio:.2f}")

        return interactions
```

---

## 3. 시간에 따라 바뀌는 기여도

인자의 기여는 시간에 따라 바뀐다. 시점마다 SHAP 값을 계산해 시계열로 이어 보면 레짐 전환이 드러난다. 아래 구현은 모델과 설명기를 하나로 고정하고 첫 `window`개 시점 뒤의 각 시점을 설명한다. 윈도마다 모델을 다시 학습하는 롤링 분석은 아니므로, 바뀐 것이 모델의 판단인지 입력인지 가르려면 윈도마다 다시 학습해 비교해야 한다.

```python
def rolling_factor_attribution(model, factor_data, factor_names, window=60):
    """시간에 따라 바뀌는 인자 기여도를 계산한다."""
    n_periods = len(factor_data) - window
    attributions = np.zeros((n_periods, len(factor_names)))

    explainer = shap.Explainer(model)

    for t in range(n_periods):
        shap_values = explainer(factor_data[t + window:t + window + 1])
        attributions[t] = shap_values.values[0]

    return attributions
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
두 인자 노출 $x_1, x_2 \in \{0, 1\}$을 받아 수익률을 예측하는 두 모델이 있다. 선형 모델은 $\hat{r} = 0.02 x_1 + 0.01 x_2$이고, 비선형 모델은 $\hat{r} = 0.02\,|x_1 - x_2|$(두 노출이 엇갈릴 때만 초과 수익)이다. 배경 데이터가 네 조합을 고르게 담는다고 할 때, 노출 $x = [1, 1]$인 자산에 대해 (가) 선형 모델의 「계수 × 노출」 분해, (나) 선형 모델의 SHAP 값, (다) 비선형 모델의 SHAP 값을 구하여라.

</div>

??? success "풀이"
    (가) 기여는 $(0.02, 0.01)$이고 합 $0.03$이 예측 그 자체다. (나) $\mathbb{E}[x_k] = 0.5$이므로 $\phi = (0.02 \times 0.5,\ 0.01 \times 0.5) = (0.01, 0.005)$이고, 합 $0.015$는 예측 $0.03$에서 배경 평균 $\phi_0 = 0.015$를 뺀 값이다. 같은 모델인데 기준점이 달라 수가 절반이 되었다. (다) 연합 값은 $f_x(\varnothing) = 0.01$, $f_x(\{1\}) = f_x(\{2\}) = 0.01$, $f_x(\{1,2\}) = 0$이므로 $\phi_1 = \phi_2 = \tfrac12[(0.01 - 0.01) + (0 - 0.01)] = -0.005$다. 두 인자 모두 수익률을 깎는 쪽으로 기여한 것처럼 보이지만, 실제로는 어느 한쪽 노출만 있었다면 $0.02$를 벌었을 것이다. 비선형 인자 모델에서는 SHAP 값만이 아니라 앞의 상호작용 분석을 함께 보아야 하는 까닭이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. 이 쪽의 두 갈래를 보면, SHAP 값은 배경 평균을 베이스라인으로 완전성을 만족하고, 선형 모델의 「계수 × 노출」은 노출이 모두 0인 입력을 베이스라인으로 두면 완전성을 만족한다($\hat{r}(x) - \hat{r}(0) = \sum_k \beta_k x_k$). 둘 다 완전성을 만족하지만 베이스라인이 달라 같은 모델에서도 수가 다르다. $\square$

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

SHAP 값을 쓰는 인자 기여도 분석은 수익률 예측을 이론적 근거 위에서 분해하며, 선형 모델과 비선형 모델에 모두 쓸 수 있다. 시간에 따라 추적하면 레짐에 따라 달라지는 인자의 움직임이 드러난다.

**살펴볼 거리**

1. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
2. Feng, G., Giglio, S., & Xiu, D. (2020). "Taming the Factor Zoo: A Test of New Factors." *Journal of Finance*.
