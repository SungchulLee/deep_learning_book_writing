# 인자 몫 매기기
## 들머리

인자 몫 매기기는 밑천 꾸러미의 돌아옴과 무릅씀을 얼개에 매인 인자의 이바지로 쪼갠다. 풀이 방법, 그 가운데서도 SHAP은 인자에 얼마나 드러났는지가 모형의 미루어 봄을 어떻게 이끄는지 알아보는 엄밀한 틀을 주어, 밑천 꾸러미를 다루는 이가 경제 속뜻을 따지고 숨은 얽힘을 짚어내게 한다.

## 인자에 드러남 알아보기

### 선형 인자 모형 읽기

선형 인자 모형에서는 계수를 그대로 읽을 수 있다.

$$
r_p = \alpha + \sum_{k=1}^{K} \beta_k f_k + \epsilon
$$

여기서 $\beta_k$은 인자 $k$에 드러난 만큼이고 $f_k$은 인자의 돌아옴이다.

### 선형이 아닌 인자 모형

돌아옴을 미루어 보는 데 신경 그물이나 나무 바탕 모형을 쓰면 인자의 이바지가 더는 단순한 계수가 아니다. SHAP 값이 알맞은 쪼갬을 준다.

$$
\hat{r}_p = \phi_0 + \sum_{k=1}^{K} \phi_k
$$

여기서 $\phi_k$은 인자 $k$의 SHAP 값이다.

## 짜보기

### 인자 모형 풀이개

```python
import numpy as np
import shap
import matplotlib.pyplot as plt

class FactorModelExplainer:
    """인자 모형의 미루어 봄을 풀이한다."""

    def __init__(self, model, factor_names):
        self.model = model
        self.factor_names = factor_names

    def explain_return_forecast(self, factor_exposures):
        """미루어 본 돌아옴의 쪼갬을 풀이한다."""
        predicted_return = self.model.predict(
            factor_exposures.reshape(1, -1)
        )[0]

        if hasattr(self.model, 'coef_'):
            # 선형 모형: 그대로 읽는다
            factor_contributions = self.model.coef_ * factor_exposures
            intercept = self.model.intercept_
        else:
            # 선형이 아닌 모형: SHAP을 쓴다
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
        """돌아옴 쪼갬을 폭포 그림으로 만든다."""
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
        ax.set_xlabel('돌아옴에 대한 이바지')
        ax.set_title(f'돌아옴 쪼갬 (모두: {explanation["predicted_return"]:.2%})')
        ax.axvline(x=0, color='black', linestyle='-')

        return fig

    def factor_interaction_analysis(self, factor_data):
        """인자끼리 서로 미치는 힘을 살핀다."""
        explainer = shap.TreeExplainer(self.model)
        interactions = explainer.shap_interaction_values(factor_data[:100])

        mean_interactions = np.abs(interactions).mean(axis=0)

        # 대각선 밖: 서로 미침, 대각선: 홑 힘
        main_effects = np.diag(mean_interactions)

        print("인자의 홑 힘 대 서로 미치는 힘:")
        print("-" * 60)
        for i, name in enumerate(self.factor_names):
            interaction_total = mean_interactions[i].sum() - main_effects[i]
            ratio = interaction_total / (main_effects[i] + 1e-10)
            print(f"{name:20s}: 홑={main_effects[i]:.4f}, "
                  f"서로 미침={interaction_total:.4f}, 비={ratio:.2f}")

        return interactions
```

## 때에 따라 바뀌는 몫

인자의 이바지는 때에 따라 바뀐다. 굴러가는 창에 걸쳐 SHAP 값을 좇으면 판이 바뀌는 것이 드러난다.

```python
def rolling_factor_attribution(model, factor_data, factor_names, window=60):
    """때에 따라 바뀌는 인자 몫을 셈한다."""
    n_periods = len(factor_data) - window
    attributions = np.zeros((n_periods, len(factor_names)))

    explainer = shap.Explainer(model)

    for t in range(n_periods):
        shap_values = explainer(factor_data[t + window:t + window + 1])
        attributions[t] = shap_values.values[0]

    return attributions
```

## 간추림

SHAP 값을 쓰는 인자 몫 매기기는 돌아옴 미루어 봄을 이론에 뿌리내려 쪼개며, 선형 모형과 선형이 아닌 모형에 모두 쓸 수 있다. 때에 따라 살피면 판에 매인 인자의 움직임이 드러난다.

## 살펴볼 거리

1. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
2. Feng, G., Giglio, S., & Xiu, D. (2020). "Taming the Factor Zoo: A Test of New Factors." *Journal of Finance*.

## 익힘 문제

**익힘 1.**
이 마디에서 밝힌 풀이 방법을, XOR 들임을 가르는 ReLU 살림의 두 켜 신경 그물에 걸어라. 들임 $x = [1, 1]$에 대한 풀이를 셈하여라.

??? success "익힘 1 풀이"
    짐이 $W_1, b_1, W_2, b_2$인 익힌 XOR 그물에서 내놓기는 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 풀이 방법은 들임 결마다 몫을 내놓는다. $x = [1, 1]$(갈래 0)이면 두 결 모두 음수 가름에 이바지한다. 몫 값은 방법마다 다르다. 기울기 바탕 방법은 $\partial f / \partial x_i$을 셈하고, 흔들어 보는 방법은 결을 가렸을 때 내놓기가 얼마나 바뀌는지 잰다. XOR 문제는 판단 금이 선형이 아니므로 선형 풀이 방법이 그르칠 수 있음을 보여 준다. $\square$

---

**익힘 2.**
이 마디의 풀이 방법이 온전함 공리를 채우는지, 곧 어떤 밑금 $x_0$에 대해 모든 결 몫의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 뒤집어라.

??? success "익힘 2 풀이"
    온전함 공리(섀플리 값 이론에서는 효율이라고도 한다)는 몫의 합이 들임에서의 모형 내놓기와 밑금에서의 내놓기의 차이와 같다는 것이다. 이 방법이 온전함을 채우는지는 그 세움새에 달렸다. 기울기 방법은 온전함을 채우지 못한다(기울기는 그 자리의 것이고 길을 따라 쌓은 것이 아니다). 쌓은 기울기는 세움새 자체로 온전함을 채운다(길을 따라 미적분의 밑정리를 쓴다). SHAP 값은 섀플리 공리로 효율을 채운다. 온전함을 어기는 방법은 몫을 너무 많거나 적게 매길 수 있어, 온 몫을 온 세상 풀이로 믿기 어렵게 만든다. $\square$

---

**익힘 3.**
이 방법이 내놓는 풀이가 얼마나 미더운지 따지는 시험을 꾸며라. 짚어 준 결이 참으로 모형에 중요한지를 넣기와 빼기 곡선으로 재어라.

??? success "익힘 3 풀이"
    절차는 이렇다. (1) 시험 그림마다 결 몫을 셈한다. (2) 빼기: 몫이 큰 차례로 결을 하나씩 가리며 모형의 자신함이 떨어지는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 떨어진다. (3) 넣기: 빈 밑금에서 시작해 몫이 큰 차례로 결을 하나씩 드러내며 자신함이 오르는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 오른다. (4) 두 곡선의 아래 넓이를 셈한다. (5) 아무렇게나 매긴 차례(밑금)와 다른 방법에 견준다. 미더운 방법이면 빼기 넓이가 작고 넣기 넓이가 커야 한다. 통계로 미더우려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

**익힘 4.**
이 풀이 방법을 신용 부도를 미루어 보는 금융 모형에 어떻게 걸 수 있는지 다루어라. 풀이가 채워야 할 규정 요건은 무엇인가?

??? success "익힘 4 풀이"
    신용 모형에는 규정(ECOA, GDPR 22조)이 불리한 판단마다 그 사람에게 맞춘 풀이를 바란다. 방법은 다음을 내놓아야 한다. (1) 물리침에 가장 크게 이바지한 인자(불리한 처분 까닭). (2) 한결같은 풀이(비슷한 신청자는 비슷한 풀이를 받는다). (3) 손에 잡히는 풀이(신청자가 무엇을 바꾸어야 하는지 안다). 이 마디의 풀이 방법으로 결의 중요함을 짚을 수 있으나, 든든함(들임이 조금 바뀌었다고 풀이가 확 달라지면 안 된다)과 옳음(중요한 결을 없애면 미루어 봄이 바뀌어야 한다)을 따져 보아야 한다. 지켜야 할 됨됨이는 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$
