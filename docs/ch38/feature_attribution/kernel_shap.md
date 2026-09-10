# 커널 SHAP

**커널 SHAP**은 짐 실은 선형 되돌이로 SHAP 값에 다가가는, 모형을 가리지 않는 방법이다. 섀플리 값 셈을 남다르게 짐 실은 되돌이 문제로 세워, LIME의 그 자리 대리 모형 길과 SHAP의 놀이 이론 밑바탕을 잇는다. 그래서 어떤 검은 상자 모형에도 쓰면서 섀플리 값의 이론 보장을 함께 얻는다.

---

## 1. 이론 밑바탕

### SHAP에서 되돌이로

SHAP 값은 그 자리 맞음, 없음, 한결같음을 채우는 하나뿐인 더해지는 결 몫으로 뜻매김된다. 룬드베리와 리(2017)는 이것을 짐 실은 최소 제곱 문제를 풀어 셈할 수 있음을 밝혔다.

모형 $f$과 결이 $d$개인 들임 $\mathbf{x}$에 대해, 단순하게 줄인 들임 $z' \in \{0, 1\}^d$을 세운다. $z'_i = 1$이면 결 $i$이 본 값을 지니고, $z'_i = 0$이면 "없다"(가장자리로 밀어냈다)는 뜻이다.

SHAP 값 $\phi_1, \ldots, \phi_d$은 다음을 가장 작게 한다.

$$
\sum_{z' \in \{0,1\}^d} \pi_x(z') \left[ f_x(z') - \left(\phi_0 + \sum_{i=1}^{d} \phi_i z'_i \right) \right]^2
$$

### SHAP 커널

고갱이 깨침은 남다른 짐 주는 커널에 있다.

$$
\pi_x(z') = \frac{d - 1}{\binom{d}{|z'|} \cdot |z'| \cdot (d - |z'|)}
$$

여기서 $|z'| = \sum_i z'_i$은 뭉치의 크기다. 이 커널은 빈 뭉치와 온 뭉치에 끝없이 큰 짐을 주어(양 끝에서 미루어 봄이 꼭 맞게 한다) 작은 뭉치와 큰 뭉치에 큰 짐을 준다. 거기서 가장자리 이바지가 가장 많은 것을 알려 주기 때문이다.

| 뭉치 크기 $|z'|$ | 짐 (d=10) | 읽는 법 |
|----------------------|---------------|----------------|
| 0이나 $d$ | $\infty$ | 꼭 맞는 금 언저리 조건 |
| 1이나 $d-1$ | 큼 | 결 하나의 가장자리 힘 |
| $d/2$ | 작음 | 뭉치가 가장 많고 하나하나로는 알려 주는 것이 적다 |

### 뭉치 값 셈하기

뭉치 $S$($z'_i = 1$인 결의 모임)에 대해 뭉치 값은 이렇다.

$$
f_x(S) = \mathbb{E}[f(X) \mid X_S = x_S]
$$

참으로는 이 기댓값을 바탕 자료 위에서 가장자리로 밀어내어 어림한다.

$$
f_x(S) \approx \frac{1}{N} \sum_{j=1}^{N} f(x_S, x^{(j)}_{\bar{S}})
$$

여기서 $x^{(j)}_{\bar{S}}$은 없는 결에 넣는 바탕 값이다.

---

## 2. 짜보기

### 온전한 커널 SHAP

```python
import numpy as np
from math import comb
from sklearn.linear_model import LinearRegression

class KernelSHAP:
    """
    커널 SHAP - 모형을 가리지 않는 SHAP 어림.

    SHAP 커널로 짐 실은 선형 되돌이를 풀어 섀플리 값에 다가간다.
    """

    def __init__(self, model, background_data):
        """
        Args:
            model: 미루어 보는 함수 (numpy 배열 -> 미루어 봄)
            background_data: 기댓값을 셈할 견줌 자료
        """
        self.model = model
        self.background_data = background_data
        self.base_value = model(background_data).mean()

    def _shap_kernel(self, n_features, coalition_size):
        """주어진 크기의 뭉치에 주는 SHAP 커널 짐."""
        if coalition_size == 0 or coalition_size == n_features:
            return 1e6  # 금 언저리 뭉치에는 큰 짐

        return (n_features - 1) / (
            comb(n_features, coalition_size) * 
            coalition_size * (n_features - coalition_size)
        )

    def _compute_coalition_value(self, instance, coalition, background):
        """
        뭉치에 대한 모형 내놓기의 기댓값을 셈한다.

        뭉치에 든 결은 이 들임의 값을 쓰고,
        들지 않은 결은 바탕 자료로 가장자리로 밀어낸다.
        """
        n_samples = len(background)
        samples = np.tile(instance, (n_samples, 1))

        # 뭉치에 들지 않은 결을 바탕 값으로 갈음한다
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
        들임 하나의 SHAP 값을 셈한다.

        Args:
            instance: 풀이할 들임(1차원 배열)
            num_samples: 뽑을 뭉치의 수

        Returns:
            결마다의 SHAP 값
        """
        n_features = len(instance)

        # 뭉치를 뽑고 값을 셈한다
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

        # 두 값 뭉치 행렬을 만든다
        Z = np.zeros((len(coalitions), n_features))
        for i, coalition in enumerate(coalitions):
            Z[i, list(coalition)] = 1

        # 짐 실은 선형 되돌이
        W = np.diag(np.clip(weights, 0, 1e10))
        y = np.array(coalition_values) - self.base_value
        ZtWZ = Z.T @ W @ Z + 1e-6 * np.eye(n_features)
        ZtWy = Z.T @ W @ y

        shap_values = np.linalg.solve(ZtWZ, ZtWy)

        return shap_values
```

### SHAP 곳집 쓰기

```python
import shap

def kernel_shap_explain(model, X_train, X_test, feature_names):
    """
    shap 곳집으로 하는 서비스 품질의 커널 SHAP.
    """
    # 바탕 자료로 풀이개를 만든다
    explainer = shap.KernelExplainer(
        model.predict, 
        shap.sample(X_train, 100)  # 잘 들게 바탕 자료를 솎아 낸다
    )

    # SHAP 값을 셈한다
    shap_values = explainer.shap_values(X_test[:100])

    # 간추린 그림
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)

    # 들임 하나의 힘 그림
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        X_test[0],
        feature_names=feature_names
    )

    return shap_values
```

---

## 3. 셈에서 헤아릴 것

### 복잡도

꼭 맞는 섀플리 값을 얻으려면 $2^d$개의 뭉치를 모두 따져야 한다. 커널 SHAP은 뭉치를 뽑아 이를 어림하며 복잡도는 이렇다.

$$
O(N_{\text{뽑음}} \cdot N_{\text{바탕}} \cdot C_{\text{모형}})
$$

여기서 $C_{\text{모형}}$은 모형을 한 번 돌리는 값이다.

### 들쭉날쭉함 줄이기

커널 SHAP은 뽑아 쓰므로 들쭉날쭉함이 들어온다. 줄이는 꾀는 이렇다.

| 꾀 | 미치는 힘 | 값 |
|----------|--------|------|
| 뭉치를 더 뽑기 | 덜 들쭉날쭉해진다 | 모형을 더 돌린다 |
| 짝지어 뽑기 | 일차 들쭉날쭉함이 지워진다 | 돌리는 횟수는 같다 |
| 층으로 나눠 뽑기 | 뭉치 크기를 고루 덮는다 | 짐이 조금 붙는다 |
| 바탕 모둠 키우기 | 가장자리 어림이 나아진다 | 기억을 더 쓴다 |

### 커널 SHAP 대 다른 것

| 자리 | 이르는 말 |
|----------|---------------|
| 나무 바탕 모형 | 나무 SHAP을 쓴다(꼭 맞고 빠르다) |
| 신경 그물 | 깊은 SHAP이나 기울기 SHAP을 쓴다 |
| 아무 검은 상자 모형 | 커널 SHAP |
| 결이 적음($d < 15$) | 커널 SHAP이 잘 듣는다 |
| 결이 많음($d > 100$) | 어림 방법을 헤아린다 |

---

## 4. 금융에 쓰기

```python
def explain_portfolio_allocation(
    allocation_model,
    market_features,
    feature_names,
    background_data
):
    """
    커널 SHAP으로 밑천 나누기 판단을 풀이한다.
    """
    explainer = shap.KernelExplainer(
        allocation_model.predict, 
        background_data
    )

    shap_values = explainer.shap_values(market_features)

    print("밑천 나누기 풀이:")
    print("-" * 50)

    sorted_idx = np.argsort(np.abs(shap_values[0]))[::-1]
    for idx in sorted_idx[:10]:
        direction = "↑ 더 담음" if shap_values[0][idx] > 0 else "↓ 덜 담음"
        print(f"{feature_names[idx]:30s}: {shap_values[0][idx]:+.4f} ({direction})")

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

커널 SHAP은 남다르게 꾸민 커널로 짐 실은 선형 되돌이를 풀어, 모형을 가리지 않고 섀플리 값을 어림한다. 어떤 검은 상자 모형에도 쓰면서 섀플리 값의 이론 됨됨이를 모두 물려받는다.

**고갱이 식:**

$$
\pi_x(z') = \frac{d - 1}{\binom{d}{|z'|} \cdot |z'| \cdot (d - |z'|)}
$$

**살펴볼 거리**

1. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.

2. Covert, I., & Lee, S. I. (2021). "Improving KernelSHAP: Practical Shapley Value Estimation Using Linear Regression." *AISTATS*.

3. Shapley, L. S. (1953). "A Value for n-Person Games." *Contributions to the Theory of Games*.
