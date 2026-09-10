# 나무 SHAP

**나무 SHAP**은 나무 바탕 모형(판단 나무, 아무 숲, 기울기 북돋운 나무)에서 섀플리 값을 꼭 맞게, 다항식 때에 셈하는 알고리즘이다. 커널 SHAP은 꼭 맞게 셈하려면 $O(TL2^M)$번을 따져야 하는 데 견주어, 나무 SHAP은 판단 나무의 되돌이 얼개를 살려 $O(TLD^2)$ 때에 꼭 맞는 섀플리 값을 셈한다. 여기서 $T$은 나무의 수, $L$은 잎의 가장 큰 수, $D$은 가장 큰 깊이다.

이렇게 잘 들기에 XGBoost, LightGBM, CatBoost, scikit-learn 나무 모형을 풀이할 때 나무 SHAP을 고른다.

---

## 1. 이론 밑바탕

### 나무의 섀플리 값

미루어 보는 함수가 $f(\mathbf{x}) = \sum_{t=1}^{T} f_t(\mathbf{x})$인 나무 모둠에서 결 $i$의 SHAP 값은 나무마다로 쪼개진다.

$$
\phi_i(\mathbf{x}) = \sum_{t=1}^{T} \phi_i^{(t)}(\mathbf{x})
$$

판단 나무 하나에서 뭉치 값 $f_x(S)$($S$에 든 결만 알 때의 미루어 봄)은 나무 얼개를 따라가며 셈할 수 있다.

- 결 $j$으로 쪼개는 속마디마다:
  - $j \in S$이면 $x_j$에 따라 알맞은 가지를 따라간다
  - $j \notin S$이면 **두 가지 모두**를 따라가되, 마디마다의 익힘 표본 비율로 짐을 싣는다

### 나무 SHAP 알고리즘

고갱이 깨침은 뿌리에서 잎까지의 길마다 쓰인 결의 모임을 좇으며, 결마다의 짐 실은 이바지를 잘 들게 셈하는 데 있다.

값이 $v_p$인 뿌리에서 잎까지의 길 $p$마다

1. 길 $p$에서 판단에 쓰인 결의 모임을 $D_p = \{j_1, j_2, \ldots, j_k\}$이라 하자
2. 결 $j_i \in D_p$마다 모든 차례를 헤아려 가장자리 이바지를 셈한다
3. 이 이바지는 마디마다의 익힘 자료 몫을 셈에 넣는다

### 복잡도

| 방법 | 복잡도 | 꼭 맞는가? |
|--------|-----------|--------|
| 마구잡이 섀플리 | $O(2^d \cdot T \cdot D)$ | 예 |
| 커널 SHAP | $O(N_{\text{뽑음}} \cdot N_{\text{바탕}})$ | 아니오 |
| **나무 SHAP** | $O(T \cdot L \cdot D^2)$ | **예** |

나무 500그루, 깊이 6, 잎 64개인 흔한 XGBoost 모형이면 나무 SHAP이 표본마다 밀리초 안에 꼭 맞는 값을 셈한다.

---

## 2. 짜보기

### SHAP 곳집 쓰기

권하는 길은 `shap` 곳집의 잘 다듬은 C++ 짜보기를 쓰는 것이다.

```python
import shap
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification

def tree_shap_example():
    """XGBoost으로 하는 온전한 나무 SHAP 보기."""

    # 보기 자료를 만든다
    X, y = make_classification(
        n_samples=1000, n_features=20,
        n_informative=10, random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(20)]

    # XGBoost 모형을 익힌다
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, random_state=42
    )
    model.fit(X[:800], y[:800])

    # 나무 SHAP 풀이개 - 꼭 맞고 빠르다
    explainer = shap.TreeExplainer(model)

    # 시험 자료의 SHAP 값을 셈한다
    shap_values = explainer.shap_values(X[800:])

    # 밑값(바라는 미루어 봄)
    print(f"밑값: {explainer.expected_value:.4f}")

    # 첫 표본의 온전함을 살핀다
    prediction = model.predict_proba(X[800:801])[0, 1]
    from scipy.special import expit
    shap_sum = shap_values[0].sum() + explainer.expected_value
    print(f"미루어 봄(낌새):  {prediction:.4f}")
    print(f"SHAP으로 되세움: {expit(shap_sum):.4f}")

    return shap_values, explainer

def tree_shap_for_lightgbm(model, X_train, X_test, feature_names):
    """LightGBM 모형의 나무 SHAP."""
    import lightgbm as lgb

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 갈래가 여럿이면 shap_values이 갈래마다의 목록이다
    if isinstance(shap_values, list):
        print(f"갈래의 수: {len(shap_values)}")
        for c, sv in enumerate(shap_values):
            importance = np.abs(sv).mean(axis=0)
            top_idx = np.argsort(importance)[::-1][:5]
            print(f"\n갈래 {c}의 앞선 결:")
            for idx in top_idx:
                print(f"  {feature_names[idx]}: {importance[idx]:.4f}")

    return shap_values
```

### SHAP 서로 미침 값

나무 SHAP만이 지닌 힘은 **꼭 맞는 서로 미침 값**, 곧 미루어 봄을 홑 힘과 짝지은 서로 미침으로 쪼개는 이차 섀플리 값을 셈하는 것이다.

$$
f(\mathbf{x}) = \phi_0 + \sum_i \phi_{ii}(\mathbf{x}) + \sum_{i < j} \phi_{ij}(\mathbf{x})
$$

여기서 $\phi_{ii}$은 결 $i$의 홑 힘이고 $\phi_{ij}$은 결 $i$과 $j$ 사이의 서로 미침을 담는다.

```python
def compute_tree_shap_interactions(model, X_test, feature_names):
    """
    나무 모형의 SHAP 서로 미침 값을 셈한다.

    interaction_values[i, j, k] = 표본 i에서 결 j와 k 사이의
    서로 미침.
    """
    explainer = shap.TreeExplainer(model)

    # 서로 미침 값 - 홑 힘보다 비싸다
    interaction_values = explainer.shap_interaction_values(X_test[:100])

    # 절댓값 서로 미침의 고른 값
    mean_interactions = np.abs(interaction_values).mean(axis=0)

    # 앞선 서로 미침
    n_features = len(feature_names)
    interactions = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions.append({
                'feature_i': feature_names[i],
                'feature_j': feature_names[j],
                'strength': mean_interactions[i, j]
            })

    interactions.sort(key=lambda x: x['strength'], reverse=True)

    print("앞선 결 서로 미침:")
    print("-" * 50)
    for inter in interactions[:10]:
        print(f"{inter['feature_i']:20s} × {inter['feature_j']:20s}: "
              f"{inter['strength']:.4f}")

    return interaction_values
```

---

## 3. 그림으로 보이기

### 간추린 그림

```python
def tree_shap_visualization(shap_values, X_test, feature_names):
    """나무 모형에 쓰는 여느 SHAP 그림."""

    # 벌떼 간추림 - 결마다 SHAP 값의 퍼짐을 보인다
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        plot_type='dot'
    )

    # 막대 그림 - 절댓값 SHAP 값의 고른 값
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        plot_type='bar'
    )

    # 매임 그림 - SHAP 값 대 결 값
    # 서로 미치는 결을 저절로 찾아 준다
    top_feature_idx = np.abs(shap_values).mean(axis=0).argmax()
    shap.dependence_plot(
        top_feature_idx, shap_values, X_test,
        feature_names=feature_names
    )
```

### 폭포 그림

```python
def explain_single_prediction(explainer, X_instance, feature_names):
    """미루어 봄 하나를 폭포 그림으로 촘촘히 풀이한다."""

    shap_values = explainer.shap_values(X_instance.reshape(1, -1))

    # 폭포 그림
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_instance,
            feature_names=feature_names
        )
    )
```

---

## 4. 본 대로 대 손대어 본 나무 SHAP

나무 SHAP에는 두 결이 있다.

### 본 대로(길에 매인 것)

나무 얼개가 뜻하는 매인 분포 $P(X_{\bar{S}} \mid X_S)$을 쓴다. 서로 얽힌 결이 몫을 나누어 받는다.

```python
explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
```

### 손대어 본

본 결과 상관없이 가장자리 분포 $P(X_{\bar{S}})$을 쓴다. 인과 이바지를 더 잘 드러낸다.

```python
explainer = shap.TreeExplainer(model, X_background, feature_perturbation='interventional')
```

| 결 | 나은 점 | 못한 점 |
|------|------|------|
| 본 대로 | 빠르고 바탕 자료가 없어도 된다 | 얽힌 결에 몫을 나눠 준다 |
| 손대어 본 | 인과로 읽을 수 있다 | 바탕 자료가 있어야 하고 느리다 |

---

## 5. 계량 금융에 쓰기

### 나무 모형으로 하는 신용 점수 매기기

```python
def explain_credit_tree_model(
    model,  # XGBoost/LightGBM 신용 점수 모형
    applicant_features: np.ndarray,
    feature_names: list,
    X_train: np.ndarray
):
    """
    신용 판단에 대한 규정에 맞는 풀이를 만든다.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(applicant_features.reshape(1, -1))

    # 두 갈래 가름이면 shap_values이 목록일 수 있다
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 맡긴 갈래

    values = shap_values[0]

    # 불리한 처분 까닭(규정이 바라는 것)
    negative_factors = []
    for idx in np.argsort(values)[::-1]:
        if values[idx] > 0:  # 부도 무릅씀을 올린다
            negative_factors.append({
                'factor': feature_names[idx],
                'impact': values[idx],
                'value': applicant_features[idx]
            })

    print("불리한 처분 까닭:")
    for i, factor in enumerate(negative_factors[:4], 1):
        print(f"  {i}. {factor['factor']}: 값={factor['value']:.2f}, "
              f"미침={factor['impact']:+.4f}")

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

나무 SHAP은 나무 바탕 모형의 섀플리 값을 다항식 때에 꼭 맞게 주므로, 서비스에서 기울기 북돋운 모형을 풀이하는 으뜸 잣대다.

**고갱이 됨됨이:**

- **꼭 맞음**: 어림 어긋남이 없다(커널 SHAP과 다르다)
- **빠름**: 표본마다 $O(TLD^2)$
- **서로 미침**: 짝지은 서로 미침 값을 꼭 맞게 셈할 수 있다
- **두 결**: 본 대로(빠름)와 손대어 본(인과)

**살펴볼 거리**

1. Lundberg, S. M., et al. (2020). "From Local Explanations to Global Understanding with Explainable AI for Trees." *Nature Machine Intelligence*.

2. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.

3. Lundberg, S. M., et al. (2018). "Consistent Individualized Feature Attribution for Tree Ensembles." *arXiv:1802.03888*.
