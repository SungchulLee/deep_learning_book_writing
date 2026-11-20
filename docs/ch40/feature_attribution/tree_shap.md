# Tree SHAP

**Tree SHAP**은 트리 기반 모델(결정 트리, 아무 랜덤 포레스트, 그래디언트 부스팅 트리)에서 섀플리 값을 꼭 맞게, 다항식 때에 계산하는 알고리즘이다. 커널 SHAP은 꼭 맞게 계산하려면 $O(TL2^M)$번을 따져야 하는 데 비교하여, Tree SHAP은 결정 트리의 재귀 구조를 살려 $O(TLD^2)$ 때에 꼭 맞는 섀플리 값을 계산한다. 여기서 $T$은 트리의 수, $L$은 리프의 가장 큰 수, $D$은 가장 큰 깊이다.

이렇게 잘 들기에 XGBoost, LightGBM, CatBoost, scikit-learn 트리 모델을 해석할 때 Tree SHAP을 고른다.

---

## 1. 이론 기초

### 트리의 섀플리 값

예측하는 함수가 $f(\mathbf{x}) = \sum_{t=1}^{T} f_t(\mathbf{x})$인 트리 앙상블에서 피처 $i$의 SHAP 값은 트리마다로 쪼개진다.

$$
\phi_i(\mathbf{x}) = \sum_{t=1}^{T} \phi_i^{(t)}(\mathbf{x})
$$

결정 트리 하나에서 뭉치 값 $f_x(S)$($S$에 든 피처만 알 때의 추론)은 트리 구조를 따라가며 계산할 수 있다.

- 피처 $j$으로 쪼개는 속마디마다:
  - $j \in S$이면 $x_j$에 따라 알맞은 가지를 따라간다
  - $j \notin S$이면 **두 가지 모두**를 따라가되, 마디마다의 학습 표본 비율로 웨이트를 싣는다

### Tree SHAP 알고리즘

핵심 통찰은 뿌리에서 리프까지의 길마다 쓰인 피처의 모임을 좇으며, 피처마다의 웨이트 실은 기여를 효율적으로 계산하는 데 있다.

값이 $v_p$인 뿌리에서 리프까지의 길 $p$마다

1. 길 $p$에서 판단에 쓰인 피처의 모임을 $D_p = \{j_1, j_2, \ldots, j_k\}$이라 하자
2. 피처 $j_i \in D_p$마다 모든 차례를 헤아려 가장자리 기여를 계산한다
3. 이 기여는 마디마다의 학습 데이터 기여도를 계산에 넣는다

### 복잡도

| 방법 | 복잡도 | 꼭 맞는가? |
|--------|-----------|--------|
| 마구잡이 섀플리 | $O(2^d \cdot T \cdot D)$ | 예 |
| 커널 SHAP | $O(N_{\text{뽑음}} \cdot N_{\text{바탕}})$ | 아니오 |
| **Tree SHAP** | $O(T \cdot L \cdot D^2)$ | **예** |

나무 500그루, 깊이 6, 잎 64개인 흔한 XGBoost 모델이면 Tree SHAP이 표본마다 밀리초 안에 꼭 맞는 값을 계산한다.

---

## 2. 구현

### SHAP 곳집 쓰기

권하는 길은 `shap` 곳집의 잘 다듬은 C++ 구현을 쓰는 것이다.

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

    # 첫 표본의 완전성을 살핀다
    prediction = model.predict_proba(X[800:801])[0, 1]
    from scipy.special import expit
    shap_sum = shap_values[0].sum() + explainer.expected_value
    print(f"미루어 봄(확률):  {prediction:.4f}")
    print(f"SHAP으로 되세움: {expit(shap_sum):.4f}")

    return shap_values, explainer

def tree_shap_for_lightgbm(model, X_train, X_test, feature_names):
    """LightGBM 모형의 나무 SHAP."""
    import lightgbm as lgb

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 클래스가 여럿이면 shap_values이 클래스마다의 목록이다
    if isinstance(shap_values, list):
        print(f"클래스의 수: {len(shap_values)}")
        for c, sv in enumerate(shap_values):
            importance = np.abs(sv).mean(axis=0)
            top_idx = np.argsort(importance)[::-1][:5]
            print(f"\n클래스 {c}의 앞선 결:")
            for idx in top_idx:
                print(f"  {feature_names[idx]}: {importance[idx]:.4f}")

    return shap_values
```

### SHAP 상호작용 값

Tree SHAP만이 지닌 힘은 **꼭 맞는 상호작용 값**, 곧 추론을 주효과와 짝지은 상호작용으로 쪼개는 이차 섀플리 값을 계산하는 것이다.

$$
f(\mathbf{x}) = \phi_0 + \sum_i \phi_{ii}(\mathbf{x}) + \sum_{i < j} \phi_{ij}(\mathbf{x})
$$

여기서 $\phi_{ii}$은 피처 $i$의 주효과이고 $\phi_{ij}$은 피처 $i$과 $j$ 사이의 상호작용을 담는다.

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

## 3. 시각화

### 간추린 그림

```python
def tree_shap_visualization(shap_values, X_test, feature_names):
    """나무 모형에 쓰는 여느 SHAP 그림."""

    # 벌떼 간추림 - 결마다 SHAP 값의 스프레드를 보인다
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

## 4. 본 대로 대 손대어 본 Tree SHAP

Tree SHAP에는 두 피처가 있다.

### 본 대로(길에 매인 것)

트리 구조가 뜻하는 매인 분포 $P(X_{\bar{S}} \mid X_S)$을 쓴다. 서로 상관된 피처가 기여도를 나누어 받는다.

```python
explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
```

### 손대어 본

본 피처와 상관없이 가장자리 분포 $P(X_{\bar{S}})$을 쓴다. 인과 기여를 더 잘 드러낸다.

```python
explainer = shap.TreeExplainer(model, X_background, feature_perturbation='interventional')
```

| 피처 | 나은 점 | 못한 점 |
|------|------|------|
| 본 대로 | 빠르고 기초 데이터가 없어도 된다 | 얽힌 피처에 기여도를 나눠 준다 |
| 손대어 본 | 인과로 읽을 수 있다 | 기초 데이터가 있어야 하고 느리다 |

---

## 5. 계량 금융 응용

### 트리 모델로 하는 신용 점수 매기기

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

    # 두 클래스 가름이면 shap_values이 목록일 수 있다
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 맡긴 클래스

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

Tree SHAP은 트리 기반 모델의 섀플리 값을 다항식 때에 꼭 맞게 주므로, 서비스에서 그래디언트 북돋운 모델을 해석하는 으뜸 기준이다.

**핵심 성능:**

- **꼭 정확도**: 추정 어긋남이 없다(커널 SHAP과 다르다)
- **빠름**: 표본마다 $O(TLD^2)$
- **상호작용**: 짝지은 상호작용 값을 꼭 맞게 계산할 수 있다
- **두 피처**: 본 대로(빠름)와 손대어 본(인과)

**살펴볼 거리**

1. Lundberg, S. M., et al. (2020). "From Local Explanations to Global Understanding with Explainable AI for Trees." *Nature Machine Intelligence*.

2. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.

3. Lundberg, S. M., et al. (2018). "Consistent Individualized Feature Attribution for Tree Ensembles." *arXiv:1802.03888*.
