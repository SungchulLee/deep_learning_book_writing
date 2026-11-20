# 피처 간 상호작용

설명 방법은 대개 **주효과**, 곧 피처마다 예측에 따로 기여한 만큼을 계산한다. 그런데 실제 현상에는 **상호작용**이 있을 때가 많다. 둘 이상의 피처가 함께 내는 효과는 개별 기여의 합으로 분해할 수 없다. 인자 간 상호작용이 포트폴리오의 움직임, 위험 집중, 시장의 비선형 동학을 만들어 내는 계량 금융에서는 상호작용을 이해하는 일이 중요하다.

---

## 1. 수학 기초

### 상호작용 정의

모델 $f(\mathbf{x})$의 예측은 주효과와 상호작용으로 분해할 수 있다.

$$
f(\mathbf{x}) = \phi_0 + \sum_i \phi_i(\mathbf{x}) + \sum_{i < j} \phi_{ij}(\mathbf{x}) + \text{더 높은 차수 항}
$$

피처 $i$와 $j$의 **SHAP 상호작용 값** $\Phi_{ij}$는 이렇다.

$$
\Phi_{ij}(\mathbf{x}) = \sum_{S \subseteq N \setminus \{i,j\}} \frac{|S|!(d - |S| - 2)!}{2(d-1)!} \nabla_{ij}(S)
$$

여기서 이산 2차 차분(discrete second-order difference)은 이렇다.

$$
\nabla_{ij}(S) = f_x(S \cup \{i,j\}) - f_x(S \cup \{i\}) - f_x(S \cup \{j\}) + f_x(S)
$$

### 성질

1. **대칭성**: $\Phi_{ij} = \Phi_{ji}$
2. **완전성**: $\sum_j \Phi_{ij} = \phi_i$
3. **대각선 = 주효과**: $\Phi_{ii}$는 상호작용을 걷어낸 뒤의 주효과를 담는다

### 프리드먼의 H 통계량

두 피처를 함께 본 부분 의존 함수 $\hat{f}_{ij}$의 분산 가운데 상호작용이 설명하는 비율을 잰다. 부분 의존 함수는 다른 피처를 데이터에 걸쳐 평균 내고 남은 함수이며, 여기서는 평균이 0이 되도록 중심화해 쓴다. $H^2_{ij}$가 0이면 두 피처의 효과는 더해질 뿐이고, 1에 가까우면 그 둘의 결합 효과가 거의 다 상호작용이다.

$$
H^2_{ij} = \frac{\sum_k \left[\hat{f}_{ij}(x_i^{(k)}, x_j^{(k)}) - \hat{f}_i(x_i^{(k)}) - \hat{f}_j(x_j^{(k)})\right]^2}{\sum_k \hat{f}_{ij}^2(x_i^{(k)}, x_j^{(k)})}
$$

---

## 2. 상호작용 계산

### SHAP 상호작용 값

```python
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_shap_interactions(model, X, feature_names, n_samples=100):
    """
    나무 모형의 SHAP 서로 미침 값을 셈한다.

    형태가 [표본 수, 결 수, 결 수]인 interaction_values을 내놓는다.
    """
    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(X[:n_samples])
    return interaction_values

def top_interactions(interaction_values, feature_names, k=10):
    """고른 크기로 앞선 k개의 결 서로 미침을 찾는다."""
    mean_interactions = np.abs(interaction_values).mean(axis=0)
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
    return interactions[:k]
```

### 시각화

```python
def plot_interaction_matrix(interaction_values, feature_names):
    """고른 서로 미침을 열 그림으로 그린다."""
    mean_interactions = np.abs(interaction_values).mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        mean_interactions,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap='Blues',
        ax=ax
    )
    ax.set_title('SHAP 결 서로 미침')
    plt.tight_layout()
    return fig

def plot_interaction_dependence(
    interaction_values, X, feature_names, feature_i, feature_j
):
    """
    두 결 사이의 서로 미침이 그 값에 따라 어떻게 바뀌는지
    그린다.
    """
    i = feature_names.index(feature_i)
    j = feature_names.index(feature_j)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 서로 미침 값 대 feature_i, feature_j으로 빛깔을 준다
    scatter = axes[0].scatter(
        X[:len(interaction_values), i],
        interaction_values[:, i, j],
        c=X[:len(interaction_values), j],
        cmap='coolwarm', alpha=0.6, s=20
    )
    axes[0].set_xlabel(feature_i)
    axes[0].set_ylabel(f'서로 미침 ({feature_i} × {feature_j})')
    plt.colorbar(scatter, ax=axes[0], label=feature_j)

    # 서로 미침 값 대 feature_j, feature_i으로 빛깔을 준다
    scatter = axes[1].scatter(
        X[:len(interaction_values), j],
        interaction_values[:, i, j],
        c=X[:len(interaction_values), i],
        cmap='coolwarm', alpha=0.6, s=20
    )
    axes[1].set_xlabel(feature_j)
    axes[1].set_ylabel(f'서로 미침 ({feature_i} × {feature_j})')
    plt.colorbar(scatter, ax=axes[1], label=feature_i)

    plt.tight_layout()
    return fig
```

---

## 3. 계량 금융 응용

### 인자 상호작용 살피기

```python
def analyze_factor_interactions(
    model,
    factor_data: np.ndarray,
    factor_names: list
):
    """
    금융 인자끼리 서로 미치는 힘을 살핀다.

    금융에서 중요한 서로 미침:
    - 밀림 × 출렁임 (출렁이는 저자에서 밀림이 무너진다)
    - 값어치 × 됨됨이 (값싼 덫 대 알맞은 값의 좋은 회사)
    - 크기 × 사고팔기 쉬움 (작은 회사의 사고팔기 어려움 웃돈)
    """
    explainer = shap.TreeExplainer(model)
    interactions = explainer.shap_interaction_values(factor_data[:200])

    # 홑 힘 대 서로 미침
    n_factors = len(factor_names)
    main_effects = np.zeros(n_factors)
    interaction_effects = np.zeros((n_factors, n_factors))

    for i in range(n_factors):
        main_effects[i] = np.abs(interactions[:, i, i]).mean()
        for j in range(n_factors):
            if i != j:
                interaction_effects[i, j] = np.abs(interactions[:, i, j]).mean()

    # 서로 미침 대 홑 힘의 비
    print("인자의 홑 힘 대 서로 미침:")
    print("-" * 60)
    for i in range(n_factors):
        total_interaction = interaction_effects[i].sum()
        ratio = total_interaction / (main_effects[i] + 1e-10)
        print(f"{factor_names[i]:20s}: 홑={main_effects[i]:.4f}, "
              f"서로 미침={total_interaction:.4f}, 비={ratio:.2f}")

    return interactions
```

### 위험 집중 탐지

```python
def detect_risk_concentrations(
    risk_model,
    portfolio_features: np.ndarray,
    feature_names: list
):
    """
    서로 미침으로 숨은 무릅씀 쏠림을 찾는다.

    무릅씀 인자끼리 크게 서로 미치면 나누어 담아 얻는 이로움을
    너무 크게 본 것일 수 있다.
    """
    interactions = compute_shap_interactions(
        risk_model, portfolio_features, feature_names
    )

    top = top_interactions(interactions, feature_names, k=5)

    print("무릅씀 쏠림 확률(앞선 서로 미침):")
    print("-" * 60)
    for inter in top:
        print(f"  {inter['feature_i']} × {inter['feature_j']}: "
              f"셈={inter['strength']:.4f}")

    return top
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
XOR $f(x) = x_1 \oplus x_2$를 정확히 계산하는 모델이 있고, 모르는 피처는 $\{0,1\}$에서 고르게 뽑는다고 하자($f_x(S)$는 그 평균이다). 입력 $x = [1, 1]$에서 이산 2차 차분 $\nabla_{12}(\varnothing)$, SHAP 상호작용 값 $\Phi_{12}$, 주효과 $\Phi_{11}$을 계산하고, 대칭성과 완전성이 성립하는지 확인하여라.

</div>

??? success "풀이"
    연합 값은 $f_x(\varnothing) = 0.5$, $f_x(\{1\}) = f_x(\{2\}) = (1 + 0)/2 = 0.5$, $f_x(\{1,2\}) = 0$이다. 따라서 $\nabla_{12}(\varnothing) = 0 - 0.5 - 0.5 + 0.5 = -0.5$다. $d = 2$이면 $S$는 공집합 하나뿐이고 가중치가 $\frac{0!\,0!}{2 \cdot 1!} = \frac12$이므로 $\Phi_{12} = -0.25$이고, 같은 계산으로 $\Phi_{21} = -0.25$라 대칭성이 성립한다. 섀플리 값은 $\phi_1 = \tfrac12[(0.5 - 0.5) + (0 - 0.5)] = -0.25$이므로 완전성에서 $\Phi_{11} = \phi_1 - \Phi_{12} = 0$이다. 주효과가 0이고 예측과 평균의 차이 $-0.5$ 전부가 상호작용에서 나온다. 주효과만 계산하는 설명 방법은 XOR 같은 구조를 원리상 못 본다는 것이 이 절의 출발점이다. $\square$

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
    절차는 이렇다. (1) 시험 표본마다 피처 기여도를 계산한다. (2) 삭제: 기여도가 큰 차례로 피처를 하나씩 가리며 모델의 확신도가 떨어지는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 떨어진다. (3) 삽입: 빈 베이스라인에서 시작해 기여도가 큰 차례로 피처를 하나씩 드러내며 확신도가 오르는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 오른다. (4) 두 곡선 아래 넓이를 계산한다. (5) 무작위 순서(베이스라인)와 다른 방법에 견준다. 믿을 만한 방법이면 삭제 곡선의 넓이가 작고 삽입 곡선의 넓이가 커야 한다. 통계적으로 믿을 만하려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 설명 방법을 신용 부도를 예측하는 금융 모델에 어떻게 적용할 수 있는지 논하라. 설명이 채워야 할 규제 요건은 무엇인가?

</div>

??? success "풀이"
    신용 모델에는 규제(ECOA, GDPR 22조)가 불리한 결정마다 그 사람에게 맞춘 설명을 요구한다. 방법은 다음을 내놓아야 한다. (1) 거절에 가장 크게 기여한 요인(불리한 처분 사유). (2) 일관된 설명(비슷한 신청자는 비슷한 설명을 받는다). (3) 실행 가능한 설명(신청자가 무엇을 바꾸어야 하는지 안다). 이 절의 설명 방법으로 피처 중요도를 짚을 수 있으나, 강건성(입력이 조금 바뀌었다고 설명이 확 달라지면 안 된다)과 충실성(중요한 피처를 없애면 예측이 바뀌어야 한다)을 평가해야 한다. 보호 속성은 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

피처 상호작용 살피기는 모델 예측에 담긴 비가법적(non-additive) 구조를 드러낸다. 계량 금융에서 상호작용은 주효과만으로는 해석할 수 없는 것, 곧 레짐에 따라 달라지는 행태, 인자 쏠림(crowding), 비선형 위험 동학을 담아낸다.

**핵심 식:**

$$
\nabla_{ij}(S) = f_x(S \cup \{i,j\}) - f_x(S \cup \{i\}) - f_x(S \cup \{j\}) + f_x(S)
$$

**살펴볼 거리**

1. Lundberg, S. M., et al. (2020). "From Local Explanations to Global Understanding with Explainable AI for Trees." *Nature Machine Intelligence*.

2. Friedman, J. H., & Popescu, B. E. (2008). "Predictive Learning via Rule Ensembles." *Annals of Applied Statistics*.

3. Tsang, M., et al. (2020). "How Does This Interaction Affect Me? Interpretable Estimation of Individual-Level Interaction Effects." *AAAI*.
