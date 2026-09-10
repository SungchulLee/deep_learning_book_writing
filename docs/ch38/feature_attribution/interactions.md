# 결 사이 서로 미침

풀이 방법은 대개 **홑 힘**, 곧 결마다 미루어 봄에 따로 이바지한 만큼을 셈한다. 그런데 참 세상의 일은 **서로 미침**을 지닐 때가 많다. 둘 이상의 결이 함께 내는 힘은 낱낱의 이바지로 쪼갤 수 없다. 인자끼리의 서로 미침이 밑천 꾸러미의 움직임, 무릅씀 쏠림, 선형 아닌 저자 결을 이끄는 계량 금융에서는 서로 미침을 아는 일이 종요롭다.

---

## 1. 수학 밑바탕

### 서로 미침 뜻매김

모형 $f(\mathbf{x})$에서 미루어 봄은 홑 힘과 서로 미침으로 쪼갤 수 있다.

$$
f(\mathbf{x}) = \phi_0 + \sum_i \phi_i(\mathbf{x}) + \sum_{i < j} \phi_{ij}(\mathbf{x}) + \text{더 높은 차수 항}
$$

결 $i$과 $j$의 **SHAP 서로 미침 값** $\Phi_{ij}$은 이렇다.

$$
\Phi_{ij}(\mathbf{x}) = \sum_{S \subseteq N \setminus \{i,j\}} \frac{|S|!(d - |S| - 2)!}{2(d-1)!} \nabla_{ij}(S)
$$

여기서 띄엄띄엄한 이차 미분은 이렇다.

$$
\nabla_{ij}(S) = f_x(S \cup \{i,j\}) - f_x(S \cup \{i\}) - f_x(S \cup \{j\}) + f_x(S)
$$

### 됨됨이

1. **맞섬**: $\Phi_{ij} = \Phi_{ji}$
2. **온전함**: $\sum_j \Phi_{ij} = \phi_i$
3. **대각선 = 홑 힘**: $\Phi_{ii}$은 서로 미침을 걷어낸 뒤의 홑 힘을 담는다

### 프리드먼의 H 셈속

서로 미침이 풀어내는 흩어짐의 몫을 잰다.

$$
H^2_{ij} = \frac{\sum_k \left[\hat{f}_{ij}(x_i^{(k)}, x_j^{(k)}) - \hat{f}_i(x_i^{(k)}) - \hat{f}_j(x_j^{(k)})\right]^2}{\sum_k \hat{f}_{ij}^2(x_i^{(k)}, x_j^{(k)})}
$$

---

## 2. 서로 미침 셈하기

### SHAP 서로 미침 값

```python
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_shap_interactions(model, X, feature_names, n_samples=100):
    """
    나무 모형의 SHAP 서로 미침 값을 셈한다.

    꼴이 [표본 수, 결 수, 결 수]인 interaction_values을 내놓는다.
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

### 그림으로 보이기

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

## 3. 계량 금융에 쓰기

### 인자 서로 미침 살피기

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

### 무릅씀 쏠림 찾기

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

    print("무릅씀 쏠림 낌새(앞선 서로 미침):")
    print("-" * 60)
    for inter in top:
        print(f"  {inter['feature_i']} × {inter['feature_j']}: "
              f"셈={inter['strength']:.4f}")

    return top
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

결 서로 미침 살피기는 모형의 미루어 봄에 든 더해지지 않는 얼개를 드러낸다. 계량 금융에서 서로 미침은 홑 힘만으로는 풀이할 수 없는 판에 매인 결, 인자 쏠림, 선형 아닌 무릅씀의 움직임을 담아낸다.

**고갱이 식:**

$$
\nabla_{ij}(S) = f_x(S \cup \{i,j\}) - f_x(S \cup \{i\}) - f_x(S \cup \{j\}) + f_x(S)
$$

**살펴볼 거리**

1. Lundberg, S. M., et al. (2020). "From Local Explanations to Global Understanding with Explainable AI for Trees." *Nature Machine Intelligence*.

2. Friedman, J. H., & Popescu, B. E. (2008). "Predictive Learning via Rule Ensembles." *Annals of Applied Statistics*.

3. Tsang, M., et al. (2020). "How Does This Interaction Affect Me? Interpretable Estimation of Individual-Level Interaction Effects." *AAAI*.
