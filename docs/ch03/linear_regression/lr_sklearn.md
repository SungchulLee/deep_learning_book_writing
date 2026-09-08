# scikit-learn으로 하는 선형 회귀

scikit-learn은 선형 회귀와 그 정칙화 변형인 릿지, 라쏘, 엘라스틱 넷의 고도로 최적화되고 충분히 검증된 구현을 제공한다. 이 스크립트는 `StandardScaler`를 포함한 파이프라인으로 이 넷을 합성 데이터에서 비교하고, `RidgeCV`, `LassoCV`, `ElasticNetCV`로 교차 검증 기반의 초매개변수 선택을 보여준다. 이런 기준선은 표 형태 데이터에서 어설프게 조율된 딥러닝 모델보다 나은 경우가 많으므로 잘 이해해 두어야 한다.

## 1. 코드

```python
"""
사이킷런으로 하는 선형 회귀
=====================================

인공 데이터에서 최소제곱, 능선, 라쏘, 엘라스틱넷을 견준다.

Demonstrates:
- sklearn.linear_model.{LinearRegression, Ridge, Lasso, ElasticNet}
- 엇갈아 검증하는 클래스 {RidgeCV, LassoCV, ElasticNetCV}
- StandardScaler을 쓰는 흐름
- 모델마다의 계수 견주기

지은이: 깊은 학습 바탕 학습 차례
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    RidgeCV,
    Lasso,
    LassoCV,
    ElasticNet,
    ElasticNetCV,
)

# ============================================================================
# 설정
# ============================================================================

parser = argparse.ArgumentParser(description="sklearn linear regression comparison")
parser.add_argument("--n-samples", type=int, default=300, help="number of samples")
parser.add_argument("--n-features", type=int, default=10, help="total features")
parser.add_argument("--n-informative", type=int, default=5, help="informative features")
parser.add_argument("--noise", type=float, default=15.0, help="noise std dev")
parser.add_argument("--seed", type=int, default=42, help="random seed")
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)

# ============================================================================
# 데이터
# ============================================================================

X, y = make_regression(
    n_samples=ARGS.n_samples,
    n_features=ARGS.n_features,
    n_informative=ARGS.n_informative,
    noise=ARGS.noise,
    random_state=ARGS.seed,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=ARGS.seed,
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================================
# 모델들
# ============================================================================

models = {
    "OLS": LinearRegression(),
    "Ridge (α=1)": Ridge(alpha=1.0),
    "Lasso (α=0.1)": Lasso(alpha=0.1),
    "ElasticNet (α=0.1, ρ=0.5)": ElasticNet(alpha=0.1, l1_ratio=0.5),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    coef = pipe.named_steps["model"].coef_
    n_nonzero = np.sum(np.abs(coef) > 1e-6)
    results[name] = {"mse": mse, "r2": r2, "coef": coef, "n_nonzero": n_nonzero}
    print(f"{name:30s}  MSE={mse:8.2f}  R²={r2:.4f}  nonzero={n_nonzero}/{len(coef)}")

# ============================================================================
# 교차 검증을 통한 초매개변수 선택
# ============================================================================

print("\n--- Cross-Validated Selection ---")

ridge_cv = Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]))])
ridge_cv.fit(X_train, y_train)
print(f"Best Ridge α: {ridge_cv.named_steps['model'].alpha_:.4f}")

lasso_cv = Pipeline([("scaler", StandardScaler()), ("model", LassoCV(cv=5, random_state=ARGS.seed))])
lasso_cv.fit(X_train, y_train)
print(f"Best Lasso α: {lasso_cv.named_steps['model'].alpha_:.4f}")

elastic_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.7, 0.9], random_state=ARGS.seed)),
])
elastic_cv.fit(X_train, y_train)
m = elastic_cv.named_steps["model"]
print(f"Best ElasticNet α: {m.alpha_:.4f}, l1_ratio: {m.l1_ratio_:.2f}")

# ============================================================================
# 계수 비교 그림
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 5))
x_pos = np.arange(ARGS.n_features)
width = 0.2

for i, (name, res) in enumerate(results.items()):
    ax.bar(x_pos + i * width, res["coef"], width, label=name, alpha=0.8)

ax.set_xlabel("Feature Index")
ax.set_ylabel("Coefficient Value")
ax.set_title("Coefficient Comparison Across Regularisation Methods")
ax.set_xticks(x_pos + 1.5 * width)
ax.set_xticklabels([f"x{i}" for i in range(ARGS.n_features)])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("sklearn_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: sklearn_comparison.png")


if __name__ == "__main__":
    pass
```

## 2. 논의

`Pipeline` 추상화는 (특징 정규화를 위한) `StandardScaler`와 회귀 모델을 하나의 추정 가능한 객체로 엮는다. `pipe.fit(X_train, y_train)`을 호출하면 먼저 특징을 표준화한 뒤 표준화된 데이터로 모델을 적합시킨다. `pipe.predict(X_test)`를 호출하면 예측 전에 시험 데이터에 같은 표준화를 적용한다. 이로써 전처리가 일관되게 유지되고, 시험 데이터로 스케일러를 적합시켜 생기는 데이터 누출을 막는다.

릿지 회귀는 모든 계수를 0 쪽으로 줄이는 L2 벌점 $\alpha \|w\|_2^2$을 더하고, 라쏘는 일부 계수를 정확히 0으로 만들 수 있는 L1 벌점 $\alpha \|w\|_1$을 더한다. 엘라스틱 넷은 두 벌점을 혼합 비율 $\rho$로 균형을 잡아 결합한다. `RidgeCV`, `LassoCV`, `ElasticNetCV`는 내부적으로 교차 검증을 수행하여 가장 좋은 정칙화 강도 $\alpha$를 고르며, 그러지 않으면 직접 격자 탐색을 해야 할 일을 자동화해 준다.

여러 방법의 계수를 비교하면 서로 다른 성질이 드러난다. OLS(정칙화 없음)는 정보가 없는 특징을 포함해 모든 특징에 0이 아닌 가중치를 준다. 릿지는 모든 가중치를 줄이되 0이 아닌 값으로 남긴다. 라쏘는 정보가 없는 특징을 0으로 만들어 사실상 특징 선택을 수행한다. 엘라스틱 넷은 특징 선택 면에서는 라쏘처럼 행동하지만 L2 성분 덕분에 상관된 특징을 더 부드럽게 다룬다. 계수들을 나란히 그려 보면 이 차이가 곧바로 눈에 들어온다.

## 연습문제

**연습문제 1.**
베이즈 릿지 회귀(`sklearn.linear_model.BayesianRidge`)를 비교에 추가하라. 성능과 계수 양상이 표준 릿지와 어떻게 다른가?

??? success "연습문제 1 풀이"
    ```python
    from sklearn.linear_model import BayesianRidge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    pipe = Pipeline([('scaler', StandardScaler()), ('model', BayesianRidge())])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    from sklearn.metrics import mean_squared_error, r2_score
    print(f'BayesianRidge MSE={mean_squared_error(y_test, y_pred):.2f} R^2={r2_score(y_test, y_pred):.4f}')
    # BayesianRidge는 정칙화를 자동으로 조율하며 대체로 RidgeCV와 비슷한
    # 성능을 내면서 불확실성 추정까지 제공한다.
    ```

---

**연습문제 2.**
라쏘로 alpha 값을 직접 훑는 것보다 LassoCV가 선호되는 이유를 계산적 관점과 통계적 관점에서 설명하라.

??? success "연습문제 2 풀이"
    LassoCV는 효율적인 웜스타트를 쓴다. 가장 큰 alpha(모든 계수가 0인 지점)에서 모델을 적합시킨 뒤 alpha를 점차 줄이면서 직전 해를 다음 시작점으로 삼는다. 이는 alpha마다 독립적인 모델을 바닥부터 적합시키는 것보다 훨씬 빠르다. 통계적으로는 교차 검증으로 alpha를 고르므로 일반화 성능에 대한 편향 없는 추정을 얻는다. 직접 훑는 방식도 같은 교차 검증이 필요하지만 실수하기 쉽고 웜스타트가 없어 더 느리다.

---

**연습문제 3.**
특징 50개 중 5개만 유용하고 나머지는 잡음인 데이터셋을 생성하라. OLS, 릿지, 라쏘, 엘라스틱 넷에 대해 0이 아닌 계수의 개수를 비교하라. 어떤 방법이 유용한 특징을 가장 잘 찾아내는가?

??? success "연습문제 3 풀이"
    ```python
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    import numpy as np
    
    X, y = make_regression(n_samples=300, n_features=50, n_informative=5, noise=15, random_state=42)
    for name, model in [('OLS', LinearRegression()), ('Ridge', Ridge(alpha=1.0)),
                         ('Lasso', Lasso(alpha=0.1)), ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5))]:
        model.fit(X, y)
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
        print(f'{name:12s}: {n_nonzero}/50 non-zero coefficients')
    # 라쏘는 대체로 0이 아닌 계수를 5개 가까이 찾아낸다.
    # OLS와 릿지는 50개를 모두 0이 아닌 채로 남긴다.
    ```

## 정리하며

**다룬 것** — scikit-learn으로 하는 선형 회귀

`Pipeline` 추상화는 (특징 정규화를 위한) `StandardScaler`와 회귀 모델을 하나의 추정 가능한 객체로 엮는다.

앞의 연습문제 3개로 직접 확인할 수 있다.
