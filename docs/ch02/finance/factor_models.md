# Factor Models

Factor models explain asset returns as linear (or non-linear) functions of common risk factors. Scikit-learn provides the infrastructure for cross-sectional regression, factor construction, and evaluation within a disciplined ML pipeline.

## Cross-Sectional Factor Regression

The classic Fama–MacBeth (1973) approach: at each time step $t$, regress cross-sectional returns on factor exposures:

$$r_{i,t} = \alpha_t + \sum_{k=1}^{K} \beta_{i,k} \lambda_{k,t} + \varepsilon_{i,t}$$

where $\beta_{i,k}$ are stock $i$'s exposures to factor $k$ and $\lambda_{k,t}$ are the factor risk premia estimated at time $t$.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

def fama_macbeth(factor_exposures, returns, alpha=0.0):
    """Cross-sectional regression at each time step.
    
    Parameters
    ----------
    factor_exposures : DataFrame, shape (n_dates, n_stocks, n_factors)
        Panel of factor exposures indexed by (date, stock).
    returns : DataFrame, shape (n_dates, n_stocks)
        Forward returns.
    alpha : float
        Ridge regularisation parameter (0 for OLS).
    
    Returns
    -------
    risk_premia : DataFrame
        Time series of estimated factor risk premia.
    """
    dates = returns.index.get_level_values(0).unique()
    results = []
    
    model = Ridge(alpha=alpha) if alpha > 0 else LinearRegression()
    
    for date in dates:
        X_t = factor_exposures.loc[date].values
        y_t = returns.loc[date].values
        
        # Drop NaN
        mask = ~np.isnan(y_t) & np.all(~np.isnan(X_t), axis=1)
        if mask.sum() < X_t.shape[1] + 1:
            continue
        
        model.fit(X_t[mask], y_t[mask])
        results.append({
            'date': date,
            **{f'lambda_{k}': model.coef_[k] for k in range(X_t.shape[1])},
            'alpha': model.intercept_,
            'r2': model.score(X_t[mask], y_t[mask]),
        })
    
    return pd.DataFrame(results).set_index('date')
```

## Factor Construction with Pipeline

Standardise exposures cross-sectionally before regression:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cross-sectional standardisation at each date
def cross_sectional_pipeline(X, y, alpha=1.0):
    """Standardise factors cross-sectionally, then Ridge regression."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=alpha)),
    ])
    pipe.fit(X, y)
    return pipe

# Apply per date
for date in dates:
    X_t = factor_data.loc[date]
    y_t = returns.loc[date]
    pipe = cross_sectional_pipeline(X_t, y_t, alpha=1.0)
    predictions[date] = pipe.predict(X_t)
```

## Feature Importance as Factor Loading

Tree-based models provide non-linear factor importance:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

gbm = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
gbm.fit(factor_exposures, forward_returns)

# Impurity-based importance
importance = pd.Series(gbm.feature_importances_, index=factor_names)
print(importance.sort_values(ascending=False).head(10))

# Permutation importance (more reliable)
perm_imp = permutation_importance(gbm, factor_exposures, forward_returns,
                                   n_repeats=10, random_state=42)
perm_df = pd.Series(perm_imp.importances_mean, index=factor_names)
print(perm_df.sort_values(ascending=False).head(10))
```

## Multi-Factor Model Comparison

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline

models = {
    'OLS': make_pipeline(StandardScaler(), LinearRegression()),
    'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    'Lasso': make_pipeline(StandardScaler(), Lasso(alpha=0.01)),
    'ElasticNet': make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5)),
    'RF': make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100)),
    'GBM': make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100)),
}

tscv = TimeSeriesSplit(n_splits=5)

for name, model in models.items():
    scores = cross_val_score(model, X_factors, y_returns, cv=tscv,
                             scoring='neg_mean_squared_error')
    print(f"{name:12s}: MSE = {-scores.mean():.6f} ± {scores.std():.6f}")
```

## Sparse Factor Selection with Lasso

Lasso naturally selects which factors have non-zero loadings:

```python
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(cv=TimeSeriesSplit(5), alphas=np.logspace(-4, 0, 50))
lasso_cv.fit(factor_exposures, forward_returns)

selected = factor_names[lasso_cv.coef_ != 0]
print(f"Selected {len(selected)} of {len(factor_names)} factors:")
print(selected)
print(f"Best alpha: {lasso_cv.alpha_:.6f}")
```

## Custom Factor Scorer: Information Coefficient

```python
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr

def information_coefficient(y_true, y_pred):
    """Rank correlation between predicted and realised returns."""
    corr, _ = spearmanr(y_pred, y_true)
    return corr

ic_scorer = make_scorer(information_coefficient, greater_is_better=True)

# Use in model selection
from sklearn.model_selection import GridSearchCV

param_grid = {'ridge__alpha': np.logspace(-3, 3, 20)}
pipe = make_pipeline(StandardScaler(), Ridge())

grid = GridSearchCV(pipe, param_grid, cv=tscv, scoring=ic_scorer)
grid.fit(X_factors, y_returns)
print(f"Best IC: {grid.best_score_:.4f}")
```

## Summary

| Task | Sklearn Tool | Finance Application |
|------|-------------|---------------------|
| Cross-sectional regression | `Ridge`, `LinearRegression` | Fama–MacBeth factor premia |
| Factor selection | `LassoCV`, `SelectFromModel` | Sparse factor models |
| Non-linear factors | `GradientBoostingRegressor` | ML alpha models |
| Factor importance | `permutation_importance` | Factor attribution |
| Walk-forward eval | `TimeSeriesSplit` + `cross_val_score` | Out-of-sample IC |
