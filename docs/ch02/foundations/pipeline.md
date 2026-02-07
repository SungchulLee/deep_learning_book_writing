# Pipeline Design

Pipelines chain preprocessing steps and a final estimator into a single object that prevents data leakage, enables grid search over the full workflow, and makes deployment a one-file affair. This section covers `Pipeline`, `ColumnTransformer`, `FeatureUnion`, and caching.

## Why Pipelines?

### The Data Leakage Problem

Without pipelines, preprocessing on the full dataset before cross-validation leaks test-fold statistics into training:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# WRONG — scaler sees all data including future validation folds
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scores = cross_val_score(LogisticRegression(), X_scaled, y, cv=5)
```

### The Pipeline Solution

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Scaler fits only on each training fold — no leakage
scores = cross_val_score(pipe, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

During `cross_val_score`, for each fold $k$ the pipeline calls `scaler.fit_transform(X_train_k)` and then `scaler.transform(X_val_k)`. The validation fold never contaminates the scaler's learned parameters.

## Creating Pipelines

### Explicit Construction

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', LogisticRegression())
])
```

Each step is a `(name, estimator)` tuple. All steps except the last must be transformers (implement `transform`); the last step can be a transformer or a predictor.

### `make_pipeline` Shortcut

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=10),
    LogisticRegression()
)
# Step names auto-generated: 'standardscaler', 'pca', 'logisticregression'
```

### Fit, Transform, Predict

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
accuracy = pipe.score(X_test, y_test)
```

Internally, `pipe.fit(X, y)` calls `fit_transform` on every step except the last, then `fit` on the final step. `pipe.predict(X)` calls `transform` on all intermediate steps, then `predict` on the last.

## Accessing Steps and Intermediate Results

```python
# By name
scaler = pipe.named_steps['standardscaler']
print(scaler.mean_[:5])

# By index
pca = pipe[1]
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Transform up to a step (exclude the final estimator)
X_preprocessed = pipe[:-1].transform(X_test)
```

## Parameter Access with `step__param` Syntax

```python
pipe.set_params(pca__n_components=5, logisticregression__C=0.1)

all_params = pipe.get_params()
print(all_params['pca__n_components'])  # 5
```

This syntax extends naturally to `GridSearchCV`:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'pca__n_components': [5, 10, 15],
    'logisticregression__C': [0.01, 0.1, 1, 10],
    'logisticregression__penalty': ['l1', 'l2'],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Best: {grid.best_params_}, Score: {grid.best_score_:.4f}")
```

## ColumnTransformer

Real datasets contain mixed types: numerical, categorical, text. `ColumnTransformer` applies different transformations to different column subsets:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['city', 'employment_status']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ]), categorical_features),
])
```

### Automatic Column Selection

```python
from sklearn.compose import make_column_selector

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), make_column_selector(dtype_include='number')),
    ('cat', OneHotEncoder(), make_column_selector(dtype_include='object')),
])
```

### Remainder Handling

```python
preprocessor = ColumnTransformer(
    transformers=[...],
    remainder='passthrough'    # keep unspecified columns as-is
    # remainder='drop'         # discard unspecified columns (default)
)
```

### Full Pipeline with ColumnTransformer

```python
full_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Works directly with raw DataFrames
full_pipe.fit(df_train, y_train)
full_pipe.predict(df_test)
```

## FeatureUnion

Concatenate outputs from multiple transformers applied to the **same** input:

```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

feature_union = FeatureUnion([
    ('pca', PCA(n_components=5)),
    ('kbest', SelectKBest(f_classif, k=5)),
])

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('features', feature_union),
    ('classifier', LogisticRegression()),
])

pipe.fit(X_train, y_train)
# Combined feature matrix: 5 PCA + 5 selected = 10 features
```

## Memory Caching

Expensive transformations (PCA on large matrices, feature engineering) can be cached to avoid recomputation during grid search:

```python
from joblib import Memory

memory = Memory(location='./cache', verbose=0)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', LogisticRegression()),
], memory=memory)

# First grid search fit: computes scaler + PCA from scratch
# Subsequent fits with same data: loads cached transformations
grid = GridSearchCV(pipe, {'classifier__C': [0.1, 1, 10]}, cv=5)
grid.fit(X_train, y_train)
```

When only the final step's parameters change, cached intermediate results are reused.

## Pipeline with Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Filter method
pipe_filter = make_pipeline(
    StandardScaler(),
    SelectKBest(f_classif, k=10),
    LogisticRegression()
)

# Wrapper method
pipe_rfe = make_pipeline(
    StandardScaler(),
    RFE(LogisticRegression(), n_features_to_select=10),
    LogisticRegression()
)
```

## Persistence

```python
import joblib

# Save the entire fitted pipeline
joblib.dump(full_pipe, 'credit_model_pipeline.pkl')

# Load and deploy
loaded_pipe = joblib.load('credit_model_pipeline.pkl')
predictions = loaded_pipe.predict(new_data)
```

## Common Pipeline Patterns

### Classification with Mixed Data

```python
from sklearn.ensemble import RandomForestClassifier

clf_pipe = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), make_column_selector(dtype_include='number')),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]), make_column_selector(dtype_include='object')),
    ])),
    ('classifier', RandomForestClassifier(n_estimators=100)),
])
```

### Regression with Feature Engineering

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

reg_pipe = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, interaction_only=True),
    Ridge(alpha=1.0)
)
```

### Text Classification

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

text_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', MultinomialNB())
])
```

## Quantitative Finance: Walk-Forward Pipeline

In finance, pipelines are critical for walk-forward validation where preprocessing must be re-fitted on each expanding or rolling training window:

```python
from sklearn.model_selection import TimeSeriesSplit

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0)),
])

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(
    pipe, X, y, cv=tscv, scoring='neg_mean_squared_error'
)
# Each fold: scaler fitted only on the expanding training window
print(f"Walk-forward MSE: {-scores.mean():.4f} ± {scores.std():.4f}")
```

## Summary

| Component | Purpose | Key Feature |
|-----------|---------|-------------|
| `Pipeline` | Chain steps sequentially | Prevents leakage, enables grid search |
| `make_pipeline` | Pipeline with auto-naming | Less boilerplate |
| `ColumnTransformer` | Different transforms per column | Handles mixed data types |
| `FeatureUnion` | Concatenate parallel transforms | Combine feature extraction methods |
| `memory` parameter | Cache intermediate results | Speed up grid search |
| `joblib.dump/load` | Persistence | Deploy full workflow as single file |
