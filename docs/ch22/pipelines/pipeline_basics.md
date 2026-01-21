# Pipeline Basics

Pipelines chain multiple processing steps together, ensuring proper data flow and preventing data leakage. They're essential for clean, reproducible machine learning workflows.

---

## Why Pipelines?

### 1. The Problem Without Pipelines

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Generate data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# WRONG: Data leakage!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on ALL data
scores = cross_val_score(LogisticRegression(), X_scaled, y, cv=5)
# Problem: Scaler saw test data during fit!
```

### 2. The Solution With Pipelines

```python
from sklearn.pipeline import Pipeline

# CORRECT: No data leakage
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

scores = cross_val_score(pipe, X, y, cv=5)
# Scaler fits only on training folds
print(f"CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## Creating Pipelines

### 1. Using Pipeline Class

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Explicit Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', LogisticRegression())
])

# Each step is a (name, transformer/estimator) tuple
```

### 2. Using make_pipeline (Shortcut)

```python
from sklearn.pipeline import make_pipeline

# Names generated automatically from class names
pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=10),
    LogisticRegression()
)

# Check step names
print(pipe.named_steps.keys())
# dict_keys(['standardscaler', 'pca', 'logisticregression'])
```

### 3. Fit and Predict

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit pipeline
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)

# Score
accuracy = pipe.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

---

## Pipeline Steps

### 1. Transformers vs Estimators

```python
# Transformers: Have fit() and transform() methods
# Examples: StandardScaler, PCA, OneHotEncoder

# Estimators: Have fit() and predict() methods  
# Examples: LogisticRegression, RandomForestClassifier

# All steps except last must be transformers
# Last step can be transformer or estimator
```

### 2. Accessing Steps

```python
# By name
scaler = pipe.named_steps['standardscaler']
print(f"Scaler mean: {scaler.mean_[:3]}")

# By index
pca = pipe[1]  # Second step
print(f"PCA components: {pca.n_components_}")

# Last estimator
classifier = pipe[-1]
print(f"Classifier: {classifier}")
```

### 3. Getting Intermediate Results

```python
# Transform up to a specific step
X_scaled = pipe[:-1].transform(X_test)  # All steps except last
print(f"After scaling + PCA: {X_scaled.shape}")

# Using named_steps
X_after_scaler = pipe.named_steps['standardscaler'].transform(X_test)
```

---

## Pipeline with Feature Selection

### 1. SelectKBest Example

```python
from sklearn.feature_selection import SelectKBest, f_classif

pipe = make_pipeline(
    StandardScaler(),
    SelectKBest(f_classif, k=10),
    LogisticRegression()
)

pipe.fit(X_train, y_train)
print(f"Accuracy: {pipe.score(X_test, y_test):.4f}")

# Check which features were selected
selector = pipe.named_steps['selectkbest']
selected_features = selector.get_support()
print(f"Selected features: {np.where(selected_features)[0]}")
```

### 2. RFE (Recursive Feature Elimination)

```python
from sklearn.feature_selection import RFE

pipe = make_pipeline(
    StandardScaler(),
    RFE(LogisticRegression(), n_features_to_select=10),
    LogisticRegression()
)

pipe.fit(X_train, y_train)
```

---

## Pipeline Parameters

### 1. Setting Parameters

```python
# Use step_name__parameter_name syntax
pipe.set_params(logisticregression__C=0.1)
pipe.set_params(pca__n_components=5)

# View all parameters
print(pipe.get_params())
```

### 2. Grid Search with Pipelines

```python
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(
    StandardScaler(),
    PCA(),
    LogisticRegression()
)

# Parameter grid uses stepname__param format
param_grid = {
    'pca__n_components': [5, 10, 15],
    'logisticregression__C': [0.01, 0.1, 1, 10],
    'logisticregression__penalty': ['l1', 'l2']
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
```

---

## Column Transformer

### 1. Different Transformations for Different Columns

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

# Create sample data with mixed types
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000],
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
    'employed': ['yes', 'no', 'yes', 'yes', 'no']
})

# Different preprocessing for different columns
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),
    ('cat', OneHotEncoder(drop='first'), ['city', 'employed'])
])

X_transformed = preprocessor.fit_transform(df)
print(f"Transformed shape: {X_transformed.shape}")
```

### 2. With Pipeline

```python
from sklearn.linear_model import LogisticRegression

# Full pipeline with column transformer
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Now works with raw DataFrame
# pipe.fit(df, y)
```

### 3. Automatic Column Selection

```python
from sklearn.compose import make_column_selector

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), make_column_selector(dtype_include='number')),
    ('cat', OneHotEncoder(drop='first'), make_column_selector(dtype_include='object'))
])
```

---

## Custom Transformers

### 1. Using FunctionTransformer

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Simple function-based transformer
log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)

pipe = make_pipeline(
    log_transformer,
    StandardScaler(),
    LogisticRegression()
)
```

### 2. Custom Transformer Class

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomScaler(BaseEstimator, TransformerMixin):
    """Custom transformer example"""
    
    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor
    
    def fit(self, X, y=None):
        # Learn parameters from data
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        # Apply transformation
        X_scaled = (X - self.mean_) / (self.std_ + 1e-8)
        return X_scaled * self.scale_factor

# Use in pipeline
pipe = make_pipeline(
    CustomScaler(scale_factor=2.0),
    LogisticRegression()
)

pipe.fit(X_train, y_train)
```

### 3. Transformer with Feature Names

```python
class FeatureAdder(BaseEstimator, TransformerMixin):
    """Add polynomial features"""
    
    def __init__(self, add_squares=True):
        self.add_squares = add_squares
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.add_squares:
            return np.hstack([X, X ** 2])
        return X
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f'x{i}' for i in range(self.n_features_in_)]
        
        if self.add_squares:
            squared_names = [f'{name}_squared' for name in input_features]
            return list(input_features) + squared_names
        return input_features
```

---

## Memory Caching

### 1. Caching Pipeline Steps

```python
from sklearn.pipeline import Pipeline
from tempfile import mkdtemp
from shutil import rmtree

# Create cache directory
cachedir = mkdtemp()

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', LogisticRegression())
], memory=cachedir)

# First fit: computes everything
pipe.fit(X_train, y_train)

# Second fit with same data: uses cached transformations
pipe.fit(X_train, y_train)

# Clean up
rmtree(cachedir)
```

### 2. Using joblib.Memory

```python
from joblib import Memory

memory = Memory(location='./cache', verbose=0)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', LogisticRegression())
], memory=memory)
```

---

## Feature Union

### 1. Combining Features

```python
from sklearn.pipeline import FeatureUnion

# Combine multiple feature extraction methods
feature_union = FeatureUnion([
    ('pca', PCA(n_components=5)),
    ('kbest', SelectKBest(f_classif, k=5))
])

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('features', feature_union),
    ('classifier', LogisticRegression())
])

pipe.fit(X_train, y_train)
print(f"Combined features: {pipe.named_steps['features'].transform(X_train[:1]).shape}")
```

---

## Common Patterns

### 1. Classification Pipeline

```python
from sklearn.ensemble import RandomForestClassifier

clf_pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),  # Keep 95% variance
    RandomForestClassifier(n_estimators=100)
)
```

### 2. Regression Pipeline

```python
from sklearn.ensemble import GradientBoostingRegressor

reg_pipe = make_pipeline(
    StandardScaler(),
    GradientBoostingRegressor()
)
```

### 3. Text Classification Pipeline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

text_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', MultinomialNB())
])
```

### 4. Mixed Data Types

```python
numeric_features = ['age', 'income', 'score']
categorical_features = ['city', 'gender']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])

full_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

---

## Saving and Loading Pipelines

```python
import joblib

# Save
joblib.dump(pipe, 'model_pipeline.pkl')

# Load
loaded_pipe = joblib.load('model_pipeline.pkl')

# Predict with loaded pipeline
y_pred = loaded_pipe.predict(X_test)
```

---

## Summary

**Benefits of pipelines:**
1. **Prevent data leakage**: Transformations fit only on training data
2. **Reproducibility**: Single object captures entire workflow
3. **Convenience**: One fit/predict call for everything
4. **Grid search**: Easy hyperparameter tuning across all steps

**Best practices:**
- Always use pipelines for cross-validation
- Name steps descriptively for clarity
- Use ColumnTransformer for mixed data types
- Create custom transformers for complex preprocessing
- Cache expensive transformations
