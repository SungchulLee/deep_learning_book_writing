# Encoding Categorical

Categorical encoding converts non-numerical labels into numerical representations suitable for machine learning algorithms, with different strategies for ordinal vs nominal variables and high-cardinality features.

---

## Label Encoding

### 1. Ordinal Encoding

**Integer encoding for ordered categories:**

```python
from sklearn.preprocessing import LabelEncoder

# Ordinal categories (has order)
sizes = ['small', 'medium', 'large', 'small', 'large']

encoder = LabelEncoder()
sizes_encoded = encoder.fit_transform(sizes)

print(sizes_encoded)  # [2, 1, 0, 2, 0]
print(encoder.classes_)  # ['large', 'medium', 'small'] (alphabetical)
```

### 2. Inverse Transform

```python
sizes_decoded = encoder.inverse_transform(sizes_encoded)
print(sizes_decoded)  # ['small', 'medium', 'large', ...]
```

### 3. Target Encoding

```python
# For target variable (y)
from sklearn.preprocessing import LabelEncoder

y = ['cat', 'dog', 'cat', 'bird', 'dog']
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(y_encoded)  # [1, 2, 1, 0, 2]
# bird=0, cat=1, dog=2
```

### 4. Unseen Labels

```python
# Handle new categories
try:
    new_size = encoder.transform(['xlarge'])
except ValueError as e:
    print("Error: Unseen label")
    # Must handle explicitly
```

### 5. Custom Mapping

```python
# Explicit order for ordinal
from sklearn.preprocessing import OrdinalEncoder

X = [['small'], ['medium'], ['large']]
categories = [['small', 'medium', 'large', 'xlarge']]

encoder = OrdinalEncoder(categories=categories)
X_encoded = encoder.fit_transform(X)

print(X_encoded)  # [[0], [1], [2]]
```

### 6. Multiple Columns

```python
# Encode multiple categorical columns
X = [['small', 'red'], ['large', 'blue'], ['medium', 'red']]

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

print(X_encoded)
# [[2, 1],
#  [0, 0],
#  [1, 1]]
```

### 7. When to Use

- Target variable encoding
- Ordinal categories (small < medium < large)
- Tree-based models (okay with arbitrary integers)
- **Avoid for:** Nominal categories with linear models (implies order)

---

## One-Hot Encoding

### 1. Binary Vectors

**Create binary column for each category:**

```python
from sklearn.preprocessing import OneHotEncoder

colors = [['red'], ['blue'], ['green'], ['red']]

encoder = OneHotEncoder(sparse_output=False)
colors_encoded = encoder.fit_transform(colors)

print(colors_encoded)
# [[0, 0, 1],  # red
#  [1, 0, 0],  # blue
#  [0, 1, 0],  # green
#  [0, 0, 1]]  # red

print(encoder.categories_)  # [array(['blue', 'green', 'red'])]
```

### 2. Get Feature Names

```python
feature_names = encoder.get_feature_names_out(['color'])
print(feature_names)
# ['color_blue', 'color_green', 'color_red']
```

### 3. Drop First (Multicollinearity)

```python
# Drop one category to avoid dummy variable trap
encoder = OneHotEncoder(drop='first', sparse_output=False)
colors_encoded = encoder.fit_transform(colors)

print(colors_encoded)
# [[0, 1],  # red (reference: blue dropped)
#  [0, 0],  # blue
#  [1, 0]]  # green
```

### 4. Handle Unknown

```python
# Ignore unknown categories during transform
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(colors)

# Transform with unknown category
new_colors = [['red'], ['yellow']]  # 'yellow' unseen
encoded = encoder.transform(new_colors)

print(encoded)
# [[0, 0, 1],  # red
#  [0, 0, 0]]  # yellow → all zeros
```

### 5. Sparse Output

```python
# Default: sparse matrix (memory efficient)
encoder = OneHotEncoder()  # sparse_output=True by default
colors_encoded_sparse = encoder.fit_transform(colors)

print(type(colors_encoded_sparse))  # scipy.sparse matrix
print(colors_encoded_sparse.toarray())  # Convert to dense
```

### 6. Multiple Columns

```python
X = [['red', 'small'],
     ['blue', 'large'],
     ['green', 'small']]

encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

print(X_encoded.shape)  # (3, 5) = 3 colors + 2 sizes
print(encoder.get_feature_names_out(['color', 'size']))
```

### 7. When to Use

- Nominal categories (no natural order)
- Linear models (logistic regression, linear SVM)
- Neural networks
- **Avoid for:** High cardinality (too many categories)

---

## Ordinal Encoding

### 1. Specify Order

**Map categories to integers with explicit order:**

```python
from sklearn.preprocessing import OrdinalEncoder

# Education levels (has order)
education = [['High School'], ['Bachelor'], ['Master'], 
             ['PhD'], ['Bachelor']]

# Specify order
categories = [['High School', 'Bachelor', 'Master', 'PhD']]

encoder = OrdinalEncoder(categories=categories)
education_encoded = encoder.fit_transform(education)

print(education_encoded)
# [[0],  # High School
#  [1],  # Bachelor
#  [2],  # Master
#  [3],  # PhD
#  [1]]  # Bachelor
```

### 2. Unknown Categories

```python
# Handle unknown
encoder = OrdinalEncoder(
    categories=categories,
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

# Unknown mapped to -1
unknown_edu = [['Associate']]  # Not in categories
encoded = encoder.transform(unknown_edu)
print(encoded)  # [[-1]]
```

### 3. vs Label Encoding

```python
# LabelEncoder: 1D arrays (single column)
# OrdinalEncoder: 2D arrays (multiple columns)

# LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(['a', 'b', 'c'])  # 1D

# OrdinalEncoder
oe = OrdinalEncoder()
X_encoded = oe.fit_transform([['a'], ['b'], ['c']])  # 2D
```

### 4. Multiple Ordered Features

```python
X = [['low', 'small'],
     ['medium', 'large'],
     ['high', 'medium']]

categories = [
    ['low', 'medium', 'high'],      # Income
    ['small', 'medium', 'large']    # Size
]

encoder = OrdinalEncoder(categories=categories)
X_encoded = encoder.fit_transform(X)

print(X_encoded)
# [[0, 0],
#  [1, 2],
#  [2, 1]]
```

### 5. Dtype

```python
# Control output type
encoder = OrdinalEncoder(dtype=np.int32)
X_encoded = encoder.fit_transform(X)
print(X_encoded.dtype)  # int32
```

### 6. Inverse Transform

```python
X_decoded = encoder.inverse_transform(X_encoded)
print(X_decoded)  # Original categories
```

### 7. When to Use

- Ordinal categories (education, ratings)
- Tree-based models (naturally handle order)
- When order matters to model
- **Better than one-hot** for ordered features

---

## Target Encoding

### 1. Mean Encoding

**Replace category with target mean:**

```python
from category_encoders import TargetEncoder
import pandas as pd

# Data
df = pd.DataFrame({
    'color': ['red', 'blue', 'red', 'green', 'blue', 'red'],
    'price': [100, 150, 110, 200, 160, 105]
})

encoder = TargetEncoder()
df['color_encoded'] = encoder.fit_transform(df['color'], df['price'])

print(df)
# color_encoded = mean(price | color)
# red → (100+110+105)/3 = 105
# blue → (150+160)/2 = 155
# green → 200
```

### 2. Smoothing

```python
# Add smoothing to prevent overfitting
encoder = TargetEncoder(smoothing=1.0)
# Blends category mean with global mean
```

### 3. Avoid Leakage

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# Use in cross-validation to prevent leakage
# Don't fit on entire dataset!

# Correct: fit encoder in each CV fold
scores = cross_val_score(
    Pipeline([
        ('encoder', TargetEncoder()),
        ('model', Ridge())
    ]),
    X, y, cv=5
)
```

### 4. Min Samples Leaf

```python
# Require minimum samples for encoding
encoder = TargetEncoder(min_samples_leaf=10)
# Categories with <10 samples use global mean
```

### 5. Hierarchical

```python
# Use hierarchical smoothing
# Blend category mean → parent category mean → global mean
```

### 6. When to Use

- High cardinality features (many categories)
- Tree-based models (especially gradient boosting)
- When category frequency varies widely
- **Risk:** Target leakage if not careful

### 7. Alternative: Leave-One-Out

```python
# Leave-one-out encoding (prevents direct leakage)
# For sample i: use mean of other samples in category
def loo_encode(df, col, target):
    global_mean = df[target].mean()
    agg = df.groupby(col)[target].agg(['sum', 'count'])
    
    # For each row: (sum - value) / (count - 1)
    loo = df[[col, target]].merge(agg, left_on=col, right_index=True)
    loo['encoded'] = (loo['sum'] - loo[target]) / (loo['count'] - 1)
    loo['encoded'].fillna(global_mean, inplace=True)
    
    return loo['encoded']
```

---

## Frequency Encoding

### 1. Count Encoding

**Replace with frequency count:**

```python
import pandas as pd

colors = ['red', 'blue', 'red', 'green', 'blue', 'red', 'red']
df = pd.DataFrame({'color': colors})

# Count encoding
count_map = df['color'].value_counts().to_dict()
df['color_count'] = df['color'].map(count_map)

print(df)
# color  color_count
# red    4
# blue   2
# red    4
# green  1
# ...
```

### 2. Frequency Encoding

```python
# Proportion instead of count
freq_map = df['color'].value_counts(normalize=True).to_dict()
df['color_freq'] = df['color'].map(freq_map)

print(df)
# color  color_freq
# red    0.571
# blue   0.286
# green  0.143
```

### 3. When to Use

- High cardinality
- Frequency is informative
- Simple baseline before complex encoding
- Tree-based models

### 4. Log Transform

```python
# Log transform counts (reduce skewness)
df['color_log_count'] = np.log1p(df['color_count'])
```

### 5. Rank Encoding

```python
# Encode by frequency rank
rank_map = df['color'].value_counts().rank(ascending=False).to_dict()
df['color_rank'] = df['color'].map(rank_map)
```

### 6. Combine with Other Encoding

```python
# Use frequency as additional feature alongside one-hot
# Provides both category identity and frequency info
```

### 7. Unseen Categories

```python
# Handle unseen in test set
# Option 1: Use count=0 or freq=0
# Option 2: Use mean frequency from training
```

---

## Binary Encoding

### 1. Hybrid Approach

**One-hot + binary representation:**

```python
from category_encoders import BinaryEncoder

colors = ['red', 'blue', 'green', 'yellow', 'red']

encoder = BinaryEncoder()
colors_df = pd.DataFrame({'color': colors})
encoded = encoder.fit_transform(colors_df)

print(encoded)
# Fewer columns than one-hot for high cardinality
# 4 categories → 2 binary columns (log₂(4))
```

### 2. Dimensionality Reduction

```python
# One-hot: n categories → n columns
# Binary: n categories → log₂(n) columns

# Example: 100 categories
# One-hot: 100 columns
# Binary: 7 columns (2^7 = 128 > 100)
```

### 3. When to Use

- High cardinality (50-1000 categories)
- Need fewer dimensions than one-hot
- Linear models
- **Better than:** Ordinal (no false order)

### 4. Similarity Preserved

```python
# Similar categories have similar encodings
# red=001, blue=010 differ by 2 bits
# Hamming distance relates categories
```

### 5. vs Hashing

```python
# Binary: Deterministic, invertible
# Hashing: Collisions possible, not invertible
```

### 6. Implementation

```python
# Manually implement
def binary_encode(categories, n_bits):
    encoding = {}
    for i, cat in enumerate(categories):
        binary = format(i, f'0{n_bits}b')
        encoding[cat] = [int(b) for b in binary]
    return encoding
```

### 7. Feature Names

```python
feature_names = encoder.get_feature_names_out()
print(feature_names)  # ['color_0', 'color_1', ...]
```

---

## Practical Examples

### 1. Mixed Encoding Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ct = ColumnTransformer([
    ('onehot', OneHotEncoder(), ['color', 'brand']),      # Nominal
    ('ordinal', OrdinalEncoder(), ['size', 'quality'])    # Ordinal
], remainder='passthrough')

X_transformed = ct.fit_transform(X)
```

### 2. High Cardinality

```python
# For features with >50 categories
from category_encoders import TargetEncoder

# Use target encoding instead of one-hot
encoder = TargetEncoder(cols=['city', 'zip_code'])
X_encoded = encoder.fit_transform(X, y)
```

### 3. Pandas get_dummies

```python
# Quick one-hot encoding in pandas
df_encoded = pd.get_dummies(df, columns=['color', 'size'])

# With prefix
df_encoded = pd.get_dummies(df, columns=['color'], prefix='col')
# Creates: col_red, col_blue, col_green

# Drop first to avoid multicollinearity
df_encoded = pd.get_dummies(df, drop_first=True)
```

### 4. Combine with Scaling

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('scaler', StandardScaler(with_mean=False)),  # Sparse-friendly
    ('model', LogisticRegression())
])
```

### 5. Feature Hashing

```python
from sklearn.feature_extraction import FeatureHasher

# For very high cardinality (>1000 categories)
hasher = FeatureHasher(n_features=10, input_type='string')
X_hashed = hasher.transform([['red'], ['blue'], ['green']])

# Pros: Fixed dimensionality, no fitting needed
# Cons: Collisions, not invertible
```

### 6. Saving Encoders

```python
import joblib

# Save fitted encoder
joblib.dump(encoder, 'encoder.pkl')

# Load and use
encoder = joblib.load('encoder.pkl')
X_test_encoded = encoder.transform(X_test)
```

### 7. Handle Rare Categories

```python
# Combine rare categories into 'other'
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    min_frequency=10,      # Categories with <10 samples
    max_categories=20,     # Keep top 20, rest → 'infrequent'
    sparse_output=False
)

X_encoded = encoder.fit_transform(X)
```

---

## Summary

| Method | Output | Use Case | Cardinality |
|--------|--------|----------|-------------|
| **LabelEncoder** | Integer | Target variable, ordinal | Any |
| **OneHotEncoder** | Binary vectors | Nominal, linear models | Low (<50) |
| **OrdinalEncoder** | Integer (ordered) | Ordinal features | Low-medium |
| **TargetEncoder** | Real (target mean) | High cardinality, trees | High (50-1000) |
| **BinaryEncoder** | Binary columns | Dimensionality reduction | High (50-1000) |
| **FrequencyEncoder** | Count/proportion | Simple baseline | Any |
| **HashingEncoder** | Fixed dimensions | Very high cardinality | Very high (>1000) |

**Key insights:**
- **One-hot:** Default for nominal with low cardinality
- **Ordinal:** When categories have natural order
- **Target:** High cardinality, but watch for leakage
- **Binary:** Compromise between one-hot and ordinal
- **Always handle unknown categories** in test data
- **Drop first** in one-hot to avoid multicollinearity
