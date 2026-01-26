# Time Series Preprocessing

## Introduction

Time series data—sequences of observations indexed by time—require specialized preprocessing to prepare for RNN modeling. This section covers windowing, normalization, feature engineering, and handling irregularities common in real-world temporal data.

## Windowing and Sequence Creation

### Sliding Window Approach

Transform a continuous time series into supervised learning examples:

```python
import torch
import numpy as np

def create_sequences(data, window_size, horizon=1):
    """
    Create input-output pairs using sliding windows.
    
    Args:
        data: Time series array of shape (T,) or (T, features)
        window_size: Number of past timesteps as input
        horizon: Number of future timesteps to predict
    
    Returns:
        X: Input sequences (num_samples, window_size, features)
        y: Target values (num_samples, horizon, features)
    """
    data = np.array(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + horizon])
    
    return np.array(X), np.array(y)

# Example: Predict next value from 10 previous values
data = np.sin(np.linspace(0, 20, 200))  # Sine wave
X, y = create_sequences(data, window_size=10, horizon=1)
print(f"X shape: {X.shape}")  # (189, 10, 1)
print(f"y shape: {y.shape}")  # (189, 1, 1)
```

### Multi-Step Prediction

For forecasting multiple future timesteps:

```python
def create_multistep_sequences(data, input_steps, output_steps):
    """
    Create sequences for multi-step forecasting.
    
    Args:
        data: Time series (T, features)
        input_steps: Number of input timesteps
        output_steps: Number of output timesteps to predict
    """
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:i + input_steps])
        y.append(data[i + input_steps:i + input_steps + output_steps])
    return np.array(X), np.array(y)

# Predict 5 steps ahead from 20 historical steps
X, y = create_multistep_sequences(data.reshape(-1, 1), input_steps=20, output_steps=5)
```

## Normalization Strategies

### Min-Max Scaling

Scale values to [0, 1]:

$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

```python
class MinMaxScaler:
    """Min-Max normalization for time series."""
    
    def __init__(self, feature_range=(0, 1)):
        self.min = None
        self.max = None
        self.feature_range = feature_range
    
    def fit(self, data):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)
        return self
    
    def transform(self, data):
        scaled = (data - self.min) / (self.max - self.min + 1e-8)
        return scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
    
    def fit_transform(self, data):
        return self.fit(data).transform(data)
    
    def inverse_transform(self, scaled_data):
        unscaled = (scaled_data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return unscaled * (self.max - self.min) + self.min
```

### Standardization (Z-Score)

Transform to zero mean and unit variance:

$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

```python
class StandardScaler:
    """Z-score standardization."""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        return self
    
    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)
    
    def fit_transform(self, data):
        return self.fit(data).transform(data)
    
    def inverse_transform(self, scaled_data):
        return scaled_data * self.std + self.mean
```

### Per-Window Normalization

Normalize each window independently (useful for non-stationary data):

```python
def normalize_windows(windows):
    """
    Normalize each window to have zero mean and unit variance.
    Useful when the absolute level varies but patterns are consistent.
    """
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True)
    return (windows - means) / (stds + 1e-8), means, stds

def denormalize_windows(normalized, means, stds):
    """Reverse per-window normalization."""
    return normalized * stds + means
```

### Log Transformation

For data with exponential growth or heavy tails:

```python
def log_transform(data, epsilon=1e-8):
    """Log transformation for positive data."""
    return np.log(data + epsilon)

def inverse_log_transform(data, epsilon=1e-8):
    """Inverse log transformation."""
    return np.exp(data) - epsilon
```

## Train-Validation-Test Split

### Temporal Split (Correct)

Never shuffle time series data—maintain temporal order:

```python
def temporal_train_val_test_split(data, train_ratio=0.7, val_ratio=0.15):
    """
    Split time series maintaining temporal order.
    
    Args:
        data: Full time series
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    
    Returns:
        train, val, test portions
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return data[:train_end], data[train_end:val_end], data[val_end:]

# Critical: Fit scalers only on training data!
train, val, test = temporal_train_val_test_split(full_data)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)  # Fit only on train
val_scaled = scaler.transform(val)          # Transform using train stats
test_scaled = scaler.transform(test)        # Transform using train stats
```

### Walk-Forward Validation

For more robust evaluation:

```python
def walk_forward_validation(data, initial_train_size, step_size, n_splits):
    """
    Walk-forward validation for time series.
    
    Yields (train_indices, val_indices) for each fold.
    """
    for i in range(n_splits):
        train_end = initial_train_size + i * step_size
        val_end = train_end + step_size
        
        if val_end > len(data):
            break
            
        yield list(range(train_end)), list(range(train_end, val_end))
```

## Feature Engineering

### Lag Features

Include past values as additional features:

```python
def create_lag_features(data, lags):
    """
    Create lagged features from time series.
    
    Args:
        data: Time series (T, features)
        lags: List of lag values, e.g., [1, 2, 3, 7, 14]
    
    Returns:
        Augmented data with lag features
    """
    df = pd.DataFrame(data)
    
    for lag in lags:
        for col in df.columns:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df.dropna().values
```

### Rolling Statistics

Capture local trends and volatility:

```python
def create_rolling_features(data, windows=[5, 10, 20]):
    """
    Create rolling mean and std features.
    """
    import pandas as pd
    df = pd.DataFrame(data, columns=['value'])
    
    for w in windows:
        df[f'rolling_mean_{w}'] = df['value'].rolling(window=w).mean()
        df[f'rolling_std_{w}'] = df['value'].rolling(window=w).std()
        df[f'rolling_min_{w}'] = df['value'].rolling(window=w).min()
        df[f'rolling_max_{w}'] = df['value'].rolling(window=w).max()
    
    return df.dropna().values
```

### Time-Based Features

Encode temporal patterns:

```python
def create_time_features(timestamps):
    """
    Extract time-based features from timestamps.
    
    Args:
        timestamps: Array of datetime objects or pandas DatetimeIndex
    """
    import pandas as pd
    ts = pd.DatetimeIndex(timestamps)
    
    features = {
        'hour': ts.hour / 23,                    # Normalized to [0, 1]
        'day_of_week': ts.dayofweek / 6,
        'day_of_month': ts.day / 31,
        'month': ts.month / 12,
        'is_weekend': (ts.dayofweek >= 5).astype(float),
        'hour_sin': np.sin(2 * np.pi * ts.hour / 24),
        'hour_cos': np.cos(2 * np.pi * ts.hour / 24),
        'month_sin': np.sin(2 * np.pi * ts.month / 12),
        'month_cos': np.cos(2 * np.pi * ts.month / 12),
    }
    
    return pd.DataFrame(features).values
```

## Handling Missing Values

### Forward/Backward Fill

```python
def fill_missing(data, method='forward'):
    """Fill missing values in time series."""
    import pandas as pd
    df = pd.DataFrame(data)
    
    if method == 'forward':
        return df.ffill().values
    elif method == 'backward':
        return df.bfill().values
    elif method == 'interpolate':
        return df.interpolate(method='linear').values
    else:
        raise ValueError(f"Unknown method: {method}")
```

### Masking for RNNs

Create attention masks for valid timesteps:

```python
def create_mask(sequences, pad_value=0):
    """
    Create mask indicating valid (non-padded) timesteps.
    
    Args:
        sequences: Padded sequences (batch, seq_len)
        pad_value: Value used for padding
    
    Returns:
        Mask tensor (batch, seq_len) with 1 for valid, 0 for padding
    """
    return (sequences != pad_value).float()
```

## Handling Irregular Time Series

### Resampling to Regular Intervals

```python
import pandas as pd

def resample_to_regular(timestamps, values, freq='1H', method='mean'):
    """
    Resample irregular time series to regular intervals.
    
    Args:
        timestamps: Irregular timestamps
        values: Corresponding values
        freq: Target frequency ('1H', '1D', etc.)
        method: Aggregation method ('mean', 'sum', 'last')
    """
    df = pd.DataFrame({'value': values}, index=pd.DatetimeIndex(timestamps))
    resampled = df.resample(freq).agg(method)
    return resampled.index, resampled['value'].values
```

### Time-Aware Encoding

Include time gaps as features:

```python
def encode_time_gaps(timestamps):
    """
    Encode time gaps between observations.
    
    Returns normalized time differences since previous observation.
    """
    timestamps = pd.to_datetime(timestamps)
    time_diffs = timestamps.diff().total_seconds()
    time_diffs[0] = 0
    
    # Normalize by median gap
    median_gap = np.median(time_diffs[1:])
    return time_diffs / (median_gap + 1e-8)
```

## Multivariate Time Series

### Handling Multiple Features

```python
def prepare_multivariate(data_dict, window_size, target_col):
    """
    Prepare multivariate time series for prediction.
    
    Args:
        data_dict: Dictionary mapping feature names to arrays
        window_size: Input window length
        target_col: Name of target variable
    
    Returns:
        X: (samples, window_size, num_features)
        y: (samples, 1) target values
    """
    # Stack features
    feature_names = list(data_dict.keys())
    data = np.column_stack([data_dict[name] for name in feature_names])
    target_idx = feature_names.index(target_col)
    
    X, y_full = create_sequences(data, window_size)
    y = y_full[:, :, target_idx]  # Only target column for y
    
    return X, y
```

## PyTorch Dataset for Time Series

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series forecasting."""
    
    def __init__(self, data, window_size, horizon=1, transform=None):
        """
        Args:
            data: Numpy array of shape (T, features)
            window_size: Number of past timesteps
            horizon: Number of future timesteps to predict
            transform: Optional transformation function
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.horizon = horizon
        self.transform = transform
    
    def __len__(self):
        return len(self.data) - self.window_size - self.horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size:idx + self.window_size + self.horizon]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y

# Usage
data = np.random.randn(1000, 5)  # 1000 timesteps, 5 features
dataset = TimeSeriesDataset(data, window_size=20, horizon=5)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

for X_batch, y_batch in dataloader:
    print(f"X: {X_batch.shape}")  # (32, 20, 5)
    print(f"y: {y_batch.shape}")  # (32, 5, 5)
    break
```

## Complete Pipeline Example

```python
class TimeSeriesPipeline:
    """End-to-end pipeline for time series preprocessing."""
    
    def __init__(self, window_size, horizon, normalize=True):
        self.window_size = window_size
        self.horizon = horizon
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
    
    def fit_transform(self, data):
        """Fit on training data and transform."""
        if self.normalize:
            data = self.scaler.fit_transform(data)
        
        X, y = create_sequences(data, self.window_size, self.horizon)
        return X, y
    
    def transform(self, data):
        """Transform new data using fitted scaler."""
        if self.normalize:
            data = self.scaler.transform(data)
        
        X, y = create_sequences(data, self.window_size, self.horizon)
        return X, y
    
    def inverse_transform_predictions(self, predictions):
        """Convert predictions back to original scale."""
        if self.normalize:
            return self.scaler.inverse_transform(predictions)
        return predictions

# Usage
pipeline = TimeSeriesPipeline(window_size=24, horizon=6, normalize=True)

# Fit on training data
X_train, y_train = pipeline.fit_transform(train_data)

# Transform validation/test data
X_val, y_val = pipeline.transform(val_data)
X_test, y_test = pipeline.transform(test_data)

# After prediction, convert back
predictions = model(X_test)
predictions_original_scale = pipeline.inverse_transform_predictions(predictions)
```

## Summary

Time series preprocessing for RNNs involves:

1. **Windowing**: Create input-output pairs with sliding windows
2. **Normalization**: Scale data (fit on training only!)
3. **Temporal splits**: Maintain time ordering in train/val/test
4. **Feature engineering**: Add lag features, rolling statistics, time encodings
5. **Handle irregularities**: Missing values, variable sampling rates

Key principles:
- Always fit scalers on training data only
- Never shuffle time series during splitting
- Consider stationarity and seasonality
- Match window size to the temporal patterns you want to capture
