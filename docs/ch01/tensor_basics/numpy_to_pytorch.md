# NumPy to PyTorch

## Tensor Basics

### 1. Conversion

```python
import numpy as np
import torch

# NumPy → PyTorch
arr = np.array([1, 2, 3, 4, 5])
tensor = torch.from_numpy(arr)
print(type(tensor))  # <class 'torch.Tensor'>

# PyTorch → NumPy
arr_back = tensor.numpy()
```

### 2. Shared Memory

```python
arr = np.array([1, 2, 3])
tensor = torch.from_numpy(arr)

tensor[0] = 999
print(arr[0])  # 999 - shared memory!
```

### 3. Copy

```python
tensor = torch.tensor(arr)  # Copies data
tensor[0] = 999
print(arr[0])  # 1 - independent
```

## Device Management

### 1. CPU and GPU

```python
# CPU tensor
tensor_cpu = torch.tensor([1, 2, 3])
print(tensor_cpu.device)  # cpu

# GPU tensor
if torch.cuda.is_available():
    tensor_gpu = tensor_cpu.to('cuda')
    print(tensor_gpu.device)  # cuda:0
```

### 2. Device Selection

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = torch.tensor([1, 2, 3]).to(device)
```

### 3. Transfer

```python
# CPU → GPU
tensor_gpu = tensor_cpu.cuda()

# GPU → CPU
tensor_cpu = tensor_gpu.cpu()
```

## Operations

### 1. Similar to NumPy

```python
# Element-wise
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b  # [5, 7, 9]

# Reduction
mean = a.float().mean()
sum_val = a.sum()
```

### 2. Autograd

```python
# Gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()
z.backward()  # Compute gradients
print(x.grad)  # [2., 4., 6.]
```

### 3. No Gradient

```python
with torch.no_grad():
    y = x * 2  # No gradient tracking
```

## Neural Network Example

### 1. Data Preparation

```python
# NumPy data
X_np = np.random.randn(100, 10)
y_np = np.random.randint(0, 2, 100)

# Convert to tensors
X = torch.from_numpy(X_np).float()
y = torch.from_numpy(y_np).long()
```

### 2. Model Definition

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2)
)
```

### 3. Training Loop

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    # Forward
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
