# [Code Source](https://github.com/patrickloeber/pytorchTutorial)

import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 0) Prepare data (Breast Cancer Wisconsin dataset: binary classification)
#    X: (n_samples, n_features)         e.g., (569, 30)  float64 (from sklearn)
#    y: (n_samples,)                     e.g., (569,)     int (0/1) → will cast to float
#    Tip: For full reproducibility, set seeds for numpy/torch and pass
#         random_state=... in data splits (already done below).
# ---------------------------------------------------------------------------
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
# X: (569, 30) float64
# y: (569,) int64
print(f"Number of Positive Cases in {y.shape[0]} Patients of Original Data :  {y.sum()}")  # e.g., 357/569 positives

n_samples, n_features = X.shape             # n_samples=569, n_features=30

# Split:
# X_train: (N_train, n_features)            (455, 30) float64
# X_test : (N_test,  n_features)            (114, 30) float64
# y_train: (N_train,)                        (455,)    int64
# y_test : (N_test,)                         (114,)    int64
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)
print(f"Number of Positive Cases in {y_train.shape[0]} Patients of Train Data :  {y_train.sum()}")  # e.g., 288
print(f"Number of Positive Cases in {y_test.shape[0]} Patients of Test Data :  {y_test.sum()}")    # e.g., 69

# Feature scaling (fit on train, apply to both):
# After transform:
# X_train: (N_train, n_features)            float64
# X_test : (N_test,  n_features)            float64
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# Convert to torch tensors (float32 to match PyTorch defaults for layers/ops):
# X_train: (N_train, n_features)            torch.float32
# X_test : (N_test,  n_features)            torch.float32
# y_train: (N_train,)                        torch.float32
# y_test : (N_test,)                         torch.float32
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test  = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test  = torch.from_numpy(y_test.astype(np.float32))

# Reshape targets for BCE:
# y_train: (N_train, 1)
# y_test : (N_test,  1)
y_train = y_train.view(y_train.shape[0], 1)
y_test  = y_test.view(y_test.shape[0],  1)

# ---------------------------------------------------------------------------
# 1) Model
#    Linear layer:
#      self.linear.weight: (1, n_features)
#      self.linear.bias  : (1,)
#    Forward:
#      input  x: (batch_size, n_features)
#      output y: (batch_size, 1)  after sigmoid → probabilities in [0,1]
#    NOTE (stability best practice):
#      In production, prefer returning logits (no sigmoid in forward) and use
#      nn.BCEWithLogitsLoss (which applies a stable sigmoid + BCE internally).
#      Here we keep sigmoid+BCELoss for didactic clarity.
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)  # (n_features → 1)
        # print(f"{self.linear.weight.shape = }") # (1,30)
        # print(f"{self.linear.bias.shape = }")   # (1,)

    def forward(self, x):  # x: (batch_size, n_features) torch.float32
        x = self.linear(x)  # x: (batch_size, 1) torch.float32  ← raw score (logit)
        y_pred = torch.sigmoid(x)  # y_pred: (batch_size, 1) torch.float32 (probabilities)
        return y_pred

model = Model(n_features)

# ---------------------------------------------------------------------------
# 2) Loss and optimizer
#    criterion (BCELoss) expects:
#      input : (batch_size, 1) probabilities in [0,1]
#      target: (batch_size, 1) in {0,1} (float OK: 0.0/1.0)
#    loss: scalar tensor with shape ()  (0-dim)
#    Tip (again): Prefer nn.BCEWithLogitsLoss with logits from the model for stability.
#    Note: learning_rate=10. is intentionally large here (didactic). In practice,
#          start with something like 0.1, 0.01, or use an optimizer like Adam.
# ---------------------------------------------------------------------------
num_epochs = 100
learning_rate = 10.
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------------------------
# 3) Training loop
#    Shapes per iteration:
#      y_pred = model(X_train)              → (N_train, 1)
#      loss = criterion(y_pred, y_train)    → () scalar
#    Best practice: call optimizer.zero_grad() **before** backward each step
#    because gradients accumulate by default in PyTorch.
# ---------------------------------------------------------------------------
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(X_train)              # (N_train, 1)
    loss = criterion(y_pred, y_train)    # () scalar tensor

    # Zero grad before new step (canonical place is BEFORE backward)
    optimizer.zero_grad()

    # Backward pass and update
    loss.backward()          # populates gradients matching parameter shapes:
                             #   linear.weight.grad: (1, n_features)
                             #   linear.bias.grad  : (1,)
    optimizer.step()         # in-place param update (SGD step)

    if (epoch + 1) % 10 == 0:
        # loss.item() → Python float (host). If on GPU, this syncs to host.
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# ---------------------------------------------------------------------------
# 4) Evaluation
#    Switch to eval mode if using layers with train/eval behavior (Dropout/BN):
#      model.eval()
#    y_predicted: (N_test, 1) probabilities in [0,1]
#    y_predicted_cls: (N_test, 1) in {0,1} after thresholding via .round()
#                     - .round() acts like threshold at 0.5:
#                       p>=0.5 → 1.0, p<0.5 → 0.0 (dtype remains float32)
#                     - To get integer type explicitly:
#                       y_predicted_cls = y_predicted.round().to(torch.int64)
#    y_test: (N_test, 1) is float32 here; equality on 0.0/1.0 floats is exact.
#    eq operation:
#      - y_predicted_cls.eq(y_test) → torch.bool mask
#      - summing booleans counts True as 1
#      - dividing by N yields accuracy in [0,1]
#    Alternative (if using logits instead of probabilities):
#      - logits = model(X_test); y_cls = (logits >= 0).to(y_test.dtype)
# ---------------------------------------------------------------------------
with torch.no_grad():
    y_predicted = model(X_test)              # (N_test, 1), probabilities
    y_predicted_cls = y_predicted.round()    # (N_test, 1) 0.0/1.0 as float32
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])  # accuracy in [0,1]
    # If you prefer explicit integer comparison:
    # y_predicted_cls = y_predicted.round().to(torch.int64)
    # y_test_int = y_test.to(torch.int64)
    # acc = y_predicted_cls.eq(y_test_int).sum() / y_test_int.shape[0]
    print(f'accuracy: {acc.item():.4f}')