# [Code Source](https://github.com/patrickloeber/pytorchTutorial)

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 0) Prepare data (Breast Cancer Wisconsin dataset: binary classification)
#    X: (n_samples, n_features)         e.g., (569, 30)  float64
#    y: (n_samples,)                    e.g., (569,)     int (0/1) → will cast to float
# ---------------------------------------------------------------------------
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target                   # X: (569, 30), y: (569,)

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Standardize features (fit on train, apply to both)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# To torch.float32
X_train = torch.from_numpy(X_train.astype(np.float32))  # (N_train, n_features)
X_test  = torch.from_numpy(X_test.astype(np.float32))   # (N_test,  n_features)
y_train = torch.from_numpy(y_train.astype(np.float32))  # (N_train,)
y_test  = torch.from_numpy(y_test.astype(np.float32))   # (N_test,)

# Reshape targets for BCE-with-logits: (N, 1)
y_train = y_train.view(-1, 1)   # (N_train, 1)
y_test  = y_test.view(-1, 1)    # (N_test,  1)

# ---------------------------------------------------------------------------
# 1) Model (logits version)
#    - Return **logits** (raw scores) from the linear layer.
#    - DO NOT apply sigmoid here. We'll use BCEWithLogitsLoss for stability.
#      (It combines a sigmoid + BCE in one stable kernel.)
#    Shapes:
#      input x: (batch_size, n_features)
#      logits : (batch_size, 1)
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)  # weight: (1, n_features), bias: (1,)

    def forward(self, x):
        logits = self.linear(x)   # (batch_size, 1), raw scores
        return logits

model = Model(n_features)

# ---------------------------------------------------------------------------
# 2) Loss and optimizer
#    - Use nn.BCEWithLogitsLoss (expects logits, NOT probabilities).
#    - For class imbalance, you can pass pos_weight=tensor([w]).
# ---------------------------------------------------------------------------
num_epochs = 100
learning_rate = 0.01
criterion = nn.BCEWithLogitsLoss()                 # ← stable sigmoid + BCE
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------------------------
# 3) Training loop
#    Shapes per iteration:
#      logits = model(X_train)                → (N_train, 1)
#      loss   = criterion(logits, y_train)    → () scalar
# ---------------------------------------------------------------------------
for epoch in range(num_epochs):
    logits = model(X_train)                   # (N_train, 1), raw scores
    loss = criterion(logits, y_train)         # () scalar tensor

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        # loss.item() returns a Python float (syncs if on GPU)
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# ---------------------------------------------------------------------------
# 4) Evaluation
#    - To get probabilities: apply sigmoid at evaluation time only.
#    - To get classes: threshold logits at 0 (since sigmoid(0)=0.5).
#      y_cls = (logits >= 0).float()
# ---------------------------------------------------------------------------
with torch.no_grad():
    logits_test = model(X_test)                       # (N_test, 1)
    probs_test  = torch.sigmoid(logits_test)          # (N_test, 1) in [0,1] (optional)
    y_predicted_cls = (logits_test >= 0).float()      # (N_test, 1) {0,1} without calling sigmoid

    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')