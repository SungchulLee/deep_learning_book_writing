"""
13_regression_task.py - Predicting Continuous Values

Build a regression model to predict house prices using the
California Housing dataset.

REGRESSION vs CLASSIFICATION:
- Regression: Predict continuous values (prices, temperature, age)
- Classification: Predict discrete categories (cat/dog, yes/no)

TIME: 30-35 minutes | DIFFICULTY: ⭐⭐⭐☆☆
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("Regression Task: California Housing Prices")
print("="*80)

# Load California Housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

print(f"Features: {housing.feature_names}")
print(f"Samples: {X.shape[0]}")
print(f"Features per sample: {X.shape[1]}")
print(f"Target: Median house value (in $100,000s)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features (important for regression!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

print(f"\nTrain samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

print("\n" + "="*80)
print("Regression Model")
print("="*80)

class RegressionNet(nn.Module):
    """Neural network for regression."""
    
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1)  # Single output for regression
        )
    
    def forward(self, x):
        return self.network(x)

model = RegressionNet(X_train.shape[1])

# MSE Loss for regression
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Model created!")
print("Loss: MSE (Mean Squared Error)")

# Training
epochs = 200
batch_size = 64
losses = []

print("\n" + "="*80)
print("Training...")
print("="*80)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    losses.append(epoch_loss / (len(X_train) // batch_size))
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] Loss: {losses[-1]:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    train_pred = model(X_train)
    test_pred = model(X_test)
    
    train_mse = criterion(train_pred, y_train).item()
    test_mse = criterion(test_pred, y_test).item()
    
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

print(f"\nFinal Results:")
print(f"Train RMSE: ${train_rmse*100000:.2f}")
print(f"Test RMSE: ${test_rmse*100000:.2f}")

# Visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Training loss
ax1.plot(losses, linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

# Predictions vs actual (train)
ax2.scatter(y_train, train_pred, alpha=0.5)
ax2.plot([y_train.min(), y_train.max()], 
         [y_train.min(), y_train.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Price')
ax2.set_ylabel('Predicted Price')
ax2.set_title('Training Set Predictions')
ax2.grid(True, alpha=0.3)

# Predictions vs actual (test)
ax3.scatter(y_test, test_pred, alpha=0.5, color='green')
ax3.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 'r--', lw=2)
ax3.set_xlabel('Actual Price')
ax3.set_ylabel('Predicted Price')
ax3.set_title('Test Set Predictions')
ax3.grid(True, alpha=0.3)

# Residuals
residuals = (test_pred - y_test).numpy()
ax4.hist(residuals, bins=50, edgecolor='black')
ax4.set_xlabel('Prediction Error')
ax4.set_ylabel('Frequency')
ax4.set_title('Residuals Distribution')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('13_regression_results.png', dpi=150)
print("\nPlots saved!")

print("\n" + "="*80)
print("KEY POINTS FOR REGRESSION")
print("="*80)
print("""
REGRESSION SPECIFICS:
✓ Use MSELoss or L1Loss (not CrossEntropy!)
✓ Single output neuron (no activation function)
✓ Normalize/standardize input features
✓ Evaluate with RMSE, MAE, R² score

LOSS FUNCTIONS:
- MSELoss: Penalizes large errors more
- L1Loss: More robust to outliers
- SmoothL1Loss: Hybrid approach

METRICS:
- MSE: Mean Squared Error
- RMSE: Root MSE (same units as target)
- MAE: Mean Absolute Error
- R²: Coefficient of determination (0-1)

TIPS:
- Always normalize features for regression
- Check for outliers in data
- Visualize predictions vs actual
- Analyze residuals distribution
""")
plt.show()
