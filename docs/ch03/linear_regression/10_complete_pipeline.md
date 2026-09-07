# 완전한 파이프라인

실전에 쓸 수 있는 학습 파이프라인은 단순한 학습 루프를 넘어 설정 관리, 난수 씨앗을 통한 재현성, 적절한 학습/검증/시험 분할, 조기 종료, 모델 체크포인트, 학습률 일정, 종합적인 평가 지표를 포함한다. 이 튜토리얼은 이 모든 구성 요소를 하나의 일관된 파이프라인으로 엮어 실제 프로젝트의 본보기로 삼는다.

## 코드

```python
"""
==============================================================================
10_complete_pipeline.py
==============================================================================
DIFFICULTY: ⭐⭐⭐⭐⭐ (Advanced)

DESCRIPTION:
    Complete production-ready training pipeline with all best practices.
    Includes train/val/test split, early stopping, model checkpointing,
    logging, and comprehensive evaluation.

다루는 것:
    - Complete training pipeline
    - Train/validation/test splits
    - Early stopping
    - Model checkpointing
    - Learning rate scheduling
    - Comprehensive evaluation
    - Reproducibility

PREREQUISITES:
    - All previous tutorials

TIME: ~40 minutes
==============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

print("=" * 70)
print("COMPLETE PRODUCTION-READY TRAINING PIPELINE")
print("=" * 70)

# ============================================================================
# 1부: 설정과 재현성
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: CONFIGURATION")
print("=" * 70)

class Config:
    """초매개변수를 담는 설정 클래스"""
    # 데이터
    test_size = 0.2
    val_size = 0.2  # From training set
    
    # 모델
    hidden_sizes = []  # Empty for linear, or [64, 32] for MLP
    
    # 학습
    batch_size = 128
    n_epochs = 200
    learning_rate = 0.01
    weight_decay = 0.001  # L2 regularization
    
    # 조기 종료
    patience = 15
    min_delta = 1e-4
    
    # 학습률 스케줄러
    use_scheduler = True
    scheduler_patience = 5
    scheduler_factor = 0.5
    
    # 경로
    checkpoint_dir = '/home/claude/pytorch_linear_regression_tutorial/checkpoints'
    log_dir = '/home/claude/pytorch_linear_regression_tutorial/logs'
    
    # 재현성
    random_seed = 42

config = Config()

# 재현성을 위해 씨앗을 설정한다
def set_seed(seed):
    """재현성을 위해 모든 난수 씨앗을 설정한다"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True  # 완전한 재현성을 원하면 주석을 푼다
    # torch.backends.cudnn.benchmark = False

set_seed(config.random_seed)
print(f"Random seed set to: {config.random_seed}")
print(f"Configuration loaded")

# 디렉터리를 만든다
os.makedirs(config.checkpoint_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)
print(f"Directories created")

# ============================================================================
# 2부: 데이터 준비
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: DATA PREPARATION")
print("=" * 70)

# 데이터를 불러온다
housing = fetch_california_housing()
X, y = housing.data, housing.target

print(f"Dataset: California Housing")
print(f"  Total samples: {len(X)}")
print(f"  Features: {X.shape[1]}")

# 학습/검증/시험으로 나눈다
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=config.test_size, random_state=config.random_seed
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=config.val_size, random_state=config.random_seed
)

print(f"\nData split:")
print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# 특징 스케일링
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# DataLoader들을 만든다
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled),
    torch.FloatTensor(y_train_scaled).reshape(-1, 1)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val_scaled),
    torch.FloatTensor(y_val_scaled).reshape(-1, 1)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test_scaled),
    torch.FloatTensor(y_test_scaled).reshape(-1, 1)
)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

print(f"\nDataLoaders created with batch_size={config.batch_size}")

# ============================================================================
# 3부: 모델 정의
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: MODEL DEFINITION")
print("=" * 70)

class RegressionModel(nn.Module):
    """유연한 회귀 모델"""
    
    def __init__(self, n_features, hidden_sizes=[]):
        super(RegressionModel, self).__init__()
        
        layers = []
        in_features = n_features
        
        # 은닉층
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        
        # 출력층
        layers.append(nn.Linear(in_features, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

model = RegressionModel(X_train.shape[1], config.hidden_sizes)
print(f"Model created:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# 4부: 학습 보조 함수들
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: TRAINING UTILITIES")
print("=" * 70)

class EarlyStopping:
    """과적합을 막기 위한 조기 종료"""
    
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop

class ModelCheckpoint:
    """가장 좋은 모델을 저장한다"""
    
    def __init__(self, filepath, mode='min'):
        self.filepath = filepath
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(self, model, val_loss):
        if self.mode == 'min':
            is_better = val_loss < self.best_score
        else:
            is_better = val_loss > self.best_score
        
        if is_better:
            self.best_score = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, self.filepath)
            return True
        return False

class Logger:
    """학습 지표를 기록한다"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        self.logs = []
    
    def log(self, epoch, train_loss, val_loss, lr):
        entry = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'learning_rate': float(lr)
        }
        self.logs.append(entry)
    
    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
checkpoint = ModelCheckpoint(
    os.path.join(config.checkpoint_dir, 'best_model.pth'),
    mode='min'
)
logger = Logger(config.log_dir)

print("Training utilities initialized:")
print(f"  Early stopping: patience={config.patience}")
print(f"  Model checkpointing: enabled")
print(f"  Logging: enabled")

# ============================================================================
# 5부: 학습 루프
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: TRAINING")
print("=" * 70)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

# 학습률 스케줄러
scheduler = None
if config.use_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
        verbose=True
    )

print(f"Optimizer: Adam (lr={config.learning_rate}, weight_decay={config.weight_decay})")
if scheduler:
    print(f"Scheduler: ReduceLROnPlateau")

# 학습 기록
history = {
    'train_loss': [],
    'val_loss': []
}

print(f"\nStarting training...")
print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'LR':<10} {'Best':<6}")
print("-" * 60)

for epoch in range(config.n_epochs):
    # 학습 단계
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # 검증 단계
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # 이력 저장
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    
    # 학습률 스케줄링
    if scheduler:
        scheduler.step(val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    # 체크포인트
    is_best = checkpoint(model, val_loss)
    
    # 기록
    logger.log(epoch + 1, train_loss, val_loss, current_lr)
    
    # 진행 상황 출력
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<6} {train_loss:<12.6f} {val_loss:<12.6f} {current_lr:<10.2e} {'✓' if is_best else '':<6}")
    
    # 조기 종료
    if early_stopping(val_loss):
        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
        break

logger.save()
print(f"\nTraining completed!")
print(f"  Logs saved to: {logger.log_file}")

# 가장 좋은 모델을 불러온다
best_checkpoint = torch.load(os.path.join(config.checkpoint_dir, 'best_model.pth'))
model.load_state_dict(best_checkpoint['model_state_dict'])
print(f"  Best model loaded (val_loss={best_checkpoint['val_loss']:.6f})")

# ============================================================================
# 6부: 평가
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: COMPREHENSIVE EVALUATION")
print("=" * 70)

def evaluate_model(model, loader, scaler_y, dataset_name=""):
    """종합적인 모델 평가"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            y_pred = model(batch_X)
            predictions.append(y_pred)
            targets.append(batch_y)
    
    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    
    # 원래 규모로 역변환한다
    predictions_orig = scaler_y.inverse_transform(predictions)
    targets_orig = scaler_y.inverse_transform(targets)
    
    # 지표를 계산한다
    mse = mean_squared_error(targets_orig, predictions_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_orig, predictions_orig)
    r2 = r2_score(targets_orig, predictions_orig)
    
    print(f"\n{dataset_name} Set Metrics:")
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MSE:       {mse:.4f}")
    print(f"  RMSE:      {rmse:.4f}")
    print(f"  MAE:       {mae:.4f} (${"%.2f" % (mae*100)}k)")
    
    return predictions_orig, targets_orig, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

# 모든 집합에서 평가한다
train_pred, train_true, train_metrics = evaluate_model(model, train_loader, scaler_y, "Train")
val_pred, val_true, val_metrics = evaluate_model(model, val_loader, scaler_y, "Validation")
test_pred, test_true, test_metrics = evaluate_model(model, test_loader, scaler_y, "Test")

# ============================================================================
# 7부: 시각화
# ============================================================================
print("\n" + "=" * 70)
print("PART 7: VISUALIZATION")
print("=" * 70)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 학습 기록
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training History')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# 2. 예측 대 실제 (시험)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(test_true, test_pred, alpha=0.5, s=20)
ax2.plot([test_true.min(), test_true.max()], [test_true.min(), test_true.max()], 
         'r--', lw=2, label='Perfect prediction')
ax2.set_xlabel('Actual Price ($100k)')
ax2.set_ylabel('Predicted Price ($100k)')
ax2.set_title(f'Test Set: Predictions vs Actual (R²={test_metrics["r2"]:.4f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 잔차 그림
ax3 = fig.add_subplot(gs[0, 2])
residuals = test_true - test_pred
ax3.scatter(test_pred, residuals, alpha=0.5, s=20)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Predicted Price ($100k)')
ax3.set_ylabel('Residuals')
ax3.set_title('Residual Plot (Test Set)')
ax3.grid(True, alpha=0.3)

# 4. 오차 분포
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Residual')
ax4.set_ylabel('Frequency')
ax4.set_title('Residual Distribution')
ax4.grid(True, alpha=0.3, axis='y')

# 5. 성능 비교
ax5 = fig.add_subplot(gs[1, 1])
datasets = ['Train', 'Val', 'Test']
r2_scores = [train_metrics['r2'], val_metrics['r2'], test_metrics['r2']]
colors = ['green', 'orange', 'blue']
bars = ax5.bar(datasets, r2_scores, color=colors, alpha=0.7)
ax5.set_ylabel('R² Score')
ax5.set_title('Model Performance Across Datasets')
ax5.set_ylim([0, 1])
ax5.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{score:.4f}', ha='center', va='bottom')

# 6. MAE 비교
ax6 = fig.add_subplot(gs[1, 2])
mae_scores = [train_metrics['mae']*100, val_metrics['mae']*100, test_metrics['mae']*100]
bars = ax6.bar(datasets, mae_scores, color=colors, alpha=0.7)
ax6.set_ylabel('MAE ($1000s)')
ax6.set_title('Mean Absolute Error')
ax6.grid(True, alpha=0.3, axis='y')

# 7. 학습 요약
ax7 = fig.add_subplot(gs[2, :])
summary = f"""
TRAINING SUMMARY

Configuration:
  - Model: {'Linear' if not config.hidden_sizes else f'MLP {config.hidden_sizes}'}
  - Optimizer: Adam (lr={config.learning_rate}, weight_decay={config.weight_decay})
  - Batch size: {config.batch_size}
  - Epochs: {len(history['train_loss'])}
  - Early stopping patience: {config.patience}

Final Metrics:
  Train - R²: {train_metrics['r2']:.4f}, MAE: ${train_metrics['mae']*100:.2f}k, RMSE: ${train_metrics['rmse']*100:.2f}k
  Val   - R²: {val_metrics['r2']:.4f}, MAE: ${val_metrics['mae']*100:.2f}k, RMSE: ${val_metrics['rmse']*100:.2f}k
  Test  - R²: {test_metrics['r2']:.4f}, MAE: ${test_metrics['mae']*100:.2f}k, RMSE: ${test_metrics['rmse']*100:.2f}k

Observations:
  - {"No significant overfitting" if abs(train_metrics['r2'] - test_metrics['r2']) < 0.05 else "Some overfitting detected"}
  - Best validation loss: {best_checkpoint['val_loss']:.6f}
  - Model saved to: {config.checkpoint_dir}/best_model.pth
"""
ax7.text(0.1, 0.9, summary, transform=ax7.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax7.axis('off')

plt.savefig('/home/claude/pytorch_linear_regression_tutorial/10_complete_pipeline_results.png', dpi=100, bbox_inches='tight')
print("Visualization saved")
plt.show()

print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)
print("""
CONGRATULATIONS! You've completed a production-ready ML pipeline!

This tutorial demonstrated:
✓ Configuration management
✓ Reproducibility (random seeds)
✓ Proper train/val/test split
✓ Feature scaling
✓ DataLoader for efficient batching
✓ Early stopping
✓ Model checkpointing
✓ Learning rate scheduling
✓ Comprehensive logging
✓ Multiple evaluation metrics
✓ Professional visualizations

다음 걸음:
1. Try with different models (add hidden layers)
2. Experiment with hyperparameters
3. Apply to your own datasets
4. Add GPU support (.to('cuda'))
5. Implement cross-validation
6. Add data augmentation (for image tasks)
7. Deploy the model

You now have a solid foundation for PyTorch ML projects!
""")


if __name__ == "__main__":
    pass
```

## 논의

재현성을 얻으려면 무작위성의 모든 원천에 씨앗을 설정해야 한다. PyTorch 연산에는 `torch.manual_seed()`, NumPy에는 `np.random.seed()`, GPU 연산에는 선택적으로 `torch.backends.cudnn.deterministic = True`를 쓴다. 설정 객체나 데이터클래스는 모든 초매개변수를 한곳에 모아 실험을 기록하고 재현하기 쉽게 해 준다. 시작할 때 체크포인트와 로그를 위한 출력 디렉터리를 만들어 두면 결과가 체계적으로 저장된다.

조기 종료는 검증 손실을 지켜보다가 더 이상 나아지지 않으면 학습을 멈춰 모델이 학습 데이터에 과적합하는 것을 막는다. 인내(patience) 매개변수는 멈추기 전에 개선이 없는 에폭을 몇 번까지 견딜지를 조절한다. 모델 체크포인트는 (검증 손실로 판단한) 가장 좋은 모델 가중치를 저장하므로, 최종 모델은 마지막으로 학습된 것이 아니라 가장 잘 일반화된 것이 된다. `ReduceLROnPlateau` 같은 학습률 일정은 진전이 멈추면 학습률을 자동으로 낮추어 최솟값 근처에서 세밀하게 최적화할 수 있게 한다.

시험 집합에 대한 종합 평가에는 여러 지표를 쓴다. 전체 설명 분산에는 $R^2$, 원래 단위의 오차에는 RMSE, 이상치에 강한 평균 오차 척도로는 MAE를 쓴다. 예측 대 실제값, 잔차 분포, 학습/검증/시험 집합 사이의 성능 비교를 그려 보면 모델이 제대로 작동하는지 눈으로 확인할 수 있다. 학습 지표와 시험 지표의 차이는 과적합 정도를 알려 준다. 차이가 작으면 일반화가 잘 된 것이다.

## 연습문제

**익힘 1.**
선형 모델을 2층 MLP로 바꾸도록 `Config` 클래스에 `hidden_sizes=[64, 32]` 설정을 추가하라. California Housing 데이터셋에서 선형 모델과 MLP의 시험 $R^2$을 비교하라.

??? success "익힘 1 풀이"
    ```python
    # Config 클래스에서 다음과 같이 설정한다:
    # hidden_sizes = [64, 32]
    # RegressionModel 클래스는 hidden_sizes 매개변수로 이미 이를 지원한다.
    # MLP로 학습하면 대체로 선형 모델(예: 0.60-0.65)보다 높은
    # R^2(예: 0.75-0.82)이 나온다. 집값 관계가
    # 비선형이기 때문이다.
    ```

---

**익힘 2.**
`ReduceLROnPlateau`의 목적을 설명하고, 고정된 학습률보다 크게 이득이 되는 상황을 서술하라.

??? success "익힘 2 풀이"
    ReduceLROnPlateau은 어떤 지표(보통 검증 손실)를 지켜보다가 지정한 에폭 수(인내) 동안 개선이 없으면 학습률에 어떤 인수(예: 0.5)를 곱해 줄인다. 이는 처음 학습률이 경사가 큰 초기 단계에는 알맞지만 최적점 근처에서 미세 조정하기에는 너무 클 때 도움이 된다. 스케줄러가 없으면 모델이 최솟값 주위를 진동한다. 스케줄러가 있으면 낮아진 학습률 덕분에 최적화기가 더 나은 해에 자리 잡는다. 구체적인 상황은 손실이 여러 규모에서 여러 번 정체되는 복잡한 데이터셋에서 모델을 학습시키는 경우이다.

---

**익힘 3.**
4개의 겹으로 학습하고 남은 겹으로 검증하는 과정을 5개 겹 전체에 대해 돌리도록 파이프라인을 수정하여 5겹 교차 검증을 구현하라. 겹들에 걸친 시험 $R^2$의 평균과 표준편차를 보고하라.

??? success "익힘 3 풀이"
    ```python
    import numpy as np
    from sklearn.model_selection import KFold
    
    # X, y가 numpy 배열이라고 가정한다
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        # ... (파이프라인에서처럼 표준화하고, 텐서를 만들고, 모델을 학습시킨다)
        # ... (검증 겹에서 R^2을 평가한다)
        # r2_scores.append(r2_val)
        print(f'Fold {fold+1}: R^2 = {r2_scores[-1]:.4f}')
    
    print(f'Mean R^2: {np.mean(r2_scores):.4f} +/- {np.std(r2_scores):.4f}')
    ```
