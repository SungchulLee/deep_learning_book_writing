# 조기 종료

조기 종료는 검증 집합에서의 모델 성능이 더 나아지지 않을 때 학습을 멈추는 정칙화 기법이다. 모델이 일반화되는 양상은 배웠지만 아직 학습 데이터의 잡음을 외우기 시작하지는 않은 지점을 찾아내어 과적합을 막는다.

!!! note "함께 볼 것"
    학습률 스케줄링과의 상호작용을 포함하여 PyTorch 학습 루프에서 조기 종료를 쓰는 간결한 실용 안내는 **5.6절 과적합과 일반화**를 보라.

---

## 1. 개념적 토대

### 과적합의 진행 경로

학습 중에 모델은 대체로 예측 가능한 경로를 따른다.

1. **초기 단계**: 학습 손실과 검증 손실이 모두 빠르게 줄어든다
2. **학습 단계**: 학습 손실은 계속 줄지만 검증 손실은 더 느리게 준다
3. **과적합 단계**: 학습 손실은 줄지만 검증 손실은 늘어난다

조기 종료는 2단계와 3단계 사이의 전환점을 찾아낸다.

### 암묵적 정칙화

조기 종료는 다음을 통해 암묵적 정칙화 장치로 작동한다.

- 최적화 반복 횟수를 제한하여 모델의 복잡도를 제한한다
- 가중치가 극단적인 값에 이르는 것을 막는다
- 모델을 초기화에 더 가까운 매개변수 공간의 영역에 머무르게 한다

선형 모델에서 경사 하강법과 함께 쓰는 조기 종료는 수학적으로 L2 정칙화와 동등하며, 실효 정칙화 강도는 반복 횟수에 반비례한다.

---

## 2. 수학적 정식화

### 검증에 기반한 종료 기준

에포크 $t$에서의 검증 손실을 $\mathcal{L}_{\text{val}}^{(t)}$이라 하자. 기본 종료 기준은 다음과 같다.

$$
\text{Stop if } \mathcal{L}_{\text{val}}^{(t)} > \mathcal{L}_{\text{val}}^{(t-1)} \text{ for } k \text{ consecutive epochs}
$$

여기서 $k$은 **인내(patience)** 매개변수이다.

### 최적 모델의 선택

가장 좋은 검증 성능을 기록해 둔다.

$$
t^* = \arg\min_{t \leq T} \mathcal{L}_{\text{val}}^{(t)}
$$

마지막 매개변수 $\theta^{(T)}$이 아니라 매개변수 $\theta^{(t^*)}$을 돌려준다.

### 일반화 한계의 관점

조기 종료는 암묵적 정칙화를 제공한다. 경사 하강법을 쓰는 선형 회귀에서 반복 $t$에서 멈추는 것은 다음을 쓰는 능선 회귀와 동등하다.

$$
\lambda_{\text{eff}} \approx \frac{1}{\eta t}
$$

여기서 $\eta$은 학습률이다.

---

## 3. PyTorch 구현

### 기본적인 조기 종료

```python
import torch
import numpy as np
from typing import Optional
import copy

class EarlyStopping:
    """
    검증 손실이 더 나아지지 않으면 학습을 멈추는 조기 종료.
    
    인수:
        patience: 마지막 개선 뒤 기다릴 에포크 수
        min_delta: 개선으로 볼 최소 변화량
        mode: 손실이면 'min'(작을수록 좋다), 정확도면 'max'
        restore_best_weights: 가장 좋았던 가중치로 되돌릴지 여부
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta
    
    def __call__(self, score: float, model: torch.nn.Module, epoch: int) -> bool:
        """
        학습을 멈춰야 하는지 확인한다.
        
        인수:
            score: 현재 검증 점수
            model: 저장할 수도 있는 모델
            epoch: 현재 에포크 번호
            
        반환값:
            학습을 멈춰야 하면 True
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False
    
    def get_best_score(self) -> Optional[float]:
        return self.best_score
    
    def get_best_epoch(self) -> int:
        return self.best_epoch
```

### 조기 종료를 쓰는 학습 루프

```python
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    max_epochs: int = 1000,
    patience: int = 20,
    min_delta: float = 1e-4,
    verbose: bool = True
) -> dict:
    """
    조기 종료와 함께 모델을 학습시킨다.
    
    인수:
        model: 신경망
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        optimizer: 최적화기
        max_epochs: 최대 학습 에포크 수
        patience: 조기 종료의 인내
        min_delta: 개선으로 볼 최소 문턱값
        verbose: 진행 상황 출력 여부
        
    반환값:
        학습 이력
    """
    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode='min',
        restore_best_weights=True
    )
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(max_epochs):
        # 학습 단계
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # 검증 단계
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # 이력 기록
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # 조기 종료 여부를 확인한다
        if early_stopping(val_loss, model, epoch):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
                print(f"Best epoch: {early_stopping.get_best_epoch()+1} "
                      f"with val_loss: {early_stopping.get_best_score():.4f}")
            break
    
    return history
```

### 여러 기준을 쓰는 고급 조기 종료

```python
class MultiMetricEarlyStopping:
    """
    여러 지표에 기반한 조기 종료.
    
    감시하는 모든 지표가 더 나아지지 않으면 멈춘다.
    """
    
    def __init__(
        self,
        metrics_config: dict,
        patience: int = 10,
        restore_best_weights: bool = True
    ):
        """
        인수:
            metrics_config: 지표 이름을 'min' 또는 'max'에 대응시키는 사전
                           예: {'val_loss': 'min', 'val_acc': 'max'}
            patience: 지표마다의 인내
            restore_best_weights: 가장 좋았던 가중치로 되돌릴지 여부
        """
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        
        self.metrics_config = metrics_config
        self.best_scores = {name: None for name in metrics_config}
        self.counters = {name: 0 for name in metrics_config}
        self.best_weights = None
        self.best_epoch = 0
    
    def _is_better(self, name: str, current: float, best: float) -> bool:
        mode = self.metrics_config[name]
        if mode == 'min':
            return current < best
        return current > best
    
    def __call__(self, metrics: dict, model: nn.Module, epoch: int) -> bool:
        """
        학습을 멈춰야 하는지 확인한다.
        
        인수:
            metrics: 현재 지표 값의 사전
            model: 저장할 모델
            epoch: 현재 에포크
            
        반환값:
            멈춰야 하면 True
        """
        any_improved = False
        
        for name in self.metrics_config:
            current = metrics[name]
            best = self.best_scores[name]
            
            if best is None or self._is_better(name, current, best):
                self.best_scores[name] = current
                self.counters[name] = 0
                any_improved = True
            else:
                self.counters[name] += 1
        
        if any_improved:
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        
        # 모든 지표가 인내를 넘겼으면 멈춘다
        all_exceeded = all(c >= self.patience for c in self.counters.values())
        
        if all_exceeded and self.restore_best_weights:
            model.load_state_dict(self.best_weights)
        
        return all_exceeded
```

### 학습률 스케줄링과 함께 쓰는 조기 종료

```python
class EarlyStoppingWithLRScheduler:
    """
    조기 종료와 정체 시 학습률 감소를 결합한다.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        patience: int = 20,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        min_lr: float = 1e-7,
        min_delta: float = 1e-4
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.lr_counter = 0
        self.num_lr_reductions = 0
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            return False
        
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.lr_counter = 0
        else:
            self.counter += 1
            self.lr_counter += 1
            
            # 정체되면 학습률을 줄인다
            if self.lr_counter >= self.lr_patience:
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * self.lr_factor, self.min_lr)
                
                if new_lr < current_lr:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"Reducing LR to {new_lr:.2e}")
                    self.num_lr_reductions += 1
                
                self.lr_counter = 0
        
        # 인내를 넘겼으면 멈춘다
        if self.counter >= self.patience:
            model.load_state_dict(self.best_weights)
            return True
        
        return False
```

---

## 4. 초매개변수에 대한 고려

### 인내 값 고르기

인내 매개변수는 다음 사이의 절충을 조절한다.

- **너무 작으면**: 일시적인 요동 중에 너무 일찍 멈출 수 있다
- **너무 크면**: 계산을 낭비하고 과적합의 위험을 안는다

지침은 다음과 같다.

- 인내 = 10~20 에포크에서 시작한다
- 검증 지표가 잡음이 많으면 늘린다
- 학습 곡선이 매끄럽고 예측 가능하면 줄인다

```python
def analyze_optimal_patience(history: dict, test_patience_values: list):
    """
    인내 값에 따라 어느 지점에서 멈추게 되는지 분석한다.
    
    인수:
        history: 'val_loss'를 담은 학습 이력
        test_patience_values: 시험해 볼 인내 값의 목록
        
    반환값:
        인내를 종료 에포크와 최적 val_loss에 대응시키는 사전
    """
    # 이미 끝난 학습의 이력을 두고 "참을성을 다르게 잡았다면 어디서
    # 멈췄을까"를 되짚어 본다. 학습을 다시 돌릴 필요가 없으므로
    # 참을성을 고르는 데 드는 비용이 사실상 0이다.
    val_losses = history['val_loss']
    results = {}

    for patience in test_patience_values:
        best_loss = float('inf')
        best_epoch = 0
        counter = 0   # 나아지지 않은 채 지난 에폭 수

        for epoch, loss in enumerate(val_losses):
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                counter = 0        # 나아졌으니 참을성을 되돌린다
            else:
                counter += 1

            if counter >= patience:
                break              # 여기서 조기 종료가 걸렸을 것이다

        # stop_epoch 와 best_epoch 를 함께 담는 것이 요점이다.
        # 둘의 차이가 곧 "헛되이 더 돌린 에폭 수"이며, 참을성을 키울수록
        # 이 간격이 벌어진다. 반대로 참을성이 너무 작으면 잠깐 나빠졌다가
        # 다시 좋아지는 구간을 못 넘기고 일찍 끊겨 best_val_loss가 나빠진다.
        #
        # 주의: for가 break 없이 끝나면 epoch는 마지막 값으로 남는다.
        # 곧 이 경우 stop_epoch는 "멈춘 곳"이 아니라 "끝까지 갔다"는 뜻이다
        results[patience] = {
            'stop_epoch': epoch,
            'best_epoch': best_epoch,
            'best_val_loss': best_loss
        }

    return results
```

### 감시할 지표 고르기

| 지표 | 쓸 때 |
|--------|-------------|
| 검증 손실 | 기본 선택. 일반화를 직접 잰다 |
| 검증 정확도 | 정확도가 주된 목표일 때 |
| F1 점수 | 불균형 분류일 때 |
| 사용자 정의 지표 | 분야에 특화된 요구가 있을 때 |

### 최소 개선폭 고르기

`min_delta` 매개변수는 무엇을 개선으로 볼지 정한다.

```python
# 손실값이 대략 0.1~1.0인 보통의 경우
min_delta = 1e-4  # 기본값

# 손실값이 아주 작은 경우 (< 0.01)
min_delta = 1e-5

# 검증 지표에 잡음이 많은 경우
min_delta = 1e-3  # 더 너그럽게
```

---

## 5. 이론적 분석

### L2 정칙화와의 관계

선형 회귀에 경사 하강법을 쓸 때 반복 $t$에서 멈추면 다음을 얻는다.

$$
\hat{w}_t = \sum_{i=1}^{t} (I - \eta X^T X)^{i-1} \eta X^T y
$$

$t \to \infty$일 때 이는 능선 회귀의 해로 수렴한다.

$$
\hat{w}_\infty = (X^T X)^{-1} X^T y
$$

실효 정칙화는 대략 다음과 같다.

$$
\hat{w}_t \approx (X^T X + \frac{1}{\eta t} I)^{-1} X^T y
$$

### 편향-분산 절충

조기 종료는 편향-분산 분해에 영향을 준다.

- **일찍 멈추면($t$이 작으면)**: 편향이 크고 분산이 작다
- **늦게 멈추면($t$이 크면)**: 편향이 작고 분산이 크다

최적의 종료 지점은 전체 일반화 오차를 최소화한다.

---

## 6. 실무 지침

### 조기 종료를 쓸 때

1. **언제나**: 사실상 공짜이고 도움이 될 때가 많다
2. **계산 자원이 적을 때**: 불필요한 학습 시간을 아낀다
3. **적절한 학습 길이를 모를 때**: 최적 에포크 수를 모를 때
4. **과적합이 보일 때**: 학습 성능과 검증 성능에 차이가 있을 때

### 좋은 관행

```python
def recommended_training_setup(model, train_loader, val_loader):
    """
    조기 종료를 다른 기법과 결합한 권장 설정.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 조기 종료
    early_stopping = EarlyStopping(
        patience=20,
        min_delta=1e-4,
        restore_best_weights=True
    )
    
    max_epochs = 500  # 상계
    
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)
        
        # 스케줄러 갱신
        scheduler.step(val_loss)
        
        # 조기 종료 여부를 확인한다
        if early_stopping(val_loss, model, epoch):
            print(f"Stopped at epoch {epoch+1}")
            break
    
    return model
```

### 검사점 저장

조기 종료와 함께 언제나 검사점을 저장하라.

```python
class CheckpointingEarlyStopping(EarlyStopping):
    """주기적인 검사점 저장과 함께 쓰는 조기 종료."""
    
    def __init__(self, checkpoint_dir: str = './checkpoints', 
                 checkpoint_freq: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def __call__(self, score, model, epoch):
        # 두 가지 저장이 서로 다른 목적을 가진다.
        #   (1) 주기적 검사점 — 학습이 죽었을 때 이어서 돌리기 위한 것
        #   (2) 최고 성적 모델 — 배포할 모델을 고르기 위한 것
        # 둘을 헷갈리면 마지막 검사점을 배포하는 실수를 하게 되는데,
        # 조기 종료가 걸린 시점의 모델은 이미 나빠지고 있던 모델이다.

        # (1) 주기적인 검사점 저장.
        # 파일 이름에 에폭을 넣어 덮어쓰지 않는다. 다만 이대로 두면
        # 파일이 계속 쌓이므로 실제로는 오래된 것을 지우는 손질이 필요하다
        if (epoch + 1) % self.checkpoint_freq == 0:
            path = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'score': score
            }, path)
            # 학습을 이어 가려면 옵티마이저 상태도 함께 담아야 한다.
            # Adam의 모멘텀 통계가 사라지면 이어받은 학습이 흔들린다

        # 부모 클래스가 최고 성적 갱신과 참을성 세기를 맡는다.
        # 성적이 나아졌으면 self.best_weights를 여기서 새로 채운다
        should_stop = super().__call__(score, model, epoch)

        # (2) 최고 성적 모델 저장. 매번 덮어쓰므로 파일은 늘 하나다.
        # 이 파일이 곧 배포 후보이며, epoch가 아니라 best_epoch를 담는다
        if self.best_weights is not None:
            path = f"{self.checkpoint_dir}/best_model.pt"
            torch.save({
                'epoch': self.best_epoch,
                'model_state_dict': self.best_weights,
                'score': self.best_score
            }, path)

        return should_stop
```

---

## 7. 시각화

```python
import matplotlib.pyplot as plt

def plot_training_with_early_stopping(history: dict, best_epoch: int):
    """
    조기 종료 지점과 함께 학습 진행을 시각화한다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 1부터 세는 까닭은 사람이 읽을 그림이기 때문이다.
    # 코드 안의 인덱스는 0부터이므로 아래에서 best_epoch에 1을 더한다
    epochs = range(1, len(history['train_loss']) + 1)

    # ── 왼쪽: 손실 곡선 ─────────────────────────────────────────────
    # 훈련과 검증을 반드시 함께 그린다. 검증 손실만 보면 왜 나빠지는지
    # 알 수 없는데, 훈련 손실이 계속 내려가는 동안 검증이 올라가면
    # 그것이 과적합의 그림이다
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss')

    # 세로 점선이 되돌릴 지점이다. 조기 종료가 걸린 곳이 아니라
    # 검증 손실이 가장 낮았던 곳을 표시해야 한다. 둘 사이의 간격이
    # 참을성만큼 벌어져 있는 것이 정상이다
    axes[0].axvline(best_epoch + 1, color='r', linestyle='--', 
                    label=f'Best Epoch ({best_epoch + 1})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── 오른쪽: 정확도 곡선(이력에 있을 때만) ───────────────────────
    # 손실과 정확도의 최적 지점이 어긋나는 일이 흔하다. 손실은
    # 확신의 정도까지 재지만 정확도는 맞고 틀림만 세기 때문이다.
    # 어느 쪽으로 조기 종료를 걸지는 실제로 무엇을 중히 여기는지에 달렸다
    if 'train_acc' in history:
        axes[1].plot(epochs, history['train_acc'], label='Train Acc')
        axes[1].plot(epochs, history['val_acc'], label='Val Acc')
        axes[1].axvline(best_epoch + 1, color='r', linestyle='--',
                        label=f'Best Epoch ({best_epoch + 1})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    # show()가 아니라 fig를 돌려준다. 부르는 쪽에서 저장할지 띄울지
    # 정할 수 있고, 시험 코드에서 그림을 띄우지 않고 검사할 수도 있다
    return fig
```

---

## 8. 흔히 빠지는 함정

1. **검증 집합을 쓰지 않기**: 조기 종료에는 따로 떼어 둔 데이터가 필요하다
2. **인내가 너무 작음**: 정상적인 요동 중에 멈춘다
3. **최적 가중치를 되돌리지 않기**: 최적 가중치 대신 마지막 가중치를 쓴다
4. **지표를 무시하기**: 과제에 맞지 않는 지표를 쓴다
5. **데이터 유출**: 검증 집합이 학습 데이터에 오염된다

---

## 연습문제

**연습문제 1.**
조기 종료가 정칙화의 한 형태로 작동하는 이유를 설명하라.

??? success "연습문제 1 풀이"
    조기 종료는 경사 하강 단계의 실효 횟수를 제한하고, 이는 가중치가 초기화에서 얼마나 멀리 움직일 수 있는지를 제한한다. 초기화는 보통 영에 가까우므로(L2 공) 이는 가중치의 L2 노름을 암묵적으로 제약한다. 형식적으로, 이차 손실에서 $t$ 단계 뒤의 조기 종료는 $\lambda \approx 1/(\eta t)$인 L2 정칙화와 동등하다.

---

**연습문제 2.**
PyTorch 학습 루프에서 인내 매개변수를 갖는 조기 종료를 구현하라.

??? success "연습문제 2 풀이"
    ```python
    best_loss, patience_counter = float('inf'), 0
    for epoch in range(max_epochs):
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    model.load_state_dict(torch.load('best.pt'))
    ```

---

**연습문제 3.**
조기 종료와 편향-분산 절충의 관계는 무엇인가?

??? success "연습문제 3 풀이"
    학습 에포크가 많아지면 편향은 줄지만(모델이 학습 데이터에 더 잘 맞는다) 분산은 커진다(모델이 학습 데이터의 잡음에 더 민감해진다). 조기 종료는 편향과 분산의 합이 최소가 되는 에포크를 찾으며, 이는 보통 검증 손실이 늘기 시작하는 지점이다.

---

**연습문제 4.**
회귀 과제에서 조기 종료와 L2 정칙화를 비교하라. 둘은 비슷한 시험 성능을 내는가?

??? success "연습문제 4 풀이"
    이차 손실을 쓰는 선형 모델에서는 동등함이 증명되어 있다(Bishop, 1995). 심층 신경망에서는 근사적으로 비슷하지만 같지는 않다. 조기 종료는 L2와 달리 학습률 스케줄이나 적응형 최적화기와 상호작용한다. 실무에서는 둘을 함께 쓰는 것이 가장 잘 통할 때가 많다.

## 정리하며

이 마당은 개념적 토대、수학적 정식화、PyTorch 구현、초매개변수에 대한 고려을 차례로 짚었다.

**참고 문헌**

1. Prechelt, L. (1998). Early Stopping - But When? *Neural Networks: Tricks of the Trade*, 55-69.
2. Yao, Y., Rosasco, L., & Caponnetto, A. (2007). On Early Stopping in Gradient Descent Learning. *Constructive Approximation*, 26(2), 289-315.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 7.
