# 학습

자기 되돌이 시계열 모델을 위한 익히기 대본. 이 대본은 다음을 보인다.

자기 되돌이 모델은 앞선 모든 낱개를 조건으로 삼아 낱개마다 미리 헤아려 자료를 만든다. 이 단원은 자기 되돌이 모델 부품의 짜기를 보이며 차례대로 만들어 내는 과정과 그 얼개의 요구를 그려 보인다.

## 코드

```python
"""
자기 되돌이 시계열 모델을 위한 익히기 대본

이 대본은 다음을 보인다.
1. 인공 시계열 자료를 만든다
2. 자기 되돌이 모델 익히기에 맞게 다듬는다
3. 선형과 신경 자기 되돌이 모델을 모두 익힌다
4. 결과를 따지고 그려 본다
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========================================================================
# 메인
# ========================================================================

# 우리가 만든 단원을 들여온다
from ar_model import ARModel, NeuralARModel
from data import generate_sine_wave, generate_ar_process, create_sequences, train_test_split_temporal


def train_model(model: nn.Module, 
                X_train: torch.Tensor, 
                y_train: torch.Tensor,
                X_test: torch.Tensor,
                y_test: torch.Tensor,
                n_epochs: int = 100,
                learning_rate: float = 0.01,
                verbose: bool = True) -> dict:
    """
    평균 제곱 어긋남 손실로 자기 되돌이 모델을 익힌다.
    
    인수:
        model: 익힐 자기 되돌이 모델(ARModel이나 NeuralARModel)
        X_train: 익히기 들임 차례
        y_train: 익히기 목표 값
        X_test: 시험 들임 차례
        y_test: 시험 목표 값
        n_epochs: 익히기 바퀴 수
        learning_rate: 최적화기의 학습률
        verbose: 나아감 막대를 보일지 여부
        
    반환값:
        익히기 지난 일(손실)을 담은 사전
    """
    
    # 손실 함수: 평균제곱오차
    # 평균 제곱 어긋남 = (1/n) * Σ(헤아린 값 - 실제 값)²
    criterion = nn.MSELoss()
    
    # 가장 좋게 하개: Adam(맞추어 가는 배움 빠르기)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 그리려고 손실을 담아 둔다
    train_losses = []
    test_losses = []
    
    # 학습 루프
    iterator = tqdm(range(n_epochs), desc="Training") if verbose else range(n_epochs)
    
    for epoch in iterator:
        # ==================== 익히기 단계 ====================
        model.train()  # 모델을 학습 모드로
        
        # 앞먹임: 예측을 셈한다
        train_predictions = model(X_train)
        
        # 손실을 셈한다: 헤아린 값이 참값에서 얼마나 먼가?
        train_loss = criterion(train_predictions, y_train)
        
        # 역전파: 경사를 계산한다
        optimizer.zero_grad()  # 이전 기울기 지우기
        train_loss.backward()  # 새 기울기 계산
        
        # 모델 매개변수를 새로 고친다
        optimizer.step()
        
        # ==================== 따지기 단계 ====================
        model.eval()  # 모델을 평가 모드로
        
        with torch.no_grad():  # 따질 때는 기울기를 셈하지 않는다
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions, y_test)
        
        # 손실 담기
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        
        # 진행 막대를 고친다
        if verbose and epoch % 10 == 0:
            iterator.set_postfix({
                'train_loss': f'{train_loss.item():.4f}',
                'test_loss': f'{test_loss.item():.4f}'
            })
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses
    }


def visualize_results(data: np.ndarray,
                     model: nn.Module,
                     X_test: torch.Tensor,
                     y_test: torch.Tensor,
                     sequence_length: int,
                     train_size: int,
                     n_forecast: int = 50,
                     title: str = "AR Model Results"):
    """
    모델 솜씨를 두루 그려 본다.
    
    인수:
        data: 본디 시계열 자료
        model: 익힌 자기 되돌이 모델
        X_test: 시험 들임 차례
        y_test: 시험 목표 값
        sequence_length: 들임 차례의 길이
        train_size: 익히기 표본의 수
        n_forecast: 앞날로 내다볼 걸음의 수
        title: 그림의 제목
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # ==================== 그림 1: 헤아린 값과 실제 값 ====================
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test).numpy().flatten()
    
    actual_values = y_test.numpy().flatten()
    
    axes[0].plot(actual_values, label='Actual', linewidth=2, alpha=0.7)
    axes[0].plot(test_predictions, label='Predicted', linewidth=2, alpha=0.7)
    axes[0].set_title(f"{title}: Test Set Predictions")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ==================== 그림 2: 앞날 내다보기 ====================
    # 시험 묶음의 마지막 차례를 출발점으로 삼는다
    initial_sequence = X_test[-1]
    
    # 앞날 헤아림을 만든다
    future_predictions = model.predict_sequence(initial_sequence, n_steps=n_forecast)
    
    # 본디 자료와 내다보기를 그린다
    forecast_start = len(data) - n_forecast
    
    axes[1].plot(range(len(data)), data, label='Historical Data', 
                linewidth=2, alpha=0.7, color='blue')
    axes[1].plot(range(forecast_start, forecast_start + n_forecast), 
                future_predictions, label='Forecast', 
                linewidth=2, alpha=0.7, color='red', linestyle='--')
    
    # 지난 일과 내다보기를 가르는 세로선을 더한다
    axes[1].axvline(x=forecast_start, color='gray', linestyle=':', 
                   linewidth=2, label='Forecast Start')
    
    axes[1].set_title(f"{title}: Future Forecasting")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """
    으뜸 익히기 물길
    """
    print("=" * 70)
    print("Autoregressive Time Series Model Training")
    print("=" * 70)
    
    # ==================== 웃매개변수 ====================
    SEQUENCE_LENGTH = 10  # AR(10) - 지난 값 10개를 쓴다
    N_SAMPLES = 1000      # 때 점 1000개를 만든다
    TRAIN_RATIO = 0.8     # 익히기 80%, 시험 20%
    N_EPOCHS = 200        # 익히기 바퀴 수
    LEARNING_RATE = 0.01  # 배움 비율
    
    print(f"\nHyperparameters:")
    print(f"  Sequence Length (AR order): {SEQUENCE_LENGTH}")
    print(f"  Number of samples: {N_SAMPLES}")
    print(f"  Train/Test split: {TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # ==================== 자료 만들기 ====================
    print(f"\n{'='*70}")
    print("Step 1: Generating synthetic time series data...")
    print(f"{'='*70}")
    
    # 어떤 갈래의 자료를 쓸지 고를 수 있다:
    # 고르기 1: 잡음 섞인 사인 물결
    data = generate_sine_wave(n_samples=N_SAMPLES, frequency=0.05, noise_std=0.2)
    data_name = "Sine Wave"
    
    # 고르기 2: 참 자기 되돌이 과정(대신 쓰려면 주석을 푼다)
    # data = generate_ar_process(n_samples=N_SAMPLES,
    #                            coefficients=[0.7, -0.3, 0.1],
    #                            noise_std=0.3)
    # data_name = "AR(3) 과정"
    
    print(f"✓ Generated {len(data)} data points ({data_name})")
    
    # ==================== 자료 준비 ====================
    print(f"\n{'='*70}")
    print("Step 2: Preparing sequences for training...")
    print(f"{'='*70}")
    
    X, y = create_sequences(data, sequence_length=SEQUENCE_LENGTH)
    print(f"✓ Created {len(X)} sequences")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, TRAIN_RATIO)
    print(f"✓ Split into train/test:")
    print(f"  Train: {len(X_train)} sequences")
    print(f"  Test: {len(X_test)} sequences")
    
    # ==================== 선형 자기 되돌이 모델 익히기 ====================
    print(f"\n{'='*70}")
    print("Step 3a: Training Linear AR Model...")
    print(f"{'='*70}")
    
    linear_model = ARModel(order=SEQUENCE_LENGTH)
    
    # 매개변수 개수 세기
    n_params = sum(p.numel() for p in linear_model.parameters())
    print(f"Model has {n_params} parameters")
    
    linear_history = train_model(
        linear_model, X_train, y_train, X_test, y_test,
        n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Final train loss: {linear_history['train_losses'][-1]:.4f}")
    print(f"  Final test loss: {linear_history['test_losses'][-1]:.4f}")
    
    # 배운 계수를 보인다
    coeffs = linear_model.get_coefficients()
    print(f"\nLearned AR coefficients:")
    for i, coef in enumerate(coeffs['coefficients']):
        print(f"  φ_{i+1} = {coef:.4f}")
    print(f"  Constant c = {coeffs['constant']:.4f}")
    
    # ==================== 신경 자기 되돌이 모델 익히기 ====================
    print(f"\n{'='*70}")
    print("Step 3b: Training Neural AR Model...")
    print(f"{'='*70}")
    
    neural_model = NeuralARModel(order=SEQUENCE_LENGTH, hidden_size=64)
    
    n_params = sum(p.numel() for p in neural_model.parameters())
    print(f"Model has {n_params} parameters")
    
    neural_history = train_model(
        neural_model, X_train, y_train, X_test, y_test,
        n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Final train loss: {neural_history['train_losses'][-1]:.4f}")
    print(f"  Final test loss: {neural_history['test_losses'][-1]:.4f}")
    
    # ==================== 결과 그려 보기 ====================
    print(f"\n{'='*70}")
    print("Step 4: Creating visualizations...")
    print(f"{'='*70}")
    
    # 익히기 곡선을 그린다
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(linear_history['train_losses'], label='Train', alpha=0.7)
    plt.plot(linear_history['test_losses'], label='Test', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Linear AR Model: Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(neural_history['train_losses'], label='Train', alpha=0.7)
    plt.plot(neural_history['test_losses'], label='Test', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Neural AR Model: Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("✓ Saved training_curves.png")
    
    # 선형 모델의 헤아림을 그린다
    fig = visualize_results(
        data, linear_model, X_test, y_test, 
        SEQUENCE_LENGTH, len(X_train),
        n_forecast=100, title="Linear AR Model"
    )
    plt.savefig('linear_ar_results.png', dpi=150)
    print("✓ Saved linear_ar_results.png")
    
    # 신경 모델의 헤아림을 그린다
    fig = visualize_results(
        data, neural_model, X_test, y_test,
        SEQUENCE_LENGTH, len(X_train),
        n_forecast=100, title="Neural AR Model"
    )
    plt.savefig('neural_ar_results.png', dpi=150)
    print("✓ Saved neural_ar_results.png")
    
    # ==================== 간추리기 ====================
    print(f"\n{'='*70}")
    print("Training Complete! Summary:")
    print(f"{'='*70}")
    print(f"\nLinear AR Model:")
    print(f"  Test MSE: {linear_history['test_losses'][-1]:.4f}")
    print(f"\nNeural AR Model:")
    print(f"  Test MSE: {neural_history['test_losses'][-1]:.4f}")
    
    if neural_history['test_losses'][-1] < linear_history['test_losses'][-1]:
        print(f"\n✓ Neural model performed better (lower test loss)")
    else:
        print(f"\n✓ Linear model performed better (lower test loss)")
    
    print(f"\nCheck the generated PNG files for visualizations!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()```

## 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 결은 더 복잡한 경우로 자연스럽게 넓혀진다. 웃매개변수와 얼개 변형, 여러 자료 묶음을 실험하면 만들어 내는 모델 일에 대한 이해가 깊어지고 실제 직관이 쌓인다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.
