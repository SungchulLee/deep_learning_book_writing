# 자기회귀 시계열 예측기의 학습

이 예제는 잡음이 있는 사인파에서 선형 AR 예측기와 비선형 뉴럴 넷트웍 예측기를 평균제곱오차로 학습하는 절차다. 다음 값 한 개의 점 예측을 비교하며 시계열의 확률밀도나 예측 구간을 학습하는 코드는 아니다. 한 스텝 테스트와 재귀 롤아웃 그림도 서로 다른 평가다.

## 1. 학습 코드

```python
"""
자기 회귀 시계열 모델을 위한 익히기 대본

이 대본은 다음을 보인다.
1. 인공 시계열 자료를 만든다
2. 자기 회귀 모델 익히기에 맞게 다듬는다
3. 선형과 뉴럴 자기 회귀 모델을 모두 익힌다
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
from typing import Optional

# --- ar_model ---
"""
시계열을 위한 자기 되돌이 모델

이 단원은 PyTorch로 단순한 AR(p) 모델을 짠다.
모델은 지난 p개 값을 바탕으로 시계열의 다음 값을 헤아리는 법을 배운다.
"""


# ========================================================================
# 메인
# ========================================================================


class ARModel(nn.Module):
    """
    시계열 헤아리기를 위한 자기 되돌이 모델 AR(p).
    
    이는 다음을 헤아리는 단순한 선형 자기 되돌이 모델이다.
    X_t = c + φ₁*X_{t-1} + φ₂*X_{t-2} + ... + φₚ*X_{t-p}
    
    신경망 말로 하면 이는 깨움 함수가 없는 선형 층일 뿐이다.
    
    구조:
        들임: [배치 크기, 차례 길이] - 지난 p개 값
        내놓기: [배치 크기, 1] - 헤아린 다음 값
    """
    
    def __init__(self, order: int):
        """
        자기 되돌이 모델을 첫자리매김한다.
        
        인수:
            order: 자기 되돌이 모델의 차수 p(지난 값을 몇 개 쓸지)
        """
        super(ARModel, self).__init__()
        
        self.order = order
        
        # 선형 층: 계수 φ₁, φ₂, ..., φₚ과 상수 c을 배운다
        # 들임 차원: 차수(지난 값 p개)
        # 내놓기 차원: 1(다음 값)
        # bias=True는 상수항 c을 배운다는 뜻이다
        self.linear = nn.Linear(order, 1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        앞먹임: 지난 값으로 다음 값을 헤아린다.
        
        인수:
            x: 꼴 [배치 크기, order]인 들임 텐서
               표본마다 지난 'order'개 값을 담는다
               
        반환값:
            헤아린 다음 값, 꼴 [배치 크기, 1]
        """
        # 단순한 선형 바꿈
        # 내놓기 = w₁*x₁ + w₂*x₂ + ... + wₚ*xₚ + b
        return self.linear(x)
    
    def get_coefficients(self) -> dict:
        """
        배운 자기 되돌이 계수를 뽑아낸다.
        
        반환값:
            다음을 담은 사전:
                - 'coefficients': 배운 φ 값 [φ₁, φ₂, ..., φₚ]
                - 'constant': 배운 상수항 c
        """
        # 선형 층에서 무게와 치우침을 얻는다
        weights = self.linear.weight.data.cpu().numpy().flatten()
        bias = self.linear.bias.data.cpu().numpy()[0]
        
        return {
            'coefficients': weights,
            'constant': bias
        }
    
    def predict_sequence(self, 
                        initial_sequence: torch.Tensor, 
                        n_steps: int) -> np.ndarray:
        """
        앞날 헤아림을 자기 되돌이로 만든다.
        
        이 함수는 다음으로 앞날 여러 걸음을 헤아린다.
        1. 아는 지난 값으로 다음 값을 헤아린다
        2. 헤아린 값을 차례에 더한다
        3. 가장 최근 값(헤아린 값 포함)으로 다음을 헤아린다
        4. 걸음 2-3을 되풀이한다
        
        인수:
            initial_sequence: 출발 값을 담은 꼴 [order]인 텐서
            n_steps: 앞날로 헤아릴 걸음의 수
            
        반환값:
            헤아린 값을 담은 길이 n_steps인 넘파이 배열
        """
        self.eval()  # 따지기 모드로 둔다
        
        # 헤아림을 담아 둔다
        predictions = []
        
        # 지금 차례(미끄러지는 창)
        current_seq = initial_sequence.clone().unsqueeze(0)  # 배치 차원을 더한다
        
        with torch.no_grad():  # 기울기 셈이 필요 없다
            for _ in range(n_steps):
                # 다음 값을 헤아린다
                pred = self.forward(current_seq)
                predictions.append(pred.item())
                
                # 차례를 새로 고친다: 가장 오래된 것을 빼고 가장 새 헤아림을 더한다
                # 왼쪽으로 밀고 끝에 새 헤아림을 더한다
                current_seq = torch.cat([current_seq[:, 1:], pred], dim=1)
        
        return np.array(predictions)


class NeuralARModel(nn.Module):
    """
    신경 자기 되돌이 모델 - AR(p)의 비선형 넓힘.
    
    단순한 선형 모델 대신 작은 신경망을 써서
    시계열의 비선형 관계를 담는다.
    
    구조:
        들임 -> 숨은 층(ReLU) -> 숨은 층(ReLU) -> 내놓기
    """
    
    def __init__(self, 
                 order: int, 
                 hidden_size: int = 64):
        """
        신경 자기 되돌이 모델을 첫자리매김한다.
        
        인수:
            order: 들임으로 쓸 지난 값의 수
            hidden_size: 숨은 층의 신경 세포 수
        """
        super(NeuralARModel, self).__init__()
        
        self.order = order
        
        # 여러 층 신경망
        self.network = nn.Sequential(
            # 첫 번째 은닉층
            nn.Linear(order, hidden_size),
            nn.ReLU(),
            
            # 두 번째 은닉층
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            
            # 출력층
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        신경망을 지나는 앞먹임.
        
        인수:
            x: 꼴 [배치 크기, order]인 들임 텐서
            
        반환값:
            헤아린 다음 값, 꼴 [배치 크기, 1]
        """
        return self.network(x)
    
    def predict_sequence(self, 
                        initial_sequence: torch.Tensor, 
                        n_steps: int) -> np.ndarray:
        """
        앞날 헤아림을 자기 되돌이로 만든다.
        
        ARModel.predict_sequence()과 같되 신경망을 쓴다.
        
        인수:
            initial_sequence: 출발 값
            n_steps: 내다볼 걸음의 수
            
        반환값:
            헤아린 값의 배열
        """
        self.eval()
        predictions = []
        current_seq = initial_sequence.clone().unsqueeze(0)
        
        with torch.no_grad():
            for _ in range(n_steps):
                pred = self.forward(current_seq)
                predictions.append(pred.item())
                current_seq = torch.cat([current_seq[:, 1:], pred], dim=1)
        
        return np.array(predictions)


if __name__ == "__main__":
    """
    보여 주기: 흉내 자료로 자기 되돌이 모델을 시험한다
    """
    
    # 임시 데이터 만들기
    batch_size = 32
    order = 5  # AR(5) 모델
    
    # 아무 들임: 배치의 표본마다 지난 값 5개
    X = torch.randn(batch_size, order)
    
    print("=" * 60)
    print("Testing Linear AR Model")
    print("=" * 60)
    
    # 선형 자기 되돌이 모델을 첫자리매김한다
    model = ARModel(order=order)
    
    # 순전파
    predictions = model(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {predictions.shape}")
    
    # 배운 계수를 보인다
    coeffs = model.get_coefficients()
    print(f"\nLearned coefficients: {coeffs['coefficients']}")
    print(f"Learned constant: {coeffs['constant']:.4f}")
    
    # 차례 헤아리기를 시험한다
    initial_seq = torch.randn(order)
    future_preds = model.predict_sequence(initial_seq, n_steps=10)
    print(f"\nGenerated {len(future_preds)} future predictions")
    
    print("\n" + "=" * 60)
    print("Testing Neural AR Model")
    print("=" * 60)
    
    # 신경 자기 되돌이 모델을 첫자리매김한다
    neural_model = NeuralARModel(order=order, hidden_size=32)
    
    # 순전파
    predictions = neural_model(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {predictions.shape}")
    
    # 매개변수 개수 세기
    n_params = sum(p.numel() for p in neural_model.parameters())
    print(f"Number of parameters: {n_params}")
    
    print("\n✓ Both models working correctly!")
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
    평균 제곱 어긋남 로스로 자기 회귀 모델을 익힌다.
    
    인수:
        model: 익힐 자기 회귀 모델(ARModel이나 NeuralARModel)
        X_train: 익히기 들임 차례
        y_train: 익히기 목표 값
        X_test: 시험 들임 차례
        y_test: 시험 목표 값
        n_epochs: 익히기 바퀴 수
        learning_rate: 최적화기의 러닝레이트
        verbose: 나아감 막대를 보일지 여부
        
    반환값:
        익히기 지난 일(로스)을 담은 사전
    """
    
    # 로스 함수: 평균제곱오차
    # 평균 제곱 어긋남 = (1/n) * Σ(헤아린 값 - 실제 값)²
    criterion = nn.MSELoss()
    
    # 가장 좋게 하개: Adam(맞추어 가는 러닝 빠르기)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 그리려고 로스를 담아 둔다
    train_losses = []
    test_losses = []
    
    # 학습 루프
    iterator = tqdm(range(n_epochs), desc="Training") if verbose else range(n_epochs)
    
    for epoch in iterator:
        # ==================== 익히기 단계 ====================
        model.train()  # 모델을 학습 모드로
        
        # 피드포워드: 예측을 셈한다
        train_predictions = model(X_train)
        
        # 로스를 셈한다: 헤아린 값이 참값에서 얼마나 먼가?
        train_loss = criterion(train_predictions, y_train)
        
        # 역전파: 경사를 계산한다
        optimizer.zero_grad()  # 이전 그래디언트 지우기
        train_loss.backward()  # 새 그래디언트 계산
        
        # 모델 파라메터를 새로 고친다
        optimizer.step()
        
        # ==================== 따지기 단계 ====================
        model.eval()  # 모델을 평가 모드로
        
        with torch.no_grad():  # 따질 때는 그래디언트를 셈하지 않는다
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions, y_test)
        
        # 로스 담기
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
        model: 익힌 자기 회귀 모델
        X_test: 시험 들임 차례
        y_test: 시험 목표 값
        sequence_length: 들임 차례의 길이
        train_size: 익히기 표본의 수
        n_forecast: 앞날로 내다볼 스텝의 수
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
    # 시험 배치의 마지막 차례를 출발점으로 삼는다
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
    
    # ==================== 웃파라메터 ====================
    SEQUENCE_LENGTH = 10  # AR(10) - 지난 값 10개를 쓴다
    N_SAMPLES = 1000      # 때 점 1000개를 만든다
    TRAIN_RATIO = 0.8     # 익히기 80%, 시험 20%
    N_EPOCHS = 200        # 익히기 바퀴 수
    LEARNING_RATE = 0.01  # 러닝 비율
    
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
    
    # 어떤 클래스의 자료를 쓸지 고를 수 있다:
    # 고르기 1: 잡음 섞인 사인 물결
    data = generate_sine_wave(n_samples=N_SAMPLES, frequency=0.05, noise_std=0.2)
    data_name = "Sine Wave"
    
    # 고르기 2: 참 자기 회귀 과정(대신 쓰려면 주석을 푼다)
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
    
    # ==================== 선형 자기 회귀 모델 익히기 ====================
    print(f"\n{'='*70}")
    print("Step 3a: Training Linear AR Model...")
    print(f"{'='*70}")
    
    linear_model = ARModel(order=SEQUENCE_LENGTH)
    
    # 파라메터 개수 세기
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
    
    # ==================== 뉴럴 자기 회귀 모델 익히기 ====================
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
    
    # 뉴럴 모델의 헤아림을 그린다
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
    main()
```

기본 main은 표본 1,000개, 주파수 0.05, 잡음 표준편차 0.2인 사인파를 만든다. 창 길이 10에서 990개 쌍을 얻고 앞 792개·뒤 198개로 나눈다. 선형 모델은 파라미터 11개, 은닉 64인 비선형 모델은 4,929개다. 두 모델을 Adam 0.01로 각각 200회 학습한다.

train_model은 미니배치가 아니라 매 에포크 전체 학습 텐서의 평균 MSE로 한 번 업데이트한다. 모델과 입력을 자동으로 장치에 옮기지 않으므로 현재 main은 기본 CPU 경로다. 표시된 import에는 ar_model과 data 모듈이 필요하다. 이 디렉터리에는 ar_model.py는 있지만 data.py는 없으므로 [데이터 절](data.md)의 함수 등을 실제 import 경로에 준비해야 한다. 표시된 코드가 단독으로 의존성을 모두 갖추었다고 가정하지 않는다.

## 2. 로스와 계수 해석

train 로스는 업데이트 **전**의 모델에서, 같은 에포크 test 로스는 업데이트 **후**의 모델에서 계산한다. 두 곡선을 같은 웨이트의 평가값처럼 직접 대응시키면 안 된다. 이 모델들에는 드롭아웃·BatchNorm이 없어 모드 차이 자체는 작지만 평가 시점 차이는 남는다.

주어진 테스트 창은 실제 과거 관측으로 다음 값을 예측하므로 rolling 한 스텝 MSE다. 고정 시점에서 여러 미래를 재귀 예측한 오차는 이 수치가 아니다. 테스트 값을 매 에포크 보는 구조이며 이를 설정·종료 선택에 이용한다면 별도 검증셋과 최종 테스트셋이 필요하다.

선형 모델의 웨이트는 오래된 관측부터 대응한다. 출력에서 순서대로 φ₁,φ₂라고 이름 붙여도 전통적 AR의 최근 시차부터의 계수와는 순서가 반대다. 또한 실제 main 자료는 AR 생성기가 아니라 사인파이므로 특정 AR 생성 계수를 복원한 실험이라고 읽을 수 없다.

## 3. 롤아웃 그림의 시간축 문제

visualize_results의 첫 그림은 테스트 타깃 순서대로 한 스텝 예측을 비교한다. 그러나 둘째 그림은 X_test[-1]에서 시작한 롤아웃을 len(data)−n_forecast 위치부터 그린다. 마지막 창은 최종 관측 직전까지 담고 있어 첫 예측의 타깃은 data의 마지막 위치 N−1인데 표시 시작은 N−n_forecast다. 따라서 곡선이 실제 예측 시점과 맞지 않는다.

관측 끝 다음 N부터 미래 예측을 보이려면 마지막 p개 실제 관측으로 창을 만들고 그 시점부터 그려야 한다. 이 코드는 보존했으므로 현재 둘째 그림을 정렬된 미래 예측이나 여러 스텝 정확도의 증거로 해석하지 않는다. sequence_length와 train_size 인수도 해당 함수 안에서 사용되지 않는다.

## 4. 비교의 조건

같은 창·분할에서 두 구조의 한 스텝 MSE를 비교할 수 있지만 초기 웨이트의 시드는 따로 고정하지 않는다. 데이터 생성 시드가 같다고 PyTorch 모델 초기화도 같아지는 것은 아니다. 여러 초기화·검증 설정과 계산비용을 함께 보고 비선형 모델이 항상 우세하다고 가정하지 않는다.

그림 파일 저장 호출은 코드의 의도이며 이 편집에서는 학습이나 시각화를 실행하지 않았다. 예측값은 결정적 경로이고 잡음 혁신을 뽑은 새 시계열 표본이 아니다. 불확실성 평가에는 조건부 분포나 별도 예측 구간 방법이 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
전체 배치 학습에서 zero_grad를 제거하면 무엇이 달라지는가?

</div>

??? success "풀이"

    각 에포크의 그래디언트가 기존 grad에 더해져 이전 웨이트 위치의 그래디언트까지 다음 Adam 업데이트에 들어간다. 이는 원래 전체 배치 경사 학습이 아니며 단순 러닝레이트 배수와 같지도 않다. 의도한 그래디언트 누적은 누적 대상·정규화·step 시점을 별도로 정의한다. 현재 구현은 에포크마다 한 번 step하므로 그 전에 초기화하는 것이 계약이다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문 옵티마이저와 러닝레이트 비교의 의미를 설명하라.

</div>

??? success "풀이"

    이미 Adam 0.01을 쓰므로 Adam 0.001은 종류 변경이 아니라 러닝레이트 변경이다. 동일 초기 웨이트·자료 분할에서 수렴·검증 MSE와 시간을 비교하되 여러 시드 변동도 살핀다. 선형과 비선형은 다른 용량·최적화 특성을 가지므로 한 러닝레이트·한 실행의 승패를 일반적인 우열로 확대하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
한 스텝 테스트 수치와 미래 예측 그림이 다른 결론을 주면 무엇을 확인하는가?

</div>

??? success "풀이"

    먼저 롤아웃의 초기 관측 시점과 표시 축을 확인한다. 본문 둘째 그림은 시작 시점이 어긋난다. 정렬한 뒤에도 실제 과거를 받는 한 스텝과 예측을 다시 넣는 여러 스텝은 다른 작업이다. 선형 계수 안정성·비선형 범위 밖 반응·전처리·시차 순서도 점검한다. 그래프의 겉모양만으로 과적합이나 뉴럴 넷트웍 구조의 우열을 확정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
조기 종료 예시를 이 전체 배치 루프에 연결하려면 무엇이 필요한가?

</div>

??? success "풀이"

    시간순 검증 분할과 평가 함수를 정하고 실제 train_model 업데이트와 연결해야 한다. 아래 evaluate·val_loader는 현재 코드에 없으며 예시만으로 학습이 수행되지 않는다. state_dict().copy는 얕은 복사라 최적 텐서가 고정되지 않으므로 독립 복제나 파일 저장이 필요하다. 인내 횟수에 의한 종료뿐 아니라 정상 종료에서도 최적 모델을 복원하고 최종 테스트는 선택에서 분리한다.

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


## 정리하며

이 절은 사인파의 한 스텝 점 예측을 전체 배치 MSE로 학습한다. 계수 순서·로스 평가 시점·의존 모듈과 롤아웃 시간축을 확인해야 한다. 재귀 미래 경로와 rolling 테스트 MSE를 구분하고 검증 자료로 구조·러닝레이트를 선택한다.
