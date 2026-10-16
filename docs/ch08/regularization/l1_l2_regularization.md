# L1/L2 정규화

선형 회귀와 회귀 MLP에서 L1·L2 패널티의 효과를 비교한다. 목적함수에 직접 패널티를 더하며 어떤 계수가 포함되는지 확인한다.

## 1. 코드

```python
"""
L1과 L2 정칙화 예제
=================================
L1(라쏘)과 L2(능선) 정칙화가 로스 함수에 벌점 항을 더해
과적합을 막는 방식을 보인다.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ========================================================================
# 메인
# ========================================================================


def build_mlp(input_dim):
    """회귀용 다층 퍼셉트론을 만든다.

    네 실험이 모두 같은 구조를 쓴다. 바뀌는 것은 벌점뿐이어야
    결과의 차이를 정칙화 탓으로 돌릴 수 있다.
    """
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        # 출력 레이어에는 활성화가 없다. 회귀이므로 출력이 음수도 될 수 있어야 한다
        nn.Linear(32, 1),
    )


def regularization_penalty(model, l1_factor=0.0, l2_factor=0.0):
    """웨이트에만 적용하는 L1/L2 벌점을 계산한다.

    l1_factor만 주면 라쏘, l2_factor만 주면 능선,
    둘 다 주면 엘라스틱 넷이 된다.
    """
    device = next(model.parameters()).device
    l1 = torch.tensor(0.0, device=device)
    l2 = torch.tensor(0.0, device=device)
    # named_parameters로 이름을 보아 바이어스를 걸러 낸다. 바이어스는 과적합의
    # 원인이 아니라 출력의 중심을 옮기는 자유도라 눌러 둘 까닭이 없다.
    # model.parameters()를 그냥 훑으면 바이어스까지 벌점을 받는다
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        # += 대신 = a + b로 누적한다. 제자리 연산은 덮어쓴 값을 역전파가
        # 필요로 할 때 탈이 나므로, 누적에는 이 형태가 안전하다
        l1 = l1 + param.abs().sum()
        l2 = l2 + param.pow(2).sum()
    return l1_factor * l1 + l2_factor * l2


def train_mlp(model, X_train, y_train, l1_factor=0.0, l2_factor=0.0,
              epochs=100, batch_size=32, lr=1e-3):
    """벌점을 걸어 모델을 학습시키고 에포크별 이력을 돌려준다."""
    # 벌점은 위의 함수로 직접 만들어 로스에 더한다. 손이 조금 더 가는
    # 대신, 무엇이 벌점을 받고 무엇이 받지 않는지가 코드에 그대로 보인다
    n_val = int(0.2 * len(X_train))
    train_ds = TensorDataset(X_train[:-n_val], y_train[:-n_val])
    val_ds = TensorDataset(X_train[-n_val:], y_train[-n_val:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'loss': [], 'val_loss': []}

    for _ in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            mse = criterion(model(xb), yb)
            loss = mse + regularization_penalty(model, l1_factor, l2_factor)
            loss.backward()
            optimizer.step()
            # 기록은 벌점을 뺀 mse로 한다. 그래야 벌점의 세기가 다른
            # 실험끼리 로스 곡선을 견줄 수 있다
            total += mse.item() * xb.size(0)
        history['loss'].append(total / len(train_ds))

        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                total += criterion(model(xb), yb).item() * xb.size(0)
        history['val_loss'].append(total / len(val_ds))

    return history


def mean_absolute_error(model, X, y):
    """평균 절대 오차를 잰다."""
    model.eval()
    with torch.no_grad():
        return (model(X) - y).abs().mean().item()


def sklearn_regularization_demo():
    """scikit-learn으로 L1/L2 정칙화를 보인다."""
    print("="*60)
    print("Scikit-learn Regularization Demo")
    print("="*60)
    
    # 데이터셋 생성
    # 피처 50개 가운데 정말로 쓸모 있는 것은 20개뿐이다. 나머지 30개는
    # 목표와 아무 관계가 없다. 라쏘가 그 30개의 계수를 0으로 만들어
    # 스스로 골라내는지 보려고 일부러 이렇게 지었다.
    # 표본이 200개인데 피처가 50개라 비율이 4:1로 낮아, 정칙화 없이는
    # 잡음까지 외우기 쉬운 조건이기도 하다
    X, y = make_regression(n_samples=200, n_features=50, n_informative=20,
                          noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 피처를 표준화한다
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 학습
    models = {
        # 강도가 서로 다르다. 두 벌점의 눈금이 달라 같은 alpha를 주면
        # 라쏘가 훨씬 세게 누르기 때문이다. 이 표에서 볼 것은
        # 강도별 성능이 아니라 계수를 다루는 방식의 차이다
        'No Regularization': LinearRegression(),
        'L2 (Ridge)': Ridge(alpha=1.0),
        'L1 (Lasso)': Lasso(alpha=0.1)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # 영이 아닌 계수 세기
        # 셋 모두에 같은 임계값을 쓴다. 라쏘는 정확히 0을 만들지만
        # 능선과 최소제곱은 그러지 못하므로, 이 줄의 결과는 라쏘만
        # 20 언저리로 떨어지고 나머지 둘은 50에 가깝게 나온다.
        # 데이터에 심어 둔 유효 피처가 20개였음을 떠올려 보라
        non_zero = np.sum(np.abs(model.coef_) > 1e-5)
        
        results[name] = {
            'train_r2': train_score,
            'test_r2': test_score,
            'non_zero_coefs': non_zero
        }
        
        print(f"\n{name}:")
        print(f"  Train R²: {train_score:.4f}")
        print(f"  Test R²: {test_score:.4f}")
        print(f"  Non-zero coefficients: {non_zero}/{len(model.coef_)}")
    
    # 계수 시각화
    plt.figure(figsize=(15, 4))
    for idx, (name, model) in enumerate(models.items(), 1):
        plt.subplot(1, 3, idx)
        plt.bar(range(len(model.coef_)), model.coef_)
        plt.title(f'{name}\nCoefficients')
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient Value')
        plt.axhline(y=0, color='r', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('l1_l2_coefficients.png')
    print("\nCoefficients plot saved as 'l1_l2_coefficients.png'")


def neural_network_regularization_demo():
    """뉴럴 넷트웍에서 L1/L2 정칙화를 보인다."""
    print("\n" + "="*60)
    print("Neural Network Regularization Demo")
    print("="*60)
    
    # 데이터셋 생성
    X, y = make_regression(n_samples=500, n_features=20, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 표준화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 텐서로 옮긴다. 목표를 (N, 1)로 맞추어야 MSELoss가 (N, 1)인 출력과
    # 방송되지 않는다. (N,)으로 두면 오류 없이 (N, N) 차이 행렬이 만들어져
    # 조용히 틀린 로스가 나온다
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)

    # 벌점의 세기만 다르고 구조는 모두 같다. l1과 l2 계수를 어떻게
    # 주느냐가 곧 라쏘, 능선, 엘라스틱 넷을 가른다
    settings = {
        'No Regularization': (0.0, 0.0),
        'L1 Regularization': (0.001, 0.0),
        'L2 Regularization': (0.0, 0.001),
        'L1+L2 (Elastic Net)': (0.001, 0.001),
    }

    histories = {}
    for name, (l1_factor, l2_factor) in settings.items():
        print(f"\nTraining {name}...")
        # 모델을 만들기 직전마다 씨앗을 심어 네 실험이 같은 초기
        # 웨이트에서 출발하게 한다
        torch.manual_seed(42)
        model = build_mlp(X_train_t.shape[1])
        histories[name] = train_mlp(model, X_train_t, y_train_t,
                                    l1_factor=l1_factor, l2_factor=l2_factor)

        # 평가한다
        train_mae = mean_absolute_error(model, X_train_t, y_train_t)
        test_mae = mean_absolute_error(model, X_test_t, y_test_t)
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
    
    # 학습 이력 그리기
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history['loss'], label=f'{name} (train)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history['val_loss'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('nn_regularization_comparison.png')
    print("\nNeural network plot saved as 'nn_regularization_comparison.png'")


def main():
    # 두 시연 모두 실행
    sklearn_regularization_demo()
    neural_network_regularization_demo()
    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("• L1 (Lasso): Pushes coefficients to zero → Feature selection")
    print("• L2 (Ridge): Shrinks coefficients smoothly → Prevents large weights")
    print("• L1+L2 (Elastic Net): Combines benefits of both approaches")
    print("• Regularization helps prevent overfitting and improves generalization")


if __name__ == "__main__":
    main()
```

**출력:**

```
============================================================
Scikit-learn Regularization Demo
============================================================

No Regularization:
  Train R²: 0.9991
  Test R²: 0.9897
  Non-zero coefficients: 50/50

L2 (Ridge):
  Train R²: 0.9990
  Test R²: 0.9897
  Non-zero coefficients: 50/50

L1 (Lasso):
  Train R²: 0.9991
  Test R²: 0.9901
  Non-zero coefficients: 48/50

Coefficients plot saved as 'l1_l2_coefficients.png'

============================================================
Neural Network Regularization Demo
============================================================

Training No Regularization...
  Train MAE: 5.5543
  Test MAE: 15.3884

Training L1 Regularization...
  Train MAE: 5.5836
  Test MAE: 15.3670

Training L2 Regularization...
  Train MAE: 5.3377
  Test MAE: 15.3074

Training L1+L2 (Elastic Net)...
  Train MAE: 5.5722
  Test MAE: 15.3700

Neural network plot saved as 'nn_regularization_comparison.png'

============================================================
Key Takeaways:
============================================================
• L1 (Lasso): Pushes coefficients to zero → Feature selection
• L2 (Ridge): Shrinks coefficients smoothly → Prevents large weights
• L1+L2 (Elastic Net): Combines benefits of both approaches
• Regularization helps prevent overfitting and improves generalization
```

## 2. 논의

regularization_penalty는 이름에 weight가 들어간 텐서에 절댓값 합과 제곱합을 적용하고 바이어스는 제외한다. 이는 이 모델의 선택이며 바이어스가 과적합에 영향을 주지 않는다는 일반 명제는 아니다.

뉴럴 넷트웍 학습은 Adam에 패널티를 직접 더한다. L2 목적의 패널티와 AdamW의 분리된 감쇠를 같은 갱신으로 해석하지 않는다. L1 항을 Adam으로 최적화한다고 정확한 0이 자동으로 보장되는 것도 아니다.

MLP의 내부 검증은 외부 학습 데이터를 scaler로 변환한 후 나누므로 검증 입력의 통계가 전처리에 반영된다. 엄격한 설정 선택에는 먼저 내부 분할한 뒤 실제 학습 부분만으로 scaler를 맞춘다. 선형 모델과 뉴럴 넷트웍은 로스 집계와 정규화 계수의 관례가 다를 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 코드에서 패널티 대상과 바이어스 제외 방식을 설명하라.

</div>

??? success "연습문제 1 풀이"
    named_parameters의 이름에 weight가 있는 텐서만 합산한다. 이 MLP에서는 선형 레이어 웨이트이며 바이어스는 빠진다. 일반 모델에서 같은 문자열 규칙이 의도한 계수 집합을 고르는지는 별도로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
선형 Ridge와 MLP의 l2_factor 수치를 그대로 비교할 수 있는가?

</div>

??? success "연습문제 2 풀이"
    데이터 로스의 합·평균과 1/2 배율, 표본 수, 패널티 대상이 다를 수 있다. 식을 같은 기준으로 환산하고 각 강도를 검증해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
L1 뉴럴 넷트웍의 희소성을 어떻게 평가해야 하는가?

</div>

??? success "연습문제 3 풀이"
    정확한 0의 수와 정한 허용오차 아래 작은 계수 수를 구분한다. 결과가 작은 값이라는 사실만으로 해당 입력 피처의 모든 연결이 제거되었다고 보지 않는다. 필요하면 근접법이나 피처 단위 그룹 패널티를 검토한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_l1 l2 regularization():
        model = L1 L2 Regularization(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

패널티 비교는 대상 계수와 로스 배율을 명시해야 한다. 선형 모형의 희소성 결과를 비볼록 MLP에 자동 적용하지 않고, 내부 검증의 전처리 순서와 최종 평가를 구분한다.
