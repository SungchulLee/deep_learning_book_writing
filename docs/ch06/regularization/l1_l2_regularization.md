# L1/L2 정칙화

L1/L2 정칙화 예제. L1(라쏘)과 L2(릿지) 정칙화가 과적합을 어떻게 막는지 보여준다

정칙화 기법을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 1. 코드

```python
"""
L1과 L2 정칙화 예제
=================================
L1(라쏘)과 L2(능선) 정칙화가 손실 함수에 벌점 항을 더해
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
        # 출력층에는 활성화가 없다. 회귀이므로 출력이 음수도 될 수 있어야 한다
        nn.Linear(32, 1),
    )


def regularization_penalty(model, l1_factor=0.0, l2_factor=0.0):
    """가중치에만 적용하는 L1/L2 벌점을 계산한다.

    l1_factor만 주면 라쏘, l2_factor만 주면 능선,
    둘 다 주면 엘라스틱 넷이 된다.
    """
    device = next(model.parameters()).device
    l1 = torch.tensor(0.0, device=device)
    l2 = torch.tensor(0.0, device=device)
    # named_parameters로 이름을 보아 편향을 걸러 낸다. 편향은 과적합의
    # 원인이 아니라 출력의 중심을 옮기는 자유도라 눌러 둘 까닭이 없다.
    # model.parameters()를 그냥 훑으면 편향까지 벌점을 받는다
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
    # 벌점은 위의 함수로 직접 만들어 손실에 더한다. 손이 조금 더 가는
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
            # 실험끼리 손실 곡선을 견줄 수 있다
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
    # 특징 50개 가운데 정말로 쓸모 있는 것은 20개뿐이다. 나머지 30개는
    # 목표와 아무 관계가 없다. 라쏘가 그 30개의 계수를 0으로 만들어
    # 스스로 골라내는지 보려고 일부러 이렇게 지었다.
    # 표본이 200개인데 특징이 50개라 비율이 4:1로 낮아, 정칙화 없이는
    # 잡음까지 외우기 쉬운 조건이기도 하다
    X, y = make_regression(n_samples=200, n_features=50, n_informative=20,
                          noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 특징을 표준화한다
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
        # 셋 모두에 같은 문턱값을 쓴다. 라쏘는 정확히 0을 만들지만
        # 능선과 최소제곱은 그러지 못하므로, 이 줄의 결과는 라쏘만
        # 20 언저리로 떨어지고 나머지 둘은 50에 가깝게 나온다.
        # 데이터에 심어 둔 유효 특징이 20개였음을 떠올려 보라
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
    """신경망에서 L1/L2 정칙화를 보인다."""
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
    # 조용히 틀린 손실이 나온다
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
        # 가중치에서 출발하게 한다
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

## 2. 논의

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 학습 최적화 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심적인 설계 결정을 찾아내라. 구체적인 구현 선택 세 가지를 나열하고, 각각이 정칙화 기법에 왜 적절한지 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
L1/L2 정칙화 구현을 검증하는 종합적인 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
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
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — L1/L2 정칙화

손실 계산은 모델의 출력을 최적화 목표와 이어 준다.

앞의 연습문제 4개로 직접 확인할 수 있다.
