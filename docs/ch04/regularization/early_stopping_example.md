# 조기 종료 예제

조기 종료 예제. 학습을 멈춤으로써 조기 종료가 과적합을 어떻게 막는지 보여준다

정칙화 기법을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 코드

```python
"""
조기 종료 예제
=======================
검증 성능이 더 나아지지 않을 때 학습을 멈추어 과적합을 막는
방식을 보인다.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ========================================================================
# 메인
# ========================================================================


def create_model(input_dim):
    """신경망 모델을 만든다."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    return model


def train_without_early_stopping(X_train, y_train, X_val, y_val, epochs=200):
    """조기 종료 없이 모델을 학습시킨다."""
    print("Training WITHOUT early stopping...")
    model = create_model(X_train.shape[1])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0
    )
    
    return model, history


def train_with_early_stopping(X_train, y_train, X_val, y_val, 
                               patience=10, epochs=200):
    """조기 종료와 함께 모델을 학습시킨다."""
    print(f"Training WITH early stopping (patience={patience})...")
    model = create_model(X_train.shape[1])
    
    # 조기 종료 콜백 만들기
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',          # 감시할 지표
        patience=patience,            # 개선이 없는 에포크의 수
        restore_best_weights=True,   # 가장 좋았던 에포크의 가중치로 되돌리기
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, history


def train_with_advanced_early_stopping(X_train, y_train, X_val, y_val, epochs=200):
    """모델 검사점 저장을 포함한 여러 콜백과 함께 학습시킨다."""
    print("Training with ADVANCED early stopping (multiple callbacks)...")
    model = create_model(X_train.shape[1])
    
    # 여러 콜백
    callback_list = [
        # 검증 손실에 기반한 조기 종료
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # 가장 좋은 모델 저장
        callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # 검증 손실이 정체되면 학습률을 줄인다
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callback_list,
        verbose=0
    )
    
    return model, history


def plot_comparison(histories, labels):
    """비교를 위해 학습 이력을 그린다."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = [
        ('loss', 'Loss'),
        ('accuracy', 'Accuracy'),
        ('val_loss', 'Validation Loss'),
        ('val_accuracy', 'Validation Accuracy')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for history, label in zip(histories, labels):
            if metric in history.history:
                epochs = range(1, len(history.history[metric]) + 1)
                ax.plot(epochs, history.history[metric], label=label, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('early_stopping_comparison.png', dpi=150)
    print("\nPlot saved as 'early_stopping_comparison.png'")


def main():
    # 합성 데이터셋 생성
    print("Generating dataset...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # 데이터 나누기
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # 표준화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}\n")
    
    # 모델 학습
    model_no_es, history_no_es = train_without_early_stopping(
        X_train, y_train, X_val, y_val, epochs=200
    )
    
    model_es_10, history_es_10 = train_with_early_stopping(
        X_train, y_train, X_val, y_val, patience=10, epochs=200
    )
    
    model_es_20, history_es_20 = train_with_early_stopping(
        X_train, y_train, X_val, y_val, patience=20, epochs=200
    )
    
    model_advanced, history_advanced = train_with_advanced_early_stopping(
        X_train, y_train, X_val, y_val, epochs=200
    )
    
    # 시험 집합에서 평가
    print("\n" + "="*60)
    print("Test Set Performance")
    print("="*60)
    
    models = {
        'No Early Stopping': model_no_es,
        'Early Stop (patience=10)': model_es_10,
        'Early Stop (patience=20)': model_es_20,
        'Advanced Early Stop': model_advanced
    }
    
    for name, model in models.items():
        loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
        print(f"{name}:")
        print(f"  Test Loss: {loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test AUC: {auc:.4f}\n")
    
    # 비교 그리기
    histories = [history_no_es, history_es_10, history_es_20, history_advanced]
    labels = [
        'No Early Stopping',
        'Early Stop (p=10)',
        'Early Stop (p=20)',
        'Advanced'
    ]
    plot_comparison(histories, labels)
    
    # 요약 출력
    print("\n" + "="*60)
    print("Key Insights")
    print("="*60)
    print(f"• Without early stopping: Trained for {len(history_no_es.history['loss'])} epochs")
    print(f"• With early stopping (p=10): Stopped at epoch {len(history_es_10.history['loss'])}")
    print(f"• With early stopping (p=20): Stopped at epoch {len(history_es_20.history['loss'])}")
    print(f"• Advanced callbacks: Stopped at epoch {len(history_advanced.history['loss'])}")
    print("\nBenefits of Early Stopping:")
    print("  1. Prevents overfitting by stopping before validation loss increases")
    print("  2. Saves computational time by not training unnecessary epochs")
    print("  3. Automatically finds optimal number of training epochs")
    print("  4. Can be combined with other techniques (learning rate reduction, etc.)")


if __name__ == "__main__":
    main()```

## 논의

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
조기 종료 예제 구현을 검증하는 종합적인 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_early stopping example():
        model = Early Stopping Example(...)
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
