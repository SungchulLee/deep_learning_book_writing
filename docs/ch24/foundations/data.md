# 데이터

시계열 자기 되돌이 모델을 위한 자료 만들기 도구. 이 단원은 인공 시계열 자료를 만드는 함수를 담는다

자기 되돌이 모델은 앞선 모든 낱개를 조건으로 삼아 낱개마다 미리 헤아려 자료를 만든다. 이 단원은 자기 되돌이 모델 부품의 짜기를 보이며 차례대로 만들어 내는 과정과 그 얼개의 요구를 그려 보인다.

## 1. 코드

```python
"""
시계열 자기 되돌이 모델을 위한 자료 만들기 도구

이 단원은 인공 시계열 자료를 만드는 함수를 담는다
자기 되돌이 모델을 익히고 시험하는 데 쓴다.
"""

import numpy as np
import torch
from typing import Tuple

# ========================================================================
# 메인
# ========================================================================


def generate_sine_wave(n_samples: int = 1000, 
                       frequency: float = 0.1, 
                       noise_std: float = 0.1,
                       seed: int = 42) -> np.ndarray:
    """
    정규 분포 잡음을 더한 사인 물결을 만든다.
    
    이는 되풀이되는 움직임을 보이는 단순한 시계열이다.
    자기 되돌이 모델은 이 결을 배워 헤아릴 수 있어야 한다.
    
    인수:
        n_samples: 만들 때 점의 수
        frequency: 사인 물결의 잦기(클수록 빨리 흔들린다)
        noise_std: 더할 정규 분포 잡음의 표준 편차
        seed: 되풀이할 수 있게 하는 아무 씨앗
        
    반환값:
        시계열을 담은 꼴 (n_samples,)인 넘파이 배열
    """
    np.random.seed(seed)
    
    # 때 점을 만든다
    t = np.arange(n_samples)
    
    # 깨끗한 사인 물결을 만든다
    signal = np.sin(2 * np.pi * frequency * t)
    
    # 가우스 잡음을 더한다
    noise = np.random.normal(0, noise_std, n_samples)
    
    return signal + noise


def generate_ar_process(n_samples: int = 1000,
                       coefficients: list = [0.6, -0.3],
                       noise_std: float = 0.5,
                       seed: int = 42) -> np.ndarray:
    """
    정한 계수로 참 AR(p) 과정을 만든다.
    
    이는 알려진 자기 되돌이 모델에서 자료를 만든다.
    X_t = φ₁*X_{t-1} + φ₂*X_{t-2} + ... + φₚ*X_{t-p} + ε_t
    
    여기서 ε_t ~ N(0, noise_std²)
    
    인수:
        n_samples: 만들 때 점의 수
        coefficients: 자기 되돌이 계수 목록 [φ₁, φ₂, ..., φₚ]
        noise_std: 잡음 항의 표준 편차
        seed: 되풀이할 수 있게 하는 아무 씨앗
        
    반환값:
        시계열을 담은 꼴 (n_samples,)인 넘파이 배열
        
    보기:
        # AR(2) 과정을 만든다: X_t = 0.6*X_{t-1} - 0.3*X_{t-2} + ε_t
        data = generate_ar_process(1000, coefficients=[0.6, -0.3])
    """
    np.random.seed(seed)
    
    p = len(coefficients)  # 자기 되돌이 과정의 차수
    series = np.zeros(n_samples)
    
    # 처음 p개 값을 작은 아무 수로 첫자리매김한다
    series[:p] = np.random.normal(0, 0.1, p)
    
    # 자기 되돌이 식으로 나머지 차례를 만든다
    for t in range(p, n_samples):
        # 자기 되돌이 항을 셈한다: (계수 * 지난 값)의 합
        ar_term = sum(coefficients[i] * series[t - i - 1] for i in range(p))
        
        # 잡음 더하기
        noise = np.random.normal(0, noise_std)
        
        # 합친다
        series[t] = ar_term + noise
    
    return series


def create_sequences(data: np.ndarray, 
                     sequence_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    자기 되돌이 모델 익히기를 위한 들임-내놓기 짝을 만든다.
    
    시계열을 이끌린 배움을 위한 차례로 바꾼다.
    자기 되돌이 모델에서는 X_{t-p}, ..., X_{t-1}으로 X_t을 헤아리려 한다
    
    인수:
        data: 시계열 값의 1차원 넘파이 배열
        sequence_length: 쓸 지난 값의 수(AR(p)의 'p'이다)
        
    반환값:
        (X, y) 짝이며 여기서:
            X: 꼴 (n_sequences, sequence_length)인 텐서 - 들임 차례
            y: 꼴 (n_sequences, 1)인 텐서 - 목표 값
            
    보기:
        data = [1, 2, 3, 4, 5]이고 sequence_length = 2이면:
        X = [[1, 2], [2, 3], [3, 4]]
        y = [[3], [4], [5]]
    """
    X, y = [], []
    
    # 시계열 위로 창을 미끄러뜨린다
    for i in range(len(data) - sequence_length):
        # 들임: 지난 'sequence_length'개 값
        X.append(data[i:i + sequence_length])
        
        # 내놓기: 다음 값
        y.append(data[i + sequence_length])
    
    # PyTorch 텐서로 변환
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y)).unsqueeze(1)  # 한결같게 하려 차원을 더한다
    
    return X, y


def train_test_split_temporal(X: torch.Tensor, 
                              y: torch.Tensor, 
                              train_ratio: float = 0.8) -> Tuple[torch.Tensor, ...]:
    """
    시계열 자료를 익히기 묶음과 시험 묶음으로 가른다.
    
    중요: 시계열에서는 자료를 섞으면 안 된다!
    때 차례대로 가른다. 곧 앞선 자료로 익히고 뒤 자료로 시험한다.
    이는 실제 내다보기 상황을 흉내 낸다.
    
    인수:
        X: 들임 차례 텐서
        y: 목표 값 텐서
        train_ratio: 익히기에 쓸 자료의 몫(나머지는 시험)
        
    반환값:
        (X_train, X_test, y_train, y_test) 짝
    """
    n_samples = len(X)
    split_idx = int(n_samples * train_ratio)
    
    # 때 차례대로 가른다(섞지 않는다!)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    """
    보여 주기: 보기 자료를 만들어 그려 본다
    """
    import matplotlib.pyplot as plt
    
    # 사인 물결을 만든다
    sine_data = generate_sine_wave(n_samples=200, frequency=0.05, noise_std=0.2)
    
    # 참 자기 되돌이 과정을 만든다
    ar_data = generate_ar_process(n_samples=200, 
                                   coefficients=[0.7, -0.2], 
                                   noise_std=0.3)
    
    # 차례를 만든다
    X, y = create_sequences(ar_data, sequence_length=5)
    print(f"Created {len(X)} training sequences")
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")
    
    # 시각화한다
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    axes[0].plot(sine_data)
    axes[0].set_title("Sine Wave with Noise")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")
    axes[0].grid(True)
    
    axes[1].plot(ar_data)
    axes[1].set_title("AR(2) Process: X_t = 0.7*X_{t-1} - 0.2*X_{t-2} + noise")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Value")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("sample_time_series.png", dpi=150)
    print("Saved visualization to sample_time_series.png")
```

## 2. 논의

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 결은 더 복잡한 경우로 자연스럽게 넓혀진다. 웃매개변수와 얼개 변형, 여러 자료 묶음을 실험하면 만들어 내는 모델 일에 대한 이해가 깊어지고 실제 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 설계 결정을 가려내라. 구체적인 짜기 고르기 셋을 적고 저마다 자기 되돌이 모델에 어울리는 까닭을 설명하라.

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
Data 짜기를 확인하는 두루 갖춘 시험 함수를 적어라. 빈 들임, 낱개 하나짜리 들임, 아주 큰 들임, 끝값(0이나 아주 큰 수)을 담은 들임 같은 가장자리 경우를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_data():
        model = Data(...)
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

**다룬 것** — 데이터

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
