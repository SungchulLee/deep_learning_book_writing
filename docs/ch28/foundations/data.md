# 시계열 생성과 예측 창 구성

이 절은 잡음을 더한 사인파와 선형 AR 시계열을 만들고, 최근 관측 창에서 다음 값 하나를 예측하는 자료로 변환한다. 데이터 생성 모형, 예측 입력의 시간 순서와 시간 분할을 명확히 해야 [자기회귀 예측기](ar_model.md)의 계수·오차를 올바르게 해석할 수 있다.

## 1. 데이터 도구 구현

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
    시계열 자료를 익히기 배치와 시험 배치로 가른다.
    
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

사인파는 sin(2π f t)에 독립 가우시안 잡음을 더한다. t는 정수 표본 인덱스이므로 frequency는 표본당 주기 수이며 물리 단위 Hz를 쓰려면 표본 주파수를 함께 지정해야 한다. 고정 위상의 결정적 주기 신호에 잡음을 더한 것으로 일반적인 정상 확률과정을 자동 생성하는 것은 아니다.

AR 생성기는 첫 p개 값을 작은 가우시안으로 초기화하고 x_t=Σ_j φ_j x_(t−j)+ε_t를 계산한다. coefficients[0]은 가장 최근 시차의 계수다. 절편은 0이며 초기 분포가 정상 분포인 것은 아니고 burn-in도 없다. 임의 계수의 안정성을 검사하지 않으므로 긴 경로가 폭주할 수 있다. 통상적인 인과적 정상 AR의 근 조건은 [모델 절](ar_model.md)을 따른다.

두 함수는 np.random.seed로 전역 난수 상태를 재설정한다. 같은 인수·시드의 재현에는 도움이 되지만 이후 다른 NumPy 난수 호출에도 영향을 준다. 독립 실험의 난수 스트림을 분리하려면 별도 생성기를 사용하는 설계를 고려한다.

**기존 출력 기록:**

```
Created 195 training sequences
Input shape: torch.Size([195, 5]), Output shape: torch.Size([195, 1])
Saved visualization to sample_time_series.png
```

실행 예는 길이 200인 두 시계열을 만들고 AR 자료에 길이 5 창을 적용한다. 따라서 X는 `(195,5)`, y는 `(195,1)`이다. 이는 표본 모양 확인이며 계수 추정·예측 성능 결과는 아니다.

## 2. 슬라이딩 창과 계수 순서

길이 N인 1차원 자료에서 p=sequence_length이면 i번째 입력은 `[data[i],…,data[i+p−1]]`, 타깃은 data[i+p]다. 유효한 창은 N−p개이며 입력은 오래된 값부터 정렬된다. 생성기의 계수는 최근 시차부터이므로 선형 예측기의 가중치와 비교할 때 순서를 뒤집어야 한다.

생성 텐서는 CPU float32다. 자료의 원래 정밀도를 그대로 보존하는 함수는 아니다. 1≤p<N, 유한한 1차원 입력을 가정해야 하며 p≥N이면 X가 기대한 `(0,p)`가 아닌 1차원 빈 텐서가 된다. 음수 길이·다변량 입력도 이 함수가 명확히 검증하지 않는다.

## 3. 시간 분할의 평가 의미

train_test_split_temporal은 이미 시간순으로 만든 창의 앞부분을 학습, 뒷부분을 평가에 둔다. train_ratio는 0~1 범위와 양쪽 자료 수를 확인해야 하며 검증셋은 따로 없다. 표준화·하이퍼파라미터 선택도 학습·검증 범위 안에서 수행한다.

경계 근처의 입력 창은 과거 관측을 공유할 수 있다. 예측 시점에 이미 알려진 과거를 쓰는 한 스텝 rolling 평가에서는 이것이 곧 미래 정보 누출은 아니다. 다만 고정된 한 시점에서 여러 미래를 예측하려는 경우, 후속 테스트 창에 실제 관측을 넣는 방식은 목적과 다르다. 그 경우에는 관측 이용 범위를 고정한 재귀 평가를 별도로 해야 한다. 겹치는 창을 무작위 분할하면 인접·중복 정보로 평가가 쉬워질 수 있어 시간 목적에 맞는 분할을 정한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 데이터 도구의 실제 설계 선택 세 가지를 설명하라.

</div>

??? success "연습문제 1 풀이"

    생성 시드를 고정해 재현 가능한 인공 경로를 만드는 점, 최근 p개에서 다음 값 하나를 예측하는 겹치는 창을 만드는 점, 창의 시간 순서를 유지해 분할하는 점이다. 전역 난수 초기화의 부수 효과, 창끼리의 의존성과 rolling 평가의 의미도 함께 설명한다. 활성화·잔차 연결처럼 이 데이터 함수에 없는 신경망 구조를 설계 근거로 들지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
각 함수의 입력 검증 조건을 설명하라.

</div>

??? success "연습문제 2 풀이"

    표본 수는 필요한 초기 구간 이상인 정수, noise_std는 유한한 비음수, 계수는 유한한 값의 목록이어야 한다. create_sequences에는 유한한 1차원 자료와 1≤p<N을 요구하거나 빈 결과의 모양을 명시한다. 시간 분할에는 X·y 길이 일치, 유효한 train_ratio와 양쪽 자료 수를 확인한다. 이미지 채널 검사나 float32 모델 입력 검사로 대체할 수 없다. 샘플링된 사인파의 주파수 해석에는 aliasing도 고려해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
성능이 비정상적으로 좋거나 생성 경로가 폭주할 때 무엇을 점검하는가?

</div>

??? success "연습문제 3 풀이"

    전자는 겹치는 창의 무작위 분할, 전체 자료로 계산한 전처리 통계, 타깃 정렬과 평가 시 사용할 수 없는 미래 정보가 들어갔는지 확인한다. 후자는 AR 계수의 안정성·자료 스케일·초기 과도 상태를 점검한다. 생성 함수에는 경사 계산이 없으므로 신경망의 경사 클리핑을 데이터 생성 문제의 직접 해결책으로 제시하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 테스트 골격 대신 어떤 함수 성질을 확인해야 하는가?

</div>

??? success "연습문제 4 풀이"

    이 코드에는 Data 클래스나 호출 가능한 학습 모듈이 없으므로 아래 model=Data와 parameters·backward 검사는 적용되지 않는다. 생성 길이·유한값·시드 재현, 작은 결정적 자료의 창·타깃 정렬과 시간 분할 경계를 확인하는 것이 맞다. 예를 들어 [0,1,2,3,4],p=2이면 입력 [0,1],[1,2],[2,3]과 타깃 2,3,4가 되어야 한다. 허용하지 않는 빈 자료·음수 길이는 명확히 거부하고 필요한 경우 빈 반환 모양을 정의한다.

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


## 정리하며

시계열 도구는 생성 과정·슬라이딩 창·시간 분할의 계약을 정한다. 창의 오래된 값부터의 순서와 AR 계수의 최근 시차부터의 순서를 구분하고, 한 스텝 rolling 평가와 고정 시점의 다중 스텝 예측도 구분해야 한다.
