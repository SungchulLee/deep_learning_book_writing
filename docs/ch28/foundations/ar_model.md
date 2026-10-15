# 선형·비선형 자기회귀 예측기

이 절은 최근 p개 관측으로 다음 값 하나를 예측하는 두 모듈을 비교한다. ARModel은 선형 함수이고 NeuralARModel은 ReLU 다층 퍼셉트론이다. 두 모델은 대안이며 함께 조립해야 하나의 모델이 되는 것은 아니다. 현재 구현은 점 예측만 제공하고 혁신 잡음이나 조건부 확률밀도는 정의하지 않는다.

## 1. 모듈 구현

```python
"""
시계열을 위한 자기 되돌이 모델

이 단원은 PyTorch로 단순한 AR(p) 모델을 짠다.
모델은 지난 p개 값을 바탕으로 시계열의 다음 값을 헤아리는 법을 배운다.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

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
```

두 forward의 입력은 `(B,p)`, 출력은 `(B,1)`이다. 데이터 창을 오래된 값부터 `[x_(t−p),…,x_(t−1)]`로 만들었다면 선형 가중치도 이 순서에 대응한다. 전통적인 AR 식 x_t=c+φ₁x_(t−1)+…+φ_px_(t−p)+ε_t의 계수와 비교하려면 가중치 순서를 뒤집어야 한다.

**기존 출력 기록:**

```
============================================================
Testing Linear AR Model
============================================================
Input shape: torch.Size([32, 5])
Output shape: torch.Size([32, 1])

Learned coefficients: [ 0.25879315  0.06611959 -0.3492072  -0.33453605  0.16079101]
Learned constant: 0.2576

Generated 10 future predictions

============================================================
Testing Neural AR Model
============================================================
Input shape: torch.Size([32, 5])
Output shape: torch.Size([32, 1])
Number of parameters: 1281

✓ Both models working correctly!
```

실행부는 order=5인 무작위 모델에 무작위 입력을 넣는다. 최적화가 없으므로 “Learned coefficients”라는 출력 이름과 달리 표시된 값은 학습으로 추정한 계수가 아니다. 구조·출력 모양 확인과 예측 정확도 검증을 구분해야 한다.

## 2. 재귀적 여러 스텝 예측

predict_sequence는 길이 p의 1차원 초기 창을 `(1,p)`로 바꾸고, 예측할 때마다 가장 오래된 값을 버리고 새 예측을 오른쪽에 붙인다. eval·no_grad를 사용하지만 난수 표본을 추가하지 않으므로 초기 창과 모델이 같으면 같은 점 예측 경로를 반환한다. 반환값은 길이 n_steps의 NumPy 배열이다.

이 경로는 확률적 AR 모형의 여러 미래 표본이나 예측 구간이 아니다. 선형·평균 0 혁신 모형에서는 재귀 평균이 조건부 평균과 연결되지만 비선형 모형에서는 평균을 반복 대입한 값이 일반적으로 여러 스텝 조건부 평균과 같지 않다. 불확실성 예측에는 조건부 잡음 모형과 그에 맞는 롤아웃이 필요하다.

초기 창의 길이·순서·장치·자료형을 모델과 맞춰야 한다. 현재 코드는 단일 시계열 경로를 가정하며 배치 경로의 동시 예측은 별도 설계다. 음수 n_steps도 명시적으로 거부하지 않으므로 호출부에서 정책을 정한다.

## 3. 안정성과 평가

선형 AR 계수를 제한 없이 학습하면 재귀 예측이 폭주할 수 있다. 통상적인 인과적 정상 AR(p)에서는 1−φ₁z−…−φ_pz^p의 근이 단위원 밖에 있어야 한다([정상성 기준](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.isstationary.html)). 계수 하나하나의 절댓값만으로 일반 p의 안정성을 판정할 수 없다. 이 클래스는 그 조건을 강제하지 않는다.

비선형 예측기에도 장기 안정성 보장은 없다. 한 스텝 검증 오차와 재귀적 여러 스텝 오차를 구분하고 시간 순서의 검증에서 둘 다 살펴본다. 표준화 통계와 하이퍼파라미터 선택은 학습·검증 구간에 한정해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
선형 모델과 비선형 모델의 파라미터 수를 계산하라.

</div>

??? success "연습문제 1 풀이"

    ARModel에는 기본 order 값이 없으며 p를 지정해야 한다. 가중치 p개와 편향 1개로 p+1개다. NeuralARModel의 은닉 폭 h에서는 첫 층 ph+h, 둘째 h²+h, 마지막 h+1로 합은 ph+h²+3h+1이다. 실행 예 p=5,h=32에서는 선형 6개, 비선형 1,281개다. 클래스 기본 은닉 64를 쓰면 p=5에서 4,609개다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
forward와 predict_sequence의 서로 다른 입력 계약을 설명하라.

</div>

??? success "연습문제 2 풀이"

    forward는 `(B,p)` 부동소수점 텐서를 받아 `(B,1)`을 만든다. predict_sequence는 길이 p의 1차원 창을 받아 내부에서 단일 배치 차원을 추가한다. 원소 수·유한값·오래된 값부터의 순서, 모델과 장치·자료형 호환성을 확인한다. 이미지용 채널 검사나 무조건 float32 강제는 이 함수의 적절한 일반 검증이 아니다. n_steps는 비음수 정수로 정하고 0일 때 빈 결과를 허용할지 명시한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
한 스텝 성능이 좋아도 재귀 예측이 나빠질 수 있는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"

    학습의 실제 관측 창과 달리 재귀 예측에서는 예측값이 새 입력에 들어가 오차가 전파된다. 선형 모델의 불안정한 계수나 비선형 모델의 학습 범위 밖 반응이 이를 키울 수 있다. 한 스텝·여러 스텝 시간 검증, 계수의 지연 순서와 선형 안정성, 입력 스케일을 함께 점검한다. 모델 용량·정규화·다중 스텝 목적을 바꾸는 방법은 목적에 맞게 검증해야 하며 기울기 클리핑 하나가 장기 안정성을 보장하지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
깊이 확장은 어느 모델에 적용하며 선형 층만 쌓으면 무엇이 달라지는가?

</div>

??? success "연습문제 4 풀이"

    비선형 NeuralARModel의 은닉 블록 수를 설정하는 것이 자연스럽다. p차원 입력과 1차원 출력을 유지하고 은닉 층 사이에 비선형성을 둔다. 편향을 포함한 선형 층만 합성하면 여전히 하나의 아핀 함수라 원래 선형 AR보다 함수 계열이 넓어지지 않는다. 아래 ModuleList는 등록 골격이며 YourBlock 구현·차원·forward 순회가 필요하다. 기존 ARModel에 num_layers 인수는 없으므로 이름만 넘기는 예시는 동작하지 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

선형 AR과 신경망 AR은 최근 관측 창을 받는 대안적인 점 예측기다. 계수 순서·입력 모양과 재귀 롤아웃의 의미를 확인하고 확률 표본·불확실성 예측과 구분한다. 한 스텝 오차와 장기 안정성도 별도로 평가해야 한다.
