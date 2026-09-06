# 자기 되돌이 모델

시계열을 위한 자기 되돌이 모델. 이 단원은 PyTorch으로 단순한 AR(p) 모델을 짠다.

자기 되돌이 모델은 앞선 모든 낱개를 조건으로 삼아 낱개마다 미리 헤아려 자료를 만든다. 이 단원은 자기 되돌이 모델 부품의 짜기를 보이며 차례대로 만들어 내는 과정과 그 얼개의 요구를 그려 보인다.

## 코드

```python
"""
시계열을 위한 자기 되돌이 모델

이 단원은 PyTorch으로 단순한 AR(p) 모델을 짠다.
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
        들임: [묶음 크기, 차례 길이] - 지난 p개 값
        내놓기: [묶음 크기, 1] - 헤아린 다음 값
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
        # bias=True은 상수항 c을 배운다는 뜻이다
        self.linear = nn.Linear(order, 1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        앞먹임: 지난 값으로 다음 값을 헤아린다.
        
        인수:
            x: 꼴 [묶음 크기, order]인 들임 텐서
               표본마다 지난 'order'개 값을 담는다
               
        반환값:
            헤아린 다음 값, 꼴 [묶음 크기, 1]
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
            x: 꼴 [묶음 크기, order]인 들임 텐서
            
        반환값:
            헤아린 다음 값, 꼴 [묶음 크기, 1]
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
    
    # 아무 들임: 묶음의 표본마다 지난 값 5개
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
    
    print("\n✓ Both models working correctly!")```

## 논의

이 짜기는 갈래 2개(`ARModel`, `NeuralARModel`)를 뜻매김하며 이들이 함께 온전한 자기 되돌이 모델 얼개를 이룬다. 갈래마다 뚜렷이 구분되는 부품을 감싸므로 코드가 조각으로 나뉘고 넓히기 쉽다. `forward` 방법은 PyTorch가 자동 미분에 쓰는 셈 그래프를 뜻매김한다.

여기서 보인 결은 더 복잡한 경우로 자연스럽게 넓혀진다. 웃매개변수와 얼개 변형, 여러 자료 묶음을 실험하면 만들어 내는 모델 일에 대한 이해가 깊어지고 실제 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 첫자리매김에서 `ARModel`의 배울 수 있는 매개변수 총수를 셈하라. 무게와 치우침을 모두 넣어 층별로 나누어 적어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

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
`ARModel`을 층이나 덩이의 수를 맞출 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 만들어라. 2, 4, 8층으로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되풀이하라. (여느 파이썬 목록이 아니라) `nn.ModuleList`을 쓰면 PyTorch가 모든 매개변수를 가장 좋게 하기에 등록한다. `for n in [2, 4, 8]: model = ARModel(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험하라.
