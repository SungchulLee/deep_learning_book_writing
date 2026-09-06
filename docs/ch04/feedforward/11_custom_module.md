# 사용자 정의 Module

05_custom_module.py - 사용자 정의 nn.Module 클래스 만들기. 상속을 통해 유연하고 재사용 가능한 신경망 구성 요소를 만드는 법을 배운다

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 코드

```python
"""
==================================================================
05_custom_module.py - 사용자 정의 nn.Module 클래스 만들기
==================================================================

nn.Module을 상속하여 유연하고 다시 쓸 수 있는 신경망 부품을 만드는 법을 배운다.
복잡한 구조에는 이 방식이 좋다!

왜 사용자 정의 nn.Module인가:
    ✓ 순전파를 온전히 제어할 수 있다
    ✓ 여러 입력/출력
    ✓ 조건부 실행
    ✓ 사용자 정의 매개변수 초기화
    ✓ 훅과 디버깅 지원

학습 목표:
    1. 올바른 nn.Module 상속
    2. 매개변수 등록
    3. forward 메서드 구현
    4. 고급 기능 (훅, 사용자 정의 초기화)
    5. 좋은 관행과 양식

난이도: ⭐⭐⭐☆☆
소요 시간: 25~30분
==================================================================
"""

import torch
import torch.nn as nn
import torch.nn.init as init

# ========================================================================
# 메인
# ========================================================================

print("="*70)
print("PART 1: Basic Custom Module")
print("="*70)

class BasicNet(nn.Module):
    """
    nn.Module의 기본 구조를 보이는 기초적인 사용자 정의 신경망.
    
    반드시 있어야 하는 메서드:
        - __init__: 층과 매개변수를 정의한다
        - forward: 계산의 흐름을 정의한다
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        # 반드시 부모 생성자를 먼저 부르라!
        super(BasicNet, self).__init__()
        
        # 층을 속성으로 정의한다
        # PyTorch가 이들을 매개변수로 자동 추적한다
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
        # 매개변수가 아닌 속성도 저장할 수 있다
        self.input_size = input_size
        
    def forward(self, x):
        """
        순전파 - 데이터가 신경망을 어떻게 흐르는지 정의한다.
        PyTorch가 계산 그래프를 자동으로 만든다!
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = BasicNet(784, 128, 10)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

print("\n"+"="*70)
print("PART 2: Module with Custom Initialization")
print("="*70)

class InitializedNet(nn.Module):
    """사용자 정의 가중치 초기화를 쓰는 신경망."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(InitializedNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
        # 사용자 정의 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier/He 초기화로 가중치를 초기화한다."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # ReLU가 뒤따르는 층에는 Xavier 균등 초기화
                init.xavier_uniform_(m.weight)
                # 편향을 작은 양수로 초기화
                init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = InitializedNet(784, 128, 10)
print("Custom initialization applied!")
print(f"First layer bias sample: {model.fc1.bias[:5]}")

print("\n"+"="*70)
print("PART 3: Module with Multiple Paths")
print("="*70)

class MultiPathNet(nn.Module):
    """
    계산 경로가 여럿인 신경망.
    왜 사용자 정의 모듈이 필요한지 보인다!
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiPathNet, self).__init__()
        
        # 경로 1: 깊은 경로
        self.deep_path = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # 경로 2: 얕은 경로 (건너뛰기 연결)
        self.shallow_path = nn.Linear(input_size, hidden_size)
        
        # 경로 합치기
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 두 경로 모두 계산
        deep = self.deep_path(x)
        shallow = self.shallow_path(x)
        
        # 잔차 연결로 합치기
        combined = deep + shallow  # 원소별 덧셈
        
        # 최종 출력
        return self.output(combined)

model = MultiPathNet(784, 128, 10)
print("Multi-path network with skip connections created!")

print("\n"+"="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. Always call super().__init__() first in __init__
2. Define layers as instance attributes (self.xxx)
3. Implement forward() to define computation
4. PyTorch automatically:
   - Tracks parameters
   - Builds computation graph
   - Enables backpropagation
5. Custom modules enable:
   - Skip connections
   - Multiple inputs/outputs
   - Conditional logic
   - Custom initialization
""")

print("\n"+"="*70)
print("EXERCISES")
print("="*70)
print("""
1. Add dropout to BasicNet
2. Create a ResNet-style block with skip connections
3. Implement custom weight initialization schemes
4. Build a network that uses different paths for different inputs
5. Add layer normalization to the network
""")


if __name__ == "__main__":
    pass```

## 논의

이 구현은 3개의 클래스(`BasicNet`, `InitializedNet`, `MultiPathNet`)를 정의하며, 이들이 함께 작동하여 완전한 순방향 신경망 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `BasicNet`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `BasicNet`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = BasicNet(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
