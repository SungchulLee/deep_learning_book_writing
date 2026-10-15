# 사용자 정의 Module

사용자 정의 `nn.Module`로 기본 MLP, 초기화를 바꾼 MLP, 두 경로를 더하는 모델을 각각 정의한다. 세 클래스는 서로 다른 설계 예이며 하나의 모델을 구성하는 세 부품은 아니다.

## 1. 코드

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
        # ReLU는 배울 것이 없는 함수라 하나를 만들어 여러 번 써도 된다.
        # 드롭아웃도 부를 때마다 새 마스크를 뽑을 뿐이라 마찬가지다.
        # 반면 배치 정규화처럼 학습되는 매개변수나 이동 통계를 지니는
        # 층은 자리마다 따로 만들어야 한다. 하나를 돌려 쓰면 서로 다른
        # 두 자리의 통계가 한 버퍼에 섞인다
        self.relu = nn.ReLU()
        
        # 매개변수가 아닌 속성도 저장할 수 있다.
        # nn.Module은 nn.Module과 nn.Parameter만 장부에 올리므로,
        # 이런 정수는 그냥 파이썬 속성으로 남고 state_dict에도 들어가지
        # 않는다. 곧 모델을 저장했다 불러올 때 따라오지 않는다
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
        """He 초기화로 가중치를 초기화한다."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # ReLU가 뒤따르는 층에는 He(카이밍) 초기화.
                # 자비에르는 시그모이드나 tanh처럼 양쪽이 대칭인 활성화를
                # 전제한다. ReLU는 입력의 절반을 0으로 죽여 분산을
                # 반토막 내므로, 그 몫까지 셈에 넣는 He 쪽이 맞다.
                # 같은 규칙을 19_weight_initialization.md에서 다룬다
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                # 편향을 작은 양수로 초기화.
                # 0이 아니라 0.01로 두면 학습 초반에 ReLU가 양수 쪽에서
                # 시작하므로, 처음부터 0만 내놓는 뉴런이 줄어든다.
                # 다만 출력층에도 똑같이 걸리는데 그쪽은 ReLU가 없어
                # 아무 뜻이 없다
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
        
        # 경로 2: 얕은 경로 (건너뛰기 연결).
        # gradient_flow.md의 잔차 블록과는 다르다. 그쪽은 입력을 손대지
        # 않고 그대로 더하는 항등 경로라 미분이 1 + f'(x)가 되어 기울기가
        # 반드시 살아남지만, 이쪽은 Linear를 거치므로 미분에 W가 곱해진다.
        # 즉 깊은 경로를 우회하는 지름길이기는 해도 기울기 소실을
        # 막아 준다는 보장은 없다.
        # 여기서 Linear가 필요한 까닭은 input_size와 hidden_size가 달라
        # 그냥 더할 수 없기 때문이다
        self.shallow_path = nn.Linear(input_size, hidden_size)
        
        # 경로 합치기
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 두 경로 모두 계산
        deep = self.deep_path(x)
        shallow = self.shallow_path(x)
        
        # 잔차 연결로 합치기
        # 이 한 줄이 이 페이지의 요점이다. nn.Sequential은 층을 한 줄로
        # 잇는 것밖에 못 하므로, 갈라졌다 합쳐지는 이런 흐름은
        # forward를 직접 써야만 만들 수 있다
        combined = deep + shallow  # 원소별 덧셈
        
        # 최종 출력
        return self.output(combined)

model = MultiPathNet(784, 128, 10)
print("Multi-path network with skip connections created!")

print("\n"+"="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. __init__에서는 늘 super().__init__()을 먼저 부른다
2. 층은 인스턴스 속성(self.xxx)으로 정의한다
3. 계산을 정의하려면 forward()를 구현한다
4. PyTorch는 다음을 자동으로 한다.
   - 매개변수를 추적한다
   - 계산 그래프를 만든다
   - 역전파를 가능하게 한다
5. 맞춤 모듈로 다음이 가능하다.
   - 건너뛰기 연결
   - 여러 입력과 출력
   - 조건부 논리
   - 맞춤 초기화
""")

print("\n"+"="*70)
print("EXERCISES")
print("="*70)
print("""
1. BasicNet에 드롭아웃을 더해 보라
2. 건너뛰기 연결을 갖춘 ResNet 방식 블록을 만들어 보라
3. 맞춤 가중치 초기화 방식을 구현해 보라
4. 입력에 따라 서로 다른 경로를 쓰는 신경망을 만들어 보라
5. 신경망에 층 정규화를 더해 보라
""")


if __name__ == "__main__":
    pass
```

**출력:**

```
======================================================================
PART 1: Basic Custom Module
======================================================================
Model created with 101770 parameters

======================================================================
PART 2: Module with Custom Initialization
======================================================================
Custom initialization applied!
First layer bias sample: tensor([0.0100, 0.0100, 0.0100, 0.0100, 0.0100], grad_fn=<SliceBackward0>)

======================================================================
PART 3: Module with Multiple Paths
======================================================================
Multi-path network with skip connections created!

======================================================================
KEY TAKEAWAYS
======================================================================

1. __init__에서는 늘 super().__init__()을 먼저 부른다
2. 층은 인스턴스 속성(self.xxx)으로 정의한다
3. 계산을 정의하려면 forward()를 구현한다
4. PyTorch는 다음을 자동으로 한다.
   - 매개변수를 추적한다
   - 계산 그래프를 만든다
   - 역전파를 가능하게 한다
5. 맞춤 모듈로 다음이 가능하다.
   - 건너뛰기 연결
   - 여러 입력과 출력
   - 조건부 논리
   - 맞춤 초기화


======================================================================
EXERCISES
======================================================================

1. BasicNet에 드롭아웃을 더해 보라
2. 건너뛰기 연결을 갖춘 ResNet 방식 블록을 만들어 보라
3. 맞춤 가중치 초기화 방식을 구현해 보라
4. 입력에 따라 서로 다른 경로를 쓰는 신경망을 만들어 보라
5. 신경망에 층 정규화를 더해 보라
```

## 2. 논의

`BasicNet`은 아핀 변환–ReLU–아핀 변환을 수행한다. `InitializedNet`은 선형층의 가중치와 편향을 직접 초기화한다. 마지막 출력층에는 ReLU가 없으므로 모든 층에 같은 초기화 규칙을 적용한 선택의 의미를 구분한다.

`MultiPathNet`은 깊은 경로와 얕은 경로의 출력을 더한다. 합산할 텐서의 모양이 같아야 하며, 얕은 경로도 학습 가능한 선형 변환이므로 단순 항등 잔차 경로와는 다르다.

모듈을 인스턴스 속성이나 ModuleList 등에 등록하면 `parameters()`와 `state_dict()`에 포함된다. 모듈 등록과 실제 순전파 실행은 별개의 역할이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
BasicNet의 입력 784, 은닉 128, 출력 10일 때 매개변수 수를 구하라.

</div>

??? success "연습문제 1 풀이"
    첫 층은 $(784+1)128=100480$개, 출력층은 $(128+1)10=1290$개로 총 $101770$개이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
BasicNet의 입력 검증에서 어떤 축과 자료형을 확인해야 하는가?

</div>

??? success "연습문제 2 풀이"
    마지막 축이 `input_size`인지, 선형층과 호환되는 부동소수점 자료형·장치인지 확인한다. 이미지의 채널 축을 가정하는 검사는 이 벡터 입력 예제에 맞지 않는다. 혼합 정밀도 등을 사용할 수 있으므로 모든 입력에 float32를 강제할 필요는 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
MultiPathNet에서 두 분기를 더하는 조건과 항등 잔차 블록과의 차이를 설명하라.

</div>

??? success "연습문제 3 풀이"
    두 출력의 배치 차원과 특징 차원이 같아야 한다. 얕은 분기는 입력을 hidden_size로 바꾸는 학습 가능한 선형층이므로, 입력을 그대로 더하는 `x + F(x)`와 야코비안이 다르다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력층과 출력층은 과제의 차원을 유지하고, 은닉층 사이에 필요한 비선형성을 넣어야 한다. 선형층만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

사용자 정의 모듈은 순차 계산뿐 아니라 분기와 합산을 명확히 표현한다. 등록된 층이 실제로 어떻게 실행되는지 `forward`를 기준으로 확인하고, 각 모델의 초기화와 출력 차원을 구분한다.
