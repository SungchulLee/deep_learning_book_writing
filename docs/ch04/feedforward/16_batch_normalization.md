# 배치 정규화

09_batch_normalization.py - 안정적이고 빠른 학습. 배치 정규화는 층의 입력을 정규화하여 다음을 가져온다:

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 코드

```python
"""
09_batch_normalization.py - 안정되고 빠른 학습

배치 정규화는 층의 입력을 정규화하여 다음을 가져온다:
- 더 빠른 학습
- 더 높은 학습률 사용 가능
- 초기화에 덜 민감함
- 정칙화 구실을 함

소요 시간: 30~35분 | 난이도: ⭐⭐⭐☆☆
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ========================================================================
# 메인
# ========================================================================

print("="*70)
print("Batch Normalization")
print("="*70)

# 배치 정규화가 없는 모델
class NoBNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 배치 정규화가 있는 모델 (권장 배치 위치)
class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.bn1(x)  # 활성화 앞의 배치 정규화
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        return self.fc3(x)

print("Models created!")
print(f"Without BN: {sum(p.numel() for p in NoBNNet().parameters())} params")
print(f"With BN: {sum(p.numel() for p in BNNet().parameters())} params")

print("\n" + "="*70)
print("HOW BATCH NORMALIZATION WORKS")
print("="*70)
print("""
미니배치마다:
1. 층 입력의 평균과 분산을 계산한다
2. Normalize: x_norm = (x - mean) / sqrt(var + ε)
3. Scale and shift: y = γ * x_norm + β
   여기서 γ와 β는 학습되는 매개변수이다

좋은 점:
✓ 내부 공변량 이동을 줄인다
✓ Allows higher learning rates (10x faster training)
✓ 초기화에 덜 민감하다
✓ Acts as regularization (slight noise from batch statistics)
✓ 때로는 드롭아웃을 대신할 수 있다

USAGE:
- 학습할 때: 배치 통계를 쓴다
- 평가할 때: 이동 평균을 쓴다

PLACEMENT:
- 보통: 선형 → 배치 정규화 → 활성화
- 대안: 선형 → 활성화 → 배치 정규화
  (both work, first is more common)

PARAMETERS:
- 입력: 정규화할 특징의 수
- For fully connected: num_features = output_dim
- For conv layers: num_features = num_channels

중요한 점:
⚠ Requires batch_size > 1 (needs multiple samples)
⚠ 학습 모드와 평가 모드에서 동작이 다르다
⚠ Remember to call model.train() and model.eval()!
""")

print("\n" + "="*70)
print("BATCH NORMALIZATION VARIANTS")
print("="*70)
print("""
BatchNorm1d: For fully connected layers (batch, features)
BatchNorm2d: For conv layers (batch, channels, height, width)
BatchNorm3d: For 3D data (batch, channels, depth, height, width)

LayerNorm: Normalizes across features (used in Transformers)
InstanceNorm: 표본마다 따로 정규화한다
GroupNorm: 층 정규화와 인스턴스 정규화의 중간형
""")

# 배치 정규화의 효과 시각화
def visualize_batch_norm():
    # 배치 정규화 전후의 층 출력 모의실험
    data = torch.randn(1000, 1) * 5 + 10  # 평균=10, 표준편차=5
    
    bn = nn.BatchNorm1d(1)
    bn.eval()  # 이동 통계 쓰기
    data_normalized = bn(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(data.numpy(), bins=50, alpha=0.7, color='blue')
    ax1.set_title('Before Batch Norm', fontweight='bold')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.axvline(data.mean(), color='r', linestyle='--', label=f'Mean={data.mean():.2f}')
    ax1.legend()
    
    ax2.hist(data_normalized.detach().numpy(), bins=50, alpha=0.7, color='green')
    ax2.set_title('After Batch Norm', fontweight='bold')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.axvline(data_normalized.mean(), color='r', linestyle='--', 
                label=f'Mean={data_normalized.mean():.2f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('09_batch_norm_effect.png', dpi=150)
    print("Visualization saved!")
    plt.show()

visualize_batch_norm()

print("\nEXERCISES:")
print("1. Compare training speed with/without BatchNorm")
print("2. Try different placement: before vs after activation")
print("3. Visualize internal activations with/without BN")
print("4. Experiment with different network depths")


if __name__ == "__main__":
    pass```

## 논의

이 구현은 2개의 클래스(`NoBNNet`, `BNNet`)를 정의하며, 이들이 함께 작동하여 완전한 순방향 신경망 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `NoBNNet`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `NoBNNet`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = NoBNNet(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
