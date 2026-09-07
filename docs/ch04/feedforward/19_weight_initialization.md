# 가중치 초기화

11_weight_initialization.py - 좋은 가중치로 시작하기. 알맞은 초기화는 학습의 성패를 좌우한다.

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 코드

```python
"""
11_weight_initialization.py - 좋은 가중치로 시작하기

알맞은 초기화는 학습 성공의 열쇠다.
나쁜 초기화 → 기울기 소실/폭발 → 느리거나 실패하는 학습

소요 시간: 25~30분 | 난이도: ⭐⭐⭐☆☆
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================

print("="*70)
print("Weight Initialization Strategies")
print("="*70)

class DemoNet(nn.Module):
    def __init__(self, init_method='xavier'):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # 초기화 적용
        self._initialize_weights(init_method)
    
    def _initialize_weights(self, method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'xavier':
                    init.xavier_uniform_(m.weight)
                elif method == 'he':
                    init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == 'normal':
                    init.normal_(m.weight, mean=0, std=0.01)
                elif method == 'zeros':
                    init.zeros_(m.weight)
                
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

print("INITIALIZATION METHODS:")
print("-"*70)

methods = ['xavier', 'he', 'normal', 'zeros']
for method in methods:
    model = DemoNet(method)
    print(f"\n{method.upper()}:")
    w = model.fc1.weight.data
    print(f"  Mean: {w.mean():.6f}")
    print(f"  Std: {w.std():.6f}")
    print(f"  Min: {w.min():.6f}, Max: {w.max():.6f}")

print("\n" + "="*70)
print("INITIALIZATION GUIDE")
print("="*70)
print("""
자비에르(글로로) 초기화:
  Formula: U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
  쓰는 곳: 시그모이드, tanh 활성화
  목표: 층 사이에서 분산을 유지한다

HE (KAIMING) INITIALIZATION:
  Formula: U(-√(6/n_in), √(6/n_in))
  쓰는 곳: ReLU와 그 변형
  목표: ReLU의 비선형성을 반영한다

정규 초기화:
  Formula: N(0, 0.01)
  쓰임: 드물다. 너무 작거나 클 수 있다

ZEROS:
  가중치에는 절대 쓰지 마라!(대칭이 깨지지 않는다)
  편향에는 괜찮다

RECOMMENDATIONS:
  ReLU 신경망 → He 초기화(PyTorch의 기본값)
  시그모이드나 tanh → 자비에르 초기화
  편향 → 0이나 작은 상수
  
PyTorch 기본값: 선형층에 카이밍 균등 초기화
""")

# 가중치 분포 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, method in enumerate(methods):
    model = DemoNet(method)
    weights = model.fc1.weight.data.flatten().numpy()
    
    axes[idx].hist(weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[idx].set_title(f'{method.upper()} Initialization', fontweight='bold')
    axes[idx].set_xlabel('Weight Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].axvline(weights.mean(), color='r', linestyle='--', 
                     label=f'Mean={weights.mean():.3f}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('11_weight_init.png', dpi=150)
print("\nWeight distribution plots saved!")

print("\nEXERCISES:")
print("1. Train models with different initializations")
print("2. Visualize gradient flow with different inits")
print("3. Compare Xavier vs He for ReLU networks")
print("4. Implement custom initialization schemes")
plt.show()


if __name__ == "__main__":
    pass```

## 논의

`DemoNet` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `DemoNet`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `DemoNet`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = DemoNet(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
