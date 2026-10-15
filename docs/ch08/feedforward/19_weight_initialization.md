# 가중치 초기화

같은 MLP에 Xavier, He, 작은 정규분포와 0 초기화를 적용하고 가중치와 활성화의 통계량을 비교한다. 초기 신호의 비교와 학습 후 성능 비교는 서로 다른 실험이다.

## 1. 코드

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
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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
        # self.modules()는 자기 자신까지 훑으므로 isinstance로 걸러 낸다.
        # 이렇게 해 두면 층을 몇 개 더 붙여도 이 함수는 고치지 않아도 된다
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'xavier':
                    # 들어오는 쪽과 나가는 쪽을 함께 본다. 순전파의 분산과
                    # 역전파의 분산을 모두 지키려는 것이라, 두 방향이
                    # 대칭인 시그모이드나 tanh에 맞는다
                    init.xavier_uniform_(m.weight)
                elif method == 'he':
                    # ReLU가 입력의 절반을 0으로 죽여 분산을 반토막 내므로,
                    # 그만큼 넓게 뽑아 메운다. 그래서 자비에르보다 크다.
                    # 주의: 이것은 nn.Linear의 기본 초기화와 같지 않다.
                    # 기본값은 kaiming_uniform_(a=sqrt(5))이라 경계가
                    # 1/sqrt(fan_in)인데, 여기 nonlinearity='relu'는
                    # sqrt(6/fan_in)이다. sqrt(6)배, 곧 약 2.45배 넓다
                    init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == 'normal':
                    # fan_in을 보지 않는 고정 표준편차다. 층의 폭이
                    # 달라져도 늘 0.01이라, 넓은 층에서는 신호가 너무
                    # 작아지고 좁은 층에서는 너무 커진다. 위의 둘이
                    # 나은 까닭이 여기 있다
                    init.normal_(m.weight, mean=0, std=0.01)
                elif method == 'zeros':
                    # 배우지 못하는 초기화다. 한 층의 모든 뉴런이 똑같은
                    # 값을 내고 똑같은 기울기를 받으므로, 갱신해도 계속
                    # 나란히 움직인다. 대칭이 영영 깨지지 않아 폭 256인
                    # 층이 사실상 뉴런 하나짜리로 주저앉는다
                    init.zeros_(m.weight)
                
                # 편향은 0으로 두어도 괜찮다. 대칭을 깨는 일은 가중치가
                # 맡으므로, 가중치만 서로 다르면 뉴런들은 갈라져 나간다
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
    # fc1만 본다. 784 → 256 층이라 fan_in이 가장 커서, 초기화 방법에
    # 따른 폭 차이가 가장 뚜렷하게 드러나는 자리다.
    # zeros에서는 평균도 표준편차도 최솟값도 최댓값도 모두 0으로 찍힌다
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
    pass
```

**출력:**

```
======================================================================
Weight Initialization Strategies
======================================================================
INITIALIZATION METHODS:
----------------------------------------------------------------------

XAVIER:
  Mean: 0.000057
  Std: 0.043842
  Min: -0.075955, Max: 0.075955

HE:
  Mean: -0.000094
  Std: 0.050494
  Min: -0.087481, Max: 0.087480

NORMAL:
  Mean: 0.000019
  Std: 0.009999
  Min: -0.047237, Max: 0.048272

ZEROS:
  Mean: 0.000000
  Std: 0.000000
  Min: 0.000000, Max: 0.000000

======================================================================
INITIALIZATION GUIDE
======================================================================

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


Weight distribution plots saved!

EXERCISES:
1. Train models with different initializations
2. Visualize gradient flow with different inits
3. Compare Xavier vs He for ReLU networks
4. Implement custom initialization schemes
```

## 2. 논의

DemoNet의 구조는 784→256→128→10이며 은닉층 두 곳에 ReLU가 있다. 초기화 방법은 매개변수의 값만 바꾸므로 개수는 같다. 코드가 입력에 맞게 평탄화하지 않으므로 마지막 축이 784인 텐서를 전달해야 한다.

He 초기화는 ReLU를 통과하는 2차 적률을 고려한다. 코드에서는 출력 선형층에도 같은 규칙을 적용하므로, 출력층에 ReLU가 없다는 점과 별도로 이해한다.

0 초기화에서는 은닉 뉴런의 대칭과 ReLU의 도함수 때문에 표현이 분화되지 않을 수 있다. 반면 선형 회귀의 계수나 잔차 블록의 마지막 일부를 0으로 초기화하는 경우까지 같은 실패로 일반화하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
초기화 방법에 따른 DemoNet의 매개변수 수 차이를 구하라.

</div>

??? success "연습문제 1 풀이"
    네 방법 모두 선형층 235146개로 같다. 초기값의 분포는 저장하는 매개변수의 수를 바꾸지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력과 초기화를 비교할 때 통제할 조건을 설명하라.

</div>

??? success "연습문제 2 풀이"
    같은 입력 배치와 구조, 자료형을 사용하고 초기화 분포와 실제 분산을 기록한다. 이미지라면 784차원으로 평탄화한다. 서로 다른 무작위 표본의 차이와 초기화 방식의 효과를 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
활성화 표준편차가 비슷한 두 초기화가 같은 학습 성능을 갖는가?

</div>

??? success "연습문제 3 풀이"
    그렇지 않다. 평균, 상관구조, 활성 비율, 경사 전파와 최적화 조건도 다를 수 있다. 초기 통계량만으로 최종 검증 성능을 보장할 수 없다.

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

초기화 비교에서는 같은 입력에서 층별 신호를 살핀다. 초기 분산을 유지하려는 목적과 최종 학습 성능을 구분하고, 출력층까지 같은 규칙을 적용할지 검토한다.
