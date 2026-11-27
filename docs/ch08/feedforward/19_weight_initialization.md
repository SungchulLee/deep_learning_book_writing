# 웨이트 초기화

같은 MLP(다층 퍼셉트론)에 Xavier, He, 작은 정규분포와 0 초기화를 적용하고 첫 레이어 웨이트의 통계량과 분포를 비교한다. 초기 신호의 비교와 트레인 후 성능 비교는 서로 다른 실험이다.

## 1. 코드

```python
"""
좋은 웨이트로 시작하기

알맞은 초기화는 트레인 성공의 핵심이다.
나쁜 초기화 → 그래디언트 소실/폭발 → 느리거나 실패하는 트레인

소요 시간: 25~30분 | 난이도: ⭐⭐⭐☆☆
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
# 무작위로 뽑는 값이 아래에 나온다. 시드를 고정해야 이 쪽에 실린
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
        # 이렇게 해 두면 레이어를 몇 개 더 붙여도 이 함수는 고치지 않아도 된다
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'xavier':
                    # 들어오는 쪽(fan_in)과 나가는 쪽(fan_out)을 함께 본다.
                    # 순전파의 분산과 역전파의 분산을 모두 유지하려는 것이며,
                    # tanh처럼 0 근처에서 선형에 가까운 엑티베이션을 전제한다
                    init.xavier_uniform_(m.weight)
                elif method == 'he':
                    # ReLU가 입력의 절반을 0으로 만들어 분산을 절반으로 줄이므로,
                    # 그만큼 넓은 분포에서 샘플링해 보정한다. 그래서 Xavier보다 크다.
                    # 주의: 이것은 nn.Linear의 기본 초기화와 같지 않다.
                    # 기본값은 kaiming_uniform_(a=sqrt(5))이라 경계가
                    # 1/sqrt(fan_in)인데, 여기 nonlinearity='relu'는
                    # sqrt(6/fan_in)이다. sqrt(6)배, 곧 약 2.45배 넓다
                    init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == 'normal':
                    # fan_in을 보지 않는 고정 표준편차다. 레이어의 너비가
                    # 달라져도 늘 0.01이라, He 기준 sqrt(2/fan_in)과 맞는 것은
                    # fan_in이 2만일 때뿐이다. 그보다 좁은 레이어에서는 신호가
                    # 레이어를 지날 때마다 줄어든다. 위의 둘이 나은 까닭이다
                    init.normal_(m.weight, mean=0, std=0.01)
                elif method == 'zeros':
                    # 트레인이 되지 않는 초기화다. 한 레이어의 모든 뉴런이 똑같은
                    # 값을 내고 똑같은 그래디언트를 받으므로, 갱신해도 계속
                    # 나란히 움직인다. 대칭이 끝내 깨지지 않아 너비 256인
                    # 레이어가 사실상 뉴런 하나짜리와 같아진다
                    init.zeros_(m.weight)
                
                # 바이어스는 0으로 두어도 괜찮다. 대칭을 깨는 일은 웨이트가
                # 맡으므로, 웨이트만 서로 다르면 뉴런들은 갈라져 나간다
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
    # fc1만 본다. 784 → 256 레이어라 fan_in이 가장 커서, 초기화 방법에
    # 따른 분포 폭의 차이가 가장 뚜렷하게 드러나는 자리다.
    # zeros에서는 평균도 표준편차도 최솟값도 최댓값도 모두 0으로 찍힌다
    w = model.fc1.weight.data
    print(f"  Mean: {w.mean():.6f}")
    print(f"  Std: {w.std():.6f}")
    print(f"  Min: {w.min():.6f}, Max: {w.max():.6f}")

print("\n" + "="*70)
print("INITIALIZATION GUIDE")
print("="*70)
print("""
Xavier(Glorot) 초기화:
  Formula: U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
  쓰는 곳: 시그모이드, tanh 엑티베이션 함수
  목표: 레이어 사이에서 분산을 유지한다

HE (KAIMING) INITIALIZATION:
  Formula: U(-√(6/n_in), √(6/n_in))
  쓰는 곳: ReLU와 그 변형
  목표: ReLU의 비선형성을 반영한다

정규분포 초기화:
  Formula: N(0, 0.01²)
  용도: 드물다. 너무 작거나 클 수 있다

ZEROS:
  웨이트에는 절대 쓰지 마라!(대칭이 깨지지 않는다)
  바이어스에는 괜찮다

RECOMMENDATIONS:
  ReLU 뉴럴 네트워크 → He 초기화
  시그모이드나 tanh → Xavier 초기화
  바이어스 → 0이나 작은 상수
  
PyTorch 기본값: 선형 레이어에 카이밍 균등 초기화(a=√5, 경계 1/√n_in)
""")

# 웨이트 분포 시각화
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

Xavier(Glorot) 초기화:
  Formula: U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
  쓰는 곳: 시그모이드, tanh 엑티베이션 함수
  목표: 레이어 사이에서 분산을 유지한다

HE (KAIMING) INITIALIZATION:
  Formula: U(-√(6/n_in), √(6/n_in))
  쓰는 곳: ReLU와 그 변형
  목표: ReLU의 비선형성을 반영한다

정규분포 초기화:
  Formula: N(0, 0.01²)
  용도: 드물다. 너무 작거나 클 수 있다

ZEROS:
  웨이트에는 절대 쓰지 마라!(대칭이 깨지지 않는다)
  바이어스에는 괜찮다

RECOMMENDATIONS:
  ReLU 뉴럴 네트워크 → He 초기화
  시그모이드나 tanh → Xavier 초기화
  바이어스 → 0이나 작은 상수
  
PyTorch 기본값: 선형 레이어에 카이밍 균등 초기화(a=√5, 경계 1/√n_in)


Weight distribution plots saved!

EXERCISES:
1. Train models with different initializations
2. Visualize gradient flow with different inits
3. Compare Xavier vs He for ReLU networks
4. Implement custom initialization schemes
```

## 2. 논의

DemoNet의 구조는 784→256→128→10이며 은닉 레이어 두 곳에 ReLU가 있다. 초기화 방법은 파라메터의 값만 바꾸므로 개수는 같다. 코드가 입력에 맞게 평탄화하지 않으므로 마지막 축이 784인 텐서를 전달해야 한다.

Xavier와 He는 레이어의 입력 수(fan_in, 식의 $n_{\text{in}}$)와 출력 수(fan_out, $n_{\text{out}}$)에 맞춰 웨이트의 분산을 정해, 신호의 크기가 레이어를 지나도 유지되게 하려는 초기화다. 출력의 첫 레이어 통계가 식과 맞는다. Xavier의 최댓값 0.0760은 $\sqrt{6/(784+256)}$, He의 0.0875는 $\sqrt{6/784}$이고, 균등분포의 표준편차는 경계를 $\sqrt3$으로 나눈 값이라 각각 0.0438과 0.0505다. 고정 표준편차 0.01은 이들의 4분의 1에서 5분의 1 수준이다. He 초기화는 ReLU가 음수 쪽을 0으로 만들어 2차 적률(제곱의 평균)을 절반으로 줄이는 것을 보정한다. 코드에서는 출력 선형 레이어에도 같은 규칙을 적용하므로, 출력 레이어에 ReLU가 없다는 점과 별도로 이해한다.

0 초기화에서는 한 레이어의 뉴런들이 모두 같은 값을 내고 같은 그래디언트를 받는 대칭 때문에 표현이 분화되지 않는다. 이 코드처럼 바이어스까지 0이면 은닉 출력이 전부 0이라, 첫 역전파에서 0이 아닌 그래디언트를 받는 것은 출력 레이어의 바이어스뿐이다. 반면 선형 회귀의 계수나 잔차 블록의 마지막 일부를 0으로 초기화하는 경우까지 같은 실패로 일반화하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
초기화 방법에 따른 DemoNet의 파라메터 수 차이를 구하라.

</div>

??? success "풀이"
    네 방법 모두 235146개로 같다. 초기값의 분포는 저장하는 파라메터의 수를 바꾸지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력과 초기화를 비교할 때 통제할 조건을 설명하라.

</div>

??? success "풀이"
    같은 입력 배치와 구조, 자료형을 사용하고 초기화 분포와 실제 분산을 기록한다. 이미지라면 784차원으로 평탄화한다. 서로 다른 무작위 표본의 차이와 초기화 방식의 효과를 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
엑티베이션 값의 표준편차가 비슷한 두 초기화가 같은 트레인 성능을 갖는가?

</div>

??? success "풀이"
    그렇지 않다. 평균, 상관구조, 활성 비율, 그래디언트 전파와 최적화 조건도 다를 수 있다. 초기 통계량만으로 최종 검증 성능을 보장할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 엑티베이션, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 엑티베이션을 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

초기화 비교에서는 같은 입력에서 레이어별 신호를 살핀다. 초기 분산을 유지하려는 목적과 최종 트레인 성능을 구분하고, 출력 레이어까지 같은 규칙을 적용할지 검토한다.
