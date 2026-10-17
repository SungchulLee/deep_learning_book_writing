# 배치 정규화

배치 정규화가 없는 MLP와 있는 MLP를 정의하고 활성화의 변화를 살펴본다. 이 코드는 구조와 정규화 동작을 시연하며 두 모델의 전체 학습 성능 비교를 수행하지는 않는다.

## 1. 코드

```python
"""
09_batch_normalization.py - 안정되고 빠른 학습

배치 정규화는 레이어의 입력을 정규화하여 다음을 가져온다:
- 더 빠른 학습
- 더 높은 러닝레이트 사용 가능
- 초기화에 덜 민감함
- 정칙화 구실을 함

소요 시간: 30~35분 | 난이도: ⭐⭐⭐☆☆
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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
        # 인자는 정규화할 피처의 수, 곧 앞 레이어의 출력 크기다. 여기서
        # 학습되는 것은 피처마다 하나씩인 gamma와 beta 둘뿐이라,
        # 아래에서 찍히는 파라메터 수는 2*(256+128)=768만 늘어난다.
        # 이동 평균과 이동 분산도 함께 저장되지만 그쪽은 학습되는
        # 파라메터가 아니라 버퍼라 이 셈에 들지 않는다
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        # 마지막 레이어 뒤에는 정규화를 걸지 않는다. 로짓을 평균 0으로
        # 맞추어 버리면 클래스별 점수의 절대 위치가 사라진다
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        # 선형 → 정규화 → 활성화 순서다. 정규화가 활성화 앞에 오면
        # ReLU에 들어가는 값이 0 언저리에 모이므로, 절반쯤은 살고
        # 절반쯤은 죽어 신호가 한쪽으로 쏠리지 않는다.
        # 이 위치라면 fc1의 바이어스는 있으나 마나다. 바로 뒤에서 평균을
        # 빼 버리기 때문이며, 그래서 실무에서는 정규화 앞 레이어에
        # bias=False를 주기도 한다
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
1. 레이어 입력의 평균과 분산을 계산한다
2. 정규화: x_norm = (x - 평균) / sqrt(분산 + ε)
3. 크기 조절과 이동: y = γ * x_norm + β
   여기서 γ와 β는 학습되는 파라메터이다

좋은 점:
✓ 내부 공변량 이동을 줄인다
✓ 더 큰 러닝레이트를 쓸 수 있다(학습이 10배 빨라진다)
✓ 초기화에 덜 민감하다
✓ 정칙화 노릇을 한다(배치 통계에서 오는 작은 잡음)
✓ 때로는 드롭아웃을 대신할 수 있다

USAGE:
- 학습할 때: 배치 통계를 쓴다
- 평가할 때: 이동 평균을 쓴다

PLACEMENT:
- 보통: 선형 → 배치 정규화 → 활성화
- 대안: 선형 → 활성화 → 배치 정규화
  (둘 다 되지만 앞의 것이 더 흔하다)

PARAMETERS:
- 입력: 정규화할 피처의 수
- 완전 연결 레이어: num_features = output_dim
- 컨볼루션 레이어: num_features = num_channels

중요한 점:
⚠ batch_size > 1이어야 한다(표본이 여럿 필요하다)
⚠ 학습 모드와 평가 모드에서 동작이 다르다
⚠ model.train()과 model.eval() 부르기를 잊지 마라!
""")

print("\n" + "="*70)
print("BATCH NORMALIZATION VARIANTS")
print("="*70)
print("""
BatchNorm1d: 완전 연결 레이어용 (batch, features)
BatchNorm2d: 컨볼루션 레이어용 (batch, channels, height, width)
BatchNorm3d: 3차원 데이터용 (batch, channels, depth, height, width)

LayerNorm: 피처 방향으로 정규화한다(트랜스포머에서 쓴다)
InstanceNorm: 표본마다 따로 정규화한다
GroupNorm: 레이어 정규화와 인스턴스 정규화의 중간형
""")

# 배치 정규화의 효과 시각화
def visualize_batch_norm():
    # 배치 정규화 전후의 레이어 출력 모의실험
    data = torch.randn(1000, 1) * 5 + 10  # 평균=10, 표준편차=5
    
    bn = nn.BatchNorm1d(1)
    # train() 모드여야 이 배치의 평균과 분산으로 정규화되어 평균 0,
    # 표준편차 1이 된다.
    # eval()을 쓰면 안 된다. 갓 만든 BatchNorm1d의 이동 통계는
    # running_mean=0, running_var=1이고 eval()은 배치가 아니라 그
    # 이동 통계를 쓰므로, (x - 0) / sqrt(1 + eps)가 되어 데이터가
    # 그대로 지나간다. 그러면 아래 "After" 히스토그램이 "Before"와
    # 똑같이 그려져 아무것도 보여 주지 못한다.
    # 이동 통계는 학습을 거치며 쌓이는 값이라, 한 번도 학습하지 않은
    # 레이어에는 쓸 만한 값이 들어 있지 않다
    bn.train()  # 이 배치의 통계 쓰기
    data_normalized = bn(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(data.detach().numpy(), bins=50, alpha=0.7, color='blue')
    ax1.set_title('Before Batch Norm', fontweight='bold')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.axvline(data.mean(), color='r', linestyle='--', label=f'Mean={data.mean():.2f}')
    ax1.legend()
    
    ax2.hist(data_normalized.detach().numpy(), bins=50, alpha=0.7, color='green')
    ax2.set_title('After Batch Norm', fontweight='bold')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    # bn의 출력은 아직 그래프에 붙어 있다. matplotlib이 내부에서 numpy()를
    # 부르므로 item()으로 파이썬 실수를 넘긴다.
    normalized_mean = data_normalized.detach().mean().item()
    ax2.axvline(normalized_mean, color='r', linestyle='--', 
                label=f'Mean={normalized_mean:.2f}')
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
    pass
```

**출력:**

```
======================================================================
Batch Normalization
======================================================================
Models created!
Without BN: 235146 params
With BN: 235914 params

======================================================================
HOW BATCH NORMALIZATION WORKS
======================================================================

미니배치마다:
1. 레이어 입력의 평균과 분산을 계산한다
2. 정규화: x_norm = (x - 평균) / sqrt(분산 + ε)
3. 크기 조절과 이동: y = γ * x_norm + β
   여기서 γ와 β는 학습되는 파라메터이다

좋은 점:
✓ 내부 공변량 이동을 줄인다
✓ 더 큰 러닝레이트를 쓸 수 있다(학습이 10배 빨라진다)
✓ 초기화에 덜 민감하다
✓ 정칙화 노릇을 한다(배치 통계에서 오는 작은 잡음)
✓ 때로는 드롭아웃을 대신할 수 있다

USAGE:
- 학습할 때: 배치 통계를 쓴다
- 평가할 때: 이동 평균을 쓴다

PLACEMENT:
- 보통: 선형 → 배치 정규화 → 활성화
- 대안: 선형 → 활성화 → 배치 정규화
  (둘 다 되지만 앞의 것이 더 흔하다)

PARAMETERS:
- 입력: 정규화할 피처의 수
- 완전 연결 레이어: num_features = output_dim
- 컨볼루션 레이어: num_features = num_channels

중요한 점:
⚠ batch_size > 1이어야 한다(표본이 여럿 필요하다)
⚠ 학습 모드와 평가 모드에서 동작이 다르다
⚠ model.train()과 model.eval() 부르기를 잊지 마라!


======================================================================
BATCH NORMALIZATION VARIANTS
======================================================================

BatchNorm1d: 완전 연결 레이어용 (batch, features)
BatchNorm2d: 컨볼루션 레이어용 (batch, channels, height, width)
BatchNorm3d: 3차원 데이터용 (batch, channels, depth, height, width)

LayerNorm: 피처 방향으로 정규화한다(트랜스포머에서 쓴다)
InstanceNorm: 표본마다 따로 정규화한다
GroupNorm: 레이어 정규화와 인스턴스 정규화의 중간형

Visualization saved!

EXERCISES:
1. Compare training speed with/without BatchNorm
2. Try different placement: before vs after activation
3. Visualize internal activations with/without BN
4. Experiment with different network depths
```

## 2. 논의

`BNNet`은 선형 레이어 뒤, ReLU 앞에 BatchNorm1d를 배치한다. 각 피처의 배치 통계로 정규화한 뒤 학습 가능한 배율과 이동을 적용한다. 정규화 직후라도 epsilon과 선형 변환 때문에 출력 분산이 항상 정확히 1은 아니다.

배치 정규화의 weight와 bias는 학습 파라메터이고 running_mean과 running_var는 버퍼이다. 기본 설정에서는 학습과 평가에 쓰는 통계량이 달라지므로 모드 전환이 중요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
NoBNNet과 배치 정규화를 추가한 모델의 학습 파라메터 수를 비교하라.

</div>

??? success "풀이"
    NoBNNet은 235146개이다. BNNet은 256, 128차원의 배율·이동 계수 768개가 더해져 235914개이다. 이동 통계량 버퍼는 학습 파라메터 수에 포함하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 벡터 입력 모델의 모양을 어떻게 점검하는가?

</div>

??? success "풀이"
    이미지를 784차원으로 펼친 후 각 선형 레이어의 입력·출력 피처 수를 확인한다. BatchNorm1d는 `(B, C)`에서 피처 C별 배치 통계를 사용한다. 학습 시 피처당 통계량을 얻을 값이 하나뿐이면 정상적인 배치 통계를 계산할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
평가에서 배치 정규화를 학습 모드로 두면 어떤 문제가 생길 수 있는가?

</div>

??? success "풀이"
    기본 설정에서 현재 배치 통계량을 사용하고 이동 통계량도 갱신하므로, 예측이 함께 들어간 표본과 배치 크기에 영향을 받을 수 있다. 평가 모드와 경사 기록 중지를 각각 설정한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

배치 정규화는 학습과 평가에서 사용하는 통계량을 구분해야 한다. 효과를 비교할 때는 계수와 버퍼, 모델 선택 시점과 최종 평가 조건을 함께 확인한다.
