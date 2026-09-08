# 다중 출력 신경망

14_multi_output_network.py - 다중 과제 학습. 하나의 신경망으로 여러 과제를! 여러 출력을 예측하는 법을 배운다

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 1. 코드

```python
"""
14_multi_output_network.py - 다중 과제 학습

신경망 하나로 여러 과제를! 같은 입력에서 여러 출력을 동시에
예측하는 법을 배운다.

예: 사람의 사진에서 다음을 예측한다:
- 나이 (회귀)
- 성별 (이진 분류)
- 감정 (다중 클래스 분류)

실제 응용에서 흔히 있는 일이다!

소요 시간: 35~45분 | 난이도: ⭐⭐⭐⭐☆
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ========================================================================
# 메인
# ========================================================================

print("="*80)
print("Multi-Task Learning Example")
print("="*80)

# 합성 데이터: 특징으로부터 나이와 이진 클래스를 함께 예측
np.random.seed(42)
torch.manual_seed(42)

# 합성 데이터 생성
n_samples = 5000
n_features = 20

X = torch.randn(n_samples, n_features)

# 과제 1: 회귀 (나이 예측, 18~80)
age = 50 + 15 * X[:, 0] + torch.randn(n_samples) * 5
age = torch.clamp(age, 18, 80).reshape(-1, 1)

# 과제 2: 이진 분류 (성별 예측)
gender_logits = 2 * X[:, 1] - X[:, 2] + torch.randn(n_samples) * 0.5
gender = (gender_logits > 0).float().reshape(-1, 1)

# 데이터 나누기
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
age_train, age_test = age[:split], age[split:]
gender_train, gender_test = gender[:split], gender[split:]

print(f"Features: {n_features}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"\nTasks:")
print(f"  Task 1: Age prediction (regression, range 18-80)")
print(f"  Task 2: Gender classification (binary, 0 or 1)")

print("\n" + "="*80)
print("Multi-Task Architecture")
print("="*80)

class MultiTaskNet(nn.Module):
    """
    공유 층과 과제별 층을 갖는 다중 과제 학습 신경망.
    
    구조:
        입력 → 공유 층 → 과제별 머리로 갈라짐
    """
    
    def __init__(self, input_size):
        super().__init__()
        
        # 공유 층 (일반적인 특징을 배운다).
        # 두 과제의 기울기가 이 층에서 더해진다. 그래서 한 과제가 배운
        # 표현이 다른 과제에도 쓰이며, 이것이 다중 과제 학습이
        # 정칙화처럼 듣는 까닭이다. 다만 두 과제가 서로 무관하면
        # 오히려 방해가 되기도 한다.
        # 주의: BatchNorm1d가 들어 있어 배치 크기가 1이면 학습 모드에서
        # 오류가 난다. 표본 수가 배치 크기로 나누어떨어지지 않아
        # 마지막 배치가 1이 되는 경우를 조심하라
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 과제 1 머리: 회귀 (나이 예측).
        # 머리를 따로 두는 까닭은 두 과제가 요구하는 출력이 다르기
        # 때문이다. 나이는 실수 하나, 성별은 로짓 하나이고 손실 함수도
        # 다르다. 공유 층이 공통된 표현을 만들고 머리가 그것을
        # 과제별 답으로 옮긴다
        self.age_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 회귀를 위한 단일 출력
        )
        
        # 과제 2 머리: 이진 분류 (성별)
        self.gender_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 출력 하나, BCEWithLogitsLoss를 쓴다
        )
    
    def forward(self, x):
        # 공유 특징 추출
        shared_features = self.shared(x)
        
        # 과제별 예측
        age_pred = self.age_head(shared_features)
        gender_logits = self.gender_head(shared_features)
        
        return age_pred, gender_logits

model = MultiTaskNet(n_features)
print("Multi-task model created!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n" + "="*80)
print("Training Setup")
print("="*80)

# 과제마다 따로 둔 손실
criterion_age = nn.MSELoss()  # 회귀
criterion_gender = nn.BCEWithLogitsLoss()  # 이진 분류

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 과제 가중치 (중요도 균형 맞추기).
# 주의: 1.0과 1.0은 균형이 아니다. 나이가 18~80 그대로라 MSE는 학습
# 초반에 2500 언저리인 반면 BCE는 0.7 안팎이라, 두 손실의 크기가
# 3000배 넘게 차이 난다. 그대로 더하면 공유 층이 받는 기울기는
# 사실상 나이 과제의 것뿐이고 성별 과제는 묻힌다.
# 고치는 길은 둘이다. 나이를 표준화해 두 손실의 눈금을 맞추거나,
# weight_age를 1e-3쯤으로 낮추는 것이다. 크기가 다른 손실을 더할 때
# 늘 따라오는 문제이며, 다중 과제 학습에서 가장 먼저 확인할 자리다
weight_age = 1.0
weight_gender = 1.0

print(f"Loss functions:")
print(f"  Age: MSELoss (weight={weight_age})")
print(f"  Gender: BCEWithLogitsLoss (weight={weight_gender})")

print("\n" + "="*80)
print("Training Multi-Task Model")
print("="*80)

epochs = 100
batch_size = 64
losses_total = []
losses_age = []
losses_gender = []

for epoch in range(epochs):
    model.train()
    epoch_loss_total = 0
    epoch_loss_age = 0
    epoch_loss_gender = 0
    
    # 미니배치 학습
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_age = age_train[i:i+batch_size]
        batch_gender = gender_train[i:i+batch_size]
        
        # 순전파
        age_pred, gender_logits = model(batch_X)
        
        # 과제마다 손실 계산
        loss_age = criterion_age(age_pred, batch_age)
        loss_gender = criterion_gender(gender_logits, batch_gender)
        
        # 결합된 손실 (가중합)
        # 두 손실을 하나로 더해 backward를 한 번만 부른다. 각각 따로
        # 부르면 공유 층의 그래프를 두 번 거슬러 올라가야 해서
        # retain_graph가 필요해진다. 더해 두면 기울기가 저절로 합쳐진다
        loss = weight_age * loss_age + weight_gender * loss_gender
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss_total += loss.item()
        epoch_loss_age += loss_age.item()
        epoch_loss_gender += loss_gender.item()
    
    # 손실 기록
    num_batches = len(X_train) // batch_size
    losses_total.append(epoch_loss_total / num_batches)
    losses_age.append(epoch_loss_age / num_batches)
    losses_gender.append(epoch_loss_gender / num_batches)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Total: {losses_total[-1]:.4f} | "
              f"Age: {losses_age[-1]:.4f} | "
              f"Gender: {losses_gender[-1]:.4f}")

print("\n" + "="*80)
print("Evaluation")
print("="*80)

model.eval()
with torch.no_grad():
    age_pred, gender_logits = model(X_test)
    gender_pred = (torch.sigmoid(gender_logits) > 0.5).float()
    
    # 나이 지표 (회귀)
    age_mse = criterion_age(age_pred, age_test).item()
    age_mae = torch.abs(age_pred - age_test).mean().item()
    
    # 성별 지표 (분류)
    gender_acc = (gender_pred == gender_test).float().mean().item() * 100

print(f"Age Prediction (Regression):")
print(f"  MSE: {age_mse:.4f}")
print(f"  MAE: {age_mae:.4f} years")

print(f"\nGender Classification (Binary):")
print(f"  Accuracy: {gender_acc:.2f}%")

# 시각화
fig = plt.figure(figsize=(16, 10))

# 손실 곡선들
ax1 = plt.subplot(2, 3, 1)
ax1.plot(losses_total, label='Total Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Total Loss', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2 = plt.subplot(2, 3, 2)
ax2.plot(losses_age, label='Age Loss', color='blue', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (MSE)')
ax2.set_title('Age Prediction Loss', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

ax3 = plt.subplot(2, 3, 3)
ax3.plot(losses_gender, label='Gender Loss', color='red', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss (BCE)')
ax3.set_title('Gender Classification Loss', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 나이 예측
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(age_test.numpy(), age_pred.numpy(), alpha=0.5)
ax4.plot([18, 80], [18, 80], 'r--', lw=2)
ax4.set_xlabel('Actual Age')
ax4.set_ylabel('Predicted Age')
ax4.set_title('Age Predictions', fontweight='bold')
ax4.grid(True, alpha=0.3)

# 나이 잔차
ax5 = plt.subplot(2, 3, 5)
residuals = (age_pred - age_test).numpy()
ax5.hist(residuals, bins=30, edgecolor='black')
ax5.set_xlabel('Prediction Error (years)')
ax5.set_ylabel('Frequency')
ax5.set_title('Age Prediction Errors', fontweight='bold')
ax5.grid(True, alpha=0.3)

# 성별 혼동 행렬
ax6 = plt.subplot(2, 3, 6)
correct = (gender_pred == gender_test).sum().item()
wrong = len(gender_test) - correct
ax6.bar(['Correct', 'Wrong'], [correct, wrong], color=['green', 'red'], alpha=0.7)
ax6.set_ylabel('Count')
ax6.set_title(f'Gender Classification\nAccuracy: {gender_acc:.1f}%', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('14_multi_task_results.png', dpi=150)
print("\nVisualization saved!")

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
다중 과제 학습:
✓ 과제 사이에서 특징 추출을 공유한다
✓ 예측에는 과제별 머리를 따로 둔다
✓ 손실을 가중합으로 합친다
✓ 일반화를 높일 수 있다(정칙화 효과)

구조:
  입력 → 공유층 → 갈라짐 → 과제 머리 → 출력

ADVANTAGES:
+ 표현을 공유한다(전이 학습 효과)
+ 표본 효율이 낫다
+ 암묵적인 정칙화 효과
+ 모델 하나만 배포하면 된다

CHALLENGES:
- 과제 가중치의 균형 잡기
- 과제끼리 충돌할 수 있다
- 학습이 더 복잡하다

APPLICATIONS:
- 다중 레이블 분류
- 공동 예측 과제
- 보조 과제 학습
- 분야 간 전이

TIPS:
1. 과제 가중치를 똑같이 두고 시작하여 필요하면 조정하라
2. 과제마다 성능을 따로 살피라
3. 필요하면 과제별 학습률을 쓰라
4. 과제 불확실성 가중치를 고려하라
""")
plt.show()


if __name__ == "__main__":
    pass
```

## 2. 논의

`MultiTaskNet` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `MultiTaskNet`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `MultiTaskNet`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = MultiTaskNet(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 다중 출력 신경망

`MultiTaskNet` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다.

핵심 클래스는 `MultiTaskNet`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
