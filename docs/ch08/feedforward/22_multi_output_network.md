# 다중 출력 뉴럴 넷트웍

공유 은닉 레이어 뒤에 회귀 헤드와 이진 분류 헤드를 연결하는 다중 과제 예제이다. 데이터는 인공적으로 생성한 두 목표이며 실제 인구 특성을 추론하는 실험은 아니다.

## 1. 코드

```python
"""
14_multi_output_network.py - 다중 과제 학습

뉴럴 넷트웍 하나로 여러 과제를! 같은 입력에서 여러 출력을 동시에
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

# 합성 데이터: 피처로부터 나이와 이진 클래스를 함께 예측
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

# 나이를 표준화한다. 이 페이지에서 가장 중요한 손질이다.
# 나이를 18~80인 원래 눈금으로 두면 MSE가 학습 초반에 2500 언저리인
# 반면 BCE는 0.7 안팎이다. 두 로스를 그대로 더하면 크기가 3000배 넘게
# 차이 나므로, 아래에서 웨이트를 1.0과 1.0으로 주어도 공유 레이어가 받는
# 그래디언트는 사실상 나이 과제의 것뿐이고 성별 과제는 묻힌다.
# 표준화해 두면 두 로스가 모두 1 언저리에서 시작해, 웨이트 1.0이
# 비로소 "두 과제를 똑같이 중요하게 본다"는 뜻이 된다.
# 평균과 표준편차는 학습 집합에서만 구한다. 시험 집합까지 넣어 구하면
# 시험 자료의 정보가 학습에 새어 든다
age_mean = age[:split].mean()
age_std = age[:split].std()
age_scaled = (age - age_mean) / age_std

X_train, X_test = X[:split], X[split:]
# 학습과 평가에는 표준화한 나이를 쓴다. 사람이 읽을 지표를 낼 때만
# 아래에서 원래 눈금(년)으로 되돌린다
age_train, age_test = age_scaled[:split], age_scaled[split:]
age_test_years = age[split:]
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
    공유 레이어와 과제별 레이어를 갖는 다중 과제 학습 뉴럴 넷트웍.
    
    구조:
        입력 → 공유 레이어 → 과제별 머리로 갈라짐
    """
    
    def __init__(self, input_size):
        super().__init__()
        
        # 공유 레이어 (일반적인 피처를 배운다).
        # 두 과제의 그래디언트가 이 레이어에서 더해진다. 그래서 한 과제가 배운
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
        # 때문이다. 나이는 실수 하나, 성별은 로짓 하나이고 로스 함수도
        # 다르다. 공유 레이어가 공통된 표현을 만들고 머리가 그것을
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
        # 공유 피처 추출
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

# 과제마다 따로 둔 로스
criterion_age = nn.MSELoss()  # 회귀
criterion_gender = nn.BCEWithLogitsLoss()  # 이진 분류

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 과제 웨이트 (중요도 균형 맞추기).
# 위에서 나이를 표준화해 두었으므로 두 로스가 모두 1 언저리에서
# 출발한다. 그래서 1.0과 1.0이 정말로 "두 과제를 똑같이 중요하게
# 본다"는 뜻이 된다.
# 표준화하지 않았다면 이 값들은 균형이 아니다. 크기가 다른 로스를
# 더할 때 늘 따라오는 문제이며, 다중 과제 학습에서 가장 먼저 확인할
# 자리다. 한쪽 과제를 일부러 더 중요하게 보고 싶을 때만 이 값을
# 1이 아닌 수로 바꾼다
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
        
        # 과제마다 로스 계산
        loss_age = criterion_age(age_pred, batch_age)
        loss_gender = criterion_gender(gender_logits, batch_gender)
        
        # 결합된 로스 (가중합)
        # 두 로스를 하나로 더해 backward를 한 번만 부른다. 각각 따로
        # 부르면 공유 레이어의 그래프를 두 번 거슬러 올라가야 해서
        # retain_graph가 필요해진다. 더해 두면 그래디언트가 저절로 합쳐진다
        loss = weight_age * loss_age + weight_gender * loss_gender
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss_total += loss.item()
        epoch_loss_age += loss_age.item()
        epoch_loss_gender += loss_gender.item()
    
    # 로스 기록
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
    
    # 나이 지표 (회귀).
    # MSE는 표준화된 눈금에서 재고, MAE는 년 단위로 되돌려 잰다.
    # 표준화 눈금의 오차는 사람이 크기를 가늠하기 어렵지만, 년으로
    # 되돌리면 "평균 몇 살쯤 빗나가는가"로 곧바로 읽힌다
    age_pred_years = age_pred * age_std + age_mean
    age_mse = criterion_age(age_pred, age_test).item()
    age_mae = torch.abs(age_pred_years - age_test_years).mean().item()
    
    # 성별 지표 (분류)
    gender_acc = (gender_pred == gender_test).float().mean().item() * 100

print(f"Age Prediction (Regression):")
print(f"  MSE: {age_mse:.4f}")
print(f"  MAE: {age_mae:.4f} years")

print(f"\nGender Classification (Binary):")
print(f"  Accuracy: {gender_acc:.2f}%")

# 시각화
fig = plt.figure(figsize=(16, 10))

# 로스 곡선들
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
✓ 과제 사이에서 피처 추출을 공유한다
✓ 예측에는 과제별 머리를 따로 둔다
✓ 로스를 가중합으로 합친다
✓ 일반화를 높일 수 있다(정칙화 효과)

구조:
  입력 → 공유 레이어 → 갈라짐 → 과제 머리 → 출력

ADVANTAGES:
+ 표현을 공유한다(전이 학습 효과)
+ 표본 효율이 낫다
+ 암묵적인 정칙화 효과
+ 모델 하나만 배포하면 된다

CHALLENGES:
- 과제 웨이트의 균형 잡기
- 과제끼리 충돌할 수 있다
- 학습이 더 복잡하다

APPLICATIONS:
- 다중 레이블 분류
- 공동 예측 과제
- 보조 과제 학습
- 분야 간 전이

TIPS:
1. 과제 웨이트를 똑같이 두고 시작하여 필요하면 조정하라
2. 과제마다 성능을 따로 살피라
3. 필요하면 과제별 러닝레이트를 쓰라
4. 과제 불확실성 웨이트를 고려하라
""")
plt.show()


if __name__ == "__main__":
    pass
```

**출력:**

```
================================================================================
Multi-Task Learning Example
================================================================================
Features: 20
Training samples: 4000
Test samples: 1000

Tasks:
  Task 1: Age prediction (regression, range 18-80)
  Task 2: Gender classification (binary, 0 or 1)

================================================================================
Multi-Task Architecture
================================================================================
Multi-task model created!
Total parameters: 15,554

================================================================================
Training Setup
================================================================================
Loss functions:
  Age: MSELoss (weight=1.0)
  Gender: BCEWithLogitsLoss (weight=1.0)

================================================================================
Training Multi-Task Model
================================================================================
Epoch [ 10/100] | Total: 0.3591 | Age: 0.1650 | Gender: 0.1941
Epoch [ 20/100] | Total: 0.3192 | Age: 0.1507 | Gender: 0.1685
Epoch [ 30/100] | Total: 0.3006 | Age: 0.1441 | Gender: 0.1565
Epoch [ 40/100] | Total: 0.2817 | Age: 0.1404 | Gender: 0.1413
Epoch [ 50/100] | Total: 0.2744 | Age: 0.1367 | Gender: 0.1377
Epoch [ 60/100] | Total: 0.2644 | Age: 0.1338 | Gender: 0.1307
Epoch [ 70/100] | Total: 0.2613 | Age: 0.1352 | Gender: 0.1261
Epoch [ 80/100] | Total: 0.2378 | Age: 0.1275 | Gender: 0.1103
Epoch [ 90/100] | Total: 0.2423 | Age: 0.1304 | Gender: 0.1119
Epoch [100/100] | Total: 0.2250 | Age: 0.1252 | Gender: 0.0999

================================================================================
Evaluation
================================================================================
Age Prediction (Regression):
  MSE: 0.1408
  MAE: 4.6654 years

Gender Classification (Binary):
  Accuracy: 91.10%

Visualization saved!

================================================================================
KEY TAKEAWAYS
================================================================================

다중 과제 학습:
✓ 과제 사이에서 피처 추출을 공유한다
✓ 예측에는 과제별 머리를 따로 둔다
✓ 로스를 가중합으로 합친다
✓ 일반화를 높일 수 있다(정칙화 효과)

구조:
  입력 → 공유 레이어 → 갈라짐 → 과제 머리 → 출력

ADVANTAGES:
+ 표현을 공유한다(전이 학습 효과)
+ 표본 효율이 낫다
+ 암묵적인 정칙화 효과
+ 모델 하나만 배포하면 된다

CHALLENGES:
- 과제 웨이트의 균형 잡기
- 과제끼리 충돌할 수 있다
- 학습이 더 복잡하다

APPLICATIONS:
- 다중 레이블 분류
- 공동 예측 과제
- 보조 과제 학습
- 분야 간 전이

TIPS:
1. 과제 웨이트를 똑같이 두고 시작하여 필요하면 조정하라
2. 과제마다 성능을 따로 살피라
3. 필요하면 과제별 러닝레이트를 쓰라
4. 과제 불확실성 웨이트를 고려하라

```

## 2. 논의

MultiTaskNet은 공유 표현을 만든 뒤 연속값 하나와 이진 분류 로짓 하나를 반환한다. 회귀에는 MSE, 분류 로짓에는 BCEWithLogitsLoss를 적용하고 가중합을 역전파한다.

회귀 목표는 학습 구간의 평균과 표준편차로 표준화하며, 보고할 때 원래 단위로 되돌린다. 이는 로스의 스케일 차이를 줄이지만, 두 로스의 크기가 비슷하거나 웨이트가 같다고 공유 레이어의 경사 기여까지 같은 것은 아니다.

공유가 두 과제에 도움이 되는지는 데이터 관계와 로스 웨이트에 달려 있다. 한쪽 과제의 개선이 다른 쪽을 해칠 수 있으므로 두 검증 지표와 단일 과제 기준 모델을 함께 비교한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
두 출력에 같은 로스와 활성화를 적용하면 안 되는 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"
    연속값은 회귀 목표이고 다른 출력은 이진 로짓이다. 연속값에는 MSE를, 로짓에는 BCEWithLogitsLoss를 적용한다. 분류 출력에 시그모이드를 먼저 적용하면 로짓 로스의 규약과 달라진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
목표 표준화에서 학습 데이터만 쓰는 이유와 평가 시의 역변환을 설명하라.

</div>

??? success "연습문제 2 풀이"
    평가 데이터의 정보를 쓰지 않고 학습 부분에서 평균과 표준편차를 구한다. 연속 예측에 표준편차를 곱하고 평균을 더하면 원래 단위로 돌아간다. 원래 단위의 MAE와 표준화 단위의 MSE를 구별한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
두 로스의 웨이트가 모두 1이면 과제의 영향이 같은가?

</div>

??? success "연습문제 3 풀이"
    아니다. 로스의 단위와 경사, 헤드의 야코비안이 다르므로 공유 레이어의 기여가 다를 수 있다. 웨이트는 검증 지표와 과제 목적에 맞게 선택하고 필요하면 과제별 경사를 분석한다.

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
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

다중 과제 학습은 공유 레이어와 과제별 출력·로스를 구분한다. 목표의 스케일을 맞추는 작업과 과제의 중요도를 정하는 작업은 같지 않으며, 각 과제의 성능을 따로 확인해야 한다.
