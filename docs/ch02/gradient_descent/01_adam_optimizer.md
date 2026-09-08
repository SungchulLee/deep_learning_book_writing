# Adam 최적화기 - 적응적 모멘트 추정

이 스크립트는 Adam 최적화기, 즉 적응적 모멘트 추정을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""
================================================================================
3단계 - 보기 1: Adam 최적화기 - 맞추어 가는 적률 어림
================================================================================

학습 목표:
- 맞추어 가는 학습률 방법을 이해한다
- Adam 최적화기를 밑바닥부터 짠다
- Adam을 SGD, SGD+여세와 견준다
- Adam을 언제 쓰고 언제 다른 것을 쓸지 배운다

어려움: ⭐⭐⭐ 앞선 수준

걸리는 때: 40~50분

PREREQUISITES:
- 1단계와 2단계를 마쳤을 것
- 여세를 이해하고 있을 것
- 경사 하강법의 여러 클래스에 익숙할 것

================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("ADAM OPTIMIZER: ADAPTIVE MOMENT ESTIMATION")
print("="*80)

# ============================================================================
# 1부: Adam 이해하기
# ============================================================================
print("\n" + "="*80)
print("PART 1: WHAT IS ADAM?")
print("="*80)

print("""
Adam(맞추어 가는 적률 어림)은 다음을 아우른다.
1. 여세(기울기의 움직이는 평균)
2. RMSprop(매개변수마다 맞추어 가는 학습률)
3. 편향 바로잡기(초기화를 헤아린다)

핵심 새로움:
----------------
• 움직이는 평균 둘을 지닌다.
  - m_t: 기울기의 첫째 적률(평균)
  - v_t: 기울기의 둘째 적률(분산)

• 매개변수마다 학습률를 맞춘다
• 기본 초매개변수로도 잘 듣는다
• 깊은 학습에서 가장 널리 쓰는 최적화기다!

ALGORITHM:
----------
Initialize: m₀ = 0, v₀ = 0, t = 0

루프마다:
1. t = t + 1
2. g_t = ∇L(θ_{t-1})                    # 기울기를 계산한다
3. m_t = β₁·m_{t-1} + (1-β₁)·g_t         # 치우친 첫째 적률을 고친다
4. v_t = β₂·v_{t-1} + (1-β₂)·g_t²        # 치우친 둘째 적률을 고친다
5. m̂_t = m_t / (1 - β₁ᵗ)                 # 편향 바로잡기
6. v̂_t = v_t / (1 - β₂ᵗ)                 # 편향 바로잡기
7. θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)  # 매개변수 고치기

기본 초매개변수:
- α(학습률): 0.001
- β₁ (momentum): 0.9
- β₂ (RMSprop): 0.999
- ε(수치 든든함): 1e-8
""")

# ============================================================================
# 2부: 까다로운 데이터셋 만들기
# ============================================================================
print("\n" + "="*80)
print("PART 2: DATASET - NON-LINEAR FUNCTION")
print("="*80)

torch.manual_seed(42)
np.random.seed(42)

# 비선형 함수: y = sin(x) + 0.5*cos(2x)
n_samples = 200
X_numpy = np.linspace(-3, 3, n_samples).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + 0.5 * np.cos(2 * X_numpy) + np.random.randn(n_samples, 1) * 0.1

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print(f"Dataset: {n_samples} samples")
print("Function: y = sin(x) + 0.5*cos(2x) + noise")
print("This is non-linear - we'll use a neural network!")

# ============================================================================
# 3부: 신경망 정의
# ============================================================================

class NeuralNetwork(nn.Module):
    """
    비선형 회귀를 위한 여러 층 신경망
    Architecture: 1 → 20 → 20 → 1
    """
    def __init__(self, input_size=1, hidden_size=20, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# ============================================================================
# 4부: 학습 함수
# ============================================================================

def train_model(model, optimizer, criterion, X, y, n_epochs=500):
    """모델을 익히고 손실의 자취를 돌려준다"""
    loss_history = []
    
    for epoch in range(n_epochs):
        # 순전파
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:4d}: Loss = {loss.item():.6f}")
    
    return loss_history

# ============================================================================
# 5부: 최적화기 비교
# ============================================================================
print("\n" + "="*80)
print("PART 5: COMPARING OPTIMIZERS")
print("="*80)

n_epochs = 500
lr_sgd = 0.01
lr_adam = 0.001  # Adam typically uses smaller learning rate

print(f"\nTraining for {n_epochs} epochs...\n")

# 1. SGD(기본형)
print("1. Training with SGD...")
model_sgd = NeuralNetwork()
optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=lr_sgd)
criterion = nn.MSELoss()
loss_sgd = train_model(model_sgd, optimizer_sgd, criterion, X, y, n_epochs)

# 2. 모멘텀을 쓰는 SGD
print("\n2. Training with SGD + Momentum...")
model_momentum = NeuralNetwork()
optimizer_momentum = torch.optim.SGD(model_momentum.parameters(), lr=lr_sgd, momentum=0.9)
loss_momentum = train_model(model_momentum, optimizer_momentum, criterion, X, y, n_epochs)

# 3. Adam
print("\n3. Training with Adam...")
model_adam = NeuralNetwork()
optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=lr_adam)
loss_adam = train_model(model_adam, optimizer_adam, criterion, X, y, n_epochs)

# 4. 사용자 지정 매개변수를 쓰는 Adam
print("\n4. Training with Adam (custom β values)...")
model_adam_custom = NeuralNetwork()
optimizer_adam_custom = torch.optim.Adam(
    model_adam_custom.parameters(), 
    lr=lr_adam, 
    betas=(0.9, 0.999),  # β₁, β₂
    eps=1e-8
)
loss_adam_custom = train_model(model_adam_custom, optimizer_adam_custom, criterion, X, y, n_epochs)

# ============================================================================
# 6부: 평가
# ============================================================================
print("\n" + "="*80)
print("PART 6: EVALUATION")
print("="*80)

print("\nFinal Loss Comparison:")
print(f"  SGD:               {loss_sgd[-1]:.6f}")
print(f"  SGD + Momentum:    {loss_momentum[-1]:.6f}")
print(f"  Adam:              {loss_adam[-1]:.6f}")
print(f"  Adam (custom):     {loss_adam_custom[-1]:.6f}")

# 시각화를 위한 예측 계산
models = {
    'SGD': model_sgd,
    'SGD+Momentum': model_momentum,
    'Adam': model_adam,
    'Adam (custom)': model_adam_custom
}

predictions = {}
with torch.no_grad():
    for name, model in models.items():
        model.eval()
        predictions[name] = model(X).numpy()

# ============================================================================
# 7부: 시각화
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

fig = plt.figure(figsize=(16, 10))

# 그림 1: 손실 곡선(선형)
ax1 = plt.subplot(2, 3, 1)
ax1.plot(loss_sgd, label='SGD', linewidth=2, alpha=0.8)
ax1.plot(loss_momentum, label='SGD+Momentum', linewidth=2, alpha=0.8)
ax1.plot(loss_adam, label='Adam', linewidth=2, alpha=0.8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss (Linear Scale)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 그림 2: 손실 곡선(로그)
ax2 = plt.subplot(2, 3, 2)
ax2.plot(loss_sgd, label='SGD', linewidth=2, alpha=0.8)
ax2.plot(loss_momentum, label='SGD+Momentum', linewidth=2, alpha=0.8)
ax2.plot(loss_adam, label='Adam', linewidth=2, alpha=0.8)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss (Log Scale)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 그림 3: 학습 초기(처음 50 에폭)
ax3 = plt.subplot(2, 3, 3)
ax3.plot(loss_sgd[:50], label='SGD', linewidth=2, alpha=0.8)
ax3.plot(loss_momentum[:50], label='SGD+Momentum', linewidth=2, alpha=0.8)
ax3.plot(loss_adam[:50], label='Adam', linewidth=2, alpha=0.8)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('Early Training (First 50 Epochs)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 그림 4: SGD의 예측
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(X.numpy(), y.numpy(), alpha=0.3, label='Data', s=20)
ax4.plot(X.numpy(), predictions['SGD'], 'r-', linewidth=2, label='SGD')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('SGD Fit')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 그림 5: SGD+모멘텀의 예측
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(X.numpy(), y.numpy(), alpha=0.3, label='Data', s=20)
ax5.plot(X.numpy(), predictions['SGD+Momentum'], 'g-', linewidth=2, label='SGD+Momentum')
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_title('SGD+Momentum Fit')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 그림 6: Adam의 예측
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(X.numpy(), y.numpy(), alpha=0.3, label='Data', s=20)
ax6.plot(X.numpy(), predictions['Adam'], 'b-', linewidth=2, label='Adam')
ax6.set_xlabel('x')
ax6.set_ylabel('y')
ax6.set_title('Adam Fit')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_3_advanced/adam_comparison.png', dpi=150)
print("\n✓ Plot saved as 'adam_comparison.png'")
print("\nClose the plot window to continue...")
plt.show()

# ============================================================================
# 8부: Adam의 동작 이해하기
# ============================================================================
print("\n" + "="*80)
print("PART 8: WHY ADAM WORKS SO WELL")
print("="*80)

print("""
Adam의 핵심 이점:
----------------------

1. 맞추어 가는 학습률
   • 매개변수마다 학습률가 다르다
   • 기울기의 자취를 바탕으로 절로 잣대를 잡는다
   • 기울기가 크거나 작은 매개변수도 알맞게 고쳐진다

2. 여세의 좋은 점
   • 잡음 섞인 기울기를 매끄럽게 한다
   • 한결같은 방향으로 빨라진다
   • 흔들림을 줄인다

3. 편향 바로잡기
   • 초기화의 편향을 헤아린다(m₀=0, v₀=0)
   • 학습 이른 판에 특히 종요롭다
   • 기본 초매개변수를 더 든든하게 만든다

4. 초매개변수에 든든하다
   • 기본값(α=0.001, β₁=0.9, β₂=0.999)이 잘 듣는다
   • 학습률를 어떻게 고르든 덜 흔들린다
   • 여러 문제에 두루 좋다

Adam이 빛나는 때:
-----------------
✓ 깊은 신경망
✓ 성긴 기울기(자연어 처리, 장려하움 학습)
✓ 흐름이 바뀌는 목표
✓ 잡음 섞인 기울기
✓ "한 번 맞춰 놓고 잊는" 성능을 바랄 때

다른 것을 쓸 때:
--------------------------
• SGD+여세: 보기 일에서 일반화이 더 낫다
• 배치 경사 하강법: 볼록 문제의 작은 데이터셋
• RMSprop: 되도는 신경망과 온라인 학습
• AdaGrad: 성긴 특징
""")

# ============================================================================
# 9부: 하이퍼파라미터 민감도
# ============================================================================
print("\n" + "="*80)
print("PART 9: ADAM HYPERPARAMETERS")
print("="*80)

print("""
Adam 초매개변수:
---------------------

1. 학습률(α)
   • Default: 0.001
   • 흔한 범위: 0.0001 ~ 0.01
   • 0.001에서 비롯하고 필요하면 손본다
   • 학습률 짜기를 쓸 수 있다

2. BETA_1(β₁) - 첫째 적률 잦아듦
   • Default: 0.9
   • 여세를 다스린다
   • 클수록 더 매끄러워진다
   • 손볼 일이 드물다

3. BETA_2(β₂) - 둘째 적률 잦아듦
   • Default: 0.999
   • 맞추어 가는 학습률를 다스린다
   • 클수록 더 든든하다
   • 기울기가 성기면 0.99을 써 보아라

4. EPSILON(ε) - 수치 든든함
   • Default: 1e-8
   • 0으로 나누는 것을 막는다
   • 대개 바꿀 일이 없다

손보기 요령:
------------
1. 기본값에서 비롯하라. 열에 여덟은 잘 듣는다
2. 학습이 흔들리면 학습률를 줄여라
3. 학습이 너무 더디면 학습률를 키워라
4. 성긴 문제에서는 β₂을 0.99으로 줄여라
5. 기울기가 아주 잡음 섞이면 β₁을 0.95으로 키워라
""")

# ============================================================================
# 10부: 핵심 요점
# ============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. Adam이 가장 널리 쓰이는 최적화기다
   • 여세와 맞추어 가는 학습률를 아우른다
   • 기본 초매개변수로도 잘 듣는다
   • 웬만한 문제에 첫 고름으로 좋다

2. Adam이 SGD보다 빨리 모여든다
   • 깊은 망에 특히 이롭다
   • 기울기 크기가 제각각이어도 잘 다룬다
   • 학습이 더 든든하다

3. 여세가 종요롭다
   • SGD+여세만으로도 맨 SGD보다 크게 낫다
   • 모여듦을 빠르게 한다
   • 흔들림을 줄인다

4. 학습률는 여전히 종요롭다
   • Adam은 SGD(0.01)보다 작은 학습률(0.001)를 쓴다
   • 여전히 가장 종요로운 초매개변수다
   • 가장 좋은 결과를 얻으려면 학습률 짜기를 써라

5. 일에 맞추어 최적화기를 골라라
   • Adam: 기본 고름이며 어디서나 잘 듣는다
   • SGD+여세: 계산기 보기, 일반화이 종요로울 때
   • RMSprop: 되도는 망
   • AdaGrad: 성긴 특징

6. 수식을 이해하라
   • 첫째 적률(m): 기울기의 평균(여세)
   • 둘째 적률(v): 기울기 제곱의 평균(잣대)
   • 편향 바로잡기: 초기화의 탈을 고친다
""")

print("="*80)
print("EXPERIMENTS TO TRY")
print("="*80)
print("""
1. 여러 학습률:
   - Adam을 lr=0.01, 0.0001으로 써 보아라
   - 모여듦에 어떤 영향을 주는가?

2. 여러 망 구조:
   - 더 깊은 망(3~4층)
   - 더 넓은 망(숨은 단위 50~100개)
   - 어느 최적화기가 더 잘 듣는가?

3. Adam의 다른 클래스:
   - AdamW: torch.optim.AdamW(가중치 줄이기)
   - AMSGrad: torch.optim.Adam(..., amsgrad=True)

4. 학습률 짜기:
   - torch.optim.lr_scheduler.StepLR
   - torch.optim.lr_scheduler.ReduceLROnPlateau
""")
print("="*80)


if __name__ == "__main__":
    pass
```

## 2. 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사 초기화, 역전파, 매개변수 갱신이다. 각 구성 요소가 결정적인 역할을 한다. 최적화기는 갱신 규칙(SGD, Adam 등)을 캡슐화하고 학습률과 모멘텀 상태를 내부에서 관리한다.

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

## 연습문제

**연습문제 1.**
SGD 대신 Adam 최적화기를 쓰도록 코드를 수정하라. 100 에폭에 걸친 수렴 속도를 비교하라.

??? success "연습문제 1 풀이"
    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Adam은 적응적 학습률과 모멘텀 덕분에 보통 SGD보다
    # 빠르게 수렴한다. 다만 Adam의 최적 학습률은
    # 보통 SGD보다 작다.
    ```

---


**연습문제 2.**
학습 루프에서 `optimizer.zero_grad()`를 없애면 어떤 일이 생기는가? 실험해 보고 학습 손실에 미치는 영향을 설명하라.

??? success "연습문제 2 풀이"
    `optimizer.zero_grad()`가 없으면 경사가 반복에 걸쳐 누적된다. 실효 경사가 매 단계 커져서 매개변수 갱신이 점점 커진다. 학습이 불안정해지고 손실은 대개 발산한다. PyTorch가 경사 누적 패턴을 지원하기 위해 기본적으로 경사를 누적하기 때문이다.

---


**연습문제 3.**
최적화기에 L2 정칙화(가중치 감쇠)를 추가하고 그것이 최종 매개변수 값에 어떤 영향을 주는지 관찰하라.

??? success "연습문제 3 풀이"
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    # weight_decay는 손실에 L2 벌점항 lambda * ||w||^2을 더한다.
    # 이는 가중치를 작게 유도하여 과적합을 막을 수 있다.
    # 최종 가중치의 크기가 조금 더 작아진다.
    ```

## 정리하며

**다룬 것** — Adam 최적화기 - 적응적 모멘트 추정

학습 루프는 표준적인 PyTorch 패턴을 따른다.

핵심 클래스는 `NeuralNetwork`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
