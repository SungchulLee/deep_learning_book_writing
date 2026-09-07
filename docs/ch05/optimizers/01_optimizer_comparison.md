# 최적화기 비교

요즘의 딥러닝에는 절충이 서로 다른 최적화기 계열이 여럿 있다. 모멘텀을 쓰는 SGD는 일반화가 좋고, Adam은 조율을 거의 하지 않고도 빠르게 수렴하며, AdamW는 트랜스포머에서 가중치 감쇠를 바로잡고, RMSprop은 비정상 문제에서 학습률을 맞춘다. 어느 최적화기를 고를지는 구조, 데이터, 계산 예산에 달렸다.

## 코드

```python
"""
================================================================================
중급 01: 널리 쓰이는 최적화기 견주기 (SGD, Adam, RMSprop, AdamW)
================================================================================

배울 내용:
- 최적화기마다 작동하는 방식
- 각 최적화기를 언제 쓸까
- Adam과 AdamW
- 모멘텀과 적응형 학습률
- 실제 문제에서의 실용적인 비교

선수 지식:
- 입문 튜토리얼을 모두 마친다
- 기본적인 최적화 개념을 이해한다

소요 시간: 약 25분
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # 비대화형 백엔드용
import matplotlib.pyplot as plt

print("=" * 80)
print("COMPARING POPULAR OPTIMIZERS")
print("=" * 80)

# ============================================================================
# 1절: 최적화 알고리즘 개관
# ============================================================================
print("\n" + "-" * 80)
print("OPTIMIZATION ALGORITHMS OVERVIEW")
print("-" * 80)

print("""
STOCHASTIC GRADIENT DESCENT (SGD):
  • Simplest optimizer: param -= learning_rate × gradient
  • Pros: Simple, works well with momentum
  • Cons: Can be slow, sensitive to learning rate
  • Best for: Well-understood problems, when you have time to tune

SGD WITH MOMENTUM:
  • Adds "velocity" to push through local minima
  • Accumulates gradients over time: v = momentum × v + gradient
  • 잡음 섞인 기울기를 매끄럽게 한다
  • Best for: Deep networks, noisy gradients

ADAM (Adaptive Moment Estimation):
  • 매개변수마다 배움 빠르기를 맞춘다
  • Combines momentum + RMSprop
  • Pros: Works well out-of-the-box, fast convergence
  • Cons: Can generalize worse than SGD, higher memory usage
  • Best for: Quick prototyping, most deep learning tasks

ADAMW (Adam with Weight Decay):
  • Adam with correct weight decay implementation
  • Better generalization than Adam
  • Fixes Adam's weight decay bug
  • Best for: Transformers, modern architectures, when using weight decay

RMSPROP (Root Mean Square Propagation):
  • Adapts learning rate using moving average of squared gradients
  • Good for non-stationary problems
  • Best for: RNNs, online learning
""")

# ============================================================================
# 2절: 만만치 않은 최적화 문제 만들기
# ============================================================================
print("\n" + "-" * 80)
print("SETUP: Non-Convex Optimization Problem")
print("-" * 80)

# 재현성을 위해 씨앗 고정
torch.manual_seed(42)

# 비선형 양상을 갖는 합성 데이터 생성
n_samples = 200
X = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
# 비선형 함수: y = sin(x) + 0.5*x + 잡음
y = torch.sin(X) + 0.5 * X + torch.randn(n_samples, 1) * 0.1

print(f"Dataset: {n_samples} samples")
print(f"Task: Learn the non-linear function y = sin(x) + 0.5*x")

# ============================================================================
# 3절: 간단한 신경망 정의
# ============================================================================
print("\n" + "-" * 80)
print("NEURAL NETWORK ARCHITECTURE")
print("-" * 80)

class SimpleNet(nn.Module):
    """
    은닉층이 2개인 간단한 순방향 신경망
    입력 → 뉴런 20개 → 뉴런 20개 → 출력
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 구조 출력
model = SimpleNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"Architecture:")
print(model)
print(f"\nTotal parameters: {total_params}")

# ============================================================================
# 4절: 학습 함수
# ============================================================================

def train_model(model, optimizer_name, optimizer, X, y, epochs=100):
    """모델을 학습시키고 손실 이력을 돌려준다"""
    criterion = nn.MSELoss()
    loss_history = []
    
    for epoch in range(epochs):
        # 순전파
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 손실 기록
        loss_history.append(loss.item())
        
        # 진행 상황 출력
        if (epoch + 1) % 20 == 0:
            print(f"  [{optimizer_name}] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    return loss_history

# ============================================================================
# 5절: 최적화기 비교
# ============================================================================
print("\n" + "-" * 80)
print("TRAINING WITH DIFFERENT OPTIMIZERS")
print("-" * 80)

epochs = 100
learning_rate = 0.01

# 비교를 위해 결과 저장
results = {}
optimizers_to_test = {}

# 1. SGD (기본)
print("\n1. SGD (Vanilla - No Momentum):")
model_sgd = SimpleNet()
opt_sgd = optim.SGD(model_sgd.parameters(), lr=learning_rate)
optimizers_to_test['SGD'] = (model_sgd, opt_sgd)

# 2. 모멘텀을 쓰는 SGD
print("\n2. SGD with Momentum (0.9):")
model_sgd_momentum = SimpleNet()
opt_sgd_momentum = optim.SGD(model_sgd_momentum.parameters(), 
                               lr=learning_rate, momentum=0.9)
optimizers_to_test['SGD+Momentum'] = (model_sgd_momentum, opt_sgd_momentum)

# 3. RMSprop
print("\n3. RMSprop:")
model_rmsprop = SimpleNet()
opt_rmsprop = optim.RMSprop(model_rmsprop.parameters(), lr=learning_rate)
optimizers_to_test['RMSprop'] = (model_rmsprop, opt_rmsprop)

# 4. Adam
print("\n4. Adam:")
model_adam = SimpleNet()
opt_adam = optim.Adam(model_adam.parameters(), lr=learning_rate)
optimizers_to_test['Adam'] = (model_adam, opt_adam)

# 5. AdamW
print("\n5. AdamW (Adam with Weight Decay):")
model_adamw = SimpleNet()
opt_adamw = optim.AdamW(model_adamw.parameters(), lr=learning_rate, 
                         weight_decay=0.01)
optimizers_to_test['AdamW'] = (model_adamw, opt_adamw)

# 모든 모델 학습
for name, (model, optimizer) in optimizers_to_test.items():
    loss_history = train_model(model, name, optimizer, X, y, epochs)
    results[name] = loss_history

# ============================================================================
# 6절: 결과 분석
# ============================================================================
print("\n" + "-" * 80)
print("RESULTS ANALYSIS")
print("-" * 80)

print("\nFinal Loss (lower is better):")
for name, loss_history in results.items():
    final_loss = loss_history[-1]
    print(f"  {name:15s}: {final_loss:.6f}")

print("\nConvergence Speed (loss after 20 epochs):")
for name, loss_history in results.items():
    early_loss = loss_history[19]  # 20번째 에포크 (0부터 셈)
    print(f"  {name:15s}: {early_loss:.6f}")

# ============================================================================
# 7절: 수렴 시각화
# ============================================================================
print("\n" + "-" * 80)
print("VISUALIZATION")
print("-" * 80)

plt.figure(figsize=(12, 5))

# 그림 1: 손실 곡선
plt.subplot(1, 2, 1)
for name, loss_history in results.items():
    plt.plot(loss_history, label=name, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 차이를 잘 보려고 로그 척도

# 그림 2: 최종 예측
plt.subplot(1, 2, 2)
# 참 데이터 그리기
plt.scatter(X.numpy(), y.numpy(), alpha=0.3, s=10, label='True Data', color='black')

# 최적화기별 예측 그리기
colors = ['blue', 'orange', 'green', 'red', 'purple']
with torch.no_grad():
    for (name, (model, _)), color in zip(optimizers_to_test.items(), colors):
        y_pred = model(X)
        plt.plot(X.numpy(), y_pred.numpy(), label=name, linewidth=2, 
                color=color, alpha=0.7)

plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Final Predictions', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = '/home/claude/optimizer_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")

# ============================================================================
# 8절: 최적화기 자세히 설명하기
# ============================================================================
print("\n" + "-" * 80)
print("DETAILED OPTIMIZER MECHANICS")
print("-" * 80)

print("""
1. SGD (STOCHASTIC GRADIENT DESCENT):
   Update: θ = θ - lr × ∇L
   
   • Simplest algorithm
   • Each step goes directly downhill
   • Can oscillate in narrow valleys
   • Very sensitive to learning rate choice

2. SGD WITH MOMENTUM:
   Velocity: v = β × v + ∇L
   Update: θ = θ - lr × v
   
   • Accumulates past gradients (β typically 0.9)
   • Builds "momentum" to push through small bumps
   • 잡음 섞인 기울기를 매끄럽게 한다
   • Can overshoot optimal values

3. RMSPROP:
   RMS: s = β × s + (1-β) × ∇L²
   Update: θ = θ - (lr / √s) × ∇L
   
   • Adapts learning rate per parameter
   • Divides by root mean square of gradients
   • Prevents learning rate from becoming too small
   • Good for non-stationary objectives

4. ADAM (Adaptive Moment Estimation):
   Momentum: m = β₁ × m + (1-β₁) × ∇L
   Velocity: v = β₂ × v + (1-β₂) × ∇L²
   Update: θ = θ - lr × (m / √v)
   
   • Combines momentum + RMSprop
   • 매개변수마다 배움 빠르기를 맞춘다
   • Includes bias correction
   • Default β₁=0.9, β₂=0.999
   • Most popular optimizer in deep learning

5. ADAMW:
   Same as Adam but with corrected weight decay:
   Update: θ = θ - lr × (m / √v) - lr × λ × θ
   
   • Fixes Adam's weight decay implementation
   • Better regularization
   • Preferred for transformers and large models
""")

# ============================================================================
# 9절: 실용적인 지침
# ============================================================================
print("\n" + "-" * 80)
print("PRACTICAL GUIDELINES: Which Optimizer to Use?")
print("-" * 80)

print("""
🎯 USE ADAM WHEN:
   ✓ Starting a new project (good default)
   ✓ Need fast convergence
   ✓ Limited time for hyperparameter tuning
   ✓ Working with RNNs or transformers
   ✓ Example: Quick prototyping, research experiments

📊 USE SGD + MOMENTUM WHEN:
   ✓ Final model training (often better generalization)
   ✓ Have time to tune learning rate
   ✓ Training CNNs (especially ResNet, VGG)
   ✓ Want most stable long-term performance
   ✓ Example: Production models, ImageNet training

🔬 USE ADAMW WHEN:
   ✓ Training transformers (BERT, GPT, etc.)
   ✓ Using weight decay regularization
   ✓ Need better generalization than Adam
   ✓ Following modern best practices
   ✓ Example: NLP models, large-scale training

⚡ USE RMSPROP WHEN:
   ✓ Training RNNs (historically popular)
   ✓ Non-stationary problems
   ✓ Online learning scenarios
   ✓ Example: Time series, online recommendations

LEARNING RATE RECOMMENDATIONS:
• SGD: 0.01 - 0.1
• SGD + Momentum: 0.01 - 0.1
• Adam: 0.001 (1e-3)
• AdamW: 0.0001 - 0.001 (1e-4 to 1e-3)
• RMSprop: 0.001

Always start with these defaults and adjust based on loss curves!
""")

# ============================================================================
# 10절: 흔히 쓰는 초매개변수
# ============================================================================
print("\n" + "-" * 80)
print("HYPERPARAMETER REFERENCE")
print("-" * 80)

print("""
LEARNING RATE (lr):
  • Most important hyperparameter
  • Too high: Training unstable, loss explodes
  • Too low: Training too slow
  • Use learning rate schedulers to adjust during training

MOMENTUM (SGD):
  • Typical: 0.9 or 0.95
  • Higher = more smoothing but can overshoot

BETAS (Adam/AdamW):
  • β₁ (momentum): typically 0.9
  • β₂ (RMS): typically 0.999
  • Rarely need to change these

WEIGHT DECAY:
  • Regularization strength
  • Typical: 0.0001 to 0.01
  • Higher = more regularization
  • Use AdamW for correct implementation

EPSILON (Adam/RMSprop):
  • Numerical stability constant
  • Default: 1e-8
  • Rarely need to change
""")

# ============================================================================
# 요약
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. Different optimizers make different trade-offs:
   • Speed vs stability
   • Ease of use vs final performance
   • Memory efficiency

2. Adam/AdamW are great defaults:
   • Fast convergence
   • Work well out-of-the-box
   • AdamW preferred for modern architectures

3. SGD + Momentum often achieves best final performance:
   • Requires more tuning
   • Better generalization
   • Preferred for production models

4. Always monitor loss curves:
   • Smooth decrease = good
   • Oscillations = learning rate too high
   • Plateau = learning rate too low or converged

5. No single "best" optimizer:
   • Depends on problem, architecture, data
   • Experiment and compare
   • Use learning rate schedulers for better results

NEXT STEPS:
→ Experiment with different learning rates
→ Learn about learning rate schedulers
→ Try combining optimizers with regularization
→ Study advanced optimizers (RAdam, Lookahead, etc.)
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 논의

이 실험은 같은 데이터로 같은 신경망 다섯 개를 최적화기만 달리하여 학습시킨다. 기본 SGD는 평활화나 적응 없이 날 기울기를 따르므로 가장 느리게 수렴한다. 모멘텀을 더하면 속도가 쌓여 수렴이 크게 빨라지며, 특히 기울기의 방향이 진동하는 좁은 골짜기에서 그렇다.

Adam과 AdamW는 기울기의 모멘텀과 분산 모두로부터 매개변수별 학습률을 맞추므로 대체로 가장 빨리 수렴한다. AdamW는 가중치 감쇠를 적용하는 방식만 Adam과 다르다. Adam은 손실에 L2 정칙화를 더하는데(적응형 학습률과 잘 어울리지 않는다), AdamW는 매개변수 갱신에 가중치 감쇠를 바로 적용한다.

실용적인 지침은 공동체의 경험을 담고 있다. 빠른 시제품 제작과 트랜스포머에는 Adam/AdamW를, 일반화 품질을 최대로 끌어올려야 하는 CNN에는 모멘텀을 쓰는 SGD를, RNN에는 RMSprop을 쓴다. 기본 학습률은 최적화기마다 크게 다르다. SGD는 0.01~0.1, Adam은 0.001, AdamW는 0.0001~0.001이다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

