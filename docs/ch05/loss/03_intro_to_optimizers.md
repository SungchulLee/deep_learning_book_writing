# 최적화기 소개

최적화기는 계산된 기울기를 써서 손실 함수를 최소화하도록 모델의 매개변수를 갱신한다. SGD는 가장 단순한 최적화기로 갱신 규칙 $\theta \leftarrow \theta - \eta \nabla L$을 적용한다. 학습률은 걸음의 크기를 조절하며 조율해야 할 가장 중요한 초매개변수이다. 최적화기가 없으면 모델은 배울 수 없다.

## 코드

```python
"""
================================================================================
입문 03: PyTorch의 최적화기 소개
================================================================================

배울 내용:
- 최적화기란 무엇이며 왜 필요한가
- 최적화기가 손실로 모델의 매개변수를 갱신하는 방식
- 학습률 이해하기
- 기본적인 SGD 최적화기
- 첫 완전한 학습 루프

선수 지식:
- 01_intro_to_loss_functions.py를 마친다
- 기본적인 신경망을 이해한다

소요 시간: 약 20분
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 80)
print("INTRODUCTION TO OPTIMIZERS")
print("=" * 80)

# ============================================================================
# 1절: 최적화기란 무엇인가?
# ============================================================================
print("\n" + "-" * 80)
print("WHAT IS AN OPTIMIZER?")
print("-" * 80)

print("""
최적화기는 손실 함수를 최소화하도록 모델의 매개변수를
갱신하는 알고리즘이다.

이렇게 생각해 보라:
1. Your model makes predictions (forward pass)
2. 손실 함수가 "이만큼 틀렸다"라고 알려 준다
3. PyTorch computes gradients (backward pass)
4. 최적화기가 기울기를 써서 매개변수를 갱신한다
5. 손실이 충분히 낮아질 때까지 되풀이한다

최적화기가 없으면 모델은 학습할 수 없다!
""")

# ============================================================================
# 2절: 간단한 예제 - 직선 배우기
# ============================================================================
print("\n" + "-" * 80)
print("EXAMPLE: Learning to Fit a Line")
print("-" * 80)

# 직선 y = 2x + 1 위의 데이터 점 생성
torch.manual_seed(42)  # 재현성을 위해
X = torch.linspace(0, 10, 50).reshape(-1, 1)  # 0부터 10까지 점 50개
y_true = 2 * X + 1 + torch.randn(50, 1) * 0.5  # y = 2x + 1 + 잡음

print(f"Data shape: X={X.shape}, y={y_true.shape}")
print(f"First 5 points:")
for i in range(5):
    print(f"  X={X[i].item():.2f}, y={y_true[i].item():.2f}")

# ============================================================================
# 3절: 간단한 모델 만들기
# ============================================================================
print("\n" + "-" * 80)
print("CREATE A SIMPLE MODEL")
print("-" * 80)

# 배우려는 것: y = w*x + b
# 이를 선형 회귀라 한다
model = nn.Linear(1, 1)  # 입력 특징 1개, 출력 1개

print("Initial model parameters:")
print(f"  Weight (w): {model.weight.item():.4f}")
print(f"  Bias (b): {model.bias.item():.4f}")

# 처음 예측하기 (형편없을 것이다!)
with torch.no_grad():  # 이 시험에서는 기울기를 추적하지 않는다
    y_pred_initial = model(X)

print(f"\nInitial prediction for X=5: {model(torch.tensor([[5.0]])).item():.2f}")
print(f"True value should be around: {2 * 5 + 1:.2f} = 11.0")

# ============================================================================
# 4절: 손실과 최적화기 준비
# ============================================================================
print("\n" + "-" * 80)
print("SETUP LOSS FUNCTION AND OPTIMIZER")
print("-" * 80)

# 손실 함수 - 얼마나 틀렸는지를 잰다
criterion = nn.MSELoss()

# 최적화기 - 손실을 줄이도록 매개변수를 갱신한다
# 학습률(lr)은 갱신 걸음의 크기를 조절한다
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")
print(f"Learning rate: {learning_rate}")

print("\nWHAT IS LEARNING RATE?")
print("- Too small: Training is very slow")
print("- Too large: Training is unstable, might not converge")
print("- Just right: Steady improvement")
print(f"- We're using: {learning_rate} (a common starting point)")

# ============================================================================
# 5절: 학습 루프 - 한 단계씩
# ============================================================================
print("\n" + "-" * 80)
print("TRAINING LOOP - DETAILED WALKTHROUGH")
print("-" * 80)

print("\nLet's train for just ONE epoch (iteration) first:\n")

# 처음 매개변수 저장
initial_weight = model.weight.item()
initial_bias = model.bias.item()

# 1단계: 순전파
y_pred = model(X)
print("Step 1: Forward pass (make predictions)")
print(f"  Predictions shape: {y_pred.shape}")

# 2단계: 손실 계산
loss = criterion(y_pred, y_true)
print(f"\nStep 2: Compute loss")
print(f"  Loss: {loss.item():.4f}")

# 3단계: 기울기 초기화 (중요!)
optimizer.zero_grad()
print(f"\nStep 3: Zero gradients")
print(f"  Why? Gradients accumulate by default in PyTorch")
print(f"  Weight gradient before zero_grad: {model.weight.grad}")

# 4단계: 역전파 (기울기 계산)
loss.backward()
print(f"\nStep 4: Backward pass (compute gradients)")
print(f"  Weight gradient: {model.weight.grad.item():.4f}")
print(f"  Bias gradient: {model.bias.grad.item():.4f}")
print(f"  Negative gradient = increase parameter, Positive = decrease")

# 5단계: 매개변수 갱신
optimizer.step()
print(f"\nStep 5: Update parameters using optimizer")
print(f"  Old weight: {initial_weight:.4f}")
print(f"  New weight: {model.weight.item():.4f}")
print(f"  Change: {model.weight.item() - initial_weight:.4f}")
print(f"\n  Old bias: {initial_bias:.4f}")
print(f"  New bias: {model.bias.item():.4f}")
print(f"  Change: {model.bias.item() - initial_bias:.4f}")

# ============================================================================
# 6절: 완전한 학습 루프
# ============================================================================
print("\n" + "-" * 80)
print("COMPLETE TRAINING - 1000 EPOCHS")
print("-" * 80)

# 모델 되돌리기
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 1000
print_every = 100  # 100 에포크마다 진행 상황 출력

print(f"Training for {num_epochs} epochs...\n")

for epoch in range(num_epochs):
    # 순전파
    y_pred = model(X)
    
    # 손실을 계산한다
    loss = criterion(y_pred, y_true)
    
    # 역전파
    optimizer.zero_grad()  # 이전 기울기 지우기
    loss.backward()         # 새 기울기 계산
    optimizer.step()        # 매개변수 갱신
    
    # 진행 상황 출력
    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ============================================================================
# 7절: 학습 뒤의 결과
# ============================================================================
print("\n" + "-" * 80)
print("RESULTS AFTER TRAINING")
print("-" * 80)

print(f"Final loss: {loss.item():.4f}")
print(f"\nLearned parameters:")
print(f"  Weight (w): {model.weight.item():.4f} (target was ~2.0)")
print(f"  Bias (b): {model.bias.item():.4f} (target was ~1.0)")

# 시험 예측
test_inputs = torch.tensor([[0.0], [5.0], [10.0]])
with torch.no_grad():
    predictions = model(test_inputs)

print(f"\nTest predictions:")
for i, (x, pred) in enumerate(zip(test_inputs, predictions)):
    expected = 2 * x.item() + 1
    print(f"  X={x.item():.1f}: Predicted={pred.item():.2f}, Expected≈{expected:.2f}")

# ============================================================================
# 8절: 학습 과정 이해하기
# ============================================================================
print("\n" + "-" * 80)
print("UNDERSTANDING WHAT HAPPENED")
print("-" * 80)

print("""
1. INITIALIZATION:
   모델은 무작위 가중치와 편향으로 시작했다
   처음 예측은 정답과 크게 어긋났다

2. ITERATION (repeated 1000 times):
   a) Forward: Model makes predictions
   b) Loss: Measure how wrong predictions are
   c) Backward: Compute gradients (direction to improve)
   d) Step: Update parameters in the right direction

3. RESULT:
   손실이 높은 값에서 낮은 값으로 줄었다
   Parameters converged to the true values (w≈2, b≈1)
   이제 모델이 정확하게 예측한다!

4. 핵심 구성 요소:
   • 손실 함수: 얼마나 틀렸는지 알려 준다
   • 최적화기: 손실을 줄이도록 매개변수를 갱신한다
   • 학습률: 갱신 보폭을 조절한다
   • 에폭: 이 과정을 되풀이하는 횟수
""")

# ============================================================================
# 9절: 최적화기 갱신 식
# ============================================================================
print("\n" + "-" * 80)
print("HOW THE OPTIMIZER UPDATES PARAMETERS")
print("-" * 80)

print("""
For Stochastic Gradient Descent (SGD):

    new_parameter = old_parameter - learning_rate × gradient

우리 학습에서 가져온 예:
    If weight gradient = 2.5 and learning_rate = 0.01
    new_weight = old_weight - 0.01 × 2.5
    new_weight = old_weight - 0.025

최적화기는 모델의 모든 매개변수에 이 일을 한다!
""")

# ============================================================================
# 10절: 흔한 함정
# ============================================================================
print("\n" + "-" * 80)
print("COMMON PITFALLS (And How to Avoid Them)")
print("-" * 80)

print("""
❌ MISTAKE 1: Forgetting optimizer.zero_grad()
   → 기울기가 쌓여 잘못된 갱신이 일어난다
   ✓ Always call optimizer.zero_grad() before loss.backward()

❌ 실수 2: 연산 순서가 틀렸다
   → 반드시 순전파 → 손실 → zero_grad → backward → step 순이어야 한다
   ✓ 학습 루프에서 이 순서를 정확히 지켜라

❌ MISTAKE 3: Using torch.no_grad() during training
   → 기울기 계산이 막혀 모델이 학습하지 못한다
   ✓ Only use no_grad() for evaluation/testing

❌ 실수 4: 손실을 추적하지 않는다
   → 모델이 학습하고 있는지 알 수 없다
   ✓ 학습을 살피려면 손실을 규칙적으로 출력하거나 기록하라

❌ 실수 5: 학습률이 너무 크거나 너무 작다
   → 모델이 발산하거나 너무 느리게 학습한다
   ✓ 0.01이나 0.001로 시작하여 손실 곡선을 보며 조정하라
""")

# ============================================================================
# 요약
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. 최적화기는 손실을 최소화하도록 모델 매개변수를 갱신한다
   
2. 기본 학습 루프 구조:
   for epoch in range(num_epochs):
       predictions = model(inputs)       # 순전파
       loss = criterion(predictions, targets)  # 손실
       optimizer.zero_grad()             # 기울기 초기화
       loss.backward()                   # 기울기 계산
       optimizer.step()                  # 매개변수 갱신

3. 학습률이 매우 중요하다.
   • 매 단계에서 매개변수가 얼마나 바뀔지 조절한다
   • Too high = unstable, too low = slow learning
   • 보통 0.01이나 0.001로 시작한다

4. SGD (Stochastic Gradient Descent) is the simplest optimizer
   • Updates: parameter -= learning_rate × gradient
   • 많은 문제에서 잘 작동한다
   • Other optimizers (Adam, RMSprop) add improvements

다음 단계:
→ 학습률을 달리하여 어떤 일이 일어나는지 보라
→ 더 복잡한 모델로 실험해 보라
→ Learn about advanced optimizers (Adam, AdamW)
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 논의

학습 루프는 다섯 단계를 엄격한 순서로 따른다. 순전파(예측 계산), 손실 계산(오차 재기), 기울기 초기화(앞 단계에서 쌓인 기울기 지우기), 역전파(역전파로 기울기 계산), 매개변수 갱신(기울기로 가중치 조정)이다. 어느 한 단계라도 빠뜨리거나 순서를 바꾸면 학습이 잘못된다.

학습률은 갱신 단계마다 각 매개변수가 얼마나 바뀔지를 정한다. 너무 크면 손실이 진동하거나 발산하고, 너무 작으면 학습이 견딜 수 없이 느려진다. SGD에서는 0.01에서 시작하여 손실 곡선을 보며 조정하는 것이 흔하다.

SGD는 가장 단순한 갱신 규칙 $\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla L$을 구현한다. 여기서 $\eta$은 학습률이고 $\nabla L$은 $\theta$에 대한 손실의 기울기이다. Adam 같은 더 발전된 최적화기는 모멘텀과 적응형 학습률을 더하지만, 모든 최적화 방법을 이해하려면 SGD를 아는 것이 근본이다.

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

