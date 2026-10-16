# 옵티마이저 소개

옵티마이저는 저장된 경사를 사용해 계수를 갱신한다. 이 예제는 SGD의 역할과 러닝레이트, 경사 초기화를 기본 학습 루프에서 연결한다.

## 1. 코드

```python
"""
================================================================================
입문 03: PyTorch의 최적화기 소개
================================================================================

배울 내용:
- 최적화기란 무엇이며 왜 필요한가
- 최적화기가 로스로 모델의 파라메터를 갱신하는 방식
- 러닝레이트 이해하기
- 기본적인 SGD 최적화기
- 첫 완전한 학습 루프

선수 지식:
- 01_intro_to_loss_functions.py를 마친다
- 기본적인 뉴럴 넷트웍을 이해한다

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
최적화기는 로스 함수를 최소화하도록 모델의 파라메터를
갱신하는 알고리즘이다.

이렇게 생각해 보라:
1. 모델이 예측한다(순전파)
2. 로스 함수가 "이만큼 틀렸다"라고 알려 준다
3. PyTorch가 그래디언트를 계산한다(역전파)
4. 최적화기가 그래디언트를 써서 파라메터를 갱신한다
5. 로스가 충분히 낮아질 때까지 되풀이한다

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
model = nn.Linear(1, 1)  # 입력 피처 1개, 출력 1개

print("Initial model parameters:")
print(f"  Weight (w): {model.weight.item():.4f}")
print(f"  Bias (b): {model.bias.item():.4f}")

# 처음 예측하기 (형편없을 것이다!)
with torch.no_grad():  # 이 시험에서는 그래디언트를 추적하지 않는다
    y_pred_initial = model(X)

print(f"\nInitial prediction for X=5: {model(torch.tensor([[5.0]])).item():.2f}")
print(f"True value should be around: {2 * 5 + 1:.2f} = 11.0")

# ============================================================================
# 4절: 로스와 최적화기 준비
# ============================================================================
print("\n" + "-" * 80)
print("SETUP LOSS FUNCTION AND OPTIMIZER")
print("-" * 80)

# 로스 함수 - 얼마나 틀렸는지를 잰다
criterion = nn.MSELoss()

# 최적화기 - 로스를 줄이도록 파라메터를 갱신한다
# 러닝레이트(lr)은 갱신 걸음의 크기를 조절한다
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

# 처음 파라메터 저장
initial_weight = model.weight.item()
initial_bias = model.bias.item()

# 1단계: 순전파
y_pred = model(X)
print("Step 1: Forward pass (make predictions)")
print(f"  Predictions shape: {y_pred.shape}")

# 2단계: 로스 계산
loss = criterion(y_pred, y_true)
print(f"\nStep 2: Compute loss")
print(f"  Loss: {loss.item():.4f}")

# 3단계: 그래디언트 초기화 (중요!)
optimizer.zero_grad()
print(f"\nStep 3: Zero gradients")
print(f"  Why? Gradients accumulate by default in PyTorch")
print(f"  Weight gradient before zero_grad: {model.weight.grad}")

# 4단계: 역전파 (그래디언트 계산)
loss.backward()
print(f"\nStep 4: Backward pass (compute gradients)")
print(f"  Weight gradient: {model.weight.grad.item():.4f}")
print(f"  Bias gradient: {model.bias.grad.item():.4f}")
print(f"  Negative gradient = increase parameter, Positive = decrease")

# 5단계: 파라메터 갱신
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
    
    # 로스를 계산한다
    loss = criterion(y_pred, y_true)
    
    # 역전파
    optimizer.zero_grad()  # 이전 그래디언트 지우기
    loss.backward()         # 새 그래디언트 계산
    optimizer.step()        # 파라메터 갱신
    
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
   모델은 무작위 웨이트와 바이어스로 시작했다
   처음 예측은 정답과 크게 어긋났다

2. 반복(1000번 되풀이):
   a) 순전파: 모델이 예측한다
   b) 로스: 예측이 얼마나 틀렸는지 잰다
   c) 역전파: 그래디언트를 계산한다(개선할 방향)
   d) 갱신: 파라메터를 옳은 방향으로 고친다

3. RESULT:
   로스가 높은 값에서 낮은 값으로 줄었다
   파라메터가 참값으로 수렴했다(w≈2, b≈1)
   이제 모델이 정확하게 예측한다!

4. 핵심 구성 요소:
   • 로스 함수: 얼마나 틀렸는지 알려 준다
   • 최적화기: 로스를 줄이도록 파라메터를 갱신한다
   • 러닝레이트: 갱신 스트라이드를 조절한다
   • 에폭: 이 과정을 되풀이하는 횟수
""")

# ============================================================================
# 9절: 최적화기 갱신 식
# ============================================================================
print("\n" + "-" * 80)
print("HOW THE OPTIMIZER UPDATES PARAMETERS")
print("-" * 80)

print("""
확률적 경사 하강법(SGD)에서는:

    new_parameter = old_parameter - learning_rate × gradient

우리 학습에서 가져온 예:
    웨이트 기울기가 2.5이고 learning_rate가 0.01이면
    new_weight = old_weight - 0.01 × 2.5
    new_weight = old_weight - 0.025

최적화기는 모델의 모든 파라메터에 이 일을 한다!
""")

# ============================================================================
# 10절: 흔한 함정
# ============================================================================
print("\n" + "-" * 80)
print("COMMON PITFALLS (And How to Avoid Them)")
print("-" * 80)

print("""
❌ 실수 1: optimizer.zero_grad()를 잊는다
   → 그래디언트가 쌓여 잘못된 갱신이 일어난다
   ✓ loss.backward() 앞에서 늘 optimizer.zero_grad()를 부르라

❌ 실수 2: 연산 순서가 틀렸다
   → 반드시 순전파 → 로스 → zero_grad → backward → step 순이어야 한다
   ✓ 학습 루프에서 이 순서를 정확히 지켜라

❌ 실수 3: 학습 중에 torch.no_grad()를 쓴다
   → 그래디언트 계산이 막혀 모델이 학습하지 못한다
   ✓ no_grad()는 평가와 시험에만 쓰라

❌ 실수 4: 로스를 추적하지 않는다
   → 모델이 학습하고 있는지 알 수 없다
   ✓ 학습을 살피려면 로스를 규칙적으로 출력하거나 기록하라

❌ 실수 5: 러닝레이트가 너무 크거나 너무 작다
   → 모델이 발산하거나 너무 느리게 학습한다
   ✓ 0.01이나 0.001로 시작하여 로스 곡선을 보며 조정하라
""")

# ============================================================================
# 요약
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. 최적화기는 로스를 최소화하도록 모델 파라메터를 갱신한다
   
2. 기본 학습 루프 구조:
   for epoch in range(num_epochs):
       predictions = model(inputs)       # 순전파
       loss = criterion(predictions, targets)  # 로스
       optimizer.zero_grad()             # 그래디언트 초기화
       loss.backward()                   # 그래디언트 계산
       optimizer.step()                  # 파라메터 갱신

3. 러닝레이트가 매우 중요하다.
   • 매 단계에서 파라메터가 얼마나 바뀔지 조절한다
   • 너무 크면 불안정하고 너무 작으면 학습이 느리다
   • 보통 0.01이나 0.001로 시작한다

4. SGD(확률적 경사 하강법)는 가장 단순한 최적화기이다
   • 갱신: parameter -= learning_rate × gradient
   • 많은 문제에서 잘 작동한다
   • 다른 최적화기(Adam, RMSprop)는 여기에 개선을 더한다

다음 단계:
→ 러닝레이트를 달리하여 어떤 일이 일어나는지 보라
→ 더 복잡한 모델로 실험해 보라
→ 고급 최적화기(Adam, AdamW)를 배워 보라
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

??? note "전체 출력 (223줄)"

    ```
    ================================================================================
    INTRODUCTION TO OPTIMIZERS
    ================================================================================

    --------------------------------------------------------------------------------
    WHAT IS AN OPTIMIZER?
    --------------------------------------------------------------------------------

    최적화기는 로스 함수를 최소화하도록 모델의 파라메터를
    갱신하는 알고리즘이다.

    이렇게 생각해 보라:
    1. 모델이 예측한다(순전파)
    2. 로스 함수가 "이만큼 틀렸다"라고 알려 준다
    3. PyTorch가 그래디언트를 계산한다(역전파)
    4. 최적화기가 그래디언트를 써서 파라메터를 갱신한다
    5. 로스가 충분히 낮아질 때까지 되풀이한다

    최적화기가 없으면 모델은 학습할 수 없다!


    --------------------------------------------------------------------------------
    EXAMPLE: Learning to Fit a Line
    --------------------------------------------------------------------------------
    Data shape: X=torch.Size([50, 1]), y=torch.Size([50, 1])
    First 5 points:
      X=0.00, y=1.96
      X=0.20, y=2.15
      X=0.41, y=2.27
      X=0.61, y=1.17
      X=0.82, y=2.97

    --------------------------------------------------------------------------------
    CREATE A SIMPLE MODEL
    --------------------------------------------------------------------------------
    Initial model parameters:
      Weight (w): 0.8294
      Bias (b): -0.5927

    Initial prediction for X=5: 3.55
    True value should be around: 11.00 = 11.0

    --------------------------------------------------------------------------------
    SETUP LOSS FUNCTION AND OPTIMIZER
    --------------------------------------------------------------------------------
    Loss function: MSELoss()
    Optimizer: SGD (
    Parameter Group 0
        dampening: 0
        differentiable: False
        foreach: None
        fused: None
        lr: 0.01
        maximize: False
        momentum: 0
        nesterov: False
        weight_decay: 0
    )
    Learning rate: 0.01

    WHAT IS LEARNING RATE?
    - Too small: Training is very slow
    - Too large: Training is unstable, might not converge
    - Just right: Steady improvement
    - We're using: 0.01 (a common starting point)

    --------------------------------------------------------------------------------
    TRAINING LOOP - DETAILED WALKTHROUGH
    --------------------------------------------------------------------------------

    Let's train for just ONE epoch (iteration) first:

    Step 1: Forward pass (make predictions)
      Predictions shape: torch.Size([50, 1])

    Step 2: Compute loss
      Loss: 68.2641

    Step 3: Zero gradients
      Why? Gradients accumulate by default in PyTorch
      Weight gradient before zero_grad: None

    Step 4: Backward pass (compute gradients)
      Weight gradient: -95.2042
      Bias gradient: -14.9871
      Negative gradient = increase parameter, Positive = decrease

    Step 5: Update parameters using optimizer
      Old weight: 0.8294
      New weight: 1.7814
      Change: 0.9520

      Old bias: -0.5927
      New bias: -0.4428
      Change: 0.1499

    --------------------------------------------------------------------------------
    COMPLETE TRAINING - 1000 EPOCHS
    --------------------------------------------------------------------------------
    Training for 1000 epochs...

    Epoch [100/1000], Loss: 0.4150
    Epoch [200/1000], Loss: 0.3224
    Epoch [300/1000], Loss: 0.2888
    Epoch [400/1000], Loss: 0.2765
    Epoch [500/1000], Loss: 0.2720
    Epoch [600/1000], Loss: 0.2704
    Epoch [700/1000], Loss: 0.2698
    Epoch [800/1000], Loss: 0.2696
    Epoch [900/1000], Loss: 0.2695
    Epoch [1000/1000], Loss: 0.2695

    --------------------------------------------------------------------------------
    RESULTS AFTER TRAINING
    --------------------------------------------------------------------------------
    Final loss: 0.2695

    Learned parameters:
      Weight (w): 1.9990 (target was ~2.0)
      Bias (b): 1.0509 (target was ~1.0)

    Test predictions:
      X=0.0: Predicted=1.05, Expected≈1.00
      X=5.0: Predicted=11.05, Expected≈11.00
      X=10.0: Predicted=21.04, Expected≈21.00

    --------------------------------------------------------------------------------
    UNDERSTANDING WHAT HAPPENED
    --------------------------------------------------------------------------------

    1. INITIALIZATION:
       모델은 무작위 웨이트와 바이어스로 시작했다
       처음 예측은 정답과 크게 어긋났다

    2. 반복(1000번 되풀이):
       a) 순전파: 모델이 예측한다
       b) 로스: 예측이 얼마나 틀렸는지 잰다
       c) 역전파: 그래디언트를 계산한다(개선할 방향)
       d) 갱신: 파라메터를 옳은 방향으로 고친다

    3. RESULT:
       로스가 높은 값에서 낮은 값으로 줄었다
       파라메터가 참값으로 수렴했다(w≈2, b≈1)
       이제 모델이 정확하게 예측한다!

    4. 핵심 구성 요소:
       • 로스 함수: 얼마나 틀렸는지 알려 준다
       • 최적화기: 로스를 줄이도록 파라메터를 갱신한다
       • 러닝레이트: 갱신 스트라이드를 조절한다
       • 에폭: 이 과정을 되풀이하는 횟수


    --------------------------------------------------------------------------------
    HOW THE OPTIMIZER UPDATES PARAMETERS
    --------------------------------------------------------------------------------

    확률적 경사 하강법(SGD)에서는:

        new_parameter = old_parameter - learning_rate × gradient

    우리 학습에서 가져온 예:
        웨이트 기울기가 2.5이고 learning_rate가 0.01이면
        new_weight = old_weight - 0.01 × 2.5
        new_weight = old_weight - 0.025

    최적화기는 모델의 모든 파라메터에 이 일을 한다!


    --------------------------------------------------------------------------------
    COMMON PITFALLS (And How to Avoid Them)
    --------------------------------------------------------------------------------

    ❌ 실수 1: optimizer.zero_grad()를 잊는다
       → 그래디언트가 쌓여 잘못된 갱신이 일어난다
       ✓ loss.backward() 앞에서 늘 optimizer.zero_grad()를 부르라

    ❌ 실수 2: 연산 순서가 틀렸다
       → 반드시 순전파 → 로스 → zero_grad → backward → step 순이어야 한다
       ✓ 학습 루프에서 이 순서를 정확히 지켜라

    ❌ 실수 3: 학습 중에 torch.no_grad()를 쓴다
       → 그래디언트 계산이 막혀 모델이 학습하지 못한다
       ✓ no_grad()는 평가와 시험에만 쓰라

    ❌ 실수 4: 로스를 추적하지 않는다
       → 모델이 학습하고 있는지 알 수 없다
       ✓ 학습을 살피려면 로스를 규칙적으로 출력하거나 기록하라

    ❌ 실수 5: 러닝레이트가 너무 크거나 너무 작다
       → 모델이 발산하거나 너무 느리게 학습한다
       ✓ 0.01이나 0.001로 시작하여 로스 곡선을 보며 조정하라


    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. 최적화기는 로스를 최소화하도록 모델 파라메터를 갱신한다
       
    2. 기본 학습 루프 구조:
       for epoch in range(num_epochs):
           predictions = model(inputs)       # 순전파
           loss = criterion(predictions, targets)  # 로스
           optimizer.zero_grad()             # 그래디언트 초기화
           loss.backward()                   # 그래디언트 계산
           optimizer.step()                  # 파라메터 갱신

    3. 러닝레이트가 매우 중요하다.
       • 매 단계에서 파라메터가 얼마나 바뀔지 조절한다
       • 너무 크면 불안정하고 너무 작으면 학습이 느리다
       • 보통 0.01이나 0.001로 시작한다

    4. SGD(확률적 경사 하강법)는 가장 단순한 최적화기이다
       • 갱신: parameter -= learning_rate × gradient
       • 많은 문제에서 잘 작동한다
       • 다른 최적화기(Adam, RMSprop)는 여기에 개선을 더한다

    다음 단계:
    → 러닝레이트를 달리하여 어떤 일이 일어나는지 보라
    → 더 복잡한 모델로 실험해 보라
    → 고급 최적화기(Adam, AdamW)를 배워 보라

    ================================================================================
    ```


## 2. 논의

예측과 로스를 계산한 뒤 backward로 경사를 만들고 step으로 갱신한다. 이전 경사는 다음 backward 전에 초기화해야 한다. zero_grad를 순전파 앞에 둘 수도 뒤에 둘 수도 있어 모든 단계가 한 가지 엄격한 순서로만 가능한 것은 아니다.

모멘텀 없는 SGD의 갱신은 계수에서 러닝레이트×경사를 빼는 것이다. 같은 코드라도 입력 스케일과 목적의 곡률에 따라 안정적인 러닝레이트는 달라진다. 0.01 같은 한 값을 모든 과제의 기본 정답으로 볼 수 없다.

optimizer.zero_grad는 그 옵티마이저가 관리하는 계수의 경사를 초기화한다. backward는 계산 그래프를 따르므로 관리하지 않는 계수에도 경사가 생길 수 있다. 실제로 어떤 계수가 갱신되는지 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
backward와 step의 차이를 설명하라.

</div>

??? success "연습문제 1 풀이"
    backward는 경사를 계산해 누적하고 step은 옵티마이저가 관리하는 계수와 상태를 갱신한다. 하나가 다른 기능을 자동으로 대신하지 않는다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
zero_grad를 순전파 앞과 로스 계산 뒤에 두는 두 방식이 모두 가능할 조건은 무엇인가?

</div>

??? success "연습문제 2 풀이"
    이전 경사가 필요한 별도 동작이 없고, 새 backward 전에 초기화한다면 가능하다. 의도적인 경사 누적에서는 여러 backward 뒤 한 번 step하는 별도 규칙을 사용한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
optimizer 객체가 없이도 수동 경사 하강법을 구현할 수 있는가?

</div>

??? success "연습문제 3 풀이"
    가능하다. 계산한 경사로 no_grad 아래에서 계수를 직접 갱신할 수 있다. 내장 옵티마이저는 이 갱신과 상태 관리를 일관되게 구현해 주는 도구이다.

## 정리하며

옵티마이저는 경사 계산과 구분된 갱신 단계이다. 초기화 시점, 관리하는 계수와 러닝레이트의 조건을 확인한다.
