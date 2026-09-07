# 기초

1단계: 소프트맥스 회귀의 기초

이 튜토리얼은 PyTorch에서 소프트맥스 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
"""
===============================================================================
LEVEL 1: Softmax Regression Fundamentals
===============================================================================
Difficulty: Beginner
Prerequisites: Basic Python, basic NumPy
Learning Goals:
  - Understand what softmax function does
  - Learn how cross-entropy loss works
  - Compare NumPy and PyTorch implementations
  - Understand the relationship between logits, probabilities, and loss

Time to complete: 20-30 minutes
===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn

print("=" * 80)
print("LEVEL 1: SOFTMAX REGRESSION FUNDAMENTALS")
print("=" * 80)

# =============================================================================
# 1부: 소프트맥스 함수 이해하기
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: Understanding Softmax")
print("=" * 80)

"""
What is Softmax?
----------------
Softmax converts a vector of real numbers (called logits) into a probability
distribution. Each output is between 0 and 1, and they all sum to 1.

Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

Why do we use it?
- Converts raw scores to interpretable probabilities
- Amplifies differences between scores (larger values get higher probabilities)
- Essential for multi-class classification
"""

def softmax_numpy(x):
    """
    Compute softmax values for a 1D array.
    
    Args:
        x (np.array): Input array of logits (raw scores)
    
    Returns:
        np.array: Probability distribution (sums to 1)
    
    Note: For numerical stability, we subtract the max value before exp.
          This doesn't change the result but prevents overflow.
    """
    # 수치적 안정성을 위해 최댓값을 뺀다 (exp에서 넘침을 막는다)
    x_shifted = x - np.max(x)
    exp_values = np.exp(x_shifted)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities


# 예 1: 기본적인 소프트맥스 계산
print("\nExample 1: Converting logits to probabilities")
print("-" * 80)
logits = np.array([2.0, 1.0, 0.1])
print(f"Input logits:         {logits}")
print(f"  (These are raw, unnormalized scores from a model)")

probabilities = softmax_numpy(logits)
print(f"\nOutput probabilities: {probabilities}")
print(f"  (These are interpretable as class probabilities)")
print(f"Sum of probabilities: {np.sum(probabilities):.6f}")
print(f"  (Should always equal 1.0)")

# 예 2: 로짓을 바꿨을 때의 효과
print("\n\nExample 2: How logits affect probabilities")
print("-" * 80)
logits_scenarios = [
    np.array([1.0, 1.0, 1.0]),    # All equal
    np.array([3.0, 1.0, 1.0]),    # One much larger
    np.array([10.0, 1.0, 1.0]),   # One extremely larger
]

for i, logits in enumerate(logits_scenarios, 1):
    probs = softmax_numpy(logits)
    print(f"Scenario {i}: logits = {logits}")
    print(f"            probs  = {probs}")
    print()

print("💡 Key Insight: Larger differences in logits lead to more confident predictions!")


# =============================================================================
# 2부: PyTorch에서의 소프트맥스
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: Softmax in PyTorch")
print("=" * 80)

# PyTorch 텐서로 바꾼다
logits_torch = torch.tensor([2.0, 1.0, 0.1])
print(f"\nInput (PyTorch tensor): {logits_torch}")

# 0번 차원을 따라 소프트맥스를 적용한다 (1차원 텐서에서는 유일한 차원)
probs_torch = torch.softmax(logits_torch, dim=0)
print(f"Output probabilities:   {probs_torch}")

# 배치 데이터에 대해 (여러 표본을 한꺼번에)
print("\n\nBatched Example (3 samples, 3 classes each):")
print("-" * 80)
# 모양: (batch_size, num_classes) = (3, 3)
batch_logits = torch.tensor([
    [2.0, 1.0, 0.1],   # Sample 1
    [0.5, 2.5, 1.0],   # Sample 2
    [1.5, 1.5, 1.5],   # Sample 3
])
print("Logits (3 samples x 3 classes):")
print(batch_logits)

# dim=1을 따라 소프트맥스를 적용한다 (표본마다 클래스에 걸쳐)
batch_probs = torch.softmax(batch_logits, dim=1)
print("\nProbabilities (after softmax):")
print(batch_probs)
print(f"\nSum for each sample: {batch_probs.sum(dim=1)}")
print("  (Each row sums to 1.0)")


# =============================================================================
# 3부: 교차 엔트로피 손실 이해하기
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: Cross-Entropy Loss")
print("=" * 80)

"""
What is Cross-Entropy Loss?
----------------------------
Cross-entropy measures how different your predicted probability distribution
is from the true distribution. Lower loss = better predictions.

For a single sample with true class k:
  Loss = -log(p_k)
  
where p_k is the predicted probability for the true class.

Why negative log?
- If p_k = 1.0 (perfect prediction), loss = -log(1.0) = 0
- If p_k = 0.5 (uncertain), loss = -log(0.5) = 0.69
- If p_k = 0.1 (wrong), loss = -log(0.1) = 2.30
- If p_k → 0 (very wrong), loss → infinity
"""

def cross_entropy_numpy(true_class, predicted_probs):
    """
    Compute cross-entropy loss for a single sample.
    
    Args:
        true_class (int): Index of the true class (0, 1, 2, ...)
        predicted_probs (np.array): Predicted probabilities for each class
    
    Returns:
        float: Cross-entropy loss value
    """
    # log(0)을 피하려고 작은 엡실론을 더한다
    eps = 1e-15
    predicted_probs = np.clip(predicted_probs, eps, 1 - eps)
    
    # 손실은 참 클래스의 음의 로그 확률이다
    loss = -np.log(predicted_probs[true_class])
    return loss


print("\nExample: Comparing good vs bad predictions")
print("-" * 80)

# 참 클래스는 0이다 (첫 번째 클래스)
true_class = 0

# 좋은 예측 (정답 클래스의 확률이 높다)
good_probs = np.array([0.8, 0.15, 0.05])
loss_good = cross_entropy_numpy(true_class, good_probs)

# 중간 정도의 예측 (정답 클래스의 확률이 보통이다)
medium_probs = np.array([0.5, 0.3, 0.2])
loss_medium = cross_entropy_numpy(true_class, medium_probs)

# 나쁜 예측 (정답 클래스의 확률이 낮다)
bad_probs = np.array([0.1, 0.6, 0.3])
loss_bad = cross_entropy_numpy(true_class, bad_probs)

print(f"True class: {true_class}")
print(f"\nGood prediction:   probs = {good_probs}   → loss = {loss_good:.4f}")
print(f"Medium prediction: probs = {medium_probs} → loss = {loss_medium:.4f}")
print(f"Bad prediction:    probs = {bad_probs}   → loss = {loss_bad:.4f}")
print("\n💡 Key Insight: Lower loss = better prediction on the true class!")


# =============================================================================
# 4부: PyTorch의 CrossEntropyLoss (올바른 방법)
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: PyTorch CrossEntropyLoss")
print("=" * 80)

"""
CRITICAL CONCEPT: PyTorch's CrossEntropyLoss
--------------------------------------------
nn.CrossEntropyLoss combines:
  1. Softmax (converts logits to probabilities)
  2. Log (takes logarithm)
  3. Negative Log Likelihood (computes loss)

INPUT:
  - Predictions: raw logits (unnormalized scores), NOT probabilities
  - Targets: class indices (0, 1, 2, ...), NOT one-hot vectors

DO NOT apply softmax before CrossEntropyLoss - it does it internally!
"""

# 손실 함수를 만든다
criterion = nn.CrossEntropyLoss()

print("\nExample 1: Single sample")
print("-" * 80)

# 참 클래스의 인덱스
y_true = torch.tensor([0])  # Shape: (1,) - true class is 0

# 예측 (로짓) - 소프트맥스를 적용하지 말 것!
# 모양: (1, 3) - 표본 1개, 클래스 3개
y_pred_good = torch.tensor([[3.0, 1.0, 0.5]])   # High score on class 0
y_pred_bad = torch.tensor([[0.5, 3.0, 2.0]])    # High score on class 1

loss_good = criterion(y_pred_good, y_true)
loss_bad = criterion(y_pred_bad, y_true)

print(f"True class: {y_true.item()}")
print(f"\nGood logits: {y_pred_good}")
print(f"  Loss: {loss_good.item():.4f}")
print(f"\nBad logits: {y_pred_bad}")
print(f"  Loss: {loss_bad.item():.4f}")

# 예측된 클래스를 보려면 argmax를 쓴다
pred_class_good = torch.argmax(y_pred_good, dim=1)
pred_class_bad = torch.argmax(y_pred_bad, dim=1)
print(f"\nPredicted class (good): {pred_class_good.item()} ✓")
print(f"Predicted class (bad):  {pred_class_bad.item()} ✗")


print("\n\nExample 2: Batch of samples")
print("-" * 80)

# 표본 4개, 각 클래스 3개의 배치
y_true_batch = torch.tensor([2, 0, 1, 2])  # Shape: (4,)

# 표본 4개의 로짓
y_pred_batch = torch.tensor([
    [0.5, 1.0, 3.0],   # Sample 0: should predict class 2 ✓
    [2.5, 0.5, 0.3],   # Sample 1: should predict class 0 ✓
    [0.2, 2.8, 0.5],   # Sample 2: should predict class 1 ✓
    [1.5, 2.0, 0.8],   # Sample 3: predicts class 1, true is 2 ✗
])  # Shape: (4, 3)

loss_batch = criterion(y_pred_batch, y_true_batch)
print(f"Batch loss (average): {loss_batch.item():.4f}")

# 예측을 얻는다
pred_classes = torch.argmax(y_pred_batch, dim=1)
print(f"\nTrue classes:      {y_true_batch.numpy()}")
print(f"Predicted classes: {pred_classes.numpy()}")

# 정확도를 계산한다
accuracy = (pred_classes == y_true_batch).float().mean()
print(f"Accuracy: {accuracy.item():.2%}")


# =============================================================================
# 5부: 전체 파이프라인 시각화
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: Complete Pipeline")
print("=" * 80)

print("""
The Complete Softmax Regression Pipeline:
------------------------------------------

1. Model Output (Logits)
   ↓
   [2.5, 1.0, 0.3]  ← Raw, unnormalized scores
   
2. Softmax (in CrossEntropyLoss)
   ↓
   [0.77, 0.17, 0.08]  ← Probabilities (sum to 1)
   
3. Cross-Entropy Loss
   ↓
   Compare with true class → Compute loss
   
4. Backpropagation
   ↓
   Update model weights to reduce loss

During TRAINING:
  - Use CrossEntropyLoss (it handles softmax internally)
  - Input: logits (raw scores)
  - Target: class indices

During INFERENCE (making predictions):
  - Get logits from model
  - Apply softmax to get probabilities (optional)
  - Use argmax to get predicted class
""")


# =============================================================================
# 6부: 흔한 실수와 모범 사례
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: Common Mistakes and Best Practices")
print("=" * 80)

print("""
❌ MISTAKE 1: Applying softmax before CrossEntropyLoss
--------------------------------------------------
# 틀린 방법:
probs = torch.softmax(logits, dim=1)
loss = criterion(probs, targets)  # Double softmax!

# 옳은 방법:
loss = criterion(logits, targets)  # CrossEntropyLoss applies softmax


❌ MISTAKE 2: Using one-hot encoded targets
--------------------------------------------------
# 틀린 방법:
targets = torch.tensor([[1, 0, 0], [0, 1, 0]])  # One-hot encoded

# 옳은 방법:
targets = torch.tensor([0, 1])  # Class indices


❌ MISTAKE 3: Wrong tensor shapes
--------------------------------------------------
# 표본 10개, 클래스 5개의 배치에 대해:
logits shape should be:  (10, 5)
targets shape should be: (10,)  NOT (10, 1) or (10, 5)


✅ BEST PRACTICES:
--------------------------------------------------
1. Return logits from your model (no softmax in forward())
2. Use CrossEntropyLoss for training
3. Apply softmax only at inference if you need probabilities
4. Use class indices for targets, not one-hot vectors
5. Check tensor shapes: logits (N, C), targets (N,)
""")


# =============================================================================
# 7부: 연습 문제
# =============================================================================
print("\n" + "=" * 80)
print("PART 7: Quick Practice")
print("=" * 80)

print("""
Try to predict the outcome:
---------------------------
Given:
  - True class: 1
  - Logits: [1.0, 5.0, 2.0]

Questions:
1. Which class will the model predict? (Hint: argmax)
2. Will the loss be high or low? (Hint: is the prediction correct?)

Let's check:
""")

true_class_exercise = torch.tensor([1])
logits_exercise = torch.tensor([[1.0, 5.0, 2.0]])

predicted_class = torch.argmax(logits_exercise, dim=1)
loss_exercise = criterion(logits_exercise, true_class_exercise)

print(f"True class: {true_class_exercise.item()}")
print(f"Predicted class: {predicted_class.item()}")
print(f"Loss: {loss_exercise.item():.4f}")
print(f"Correct prediction? {predicted_class.item() == true_class_exercise.item()}")


# =============================================================================
# 요약
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY - What You Learned")
print("=" * 80)

print("""
✅ Softmax converts logits into probabilities
✅ Cross-entropy measures prediction quality
✅ Lower loss = better predictions
✅ PyTorch's CrossEntropyLoss:
   - Takes logits as input (NOT probabilities)
   - Takes class indices as targets (NOT one-hot)
   - Combines softmax + log + NLL internally

다음 걸음:
-----------
→ Level 2: Build a simple neural network for classification
→ Level 3: Train on real datasets (MNIST)
→ Level 4: Advanced techniques and optimizations

🎉 Congratulations! You've mastered the fundamentals!
""")


if __name__ == "__main__":
    pass
```

## 논의

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 다중 클래스 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심적인 설계 결정을 찾아내라. 구체적인 구현 선택 세 가지를 나열하고, 각각이 소프트맥스 회귀에 왜 적절한지 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

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
기초 구현을 검증하는 종합적인 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_fundamentals():
        model = Fundamentals(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.
