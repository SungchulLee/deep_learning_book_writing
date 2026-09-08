# 기초

1단계: 소프트맥스 회귀의 기초

이 튜토리얼은 PyTorch에서 소프트맥스 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 1. 코드

```python
"""
===============================================================================
1단계: 소프트맥스 회귀 기초
===============================================================================
어려움: 첫걸음
미리 알아 둘 것: 기본 파이썬, 기본 넘파이
학습 목표:
  - 소프트맥스 함수가 무엇을 하는지 이해한다
  - 교차 엔트로피 손실이 어떻게 도는지 배운다
  - 넘파이와 PyTorch 짜보기를 견준다
  - 로짓, 확률, 손실 사이의 사이를 이해한다

소요 시간: 20~30분
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
소프트맥스란 무엇인가?
----------------
소프트맥스는 실수 벡터(로짓이라 한다)를 확률 분포로 바꾼다. 출력마다
0과 1 사이이고 모두 더하면 1이다.

식: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

왜 쓰는가?
- 날것 점수를 읽을 수 있는 확률로 바꾼다
- 점수 사이의 차이를 키운다(큰 값일수록 확률이 높아진다)
- 여러 클래스 분류에 꼭 필요하다
"""

def softmax_numpy(x):
    """
    1차원 배열의 소프트맥스 값을 계산한다.
    
    Args:
        x (np.array): 로짓(날것 점수) 입력 배열
    
    Returns:
        np.array: 확률 분포(더하면 1)
    
    눈여겨볼 것: 수치가 든든하도록 exp 앞에 최댓값을 뺀다.
          결과는 그대로이면서 넘침을 막는다.
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
교차 엔트로피 손실이란 무엇인가?
----------------------------
교차 엔트로피는 예측한 확률 분포가 참 분포와 얼마나 다른지 잰다.
손실이 작을수록 예측이 좋다.

참 클래스가 k인 표본 하나에 대해
  손실 = -log(p_k)
  
여기서 p_k은 참 클래스에 매긴 예측 확률이다.

왜 음의 로그인가?
- p_k = 1.0이면(완벽한 예측) 손실 = -log(1.0) = 0
- p_k = 0.5이면(아리송함) 손실 = -log(0.5) = 0.69
- p_k = 0.1이면(틀림) 손실 = -log(0.1) = 2.30
- p_k → 0이면(크게 틀림) 손실 → 끝없이 커진다
"""

def cross_entropy_numpy(true_class, predicted_probs):
    """
    표본 하나의 교차 엔트로피 손실을 계산한다.
    
    Args:
        true_class (int): 참 클래스의 번호(0, 1, 2, ...)
        predicted_probs (np.array): 클래스마다의 예측 확률
    
    Returns:
        float: 교차 엔트로피 손실 값
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
종요로운 개념: PyTorch의 CrossEntropyLoss
--------------------------------------------
nn.CrossEntropyLoss은 다음을 아우른다.
  1. 소프트맥스(로짓을 확률로 바꾼다)
  2. 로그(로그를 취한다)
  3. 음의 로그 가능도(손실을 계산한다)

INPUT:
  - 예측: 날것 로짓(척도 맞추지 않은 점수)이지 확률이 아니다
  - 과녁: 클래스 번호(0, 1, 2, ...)이지 원핫 벡터가 아니다

CrossEntropyLoss 앞에 소프트맥스를 걸지 마라. 안에서 절로 한다!
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
온전한 소프트맥스 회귀 흐름:
------------------------------------------

1. 모델 출력(로짓)
   ↓
   [2.5, 1.0, 0.3]  ← 날것, 척도 맞추지 않은 점수
   
2. 소프트맥스(CrossEntropyLoss 안에서)
   ↓
   [0.77, 0.17, 0.08]  ← 확률(합이 1이다)
   
3. 교차 엔트로피 손실
   ↓
   참 클래스와 견준다 → 손실을 계산한다
   
4. Backpropagation
   ↓
   손실을 줄이도록 모델 가중치를 고친다

익힐 때는
  - CrossEntropyLoss을 쓴다(안에서 소프트맥스를 다룬다)
  - 입력: 로짓(날것 점수)
  - 과녁: 클래스 번호

추론(예측)에서는
  - 모델에서 로짓을 얻는다
  - 확률이 필요하면 소프트맥스를 건다(골라 쓴다)
  - argmax으로 예측 클래스를 얻는다
""")


# =============================================================================
# 6부: 흔한 실수와 모범 사례
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: Common Mistakes and Best Practices")
print("=" * 80)

print("""
❌ 잘못 1: CrossEntropyLoss 앞에 소프트맥스를 거는 것
--------------------------------------------------
# 틀린 방법:
probs = torch.softmax(logits, dim=1)
loss = criterion(probs, targets)  # 소프트맥스를 두 번!

# 옳은 방법:
loss = criterion(logits, targets)  # CrossEntropyLoss이 소프트맥스를 건다


❌ 잘못 2: 원핫으로 바꾼 과녁을 쓰는 것
--------------------------------------------------
# 틀린 방법:
targets = torch.tensor([[1, 0, 0], [0, 1, 0]])  # 원핫으로 매김

# 옳은 방법:
targets = torch.tensor([0, 1])  # 클래스 번호


❌ 잘못 3: 텐서 모양이 틀린 것
--------------------------------------------------
# 표본 10개, 클래스 5개의 배치에 대해:
로짓의 모양은 이래야 한다:  (10, 5)
과녁의 모양은 이래야 한다: (10,)이며 (10, 1)이나 (10, 5)가 아니다


✅ 좋은 버릇:
--------------------------------------------------
1. 모델은 로짓을 돌려준다(forward()에 소프트맥스를 두지 않는다)
2. 학습에는 CrossEntropyLoss을 쓴다
3. 확률이 필요하면 추론에서만 소프트맥스를 건다
4. 과녁으로 원핫 벡터가 아니라 클래스 번호를 쓴다
5. 텐서 모양을 살핀다: 로짓 (N, C), 과녁 (N,)
""")


# =============================================================================
# 7부: 연습 문제
# =============================================================================
print("\n" + "=" * 80)
print("PART 7: Quick Practice")
print("=" * 80)

print("""
결과를 미리 짚어 보아라.
---------------------------
Given:
  - 참 클래스: 1
  - Logits: [1.0, 5.0, 2.0]

Questions:
1. 모델은 어느 클래스를 예측하겠는가?(실마리: argmax)
2. 손실은 클까 작을까?(실마리: 예측이 맞는가?)

살펴보자.
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
✅ 소프트맥스가 로짓을 확률로 바꾼다
✅ 교차 엔트로피가 예측의 좋음을 잰다
✅ 손실이 작을수록 예측이 좋다
✅ PyTorch의 CrossEntropyLoss:
   - 입력으로 로짓을 받는다(확률이 아니다)
   - 과녁으로 클래스 번호를 받는다(원핫이 아니다)
   - 안에서 소프트맥스 + 로그 + NLL을 아우른다

다음 걸음:
-----------
→ 2단계: 분류을 위한 단순한 신경망 짓기
→ 3단계: 참 데이터셋으로 익히기(MNIST)
→ 4단계: 앞선 기법와 다듬기

🎉 잘했다! 기초를 익혔다!
""")


if __name__ == "__main__":
    pass
```

## 2. 논의

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

## 정리하며

**다룬 것** — 기초

손실 계산은 모델의 출력을 최적화 목표와 이어 준다.

앞의 연습문제 4개로 직접 확인할 수 있다.
