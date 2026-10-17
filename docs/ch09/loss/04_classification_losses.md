# 분류 로스

분류 로스는 모델의 로짓·확률과 목표 표현을 연결한다. BCE와 로짓 BCE, 다중 클래스 교차 엔트로피의 입력 규약을 비교한다.

## 1. 코드

```python
"""
================================================================================
입문 04: 분류 로스 함수
================================================================================

배울 내용:
- 회귀와 분류의 차이
- 이진 분류를 위한 이진 교차 엔트로피 (BCE)
- 다중 클래스 분류를 위한 교차 엔트로피 로스
- 로짓과 확률 이해하기
- 원-핫 부호화와 클래스 레이블

선수 지식:
- 앞선 입문 튜토리얼을 마친다
- 기본적인 분류 개념을 이해한다

소요 시간: 약 20분
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("CLASSIFICATION LOSS FUNCTIONS")
print("=" * 80)

# ============================================================================
# 1절: 분류와 회귀
# ============================================================================
print("\n" + "-" * 80)
print("CLASSIFICATION VS REGRESSION")
print("-" * 80)

print("""
회귀(앞선 튜토리얼에서 다루었다):
  • 연속값을 예측한다
  • 예: 집값(\$150,000), 온도(25.3°C)
  • 로스: MSE, MAE, 후버
  
CLASSIFICATION:
  • 이산 범주나 클래스를 예측한다
  • 예: 메일이 스팸인지 아닌지, 이미지가 고양이/개/새인지
  • 로스: 교차 엔트로피, 이진 교차 엔트로피
  
핵심 차이: 분류는 클래스마다 확률을 출력한다!
""")

# ============================================================================
# 2절: 이진 분류 - 이메일 스팸 탐지
# ============================================================================
print("\n" + "-" * 80)
print("BINARY CLASSIFICATION EXAMPLE: Spam Detection")
print("-" * 80)

print("Let's classify 5 emails as Spam (1) or Not Spam (0):\n")

# 참 레이블 (0 = 정상, 1 = 스팸)
true_labels = torch.tensor([0, 1, 0, 1, 1], dtype=torch.float32)
print(f"True labels: {true_labels}")
print("  0 = Not Spam, 1 = Spam\n")

# 모델의 예측 (0과 1 사이의 확률)
# 시그모이드 활성화를 거친 뒤
predicted_probs = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.6])
print(f"Predicted probabilities: {predicted_probs}")
print("  Higher = More likely to be spam\n")

# 예측 해석
print("Interpretation:")
for i, (true_label, pred_prob) in enumerate(zip(true_labels, predicted_probs)):
    true_class = "Spam" if true_label == 1 else "Not Spam"
    pred_class = "Spam" if pred_prob > 0.5 else "Not Spam"
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
    correct = "✓" if true_class == pred_class else "✗"
    
    print(f"  Email {i+1}: True={true_class:8s}, Predicted={pred_class:8s} "
          f"({confidence*100:.0f}% confident) {correct}")

# ============================================================================
# 3절: 이진 교차 엔트로피 로스 (BCE)
# ============================================================================
print("\n" + "-" * 80)
print("BINARY CROSS-ENTROPY LOSS (BCE)")
print("-" * 80)

# BCE 로스 계산
bce_criterion = nn.BCELoss()
bce_loss = bce_criterion(predicted_probs, true_labels)

print(f"BCE Loss: {bce_loss.item():.4f}\n")

print("WHAT IS BCE?")
print("  Formula: -[y × log(p) + (1-y) × log(1-p)]")
print("  where y = true label (0 or 1), p = predicted probability")

print("\nWHY THIS FORMULA?")
print("  • When true label = 1 (Spam):")
print("    Loss = -log(p) → Low if p is high (correct!)")
print("  • When true label = 0 (Not Spam):")
print("    Loss = -log(1-p) → Low if p is low (correct!)")

# 표본마다 로스 계산
print("\nPer-sample losses:")
for i in range(len(true_labels)):
    y = true_labels[i].item()
    p = predicted_probs[i].item()
    
    # BCE를 직접 계산
    if y == 1:
        sample_loss = -torch.log(torch.tensor(p))
    else:
        sample_loss = -torch.log(torch.tensor(1 - p))
    
    print(f"  Email {i+1}: True={int(y)}, Pred={p:.2f} → Loss={sample_loss.item():.4f}")

# ============================================================================
# 4절: 로짓 이해하기
# ============================================================================
print("\n" + "-" * 80)
print("UNDERSTANDING LOGITS (RAW OUTPUTS)")
print("-" * 80)

print("""
실제로 뉴럴 넷트웍은 "로짓"(가공하지 않은 무계 값)을 출력한다.
시그모이드 함수로 로짓을 확률로 바꾼다.

로짓(원본) → 시그모이드 → 확률(0에서 1)
""")

# 예시 로짓 (어떤 값이든 될 수 있다)
logits = torch.tensor([-2.0, 3.0, -1.5, 2.5, 0.5])
print(f"Raw logits: {logits}\n")

# 시그모이드로 확률로 바꾸기
probabilities = torch.sigmoid(logits)
print(f"After sigmoid: {probabilities}")

print("\nSigmoid function properties:")
print("  • logit = 0 → probability = 0.5 (uncertain)")
print("  • logit > 0 → probability > 0.5 (likely class 1)")
print("  • logit < 0 → probability < 0.5 (likely class 0)")
print("  • More extreme logits = more confident predictions")

# ============================================================================
# 5절: BCEWithLogitsLoss (더 안정적이다!)
# ============================================================================
print("\n" + "-" * 80)
print("BCEWithLogitsLoss - RECOMMENDED FOR TRAINING")
print("-" * 80)

print("""
모델 → 시그모이드 → BCE 로스 대신
쓰기:       모델 → BCEWithLogitsLoss(둘을 합친다!)

Benefits:
  ✓ 수치가 더 든든하다
  ✓ 계산이 더 빠르다
  ✓ 그래디언트 문제를 막는다
""")

# BCEWithLogitsLoss 쓰기
bce_with_logits = nn.BCEWithLogitsLoss()
loss_from_logits = bce_with_logits(logits, true_labels)

print(f"Loss using BCEWithLogitsLoss: {loss_from_logits.item():.4f}")

# 직접 계산한 것과 비교
manual_probs = torch.sigmoid(logits)
manual_loss = bce_criterion(manual_probs, true_labels)
print(f"Loss using BCE(sigmoid(logits)): {manual_loss.item():.4f}")
print("→ Same result! But BCEWithLogitsLoss is more stable\n")

# ============================================================================
# 6절: 다중 클래스 분류 - 이미지 분류
# ============================================================================
print("\n" + "-" * 80)
print("MULTI-CLASS CLASSIFICATION: Image Classification")
print("-" * 80)

print("Classifying 4 images into 3 categories: Cat, Dog, Bird\n")

# 참 레이블 (클래스 인덱스)
true_classes = torch.tensor([0, 2, 1, 0])  # 0=고양이, 1=개, 2=새
print(f"True classes: {true_classes}")
print("  Image 1: Cat (0)")
print("  Image 2: Bird (2)")
print("  Image 3: Dog (1)")
print("  Image 4: Cat (0)\n")

# 모델의 출력 (클래스별 로짓)
# 모양: (batch_size, num_classes) = (4, 3)
logits_multi = torch.tensor([
    [3.0, 1.0, 0.5],   # 이미지 1: 고양이에 대한 확신이 높다
    [0.5, 0.8, 2.5],   # 이미지 2: 새에 대한 확신이 높다
    [1.0, 2.0, 0.5],   # 이미지 3: 개에 대한 확신이 높다
    [2.5, 1.5, 1.0],   # 이미지 4: 고양이에 대한 확신이 높다
])

print(f"Model logits (raw outputs):")
print(logits_multi)
print(f"Shape: {logits_multi.shape} (4 images, 3 classes)\n")

# 소프트맥스로 로짓을 확률로 바꾸기
probs_multi = F.softmax(logits_multi, dim=1)
print(f"Probabilities after softmax:")
print(probs_multi)
print("\nNote: Each row sums to 1.0 (100% probability distributed across classes)")

# 예측 보이기
class_names = ['Cat', 'Dog', 'Bird']
print("\nPredictions:")
for i in range(len(true_classes)):
    predicted_class = torch.argmax(probs_multi[i]).item()
    confidence = probs_multi[i, predicted_class].item()
    true_class_name = class_names[true_classes[i]]
    pred_class_name = class_names[predicted_class]
    correct = "✓" if predicted_class == true_classes[i] else "✗"
    
    print(f"  Image {i+1}: True={true_class_name:4s}, Predicted={pred_class_name:4s} "
          f"({confidence*100:.1f}% confident) {correct}")

# ============================================================================
# 7절: 다중 클래스를 위한 교차 엔트로피 로스
# ============================================================================
print("\n" + "-" * 80)
print("CROSS-ENTROPY LOSS (Multi-Class)")
print("-" * 80)

# 중요: CrossEntropyLoss는 확률이 아니라 날 로짓을 받는다!
ce_criterion = nn.CrossEntropyLoss()
ce_loss = ce_criterion(logits_multi, true_classes)

print(f"Cross-Entropy Loss: {ce_loss.item():.4f}\n")

print("KEY POINTS:")
print("  1. CrossEntropyLoss takes RAW LOGITS (not probabilities!)")
print("  2. It applies softmax internally (more stable)")
print("  3. True labels are class indices (not one-hot encoded)")
print("  4. Formula: -log(probability of true class)")

# 표본별 로스 보이기
print("\nPer-sample losses:")
for i in range(len(true_classes)):
    true_class = true_classes[i].item()
    true_prob = probs_multi[i, true_class].item()
    sample_loss = -torch.log(torch.tensor(true_prob))
    
    print(f"  Image {i+1}: True class={class_names[true_class]}, "
          f"Probability={true_prob:.4f}, Loss={sample_loss:.4f}")

print("\n→ Lower probability for true class = Higher loss")

# ============================================================================
# 8절: 원-핫 부호화와 클래스 인덱스
# ============================================================================
print("\n" + "-" * 80)
print("ONE-HOT ENCODING VS CLASS INDICES")
print("-" * 80)

print("PyTorch CrossEntropyLoss uses CLASS INDICES (simpler!)")
print(f"Class indices: {true_classes}\n")

print("But you might see ONE-HOT ENCODING in other frameworks:")
one_hot = F.one_hot(true_classes, num_classes=3)
print("One-hot encoded:")
print(one_hot)
print("\nEach row has a 1 in the position of the true class, 0s elsewhere")

# 원-핫 부호화된 레이블이 있다면 바꾼다:
classes_from_one_hot = torch.argmax(one_hot, dim=1)
print(f"\nConverting back: {classes_from_one_hot}")
print(f"Same as original: {torch.equal(classes_from_one_hot, true_classes)}")

# ============================================================================
# 9절: 이진과 다중 클래스 비교
# ============================================================================
print("\n" + "-" * 80)
print("COMPARISON: Binary vs Multi-Class Classification")
print("-" * 80)

print("""
╔═══════════════════╦════════════════════╦═════════════════════════╗
║                   ║ 이진               ║ 다중 클래스             ║
╠═══════════════════╬════════════════════╬═════════════════════════╣
║ 클래스 수         ║ 2(예: 예/아니오)   ║ 3 이상(예: 고양이/개/새)║
║ 모델 출력         ║ 로짓 1개           ║ 로짓 N개(N = 클래스 수) ║
║ 활성화            ║ 시그모이드         ║ 소프트맥스              ║
║ 출력 범위         ║ [0, 1]             ║ [0, 1](합이 1)          ║
║ 로스 함수         ║ BCEWithLogitsLoss  ║ CrossEntropyLoss        ║
║ 참 레이블 형식    ║ 0 또는 1           ║ 클래스 인덱스(0~N-1)    ║
╚═══════════════════╩════════════════════╩═════════════════════════╝
""")

# ============================================================================
# 10절: 실용적인 조언
# ============================================================================
print("\n" + "-" * 80)
print("PRACTICAL TIPS")
print("-" * 80)

print("""
✓ DO:
  • 이진 분류에는 BCEWithLogitsLoss를 쓴다
  • 다중 클래스 분류에는 CrossEntropyLoss를 쓴다
  • 활성화(시그모이드나 소프트맥스)는 로스 함수가 안에서 처리하게 하라
  • CrossEntropyLoss의 레이블에는 클래스 인덱스를 쓴다
  • 수렴을 확인하려면 학습 중 로스를 살핀다

✗ DON'T:
  • BCEWithLogitsLoss 앞에 시그모이드를 걸기(안에서 이미 한다!)
  • CrossEntropyLoss 앞에 소프트맥스를 걸기(안에서 이미 한다!)
  • 원본 로짓에 BCELoss 쓰기(대신 BCEWithLogitsLoss를 쓰라)
  • CrossEntropyLoss에 원-핫 레이블 쓰기(클래스 인덱스를 쓰라)

흔한 모델 구조:
  이진:       [...레이어...] → Linear(in_features, 1) → BCEWithLogitsLoss
  다중 클래스: [...레이어...] → Linear(in_features, num_classes) → CrossEntropyLoss
""")

# ============================================================================
# 요약
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. 분류는 연속값이 아니라 이산 범주를 예측한다

2. 이진 분류(클래스 2개):
   • BCEWithLogitsLoss를 쓴다
   • 모델이 값 하나(로짓)를 출력한다
   • 시그모이드가 로짓을 확률로 바꾼다
   • 레이블은 0 또는 1이다

3. 다중 클래스 분류(클래스 3개 이상):
   • CrossEntropyLoss를 쓴다
   • 모델이 클래스마다 하나씩 N개의 값(로짓)을 출력한다
   • 소프트맥스가 로짓을 확률 분포로 바꾼다
   • 레이블은 클래스 인덱스(0, 1, 2, ...)이다

4. 두 로스 함수 모두 활성화를 안에서 처리한다
   • 로스 앞에서 시그모이드나 소프트맥스를 직접 걸지 마라!
   • 이렇게 하면 수치적으로 더 안정적이다

5. 추론할 때(예측할 때):
   • 이진: 시그모이드 뒤 0.5를 기준으로 나눈다
   • 다중 클래스: 소프트맥스 뒤 argmax를 취한다

다음 단계:
→ 간단한 이미지 분류기를 만들어 보라
→ 클래스 수를 달리하여 실험해 보라
→ 클래스 불균형과 가중 로스를 배워 보라
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

??? note "전체 출력 (236줄)"

    ```
    ================================================================================
    CLASSIFICATION LOSS FUNCTIONS
    ================================================================================

    --------------------------------------------------------------------------------
    CLASSIFICATION VS REGRESSION
    --------------------------------------------------------------------------------

    회귀(앞선 튜토리얼에서 다루었다):
      • 연속값을 예측한다
      • 예: 집값(\$150,000), 온도(25.3°C)
      • 로스: MSE, MAE, 후버
      
    CLASSIFICATION:
      • 이산 범주나 클래스를 예측한다
      • 예: 메일이 스팸인지 아닌지, 이미지가 고양이/개/새인지
      • 로스: 교차 엔트로피, 이진 교차 엔트로피
      
    핵심 차이: 분류는 클래스마다 확률을 출력한다!


    --------------------------------------------------------------------------------
    BINARY CLASSIFICATION EXAMPLE: Spam Detection
    --------------------------------------------------------------------------------
    Let's classify 5 emails as Spam (1) or Not Spam (0):

    True labels: tensor([0., 1., 0., 1., 1.])
      0 = Not Spam, 1 = Spam

    Predicted probabilities: tensor([0.1000, 0.9000, 0.2000, 0.8000, 0.6000])
      Higher = More likely to be spam

    Interpretation:
      Email 1: True=Not Spam, Predicted=Not Spam (90% confident) ✓
      Email 2: True=Spam    , Predicted=Spam     (90% confident) ✓
      Email 3: True=Not Spam, Predicted=Not Spam (80% confident) ✓
      Email 4: True=Spam    , Predicted=Spam     (80% confident) ✓
      Email 5: True=Spam    , Predicted=Spam     (60% confident) ✓

    --------------------------------------------------------------------------------
    BINARY CROSS-ENTROPY LOSS (BCE)
    --------------------------------------------------------------------------------
    BCE Loss: 0.2336

    WHAT IS BCE?
      Formula: -[y × log(p) + (1-y) × log(1-p)]
      where y = true label (0 or 1), p = predicted probability

    WHY THIS FORMULA?
      • When true label = 1 (Spam):
        Loss = -log(p) → Low if p is high (correct!)
      • When true label = 0 (Not Spam):
        Loss = -log(1-p) → Low if p is low (correct!)

    Per-sample losses:
      Email 1: True=0, Pred=0.10 → Loss=0.1054
      Email 2: True=1, Pred=0.90 → Loss=0.1054
      Email 3: True=0, Pred=0.20 → Loss=0.2231
      Email 4: True=1, Pred=0.80 → Loss=0.2231
      Email 5: True=1, Pred=0.60 → Loss=0.5108

    --------------------------------------------------------------------------------
    UNDERSTANDING LOGITS (RAW OUTPUTS)
    --------------------------------------------------------------------------------

    실제로 뉴럴 넷트웍은 "로짓"(가공하지 않은 무계 값)을 출력한다.
    시그모이드 함수로 로짓을 확률로 바꾼다.

    로짓(원본) → 시그모이드 → 확률(0에서 1)

    Raw logits: tensor([-2.0000,  3.0000, -1.5000,  2.5000,  0.5000])

    After sigmoid: tensor([0.1192, 0.9526, 0.1824, 0.9241, 0.6225])

    Sigmoid function properties:
      • logit = 0 → probability = 0.5 (uncertain)
      • logit > 0 → probability > 0.5 (likely class 1)
      • logit < 0 → probability < 0.5 (likely class 0)
      • More extreme logits = more confident predictions

    --------------------------------------------------------------------------------
    BCEWithLogitsLoss - RECOMMENDED FOR TRAINING
    --------------------------------------------------------------------------------

    모델 → 시그모이드 → BCE 로스 대신
    쓰기:       모델 → BCEWithLogitsLoss(둘을 합친다!)

    Benefits:
      ✓ 수치가 더 든든하다
      ✓ 계산이 더 빠르다
      ✓ 그래디언트 문제를 막는다

    Loss using BCEWithLogitsLoss: 0.1860
    Loss using BCE(sigmoid(logits)): 0.1860
    → Same result! But BCEWithLogitsLoss is more stable


    --------------------------------------------------------------------------------
    MULTI-CLASS CLASSIFICATION: Image Classification
    --------------------------------------------------------------------------------
    Classifying 4 images into 3 categories: Cat, Dog, Bird

    True classes: tensor([0, 2, 1, 0])
      Image 1: Cat (0)
      Image 2: Bird (2)
      Image 3: Dog (1)
      Image 4: Cat (0)

    Model logits (raw outputs):
    tensor([[3.0000, 1.0000, 0.5000],
            [0.5000, 0.8000, 2.5000],
            [1.0000, 2.0000, 0.5000],
            [2.5000, 1.5000, 1.0000]])
    Shape: torch.Size([4, 3]) (4 images, 3 classes)

    Probabilities after softmax:
    tensor([[0.8214, 0.1112, 0.0674],
            [0.1027, 0.1386, 0.7587],
            [0.2312, 0.6285, 0.1402],
            [0.6285, 0.2312, 0.1402]])

    Note: Each row sums to 1.0 (100% probability distributed across classes)

    Predictions:
      Image 1: True=Cat , Predicted=Cat  (82.1% confident) ✓
      Image 2: True=Bird, Predicted=Bird (75.9% confident) ✓
      Image 3: True=Dog , Predicted=Dog  (62.9% confident) ✓
      Image 4: True=Cat , Predicted=Cat  (62.9% confident) ✓

    --------------------------------------------------------------------------------
    CROSS-ENTROPY LOSS (Multi-Class)
    --------------------------------------------------------------------------------
    Cross-Entropy Loss: 0.3504

    KEY POINTS:
      1. CrossEntropyLoss takes RAW LOGITS (not probabilities!)
      2. It applies softmax internally (more stable)
      3. True labels are class indices (not one-hot encoded)
      4. Formula: -log(probability of true class)

    Per-sample losses:
      Image 1: True class=Cat, Probability=0.8214, Loss=0.1967
      Image 2: True class=Bird, Probability=0.7587, Loss=0.2761
      Image 3: True class=Dog, Probability=0.6285, Loss=0.4644
      Image 4: True class=Cat, Probability=0.6285, Loss=0.4644

    → Lower probability for true class = Higher loss

    --------------------------------------------------------------------------------
    ONE-HOT ENCODING VS CLASS INDICES
    --------------------------------------------------------------------------------
    PyTorch CrossEntropyLoss uses CLASS INDICES (simpler!)
    Class indices: tensor([0, 2, 1, 0])

    But you might see ONE-HOT ENCODING in other frameworks:
    One-hot encoded:
    tensor([[1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]])

    Each row has a 1 in the position of the true class, 0s elsewhere

    Converting back: tensor([0, 2, 1, 0])
    Same as original: True

    --------------------------------------------------------------------------------
    COMPARISON: Binary vs Multi-Class Classification
    --------------------------------------------------------------------------------

    ╔═══════════════════╦════════════════════╦═════════════════════════╗
    ║                   ║ 이진               ║ 다중 클래스             ║
    ╠═══════════════════╬════════════════════╬═════════════════════════╣
    ║ 클래스 수         ║ 2(예: 예/아니오)   ║ 3 이상(예: 고양이/개/새)║
    ║ 모델 출력         ║ 로짓 1개           ║ 로짓 N개(N = 클래스 수) ║
    ║ 활성화            ║ 시그모이드         ║ 소프트맥스              ║
    ║ 출력 범위         ║ [0, 1]             ║ [0, 1](합이 1)          ║
    ║ 로스 함수         ║ BCEWithLogitsLoss  ║ CrossEntropyLoss        ║
    ║ 참 레이블 형식    ║ 0 또는 1           ║ 클래스 인덱스(0~N-1)    ║
    ╚═══════════════════╩════════════════════╩═════════════════════════╝


    --------------------------------------------------------------------------------
    PRACTICAL TIPS
    --------------------------------------------------------------------------------

    ✓ DO:
      • 이진 분류에는 BCEWithLogitsLoss를 쓴다
      • 다중 클래스 분류에는 CrossEntropyLoss를 쓴다
      • 활성화(시그모이드나 소프트맥스)는 로스 함수가 안에서 처리하게 하라
      • CrossEntropyLoss의 레이블에는 클래스 인덱스를 쓴다
      • 수렴을 확인하려면 학습 중 로스를 살핀다

    ✗ DON'T:
      • BCEWithLogitsLoss 앞에 시그모이드를 걸기(안에서 이미 한다!)
      • CrossEntropyLoss 앞에 소프트맥스를 걸기(안에서 이미 한다!)
      • 원본 로짓에 BCELoss 쓰기(대신 BCEWithLogitsLoss를 쓰라)
      • CrossEntropyLoss에 원-핫 레이블 쓰기(클래스 인덱스를 쓰라)

    흔한 모델 구조:
      이진:       [...레이어...] → Linear(in_features, 1) → BCEWithLogitsLoss
      다중 클래스: [...레이어...] → Linear(in_features, num_classes) → CrossEntropyLoss


    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. 분류는 연속값이 아니라 이산 범주를 예측한다

    2. 이진 분류(클래스 2개):
       • BCEWithLogitsLoss를 쓴다
       • 모델이 값 하나(로짓)를 출력한다
       • 시그모이드가 로짓을 확률로 바꾼다
       • 레이블은 0 또는 1이다

    3. 다중 클래스 분류(클래스 3개 이상):
       • CrossEntropyLoss를 쓴다
       • 모델이 클래스마다 하나씩 N개의 값(로짓)을 출력한다
       • 소프트맥스가 로짓을 확률 분포로 바꾼다
       • 레이블은 클래스 인덱스(0, 1, 2, ...)이다

    4. 두 로스 함수 모두 활성화를 안에서 처리한다
       • 로스 앞에서 시그모이드나 소프트맥스를 직접 걸지 마라!
       • 이렇게 하면 수치적으로 더 안정적이다

    5. 추론할 때(예측할 때):
       • 이진: 시그모이드 뒤 0.5를 기준으로 나눈다
       • 다중 클래스: 소프트맥스 뒤 argmax를 취한다

    다음 단계:
    → 간단한 이미지 분류기를 만들어 보라
    → 클래스 수를 달리하여 실험해 보라
    → 클래스 불균형과 가중 로스를 배워 보라

    ================================================================================
    ```


## 2. 논의

BCELoss는 확률, BCEWithLogitsLoss는 로짓을 받는다. 후자는 시그모이드와 로그 항을 안정적으로 결합한다. BCE라는 이름만 보고 시그모이드가 항상 내장되어 있다고 생각하면 안 된다.

다중 클래스 CrossEntropyLoss는 로짓과 정수 인덱스 목표를 흔히 사용한다. 로짓과 같은 모양의 부동소수점 확률 목표도 지원하므로 원핫을 모두 금지하지 않는다. 이진·다중 레이블의 로짓 BCE 목표는 보통 출력과 같은 모양이다.

레이블 1인 관측과 혼동 행렬의 참양성은 다른 표현이다. 로스는 정답 클래스의 확률을 평가하며, 예측 임계값을 적용한 뒤의 TP·FP 등의 사건과 구분한다. 포화·클리핑과 집계 규약도 구현별로 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
시그모이드가 포함된 로스와 포함되지 않은 로스를 구분하라.

</div>

??? success "풀이"
    BCEWithLogitsLoss는 로짓을 받아 시그모이드와 BCE를 결합하고 BCELoss는 이미 계산한 확률을 받는다. 두 번 적용하지 않도록 출력 레이어와 함께 선택한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
다중 클래스 인덱스 목표와 확률 목표의 모양·자료형을 설명하라.

</div>

??? success "풀이"
    로짓 `(B, K)`에서 인덱스는 `(B,)` long이고, 확률 목표는 `(B, K)`의 부동소수점 텐서이다. 각 확률 목표가 유효한 분포인지 사용자가 확인해야 한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
같은 정확도에서 BCE가 달라질 수 있는 이유를 설명하라.

</div>

??? success "풀이"
    같은 임계값의 클래스 예측이어도 확률의 크기는 다를 수 있다. 로스는 확신도와 정답의 관계를 반영하므로 정확도 하나와 같지 않다.

## 정리하며

분류 학습에서는 출력과 목표 규약을 먼저 맞춘다. 예측 로스와 임계값 기반 정확도를 구분하고, 확률·로짓을 해당 API에 전달한다.
