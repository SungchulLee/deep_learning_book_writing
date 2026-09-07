# 분류 손실

분류 손실은 연속값이 아니라 이산적인 범주를 다루므로 회귀 손실과 근본적으로 다르다. 이진 교차 엔트로피는 시그모이드 활성화로 두 클래스 문제를 다루고, 교차 엔트로피 손실은 소프트맥스로 다중 클래스 문제를 다룬다. 두 손실 함수 모두 수치 안정성을 위해 활성화를 내부에 품고 있다.

## 코드

```python
"""
================================================================================
입문 04: 분류 손실 함수
================================================================================

배울 내용:
- 회귀와 분류의 차이
- 이진 분류를 위한 이진 교차 엔트로피 (BCE)
- 다중 클래스 분류를 위한 교차 엔트로피 손실
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
REGRESSION (Covered in previous tutorials):
  • Predict continuous values
  • Example: House price ($150,000), Temperature (25.3°C)
  • Loss: MSE, MAE, Huber
  
CLASSIFICATION:
  • Predict discrete categories/classes
  • Example: Email is Spam/Not Spam, Image is Cat/Dog/Bird
  • Loss: Cross-Entropy, Binary Cross-Entropy
  
Key difference: Classification outputs probabilities for each class!
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
# 3절: 이진 교차 엔트로피 손실 (BCE)
# ============================================================================
print("\n" + "-" * 80)
print("BINARY CROSS-ENTROPY LOSS (BCE)")
print("-" * 80)

# BCE 손실 계산
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

# 표본마다 손실 계산
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
In practice, neural networks output "logits" (raw, unbounded values).
We convert logits to probabilities using the Sigmoid function.

Logit (raw) → Sigmoid → Probability (0 to 1)
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
Instead of: Model → Sigmoid → BCE Loss
Use:        Model → BCEWithLogitsLoss (combines both!)

Benefits:
  ✓ 수치가 더 든든하다
  ✓ Faster computation
  ✓ Prevents gradient problems
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
# 7절: 다중 클래스를 위한 교차 엔트로피 손실
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

# 표본별 손실 보이기
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
║                   ║ BINARY             ║ MULTI-CLASS             ║
╠═══════════════════╬════════════════════╬═════════════════════════╣
║ Classes           ║ 2 (e.g., Yes/No)   ║ 3+ (e.g., Cat/Dog/Bird) ║
║ Model Output      ║ 1 logit            ║ N logits (N = classes)  ║
║ Activation        ║ Sigmoid            ║ Softmax                 ║
║ Output Range      ║ [0, 1]             ║ [0, 1] (sum to 1)       ║
║ Loss Function     ║ BCEWithLogitsLoss  ║ CrossEntropyLoss        ║
║ True Label Format ║ 0 or 1             ║ Class index (0 to N-1)  ║
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
  • Use BCEWithLogitsLoss for binary classification
  • Use CrossEntropyLoss for multi-class classification
  • Let loss functions handle activation (sigmoid/softmax) internally
  • Use class indices for labels in CrossEntropyLoss
  • Monitor loss during training to check convergence

✗ DON'T:
  • Apply sigmoid before BCEWithLogitsLoss (it does it internally!)
  • Apply softmax before CrossEntropyLoss (it does it internally!)
  • Use BCELoss with raw logits (use BCEWithLogitsLoss instead)
  • One-hot encode labels for CrossEntropyLoss (use class indices)

COMMON MODEL ARCHITECTURES:
  Binary:     [...layers...] → Linear(in_features, 1) → BCEWithLogitsLoss
  Multi-Class: [...layers...] → Linear(in_features, num_classes) → CrossEntropyLoss
""")

# ============================================================================
# 요약
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. Classification predicts discrete categories, not continuous values

2. Binary Classification (2 classes):
   • Use BCEWithLogitsLoss
   • Model outputs 1 value (logit)
   • Sigmoid converts logit → probability
   • Labels are 0 or 1

3. Multi-Class Classification (3+ classes):
   • Use CrossEntropyLoss
   • Model outputs N values (logits), one per class
   • Softmax converts logits → probability distribution
   • Labels are class indices (0, 1, 2, ...)

4. Both loss functions handle activation internally
   • Don't manually apply sigmoid/softmax before loss!
   • More numerically stable this way

5. For inference (making predictions):
   • Binary: threshold at 0.5 after sigmoid
   • Multi-class: take argmax after softmax

NEXT STEPS:
→ Build a simple image classifier
→ Experiment with different numbers of classes
→ Learn about class imbalance and weighted losses
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 논의

이진 교차 엔트로피는 예측 확률이 이진 레이블과 얼마나 잘 맞는지를 잰다. 참 양성(레이블 = 1)에서는 손실이 $-\log(p)$이며 $p \to 1$이면 0에, $p \to 0$이면 무한대에 가까워진다. 참 음성(레이블 = 0)에서는 손실이 $-\log(1-p)$이며 거동이 반대이다. 이러한 로그 벌점은 확신에 찬 오답을 강하게 말린다.

시그모이드를 적용한 뒤 `BCELoss`을 쓰는 것보다 `BCEWithLogitsLoss`을 선호한다. 수치 안정성을 위해 로그-합-지수 기법을 쓰기 때문이다. 로짓이 아주 크거나 아주 작으면 시그모이드를 먼저 계산할 때 부동소수점 정밀도 때문에 정확히 0이나 1이 되어 $\log(0) = -\infty$이 될 수 있다. 합쳐진 식은 시그모이드를 겉으로 계산하지 않으므로 이를 피한다.

다중 클래스 분류에서 `CrossEntropyLoss`은 소프트맥스와 음의 로그가능도를 한 연산으로 합친다. (확률이 아니라) 날 로짓과 (원-핫 벡터가 아니라) 정수 클래스 인덱스를 받는다. 표본마다의 손실은 그저 $-\log(p_{\text{true class}})$이며, 여기서 $p$은 옳은 클래스의 소프트맥스 확률이다.

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

