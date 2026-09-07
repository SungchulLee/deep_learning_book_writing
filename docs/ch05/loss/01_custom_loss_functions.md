# 사용자 정의 손실 함수

표준 손실 함수가 언제나 분야의 목표와 들어맞는 것은 아니다. 사용자 정의 손실 함수를 쓰면 분할의 겹침을 최적화하거나(다이스 손실), 클래스 불균형을 다루거나(초점 손실), 여러 목표를 결합하거나, 물리 법칙에 기반한 제약을 담을 수 있다. 사용자 정의 손실을 만들 때에는 모든 연산이 미분 가능하고 수치적으로 안정해야 한다.

## 코드

```python
"""
================================================================================
고급 01: 사용자 정의 손실 함수 만들기
================================================================================

배울 내용:
- 사용자 정의 손실을 언제 왜 만드는가
- 사용자 정의 손실 함수를 구현하는 법
- 여러 손실 항 결합하기
- 불균형한 데이터를 위한 가중 손실
- 초점 손실, 다이스 손실을 비롯한 고급 손실

선수 지식:
- 입문과 중급 튜토리얼을 마친다
- PyTorch 자동 미분을 잘 이해한다

소요 시간: 약 30분
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("CREATING CUSTOM LOSS FUNCTIONS")
print("=" * 80)

# ============================================================================
# 1절: 왜 사용자 정의 손실 함수를 만드는가?
# ============================================================================
print("\n" + "-" * 80)
print("WHY CREATE CUSTOM LOSS FUNCTIONS?")
print("-" * 80)

print("""
Standard losses (MSE, CrossEntropy) don't always match your goals:

1. 분야에 특화된 목적:
   • 의료 영상: 분할 겹침을 재는 다이스 손실
   • 객체 탐지: 경계 상자를 위한 IoU 손실
   • GAN: 적대적 손실

2. 데이터 불균형 다루기:
   • 어려운 예제를 위한 초점 손실
   • 드문 클래스를 위한 가중 손실

3. 다중 과제 학습:
   • 여러 손실을 결합한다
   • 서로 다른 목적의 균형을 잡는다

4. 맞춤 제약:
   • 물리 지식을 반영한 손실
   • 특정 성질을 강제한다

5. 연구와 실험:
   • 새로운 착상을 시험한다
   • 기존 방법을 개선한다
""")

# ============================================================================
# 2절: 기본적인 사용자 정의 손실 - 함수 방식
# ============================================================================
print("\n" + "-" * 80)
print("METHOD 1: Custom Loss as a Function")
print("-" * 80)

def custom_mse_loss(predictions, targets):
    """
    평균제곱오차를 직접 구현하기
    시연을 위한 것이다. 실무에서는 nn.MSELoss()를 쓰라!
    """
    # 차이의 제곱 계산
    squared_diff = (predictions - targets) ** 2
    
    # 평균 내기
    loss = torch.mean(squared_diff)
    
    return loss

# 시험해 보기
pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.5, 3.5])

loss = custom_mse_loss(pred, target)
print(f"Custom MSE Loss: {loss.item():.4f}")

# PyTorch의 MSE와 비교
pytorch_loss = F.mse_loss(pred, target)
print(f"PyTorch MSE Loss: {pytorch_loss.item():.4f}")
print(f"Match: {torch.allclose(loss, pytorch_loss)}\n")

print("KEY POINTS:")
print("  ✓ Use torch operations (not numpy) for autograd")
print("  ✓ Make sure output is a scalar (for backpropagation)")
print("  ✓ All operations must be differentiable")

# ============================================================================
# 3절: 사용자 정의 손실 - 클래스 방식 (권장)
# ============================================================================
print("\n" + "-" * 80)
print("METHOD 2: Custom Loss as a Class (Recommended)")
print("-" * 80)

class WeightedMSELoss(nn.Module):
    """
    표본별 가중치를 쓰는 MSE 손실
    어떤 표본이 다른 것보다 중요할 때 쓸모 있다
    """
    def __init__(self, reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, predictions, targets, weights=None):
        """
        인수:
            predictions: 모델의 예측
            targets: 참값
            weights: 표본별 가중치 (선택 사항, 기본값은 모두 1)
        """
        # 제곱 오차 계산
        squared_error = (predictions - targets) ** 2
        
        # 가중치가 주어지면 적용
        if weights is not None:
            squared_error = squared_error * weights
        
        # 줄이기 적용
        if self.reduction == 'mean':
            return torch.mean(squared_error)
        elif self.reduction == 'sum':
            return torch.sum(squared_error)
        else:  # 'none'
            return squared_error

# 시험해 보기
criterion = WeightedMSELoss()

# 예: 뒤쪽 표본이 더 중요하다
weights = torch.tensor([0.5, 1.0, 2.0])  # 중요도 올리기
weighted_loss = criterion(pred, target, weights)

print(f"Weighted MSE Loss: {weighted_loss.item():.4f}")
print(f"Unweighted MSE Loss: {loss.item():.4f}")
print(f"\nThe weighted loss is higher because we emphasized the later samples")

# ============================================================================
# 4절: 초점 손실 - 불균형 분류를 위해
# ============================================================================
print("\n" + "-" * 80)
print("FOCAL LOSS: Handling Class Imbalance")
print("-" * 80)

print("""
PROBLEM: Imbalanced datasets (e.g., 95% negative, 5% positive)
  • 늘 음성으로 예측해도 정확도 95%가 나온다
  • 분류하기 어려운 예제가 무시된다
  
해법: 초점 손실
  • 쉬운 예제의 가중치를 낮춘다
  • 어려운 예제에 집중한다
  • Formula: FL = -α(1-p)^γ log(p)
    where γ controls focusing (typical: 2)
""")

class FocalLoss(nn.Module):
    """
    이진 분류를 위한 초점 손실
    
    논문: "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        인수:
            alpha: 클래스 균형을 위한 가중 인수
            gamma: 집중 매개변수 (클수록 어려운 예에 더 집중한다)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        인수:
            predictions: 모델의 예측 (로짓)
            targets: 참값 (0 또는 1)
        """
        # 로짓을 확률로 바꾸기
        probs = torch.sigmoid(predictions)
        
        # 초점 가중치 계산
        # 양성 클래스: (1-p)^γ
        # 음성 클래스: p^γ
        focal_weight = torch.where(
            targets == 1,
            (1 - probs) ** self.gamma,
            probs ** self.gamma
        )
        
        # BCE 손실 계산
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        
        # 초점 가중치와 alpha 적용
        focal_loss = self.alpha * focal_weight * bce_loss
        
        return torch.mean(focal_loss)

# 초점 손실 시연
print("\nExample: Imbalanced binary classification")
print("Dataset: 90% class 0, 10% class 1\n")

# 초점 손실 만들기
focal_criterion = FocalLoss(alpha=0.25, gamma=2.0)
bce_criterion = nn.BCEWithLogitsLoss()

# 예시 예측과 목푯값
logits = torch.tensor([2.0, -1.5, -2.0, 3.0, -1.0])  # 모델의 날 출력
targets = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])     # 참 레이블

# 해석을 위해 로짓을 확률로 바꾸기
probs = torch.sigmoid(logits)

print("Sample predictions and difficulty:")
for i, (logit, prob, target) in enumerate(zip(logits, probs, targets)):
    correct = (prob > 0.5 and target == 1) or (prob < 0.5 and target == 0)
    confidence = prob if target == 1 else 1 - prob
    difficulty = "EASY" if confidence > 0.8 else "HARD"
    status = "✓" if correct else "✗"
    
    print(f"  Sample {i+1}: Target={int(target)}, Prob={prob:.3f}, "
          f"{difficulty} {status}")

# 손실 계산
focal_loss = focal_criterion(logits, targets)
bce_loss = bce_criterion(logits, targets)

print(f"\nStandard BCE Loss: {bce_loss.item():.4f}")
print(f"Focal Loss: {focal_loss.item():.4f}")
print("\nFocal loss emphasizes the hard examples more!")

# ============================================================================
# 5절: 다이스 손실 - 분할을 위해
# ============================================================================
print("\n" + "-" * 80)
print("DICE LOSS: For Segmentation Tasks")
print("-" * 80)

print("""
다이스 계수: 예측과 참값이 얼마나 겹치는지 잰다
  • 의료 영상 분할에 쓴다
  • Range: 0 (no overlap) to 1 (perfect overlap)
  • Formula: Dice = 2|A ∩ B| / (|A| + |B|)
  
DICE LOSS = 1 - Dice Coefficient
""")

class DiceLoss(nn.Module):
    """
    이진 분할을 위한 다이스 손실
    예측과 목푯값의 겹침을 잰다
    """
    def __init__(self, smooth=1.0):
        """
        인수:
            smooth: 0으로 나누는 것을 막는 평활 인수
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        인수:
            predictions: 모델의 예측 (시그모이드를 거친 0~1의 값)
            targets: 참 이진 마스크 (0 또는 1)
        """
        # 텐서 펼치기
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # 교집합과 합집합 계산
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        # 다이스 계수
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 다이스 손실
        return 1 - dice

# 다이스 손실 시연
print("\nExample: Binary segmentation")

# 간단한 5x5 "이미지" 만들기
true_mask = torch.tensor([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=torch.float32)

# 좋은 예측 (참값에 가깝다)
good_pred = torch.tensor([
    [0, 0, 0, 0, 0],
    [0, 0.9, 0.8, 0.9, 0],
    [0, 0.85, 0.95, 0.85, 0],
    [0, 0.9, 0.8, 0.9, 0],
    [0, 0, 0, 0, 0]
], dtype=torch.float32)

# 나쁜 예측 (겹침이 적다)
bad_pred = torch.tensor([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0.8, 0.9, 0.85, 0, 0],
    [0.9, 0.8, 0.9, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=torch.float32)

dice_criterion = DiceLoss()

good_loss = dice_criterion(good_pred, true_mask)
bad_loss = dice_criterion(bad_pred, true_mask)

print(f"Good prediction Dice Loss: {good_loss.item():.4f}")
print(f"Bad prediction Dice Loss: {bad_loss.item():.4f}")
print("\nLower loss = better overlap!")

# ============================================================================
# 6절: 여러 손실 결합하기
# ============================================================================
print("\n" + "-" * 80)
print("COMBINING MULTIPLE LOSS TERMS")
print("-" * 80)

print("""
여러 목적을 한꺼번에 최적화하고 싶을 때가 많다.
  • 재구성 + 정칙화
  • 과제 손실 + 일관성 손실
  • Multiple task losses (multi-task learning)
  
접근: 손실의 가중합
  Total Loss = α₁ × Loss₁ + α₂ × Loss₂ + ...
""")

class CombinedLoss(nn.Module):
    """
    학습 가능하거나 고정된 가중치로 여러 손실 함수를 결합한다
    """
    def __init__(self, loss_weights=None):
        """
        인수:
            loss_weights: 손실 이름을 가중치에 대응시키는 사전
                         None이면 가중치를 모두 같게 한다
        """
        super(CombinedLoss, self).__init__()
        self.loss_weights = loss_weights or {}
        
        # 개별 손실 정의
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, predictions, targets):
        """손실의 가중 결합을 계산한다"""
        # 가중치 얻기 (기본값 1.0)
        w_mse = self.loss_weights.get('mse', 1.0)
        w_l1 = self.loss_weights.get('l1', 1.0)
        
        # 개별 손실 계산
        mse = self.mse_loss(predictions, targets)
        l1 = self.l1_loss(predictions, targets)
        
        # 합친다
        total_loss = w_mse * mse + w_l1 * l1
        
        # 전체와 성분을 돌려준다 (기록에 쓸모 있다)
        return total_loss, {'mse': mse.item(), 'l1': l1.item()}

# 결합된 손실 시험
combined_criterion = CombinedLoss(loss_weights={'mse': 0.7, 'l1': 0.3})

pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.5, 3.5])

total_loss, components = combined_criterion(pred, target)

print("\nCombined Loss Example:")
print(f"  MSE component: {components['mse']:.4f} (weight: 0.7)")
print(f"  L1 component: {components['l1']:.4f} (weight: 0.3)")
print(f"  Total loss: {total_loss.item():.4f}")

print("\nWHY COMBINE LOSSES?")
print("  • MSE: Smooth gradients, penalizes large errors")
print("  • L1: Robust to outliers")
print("  • Combination: Balance both properties!")

# ============================================================================
# 7절: 사용자 정의 손실의 좋은 관행
# ============================================================================
print("\n" + "-" * 80)
print("BEST PRACTICES FOR CUSTOM LOSSES")
print("-" * 80)

print("""
✓ DO:
  1. 손실에는 nn.Module 기반 클래스를 쓴다
  2. Keep all operations in PyTorch (not numpy)
  3. 작은 예제로 기울기 흐름을 시험한다
  4. Add numerical stability (smooth terms, clamps)
  5. 손실 함수를 잘 문서화한다
  6. 기본 초매개변수를 제공한다
  7. 역전파를 위해 스칼라를 돌려준다
  8. Consider numerical stability (avoid log(0), div by 0)

✗ DON'T:
  1. Use .item() inside loss (breaks gradients)
  2. 제자리 연산을 함부로 쓴다
  3. 경계 상황 처리를 잊는다
  4. 손실이 텐서가 아닌 상수에 의존하게 만든다
  5. 기울기가 없는 연산을 쓴다

손실 시험하기:
  1. Check it returns correct shape (scalar)
  2. Verify gradients flow: loss.backward()
  3. 이미 아는 입력과 출력으로 시험한다
  4. 참조 구현이 있으면 견주어 본다
  5. 경계 상황에서 수치 안정성을 확인한다
""")

# 예: 사용자 정의 손실 시험하기
def test_custom_loss():
    """사용자 정의 손실을 시험하는 틀"""
    print("\nTesting Custom Loss:")
    
    # 1. 손실 만들기
    loss_fn = WeightedMSELoss()
    
    # 2. 기울기를 갖는 시험 입력 만들기
    pred = torch.randn(10, requires_grad=True)
    target = torch.randn(10)
    
    # 3. 손실 계산
    loss = loss_fn(pred, target)
    
    # 4. 모양 확인
    assert loss.dim() == 0, "Loss should be scalar!"
    print(f"  ✓ Shape check passed: {loss.shape}")
    
    # 5. 역전파 시험
    loss.backward()
    assert pred.grad is not None, "Gradients should flow!"
    print(f"  ✓ Gradient check passed")
    
    # 6. 수치 안정성 시험
    edge_pred = torch.tensor([0.0, 1e-10, 1e10])
    edge_target = torch.tensor([0.0, 0.0, 1e10])
    edge_loss = loss_fn(edge_pred, edge_target)
    assert not torch.isnan(edge_loss), "Loss should handle edge cases!"
    print(f"  ✓ Numerical stability check passed")
    
    print("  All tests passed! ✓\n")

test_custom_loss()

# ============================================================================
# 요약
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. 표준 손실이 맞지 않을 때 맞춤 손실을 만든다.
   • 분야에 특화된 목적
   • 데이터 불균형 다루기
   • 다중 과제 학습
   • 연구 실험

2. 두 가지 접근:
   • 함수: 단순하며 빠른 실험에 알맞다
   • Class (nn.Module): Professional, configurable, recommended

3. 고급 손실 예:
   • 초점 손실: 불균형 분류용
   • 다이스 손실: 분할 겹침용
   • 결합 손실: 여러 목적

4. 구현 요령:
   • PyTorch 연산만 쓴다
   • 역전파를 위해 스칼라를 돌려준다
   • 수치 안정성을 더한다
   • 철저히 시험한다

5. 흔한 패턴:
   • 중요도를 반영한 가중 손실
   • 여러 항 결합
   • 클래스 균형 가중치
   • 어려운 예제 캐기

다음 단계:
→ 불균형 데이터셋에 초점 손실을 구현해 보라
→ 손실 조합을 실험해 보라
→ 문제에 맞는 분야 특화 손실을 만들어 보라
→ 새로운 손실 함수를 다룬 논문을 살펴보라
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 논의

사용자 정의 손실은 간단한 함수로도, `nn.Module`을 상속한 클래스로도 구현할 수 있다. 클래스 방식을 권한다. 설정(초매개변수를 속성으로 저장하기)을 지원하고, 모델 파이프라인과 잘 어울리며, 관심사가 뚜렷이 나뉘기 때문이다. 자동 미분과 어울리려면 모든 연산에 (NumPy가 아니라) PyTorch 텐서를 써야 한다.

초점 손실은 쉬운 예의 비중을 낮추고 어려운 예에 집중하여 클래스 불균형을 다룬다. 조절 인수 $(1-p_t)^\gamma$은 잘 분류된 예가 손실에 기여하는 몫을 지수적으로 줄인다. $\gamma=2$이면 90%의 확신으로 분류된 표본은 50%의 확신으로 분류된 표본보다 손실에 100배 적게 기여한다.

다이스 손실은 예측한 분할 마스크와 참 마스크의 겹침을 잰다. 화소별 교차 엔트로피와 달리 다이스 손실은 예측의 전체 구조를 고려하므로, 전경과 배경의 넓이가 크게 불균형한 분할 과제에 특히 효과적이다.

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

