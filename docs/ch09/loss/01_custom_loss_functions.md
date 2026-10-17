# 사용자 정의 로스 함수

사용자 정의 로스를 함수와 nn.Module로 작성하는 예제이다. 자동 미분이 필요한 예측 경로를 텐서 연산으로 유지하고, 가중·마스킹·집계와 수치 범위를 명확히 정한다.

## 1. 코드

??? note "코드 (515줄)"

    ```python
    """
    ================================================================================
    고급 01: 사용자 정의 로스 함수 만들기
    ================================================================================

    배울 내용:
    - 사용자 정의 로스를 언제 왜 만드는가
    - 사용자 정의 로스 함수를 구현하는 법
    - 여러 로스 항 결합하기
    - 불균형한 데이터를 위한 가중 로스
    - 초점 로스, 다이스 로스를 비롯한 고급 로스

    선수 지식:
    - 입문과 중급 튜토리얼을 마친다
    - PyTorch 자동 미분을 잘 이해한다

    소요 시간: 약 30분
    ================================================================================
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    # 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
    # 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
    torch.manual_seed(0)

    print("=" * 80)
    print("CREATING CUSTOM LOSS FUNCTIONS")
    print("=" * 80)

    # ============================================================================
    # 1절: 왜 사용자 정의 로스 함수를 만드는가?
    # ============================================================================
    print("\n" + "-" * 80)
    print("WHY CREATE CUSTOM LOSS FUNCTIONS?")
    print("-" * 80)

    print("""
    표준 로스(MSE, 교차 엔트로피)이 늘 목표에 들어맞지는 않는다.

    1. 분야에 특화된 목적:
       • 의료 영상: 분할 겹침을 재는 다이스 로스
       • 객체 탐지: 경계 상자를 위한 IoU 로스
       • GAN: 적대적 로스

    2. 데이터 불균형 다루기:
       • 어려운 예제를 위한 초점 로스
       • 드문 클래스를 위한 가중 로스

    3. 다중 과제 학습:
       • 여러 로스를 결합한다
       • 서로 다른 목적의 균형을 잡는다

    4. 맞춤 제약:
       • 물리 지식을 반영한 로스
       • 특정 성질을 강제한다

    5. 연구와 실험:
       • 새로운 착상을 시험한다
       • 기존 방법을 개선한다
    """)

    # ============================================================================
    # 2절: 기본적인 사용자 정의 로스 - 함수 방식
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
    # 3절: 사용자 정의 로스 - 클래스 방식 (권장)
    # ============================================================================
    print("\n" + "-" * 80)
    print("METHOD 2: Custom Loss as a Class (Recommended)")
    print("-" * 80)

    class WeightedMSELoss(nn.Module):
        """
        표본별 웨이트를 쓰는 MSE 로스
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
                weights: 표본별 웨이트 (선택 사항, 기본값은 모두 1)
            """
            # 제곱 오차 계산
            squared_error = (predictions - targets) ** 2

            # 웨이트가 주어지면 적용
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
    # 4절: 초점 로스 - 불균형 분류를 위해
    # ============================================================================
    print("\n" + "-" * 80)
    print("FOCAL LOSS: Handling Class Imbalance")
    print("-" * 80)

    print("""
    문제: 불균형한 데이터셋(예: 음성 95%, 양성 5%)
      • 늘 음성으로 예측해도 정확도 95%가 나온다
      • 분류하기 어려운 예제가 무시된다

    해법: 초점 로스
      • 쉬운 예제의 웨이트를 낮춘다
      • 어려운 예제에 집중한다
      • Formula: FL = -α(1-p)^γ log(p)
        여기서 γ는 초점의 세기를 정한다(흔히 2)
    """)

    class FocalLoss(nn.Module):
        """
        이진 분류를 위한 초점 로스

        논문: "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002
        """
        def __init__(self, alpha=0.25, gamma=2.0):
            """
            인수:
                alpha: 클래스 균형을 위한 가중 인수
                gamma: 집중 파라메터 (클수록 어려운 예에 더 집중한다)
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

            # 초점 웨이트 계산
            # 양성 클래스: (1-p)^γ
            # 음성 클래스: p^γ
            focal_weight = torch.where(
                targets == 1,
                (1 - probs) ** self.gamma,
                probs ** self.gamma
            )

            # BCE 로스 계산
            bce_loss = F.binary_cross_entropy_with_logits(
                predictions, targets, reduction='none'
            )

            # 초점 웨이트와 alpha 적용
            focal_loss = self.alpha * focal_weight * bce_loss

            return torch.mean(focal_loss)

    # 초점 로스 시연
    print("\nExample: Imbalanced binary classification")
    print("Dataset: 90% class 0, 10% class 1\n")

    # 초점 로스 만들기
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

    # 로스 계산
    focal_loss = focal_criterion(logits, targets)
    bce_loss = bce_criterion(logits, targets)

    print(f"\nStandard BCE Loss: {bce_loss.item():.4f}")
    print(f"Focal Loss: {focal_loss.item():.4f}")
    print("\nFocal loss emphasizes the hard examples more!")

    # ============================================================================
    # 5절: 다이스 로스 - 분할을 위해
    # ============================================================================
    print("\n" + "-" * 80)
    print("DICE LOSS: For Segmentation Tasks")
    print("-" * 80)

    print("""
    다이스 계수: 예측과 참값이 얼마나 겹치는지 잰다
      • 의료 영상 분할에 쓴다
      • 범위: 0(전혀 안 겹침)부터 1(완전히 겹침)까지
      • Formula: Dice = 2|A ∩ B| / (|A| + |B|)

    다이스 로스 = 1 - 다이스 계수
    """)

    class DiceLoss(nn.Module):
        """
        이진 분할을 위한 다이스 로스
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

            # 다이스 로스
            return 1 - dice

    # 다이스 로스 시연
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
    # 6절: 여러 로스 결합하기
    # ============================================================================
    print("\n" + "-" * 80)
    print("COMBINING MULTIPLE LOSS TERMS")
    print("-" * 80)

    print("""
    여러 목적을 한꺼번에 최적화하고 싶을 때가 많다.
      • 재구성 + 정칙화
      • 과제 로스 + 일관성 로스
      • 여러 과제 로스(다중 과제 학습)

    접근: 로스의 가중합
      전체 로스 = α₁ × 로스₁ + α₂ × 로스₂ + ...
    """)

    class CombinedLoss(nn.Module):
        """
        학습 가능하거나 고정된 웨이트로 여러 로스 함수를 결합한다
        """
        def __init__(self, loss_weights=None):
            """
            인수:
                loss_weights: 로스 이름을 웨이트에 대응시키는 사전
                             None이면 웨이트를 모두 같게 한다
            """
            super(CombinedLoss, self).__init__()
            self.loss_weights = loss_weights or {}

            # 개별 로스 정의
            self.mse_loss = nn.MSELoss()
            self.l1_loss = nn.L1Loss()

        def forward(self, predictions, targets):
            """로스의 가중 결합을 계산한다"""
            # 웨이트 얻기 (기본값 1.0)
            w_mse = self.loss_weights.get('mse', 1.0)
            w_l1 = self.loss_weights.get('l1', 1.0)

            # 개별 로스 계산
            mse = self.mse_loss(predictions, targets)
            l1 = self.l1_loss(predictions, targets)

            # 합친다
            total_loss = w_mse * mse + w_l1 * l1

            # 전체와 성분을 돌려준다 (기록에 쓸모 있다)
            return total_loss, {'mse': mse.item(), 'l1': l1.item()}

    # 결합된 로스 시험
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
    # 7절: 사용자 정의 로스의 좋은 관행
    # ============================================================================
    print("\n" + "-" * 80)
    print("BEST PRACTICES FOR CUSTOM LOSSES")
    print("-" * 80)

    print("""
    ✓ DO:
      1. 로스에는 nn.Module 기반 클래스를 쓴다
      2. 모든 연산을 PyTorch로 한다(넘파이 말고)
      3. 작은 예제로 그래디언트 흐름을 시험한다
      4. 수치 안정성을 더한다(완충 항, 값 제한)
      5. 로스 함수를 잘 문서화한다
      6. 기본 초파라메터를 제공한다
      7. 역전파를 위해 스칼라를 돌려준다
      8. 수치 안정성을 살핀다(log(0)과 0으로 나누기를 피한다)

    ✗ DON'T:
      1. 로스 안에서 .item()을 쓴다(그래디언트가 끊긴다)
      2. 제자리 연산을 함부로 쓴다
      3. 경계 상황 처리를 잊는다
      4. 로스가 텐서가 아닌 상수에 의존하게 만든다
      5. 그래디언트가 없는 연산을 쓴다

    로스 시험하기:
      1. 올바른 모양(스칼라)을 돌려주는지 확인한다
      2. 그래디언트가 흐르는지 확인한다: loss.backward()
      3. 이미 아는 입력과 출력으로 시험한다
      4. 참조 구현이 있으면 견주어 본다
      5. 경계 상황에서 수치 안정성을 확인한다
    """)

    # 예: 사용자 정의 로스 시험하기
    def test_custom_loss():
        """사용자 정의 로스를 시험하는 틀"""
        print("\nTesting Custom Loss:")

        # 1. 로스 만들기
        loss_fn = WeightedMSELoss()

        # 2. 그래디언트를 갖는 시험 입력 만들기
        pred = torch.randn(10, requires_grad=True)
        target = torch.randn(10)

        # 3. 로스 계산
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
    1. 표준 로스가 맞지 않을 때 맞춤 로스를 만든다.
       • 분야에 특화된 목적
       • 데이터 불균형 다루기
       • 다중 과제 학습
       • 연구 실험

    2. 두 가지 접근:
       • 함수: 단순하며 빠른 실험에 알맞다
       • 클래스(nn.Module): 전문적이고 설정 가능하며 권장된다

    3. 고급 로스 예:
       • 초점 로스: 불균형 분류용
       • 다이스 로스: 분할 겹침용
       • 결합 로스: 여러 목적

    4. 구현 요령:
       • PyTorch 연산만 쓴다
       • 역전파를 위해 스칼라를 돌려준다
       • 수치 안정성을 더한다
       • 철저히 시험한다

    5. 흔한 패턴:
       • 중요도를 반영한 가중 로스
       • 여러 항 결합
       • 클래스 균형 웨이트
       • 어려운 예제 캐기

    다음 단계:
    → 불균형 데이터셋에 초점 로스를 구현해 보라
    → 로스 조합을 실험해 보라
    → 문제에 맞는 분야 특화 로스를 만들어 보라
    → 새로운 로스 함수를 다룬 논문을 살펴보라
    """)
    print("=" * 80)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (197줄)"

    ```
    ================================================================================
    CREATING CUSTOM LOSS FUNCTIONS
    ================================================================================

    --------------------------------------------------------------------------------
    WHY CREATE CUSTOM LOSS FUNCTIONS?
    --------------------------------------------------------------------------------

    표준 로스(MSE, 교차 엔트로피)이 늘 목표에 들어맞지는 않는다.

    1. 분야에 특화된 목적:
       • 의료 영상: 분할 겹침을 재는 다이스 로스
       • 객체 탐지: 경계 상자를 위한 IoU 로스
       • GAN: 적대적 로스

    2. 데이터 불균형 다루기:
       • 어려운 예제를 위한 초점 로스
       • 드문 클래스를 위한 가중 로스

    3. 다중 과제 학습:
       • 여러 로스를 결합한다
       • 서로 다른 목적의 균형을 잡는다

    4. 맞춤 제약:
       • 물리 지식을 반영한 로스
       • 특정 성질을 강제한다

    5. 연구와 실험:
       • 새로운 착상을 시험한다
       • 기존 방법을 개선한다


    --------------------------------------------------------------------------------
    METHOD 1: Custom Loss as a Function
    --------------------------------------------------------------------------------
    Custom MSE Loss: 0.2500
    PyTorch MSE Loss: 0.2500
    Match: True

    KEY POINTS:
      ✓ Use torch operations (not numpy) for autograd
      ✓ Make sure output is a scalar (for backpropagation)
      ✓ All operations must be differentiable

    --------------------------------------------------------------------------------
    METHOD 2: Custom Loss as a Class (Recommended)
    --------------------------------------------------------------------------------
    Weighted MSE Loss: 0.2917
    Unweighted MSE Loss: 0.2500

    The weighted loss is higher because we emphasized the later samples

    --------------------------------------------------------------------------------
    FOCAL LOSS: Handling Class Imbalance
    --------------------------------------------------------------------------------

    문제: 불균형한 데이터셋(예: 음성 95%, 양성 5%)
      • 늘 음성으로 예측해도 정확도 95%가 나온다
      • 분류하기 어려운 예제가 무시된다
      
    해법: 초점 로스
      • 쉬운 예제의 웨이트를 낮춘다
      • 어려운 예제에 집중한다
      • Formula: FL = -α(1-p)^γ log(p)
        여기서 γ는 초점의 세기를 정한다(흔히 2)


    Example: Imbalanced binary classification
    Dataset: 90% class 0, 10% class 1

    Sample predictions and difficulty:
      Sample 1: Target=1, Prob=0.881, EASY ✓
      Sample 2: Target=0, Prob=0.182, EASY ✓
      Sample 3: Target=0, Prob=0.119, EASY ✓
      Sample 4: Target=1, Prob=0.953, EASY ✓
      Sample 5: Target=0, Prob=0.269, HARD ✓

    Standard BCE Loss: 0.1634
    Focal Loss: 0.0017

    Focal loss emphasizes the hard examples more!

    --------------------------------------------------------------------------------
    DICE LOSS: For Segmentation Tasks
    --------------------------------------------------------------------------------

    다이스 계수: 예측과 참값이 얼마나 겹치는지 잰다
      • 의료 영상 분할에 쓴다
      • 범위: 0(전혀 안 겹침)부터 1(완전히 겹침)까지
      • Formula: Dice = 2|A ∩ B| / (|A| + |B|)
      
    다이스 로스 = 1 - 다이스 계수


    Example: Binary segmentation
    Good prediction Dice Loss: 0.0644
    Bad prediction Dice Loss: 0.4785

    Lower loss = better overlap!

    --------------------------------------------------------------------------------
    COMBINING MULTIPLE LOSS TERMS
    --------------------------------------------------------------------------------

    여러 목적을 한꺼번에 최적화하고 싶을 때가 많다.
      • 재구성 + 정칙화
      • 과제 로스 + 일관성 로스
      • 여러 과제 로스(다중 과제 학습)
      
    접근: 로스의 가중합
      전체 로스 = α₁ × 로스₁ + α₂ × 로스₂ + ...


    Combined Loss Example:
      MSE component: 0.2500 (weight: 0.7)
      L1 component: 0.5000 (weight: 0.3)
      Total loss: 0.3250

    WHY COMBINE LOSSES?
      • MSE: Smooth gradients, penalizes large errors
      • L1: Robust to outliers
      • Combination: Balance both properties!

    --------------------------------------------------------------------------------
    BEST PRACTICES FOR CUSTOM LOSSES
    --------------------------------------------------------------------------------

    ✓ DO:
      1. 로스에는 nn.Module 기반 클래스를 쓴다
      2. 모든 연산을 PyTorch로 한다(넘파이 말고)
      3. 작은 예제로 그래디언트 흐름을 시험한다
      4. 수치 안정성을 더한다(완충 항, 값 제한)
      5. 로스 함수를 잘 문서화한다
      6. 기본 초파라메터를 제공한다
      7. 역전파를 위해 스칼라를 돌려준다
      8. 수치 안정성을 살핀다(log(0)과 0으로 나누기를 피한다)

    ✗ DON'T:
      1. 로스 안에서 .item()을 쓴다(그래디언트가 끊긴다)
      2. 제자리 연산을 함부로 쓴다
      3. 경계 상황 처리를 잊는다
      4. 로스가 텐서가 아닌 상수에 의존하게 만든다
      5. 그래디언트가 없는 연산을 쓴다

    로스 시험하기:
      1. 올바른 모양(스칼라)을 돌려주는지 확인한다
      2. 그래디언트가 흐르는지 확인한다: loss.backward()
      3. 이미 아는 입력과 출력으로 시험한다
      4. 참조 구현이 있으면 견주어 본다
      5. 경계 상황에서 수치 안정성을 확인한다


    Testing Custom Loss:
      ✓ Shape check passed: torch.Size([])
      ✓ Gradient check passed
      ✓ Numerical stability check passed
      All tests passed! ✓


    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. 표준 로스가 맞지 않을 때 맞춤 로스를 만든다.
       • 분야에 특화된 목적
       • 데이터 불균형 다루기
       • 다중 과제 학습
       • 연구 실험

    2. 두 가지 접근:
       • 함수: 단순하며 빠른 실험에 알맞다
       • 클래스(nn.Module): 전문적이고 설정 가능하며 권장된다

    3. 고급 로스 예:
       • 초점 로스: 불균형 분류용
       • 다이스 로스: 분할 겹침용
       • 결합 로스: 여러 목적

    4. 구현 요령:
       • PyTorch 연산만 쓴다
       • 역전파를 위해 스칼라를 돌려준다
       • 수치 안정성을 더한다
       • 철저히 시험한다

    5. 흔한 패턴:
       • 중요도를 반영한 가중 로스
       • 여러 항 결합
       • 클래스 균형 웨이트
       • 어려운 예제 캐기

    다음 단계:
    → 불균형 데이터셋에 초점 로스를 구현해 보라
    → 로스 조합을 실험해 보라
    → 문제에 맞는 분야 특화 로스를 만들어 보라
    → 새로운 로스 함수를 다룬 논문을 살펴보라

    ================================================================================
    ```


## 2. 논의

WeightedMSELoss의 mean은 가중 오차를 원소 수로 나눈다. 웨이트 합으로 나눈 평균과는 다르므로 표본 웨이트의 스케일에 따라 값도 달라진다. 모양이 잘못되어 브로드캐스팅되지 않는지 확인한다.

여기의 FocalLoss는 모든 표본에 같은 alpha를 곱하므로 양·음성의 서로 다른 alpha_t로 클래스 비율을 조정하는 구현은 아니다. 조절 인수는 확신도에 따라 다르지만 alpha 자체는 전체 배율이다. 감마 2에서 확률 0.9와 0.5의 조절 인수 비율은 0.01/0.25=1/25이며 전체 로스 비율에는 로그 항도 반영된다.

DiceLoss의 부드러운 겹침은 전경 크기를 반영하지만 픽셀의 연결성이나 형상 구조를 직접 보장하지 않는다. 빈 목표와 예측에서의 smoothing, 배치·채널별 집계와 BCE 혼합 배율을 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
가중 MSE에서 mean(weights * error²)와 sum(weights * error²)/sum(weights)는 같은가?

</div>

??? success "풀이"
    모든 웨이트 합이 원소 수와 같을 때 등 특수한 경우를 제외하면 다르다. 웨이트가 두 배가 되면 전자는 두 배지만 후자는 유지된다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 코드의 FocalLoss alpha가 클래스 균형을 바꾸는가?

</div>

??? success "풀이"
    모든 표본에 같은 상수를 곱하므로 양·음성 사이 상대 비중을 바꾸지 않는다. 서로 다른 클래스 웨이트가 필요하면 alpha_t를 목표에 따라 정해야 한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
argmax한 예측을 로스에 넣으면 경로를 점검해야 하는 이유를 설명하라.

</div>

??? success "풀이"
    argmax의 이산 선택은 일반적으로 필요한 경사를 전달하지 않는다. 학습에는 확률·로짓 등 적절한 부드러운 표현을 사용하고, 클래스 선택은 지표 계산 경로와 구분한다.

## 정리하며

사용자 정의 로스는 식뿐 아니라 집계와 경사 경로를 검증해야 한다. 웨이트 합의 분모와 클래스별 배율, 빈 목표와 수치 범위를 명시한다.
