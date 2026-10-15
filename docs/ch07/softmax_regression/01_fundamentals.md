# 기초

특징 벡터를 클래스별 로짓으로 변환하고 소프트맥스로 확률을 나타낸다. 파라미터의 모양, 손실의 입력과 선형 결정 영역을 연결한 뒤 같은 모델의 데이터셋별 기록을 비교한다.

## 1. 특징에서 클래스 점수로

소프트맥스 회귀는 이름에 '회귀'가 붙어 있지만 분류기이다. 특징 벡터 하나를 받아 클래스마다 점수를 매기고, 그 점수를 확률로 바꾼다.

### 아핀 변환

입력 $\mathbf{x} \in \mathbb{R}^{1 \times D}$에 행렬을 곱하고 편향을 더해 클래스마다 점수 하나씩, 모두 $K$개를 만든다.

$$\mathbf{z} = \mathbf{x}A + \mathbf{b}$$

모양은 다음과 같다.

$$\mathbf{x} \in \mathbb{R}^{1 \times D},\quad A \in \mathbb{R}^{D \times K},\quad \mathbf{b} \in \mathbb{R}^{K},\quad \mathbf{z} \in \mathbb{R}^{1 \times K}$$

여기에는 은닉층이나 중간 비선형 활성화가 없다. 확률 변환에 소프트맥스를 사용한다. 이 한 겹이 모델의 전부이며, 그래서 앞 절에서 본 대로 결정 경계가 초평면으로 묶인다.

이 페이지는 행벡터 입력과 $A\in\mathbb R^{D\times K}$를 사용해 $xA+b$로 쓴다. PyTorch의 `nn.Linear(D,K)`는 $W\in\mathbb R^{K\times D}$를 저장하고 $xW^\top+b$를 계산하므로 $A=W^\top$다. 같은 기호 $A$를 두 모양에 동시에 사용하지 않는다. 프레임워크 전체를 행벡터·열벡터 관례 하나로 단정하기보다 각 연산의 모양을 확인한다.

### 매개변수 세어 보기

$$\underbrace{D \times K}_{A} + \underbrace{K}_{\mathbf{b}} = K(D + 1)$$

"모델을 학습한다"는 말은 결국 이 $K(D+1)$개의 수를 찾는다는 뜻이다. 오늘날의 큰 모델이 다루는 수가 수십억 개일 뿐, 하는 일은 같다.

!!! note "MNIST에서는 어떤 값이 되는가"
    784차원 입력에 클래스가 10개이면 $10 \times (784 + 1) = 7850$개이다.
    이 값으로 실제 학습을 돌려 92.51%를 얻는 과정은 [3.2 구현과 결과](../../ch03/linear_softmax/06_implementation.md)에 있다. 이 절은 그 특수한 경우가 아니라 모델 자체의 정의를 다룬다.

### 점수를 확률로 바꾸기

$\mathbf{z}$의 성분은 아직 확률이 아니다. 음수일 수도 있고 합이 1도 아니다. 이 날것의 점수를 **로짓**이라 부른다. 소프트맥스가 이를 확률로 바꾼다.

$$p_k = \frac{e^{z_k}}{Z}, \qquad Z = \sum_{j=1}^{K} e^{z_j}$$

지수함수가 모든 값을 양수로 만들고, $Z$로 나누어 합을 1로 맞춘다. 지수를 쓰므로 점수의 작은 차이가 확률에서는 크게 벌어진다.

### 예측 고르기

$$\widehat{y} = \operatorname*{arg\,max}_{k \in \{1,\ldots,K\}} p_k$$

소프트맥스는 순서를 바꾸지 않으므로, 예측만 얻을 목적이라면 로짓에 바로 `argmax`를 취해도 결과가 같다. 소프트맥스가 정말로 필요한 곳은 확률값 자체를 읽거나 손실을 계산할 때이다.

### 그러면 $(A, \mathbf{b})$는 어떻게 찾는가

여기까지가 모델이고, 남은 물음은 $K(D+1)$개의 수를 어떻게 정하느냐이다. 답은 최대가능도 추정이며, 거기서 교차 엔트로피 손실과 경사 하강법이 따라 나온다.

그 사슬을 처음부터 끝까지 따라가는 것은 [8.1 최대가능도 추정](../mle/mle.md)이고, 우도라는 개념 자체는 [표지-재포획 MLE](../mle/capture_recapture_mle.md)가 매개변수 하나짜리 예제로 보인다. 이 절은 모델의 정의와 소프트맥스의 계산에 집중한다.

---

## 2. 코드

??? note "코드 1 (424줄)"

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
    → 2단계: 분류를 위한 단순한 신경망 짓기
    → 3단계: 참 데이터셋으로 익히기(MNIST)
    → 4단계: 앞선 기법와 다듬기

    🎉 잘했다! 기초를 익혔다!
    """)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (209줄)"

    ```
    ================================================================================
    LEVEL 1: SOFTMAX REGRESSION FUNDAMENTALS
    ================================================================================

    ================================================================================
    PART 1: Understanding Softmax
    ================================================================================

    Example 1: Converting logits to probabilities
    --------------------------------------------------------------------------------
    Input logits:         [2.  1.  0.1]
      (These are raw, unnormalized scores from a model)

    Output probabilities: [0.65900114 0.24243297 0.09856589]
      (These are interpretable as class probabilities)
    Sum of probabilities: 1.000000
      (Should always equal 1.0)


    Example 2: How logits affect probabilities
    --------------------------------------------------------------------------------
    Scenario 1: logits = [1. 1. 1.]
                probs  = [0.33333333 0.33333333 0.33333333]

    Scenario 2: logits = [3. 1. 1.]
                probs  = [0.78698604 0.10650698 0.10650698]

    Scenario 3: logits = [10.  1.  1.]
                probs  = [9.99753241e-01 1.23379352e-04 1.23379352e-04]

    💡 Key Insight: Larger differences in logits lead to more confident predictions!

    ================================================================================
    PART 2: Softmax in PyTorch
    ================================================================================

    Input (PyTorch tensor): tensor([2.0000, 1.0000, 0.1000])
    Output probabilities:   tensor([0.6590, 0.2424, 0.0986])


    Batched Example (3 samples, 3 classes each):
    --------------------------------------------------------------------------------
    Logits (3 samples x 3 classes):
    tensor([[2.0000, 1.0000, 0.1000],
            [0.5000, 2.5000, 1.0000],
            [1.5000, 1.5000, 1.5000]])

    Probabilities (after softmax):
    tensor([[0.6590, 0.2424, 0.0986],
            [0.0996, 0.7361, 0.1643],
            [0.3333, 0.3333, 0.3333]])

    Sum for each sample: tensor([1.0000, 1.0000, 1.0000])
      (Each row sums to 1.0)

    ================================================================================
    PART 3: Cross-Entropy Loss
    ================================================================================

    Example: Comparing good vs bad predictions
    --------------------------------------------------------------------------------
    True class: 0

    Good prediction:   probs = [0.8  0.15 0.05]   → loss = 0.2231
    Medium prediction: probs = [0.5 0.3 0.2] → loss = 0.6931
    Bad prediction:    probs = [0.1 0.6 0.3]   → loss = 2.3026

    💡 Key Insight: Lower loss = better prediction on the true class!

    ================================================================================
    PART 4: PyTorch CrossEntropyLoss
    ================================================================================

    Example 1: Single sample
    --------------------------------------------------------------------------------
    True class: 0

    Good logits: tensor([[3.0000, 1.0000, 0.5000]])
      Loss: 0.1967

    Bad logits: tensor([[0.5000, 3.0000, 2.0000]])
      Loss: 2.8715

    Predicted class (good): 0 ✓
    Predicted class (bad):  1 ✗


    Example 2: Batch of samples
    --------------------------------------------------------------------------------
    Batch loss (average): 0.6059

    True classes:      [2 0 1 2]
    Predicted classes: [2 0 1 1]
    Accuracy: 75.00%

    ================================================================================
    PART 5: Complete Pipeline
    ================================================================================

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


    ================================================================================
    PART 6: Common Mistakes and Best Practices
    ================================================================================

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


    ================================================================================
    PART 7: Quick Practice
    ================================================================================

    결과를 미리 짚어 보아라.
    ---------------------------
    Given:
      - 참 클래스: 1
      - Logits: [1.0, 5.0, 2.0]

    Questions:
    1. 모델은 어느 클래스를 예측하겠는가?(실마리: argmax)
    2. 손실은 클까 작을까?(실마리: 예측이 맞는가?)

    살펴보자.

    True class: 1
    Predicted class: 1
    Loss: 0.0659
    Correct prediction? True

    ================================================================================
    SUMMARY - What You Learned
    ================================================================================

    ✅ 소프트맥스가 로짓을 확률로 바꾼다
    ✅ 교차 엔트로피가 예측의 좋음을 잰다
    ✅ 손실이 작을수록 예측이 좋다
    ✅ PyTorch의 CrossEntropyLoss:
       - 입력으로 로짓을 받는다(확률이 아니다)
       - 과녁으로 클래스 번호를 받는다(원핫이 아니다)
       - 안에서 소프트맥스 + 로그 + NLL을 아우른다

    다음 걸음:
    -----------
    → 2단계: 분류를 위한 단순한 신경망 짓기
    → 3단계: 참 데이터셋으로 익히기(MNIST)
    → 4단계: 앞선 기법와 다듬기

    🎉 잘했다! 기초를 익혔다!

    ```


## 3. 같은 모델, 더 어려운 데이터

앞의 코드는 소프트맥스와 교차 엔트로피의 계산을 손으로 따라가려고 작은 데이터를 썼다. 이제 같은 모델을 실제 데이터 두 벌에 그대로 걸어 본다. 바뀌는 것은 오직 **데이터**뿐이다. 모델도, 매개변수 수도, 학습 설정도 모두 같다.

- **MNIST** — 손으로 쓴 숫자. 획이 굵고 가운데 놓여 있다
- **Fashion-MNIST** — 옷과 신발 사진. 크기와 모양이 같은 $28 \times 28$ 회색조 10클래스이지만, 무늬와 질감이 훨씬 복잡하다

두 데이터 모두 입력과 클래스 수가 같아 파라미터 수는 7,850개다. 그러나 같은 학습 설정이 두 데이터에서 같은 정도의 수렴이나 최적 성능을 보장하지는 않는다. 관측한 격차에는 데이터 특성과 이 모델·학습 설정의 상호작용이 함께 들어 있다.

??? note "코드 2 (49줄)"

    ```python
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms


    def run(name, dataset_cls, mean, std, epochs=10):
        """소프트맥스 회귀 하나를 주어진 자료에 학습시키고 시험 정확도를 돌려준다."""
        torch.manual_seed(42)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ])
        train_set = dataset_cls('./data', train=True, download=True, transform=transform)
        test_set = dataset_cls('./data', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

        # 은닉층 없는 아핀 변환 하나. D=784, K=10 이므로 10*(784+1)=7850개.
        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(epochs):
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                criterion(model(images), labels).backward()
                optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.size(0)
        n_params = sum(p.numel() for p in model.parameters())
        return 100 * correct / total, n_params


    # 두 자료 각각의 화소 평균과 표준편차로 고르게 한다
    acc_mnist, n1 = run("MNIST", datasets.MNIST, 0.1307, 0.3081)
    acc_fashion, n2 = run("Fashion-MNIST", datasets.FashionMNIST, 0.2860, 0.3530)

    print(f"{'자료':<16} {'매개변수':>10} {'시험 정확도':>12}")
    print("-" * 42)
    print(f"{'MNIST':<16} {n1:>10,} {acc_mnist:>11.2f}%")
    print(f"{'Fashion-MNIST':<16} {n2:>10,} {acc_fashion:>11.2f}%")
    print(f"\n같은 모델, 같은 매개변수 수인데 {acc_mnist - acc_fashion:.2f}%포인트 차이가 난다.")
    ```


**출력:**

```
자료                     매개변수       시험 정확도
------------------------------------------
MNIST                 7,850       92.11%
Fashion-MNIST         7,850       84.31%

같은 모델, 같은 매개변수 수인데 7.80%포인트 차이가 난다.
```

### 무엇을 뜻하는가

매개변수 수가 한 개도 다르지 않은데 정확도가 8%포인트 가까이 벌어진다. 모델이 나빠진 것이 아니라 **문제가 어려워진 것**이다.

까닭은 두 데이터의 성격에 있다. 손으로 쓴 숫자는 획이 놓이는 자리가 클래스마다 꽤 일정해서, 화소마다 가중치 하나씩만 두어도 상당히 가려낸다. 반면 셔츠와 외투는 윤곽이 비슷하고 차이가 무늬와 질감에 있는데, 이는 화소 하나하나의 밝기를 따로 저울질하는 방식으로는 잡아내기 어렵다. 이웃한 화소들이 이루는 무늬를 보아야 하며, 그것이 [11장 합성곱 신경망](../../ch10/index.md)이 하는 일이다.

!!! note "3장의 값과 조금 다른 까닭"
    3장의 기록과 이 페이지의 수치가 조금 다르다면 데이터 순서, 초기값, 학습 설정과 실행 환경을 함께 확인해야 한다. 표에 있는 차이를 재실행 변동 하나로 확정할 수는 없다. 다른 데이터셋 간 격차가 크다는 사실도 반복 실험의 분산을 직접 측정한 것과는 다르다.

---


## 4. 논의

단일 선형 계층은 로짓을 만들고 교차 엔트로피는 로짓과 정답 클래스의 관계를 평가한다. `CrossEntropyLoss`를 사용할 때 입력은 확률이 아니라 로짓이다. 확률을 명시적으로 만든 뒤 로그를 취하는 계산과 수학적 목적은 같아도 수치 안정성은 다를 수 있다.

소프트맥스는 모든 로짓에 공통 상수를 더해도 변하지 않는다. 따라서 $K(D+1)$은 저장된 파라미터 수이며 독립적으로 식별 가능한 확률 모형의 자유도와 같지는 않다. 기준 클래스를 고정하는 등 추가 관례로 같은 확률을 나타내는 중복을 제거할 수 있다.

MNIST와 Fashion-MNIST의 수치는 해당 학습 설정에서의 결과다. 모델의 표현력과 학습 수렴을 분리해 살펴보고, 더 어려운 데이터라는 설명도 클래스별 오류와 적절히 조정한 기준 모델을 함께 확인해 뒷받침한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
소프트맥스 회귀의 파라미터 모양, 학습 손실의 입력, 확률 변환의 역할을 설명하라.

</div>

??? success "연습문제 1 풀이"
    입력 $D$개와 클래스 $K$개에서 PyTorch 가중치는 $(K,D)$, 편향은 $(K,)$다. 교차 엔트로피에는 로짓과 정답을 전달하고, 확률이 필요할 때 소프트맥스를 적용한다. 은닉층과 ReLU는 단일 선형 소프트맥스 회귀의 필수 구성요소가 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
클래스 인덱스를 정답으로 사용하는 교차 엔트로피에서 입력 조건을 설명하라.

</div>

??? success "연습문제 2 풀이"
    일반 배치 분류에서 로짓은 $(B,K)$의 실수 텐서, 정답은 $(B,)$의 long 텐서이며 값은 0부터 $K-1$까지다. 이 페이지의 인덱스 모드와 실수형 확률 목표 모드는 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
동일한 파라미터 수에서 데이터셋별 성능이 다르면 표현력 한계라고 바로 결론낼 수 있는가?

</div>

??? success "연습문제 3 풀이"
    아니다. 데이터 분포, 잡음과 클래스 중첩뿐 아니라 학습률과 반복 수에 따른 수렴 차이도 영향을 준다. 같은 설정의 비교와 각 데이터에서 충분히 조정한 비교를 구분하고 클래스별 오류를 확인한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
기초 구현을 검증하는 종합적인 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하라.

</div>

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

로짓의 모양과 소프트맥스 확률을 구분하고 안정적인 교차 엔트로피를 사용한다. 저장 파라미터 수와 식별 가능한 자유도, 같은 설정의 비교와 모델의 최선 성능은 별개의 개념이다.
