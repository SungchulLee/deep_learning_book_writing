# 인증 가능한 방어

심화 단계의 구현이다. 증명할 수 있는 보장을 주는 인증 방어 가운데 하나인 **랜덤 스무딩**(randomized smoothing)을 클래스로 구현한다 — 입력에 가우스 잡음을 여러 번 더해 분류기의 다수결을 내고, 그 다수결이 얼마나 압도적인지로 $\ell_2$ 인증 반지름을 계산하는 방법이다. 경험적 방어(적대적 학습 따위)가 알려진 공격으로만 시험받는 것과 달리, 인증 반지름 안에서는 어떤 공격도 예측을 바꾸지 못함을 확률적으로 보장한다.

적대적 강건성은 안전이 걸린 곳에 뉴럴 네트워크를 배포할 때 중요한 문제다. 다만 `__main__` 블록은 안내 메시지만 출력하고, 실제로 인증을 돌리지는 않는다(아래 출력 참고).

## 1. 코드

??? note "코드 (566줄)"

    ```python
    """
    62.5 배치: 밝혀 낸 막이 - 심화

    이 배치는 모형의 미루어 봄에 증명할 수 있는 보장을 주는 밝혀 낸 강건성 막이를
    구현했다. 남다른 치기로 시험해 보는 겪은 막이(맞서며 익히기 따위)와 달리,
    밝혀 낸 막이는 주어진 반지름 안의 모든 섭동에 대해
    수학의 보장을 준다.

    수학 밑그림:
    =======================

    겪은 강건성과 밝혀 낸 강건성:
    -----------------------------------
    **겪은 강건성**:
    - 아는 치기로 시험한다(FGSM, PGD, C&W)
    - 모르는 치기에는 보장이 없다
    - 그래디언트 가리기에 걸릴 수 있다

    **밝혀 낸 강건성**:
    - 미루어 봄이 강건하다는 수학의 증명
    - ε 공 안의 모든 섭동에 보장이 선다
    - 밝혀 낸 반지름 안에서는 어떤 치기에도 속지 않는다

    아무렇게나 매끄럽게 하기:
    ====================
    아무렇게나 매끄럽게 하기는 본디 분류기를 가우스 잡음으로 매끄럽게 하여
    증명할 수 있게 강건한 분류기를 만든다.

    짓기:
    -------------
    밑 분류기 f: R^d → {1,...,k}으로 매끄럽게 한 분류기 g을 짓는다:

        g(x) = argmax_c P(f(x + ε) = c)  where ε ~ N(0, σ²I)

    느낌으로 말하면:
    - 들임에 가우스 잡음을 더한다
    - 잡음 섞인 미루어 봄에서 많은 쪽을 고른다
    - 이것이 분류기를 "매끄럽게" 한다

    밝히기 정리:
    =====================
    코언 등(2019)이 증명했다:

    어떤 들임 x에서
        P(f(x + ε) = c_A) ≥ p_A  [으뜸 클래스의 확률]
        P(f(x + ε) = c_B) ≤ p_B  [다음 클래스의 확률]

    이면 g(x) = c_A은 다음 L2 반지름 안에서 밝혀 낸 강건성을 지닌다:

        R = σ/2 * (Φ^(-1)(p_A) - Φ^(-1)(p_B))

    여기서 Φ^(-1)은 기준 정규 분포의 쌓인 분포 함수의 거꿀이다.

    느낌으로 알기:
    ----------
    - p_A은 클래스 c_A이 많은 쪽이 될 확률
    - p_B은 다른 클래스가 이길 확률
    - p_A >> p_B이면(크게 자신하면) R이 크다
    - R은 밝혀 낸 강건성의 반지름이다

    짜기:
    ===============

    두 도막:
    ------------------
    1. **고르기**: 미루어 볼 클래스 c_A을 찾는다
       - 잡음 섞인 미루어 봄을 n0번 뽑는다
       - 많은 쪽을 고른다

    2. **밝히기**: p_A을 어림하고 반지름을 밝힌다
       - 잡음 섞인 미루어 봄을 n번 더 뽑는다
       - p_A, p_B의 믿음 구간을 셈한다
       - 밝혀 낸 반지름 R을 셈한다

    몬테카를로 뽑기:
    --------------------
    확률은 몬테카를로로 어림한다:

        P(f(x + ε) = c) ≈ (# times f(x + ε_i) = c) / N

    여기서 ε_1, ..., ε_N ~ N(0, σ²I)

    통계의 보장:
    ----------------------
    클로퍼-피어슨 믿음 구간을 쓰면 확률의 보장을 얻는다:

        확률 ≥ 1-α으로: p_A ≥ p̂_A이고 p_B ≤ p̂_B

    이로써 확률 ≥ 1-α으로 밝혀 낸 반지름 R을 얻는다.

    핵심 파라메터:
    ===============
    1. **σ(시그마)**: 잡음의 표준편차
       - σ이 크면: 밝혀 낸 반지름은 크고 정확도는 낮다
       - 흔함: σ ∈ [0.12, 1.0]

    2. **n0**: 고르는 데 쓸 표본 수
       - 흔함: n0 = 100

    3. **n**: 밝히는 데 쓸 표본 수
       - 표본이 많을수록 믿음 구간이 촘촘하다
       - 흔함: n = 10,000 이상

    4. **α(알파)**: 신뢰 수준
       - 흔함: α = 0.001(믿음 99.9%)

    맞바꿈:
    ==========
    1. **정확도와 밝혀 낸 반지름**:
       - σ이 크면: R은 크고 맑은 정확도는 낮다
       - σ이 작으면: 정확도는 높고 R은 작다

    2. **셈 값**:
       - 밝히려면 앞으로 스텝이 많이 든다(표본 N개)
       - 여느 미루어 봄보다 훨씬 느리다
       - 나란히 할 수 있다

    3. **L2과 L∞**:
       - 아무렇게나 매끄럽게 하기는 L2을 밝힌다
       - L∞을 밝히기는 더 어렵다

    지은이: 가르침 감
    날짜: 2025년 11월
    어려움: 심화
    먼저 알 것: 확률 이론, 통계, 믿음 구간
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from typing import Optional, Tuple, Dict
    from scipy.stats import norm, binom
    from tqdm import tqdm
    import math

    # ========================================================================
    # 메인
    # ========================================================================


    class RandomizedSmoothing:
        """
        밝혀 낸 강건성을 위한 아무렇게나 매끄럽게 하기

        이 클래스는 어떤 분류기에도 증명할 수 있는 L2 강건성 보장을 주는
        아무렇게나 매끄럽게 하기를 구현했다.

        수학 형태:
        -------------------------
        밑 분류기 f으로 매끄럽게 한 분류기를 짓는다:

            g(x) = argmax_c E_{ε~N(0,σ²I)}[1{f(x + ε) = c}]

        밝히기: P(f(x + ε) = c_A) ≥ p_A이면 g(x)은 다음 L2 반지름 안에서
        밝혀 낸 강건성을 지닌다:

            R = σ/2 * (Φ^(-1)(p_A) - Φ^(-1)(p_B))

        여기서 c_B은 다음 클래스이고 Φ^(-1)은 기준 정규 분포의 쌓인 분포 함수의 거꿀이다.

        속성:
        -----------
        base_classifier : nn.Module
            매끄럽게 할 밑 분류기
        sigma : float
            가우스 잡음의 표준편차
        device : torch.device
            셈할 장치
        """

        def __init__(
            self,
            base_classifier: nn.Module,
            sigma: float = 0.25,
            device: Optional[torch.device] = None
        ):
            """
            아무렇게나 매끄럽게 하기의 첫자리를 잡는다.

            파라메터:
            -----------
            base_classifier : nn.Module
                매끄럽게 할 밑 분류기
                로짓(소프트맥스 앞의 날임)을 내야 한다
            sigma : float, 기본값=0.25
                가우스 잡음의 표준편차
                σ이 크면: 더 매끄럽고 밝혀 낸 반지름이 크다
                σ이 작으면: 덜 매끄럽고 정확도가 높다
                흔한 값: σ ∈ [0.12, 0.25, 0.50, 1.0]
            device : torch.device, 골라 씀
                셈할 장치
            """
            self.base_classifier = base_classifier
            self.sigma = sigma
            self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 밑 분류기를 평가 모드로 둔다
            self.base_classifier.eval()
            self.base_classifier = self.base_classifier.to(self.device)

            print(f"아무렇게나 매끄럽게 하기 차림:")
            print(f"  시그마 (σ): {self.sigma}")
            print(f"  장치: {self.device}")

        def _sample_noise(
            self,
            x: torch.Tensor,
            num_samples: int,
            batch_size: int = 1000
        ) -> torch.Tensor:
            """
            잡음 섞인 표본의 미루어 봄을 만든다.

            이것이 몬테카를로 어림 스텝이다. 다음을 한다:
            1. 가우스 잡음을 더한다: x + ε, ε ~ N(0, σ²I)
            2. 미루어 봄을 얻는다: f(x + ε)
            3. N번 되풀이한다

            기억을 아끼려 배치로 다룬다.

            파라메터:
            -----------
            x : torch.Tensor
                들임 그림(그림 하나)
            num_samples : int
                만들 잡음 섞인 표본의 수
            batch_size : int, 기본값=1000
                표본을 다룰 배치 크기

            돌려주는 것:
            --------
            counts : torch.Tensor
                클래스마다의 미루어 봄 셈
                형태: (num_classes,)
            """
            with torch.no_grad():
                # 앞으로 스텝으로 클래스의 수를 얻는다
                if not hasattr(self, 'num_classes'):
                    test_output = self.base_classifier(x.unsqueeze(0))
                    self.num_classes = test_output.size(1)

                # 클래스마다의 셈을 마련한다
                counts = torch.zeros(self.num_classes, device=self.device)

                # 기억을 아끼려 배치로 다룬다
                num_batches = math.ceil(num_samples / batch_size)

                for _ in range(num_batches):
                    # 참 배치 크기를 정한다(마지막 배치는 작을 수 있다)
                    current_batch_size = min(batch_size, num_samples - len(counts.nonzero()))

                    # 들임을 되풀이해 배치를 만든다
                    batch = x.repeat(current_batch_size, 1, 1, 1)

                    # 가우스 잡음을 더한다: ε ~ N(0, σ²I)
                    noise = torch.randn_like(batch) * self.sigma
                    noisy_batch = batch + noise

                    # 잡음 섞인 표본의 미루어 봄을 얻는다
                    outputs = self.base_classifier(noisy_batch)
                    predictions = outputs.argmax(dim=1)

                    # 클래스마다 미루어 봄을 센다
                    for pred in predictions:
                        counts[pred] += 1

            return counts

        def predict(
            self,
            x: torch.Tensor,
            n: int = 1000,
            alpha: float = 0.001,
            batch_size: int = 1000
        ) -> Tuple[int, float]:
            """
            들임 하나의 클래스를 미루어 보고 강건성을 밝힌다.

            두 도막으로 이루어진다:
            ------------------
            1. 고르기: 표본 n0개로 미루어 볼 클래스를 찾는다
            2. 밝히기: 확률을 어림하고 반지름을 셈한다

            파라메터:
            -----------
            x : torch.Tensor
                들임 그림(그림 하나, 형태: (C, H, W))
            n : int, 기본값=1000
                밝히는 데 쓸 표본 수
                표본이 많을수록 믿음 구간이 촘촘하다
                흔함: 좋은 보장에는 n ≥ 10,000
            alpha : float, 기본값=0.001
                신뢰 수준(어긋날 확률)
                흔함: α = 0.001(믿음 99.9%)
            batch_size : int, 기본값=1000
                표본을 다룰 배치 크기

            돌려주는 것:
            --------
            prediction : int
                미루어 본 클래스 이름표
            radius : float
                밝혀 낸 L2 반지름
                반지름 = 0이면 밝히기가 어그러진 것이다
            """
            # 1도막: 고르기(으뜸 클래스 찾기)
            # 잘 들도록 표본 n/10개를 쓴다
            n_selection = max(100, n // 10)
            counts_selection = self._sample_noise(x, n_selection, batch_size)
            top_class = counts_selection.argmax().item()

            # 2도막: 밝히기(확률 어림)
            counts_cert = self._sample_noise(x, n, batch_size)

            # 확률의 믿음 구간을 셈한다
            # 클로퍼-피어슨(정확한) 두 값 믿음 구간을 쓴다

            # 으뜸 클래스의 셈
            count_top = counts_cert[top_class].item()

            # p_A(으뜸 클래스의 확률)의 믿음 아래끝
            p_A_lower = self._lower_confidence_bound(count_top, n, alpha)

            # 다음 클래스를 찾는다(으뜸 클래스는 뺀다)
            counts_cert[top_class] = -1  # Temporarily remove top class
            runner_up_class = counts_cert.argmax().item()
            count_runner_up = counts_cert[runner_up_class].item()

            # p_B(다음 클래스의 확률)의 믿음 위끝
            p_B_upper = self._upper_confidence_bound(count_runner_up, n, alpha)

            # 밝혀 낸 반지름을 셈한다
            # R = σ/2 * (Φ^(-1)(p_A) - Φ^(-1)(p_B))
            if p_A_lower > p_B_upper:
                # 밝히기가 되었다
                radius = self._compute_radius(p_A_lower, p_B_upper)
            else:
                # 밝히기가 어그러졌다(믿음이 모자라다)
                radius = 0.0

            return top_class, radius

        def _lower_confidence_bound(
            self,
            count: int,
            n: int,
            alpha: float
        ) -> float:
            """
            두 값 기여도의 믿음 아래끝을 셈한다.

            클로퍼-피어슨 길(정확한 두 값 믿음 구간)을 쓴다.

            수학 형태:
            -------------------------
            n번 가운데 count번 들어맞았다고 하자.
            참 확률 p은 다음을 채운다:
                P(count | p) ≥ α/2

            이로써 다음을 채우는 아래끝 p_lower을 얻는다:
                P(p ≥ p_lower) ≥ 1 - α/2

            파라메터:
            -----------
            count : int
                들어맞은 횟수
            n : int
                해 본 횟수
            alpha : float
                뜻있음 레이어

            돌려주는 것:
            --------
            p_lower : float
                믿음 아래끝
            """
            return binom.ppf(alpha/2, n, count/n) / n if count > 0 else 0.0

        def _upper_confidence_bound(
            self,
            count: int,
            n: int,
            alpha: float
        ) -> float:
            """
            두 값 기여도의 믿음 위끝을 셈한다.

            파라메터:
            -----------
            count : int
                들어맞은 횟수
            n : int
                해 본 횟수
            alpha : float
                뜻있음 레이어

            돌려주는 것:
            --------
            p_upper : float
                믿음 위끝
            """
            return binom.ppf(1 - alpha/2, n, count/n) / n if count < n else 1.0

        def _compute_radius(
            self,
            p_A: float,
            p_B: float
        ) -> float:
            """
            확률으로 밝혀 낸 반지름을 셈한다.

            식:
            --------
            R = σ/2 * (Φ^(-1)(p_A) - Φ^(-1)(p_B))

            여기서:
            - Φ^(-1)은 기준 정규 분포의 쌓인 분포 함수의 거꿀
            - p_A은 으뜸 클래스 확률의 아래끝
            - p_B은 다음 클래스 확률의 위끝

            느낌으로 알기:
            ----------
            - p_A이 1에 가깝고 p_B이 0에 가까우면: R이 크다(크게 확신도)
            - p_A ≈ p_B이면: R이 작다(머뭇거림)
            - σ이 반지름의 스케일을 맞춘다. σ이 크면 R도 크다

            파라메터:
            -----------
            p_A : float
                으뜸 클래스 확률의 아래끝
            p_B : float
                다음 클래스 확률의 위끝

            돌려주는 것:
            --------
            radius : float
                밝혀 낸 L2 반지름
            """
            # 쌓인 분포 함수의 거꿀 값을 셈한다
            # norm.ppf은 기준 정규 쌓인 분포 함수의 거꿀이다
            if p_A >= 1.0:
                p_A = 0.999999  # Avoid infinity
            if p_B <= 0.0:
                p_B = 0.000001

            phi_inv_pA = norm.ppf(p_A)
            phi_inv_pB = norm.ppf(p_B)

            # 반지름을 셈한다
            radius = (self.sigma / 2.0) * (phi_inv_pA - phi_inv_pB)

            return max(0.0, radius)  # Ensure non-negative

        def certify_batch(
            self,
            images: torch.Tensor,
            labels: torch.Tensor,
            n: int = 10000,
            alpha: float = 0.001,
            batch_size: int = 1000,
            verbose: bool = True
        ) -> Dict[str, float]:
            """
            그림 배치를 밝히고 자를 셈한다.

            그림마다:
            1. 클래스와 밝혀 낸 반지름을 미루어 본다
            2. 미루어 봄이 맞는지 살핀다
            3. 치기에 대해 밝혀졌는지 살핀다

            자:
            --------
            - 맑은 정확도: 옳게 미루어 본 기여도
            - 반지름 r의 밝혀 낸 정확도: 옳으면서 밝혀 낸 반지름 ≥ r인 기여도

            파라메터:
            -----------
            images : torch.Tensor
                그림 배치
            labels : torch.Tensor
                참 이름표
            n : int, 기본값=10000
                그림마다의 표본 수
            alpha : float, 기본값=0.001
                신뢰 수준
            batch_size : int, 기본값=1000
                잡음을 뽑을 배치 크기
            verbose : bool, 기본값=True
                나아가는 것을 찍는다

            돌려주는 것:
            --------
            results : Dict[str, float]
                밝히기 자
            """
            num_images = len(images)
            predictions = []
            radii = []

            if verbose:
                pbar = tqdm(range(num_images), desc="Certifying")
            else:
                pbar = range(num_images)

            for i in pbar:
                pred, radius = self.predict(images[i], n, alpha, batch_size)
                predictions.append(pred)
                radii.append(radius)

            # 텐서로 옮긴다
            predictions = torch.tensor(predictions, device=labels.device)
            radii = torch.tensor(radii)

            # 자를 셈한다
            correct = (predictions == labels)

            # 맑은 정확도
            clean_accuracy = correct.float().mean().item()

            # 반지름마다 밝혀 낸 정확도
            radius_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
            certified_accuracies = {}

            for r in radius_levels:
                # 밝혀짐: 미루어 봄이 옳고 밝혀 낸 반지름 ≥ r
                certified = correct & (radii >= r)
                certified_accuracies[f'certified_acc_r={r}'] = certified.float().mean().item()

            # 밝혀 낸 반지름의 평균(옳게 가른 보기에 대해)
            avg_radius = radii[correct].mean().item() if correct.any() else 0.0

            results = {
                'clean_accuracy': clean_accuracy,
                'avg_certified_radius': avg_radius,
                **certified_accuracies
            }

            if verbose:
                print("\n" + "=" * 60)
                print("밝히기 결과")
                print("=" * 60)
                print(f"맑은 정확도: {clean_accuracy:.2%}")
                print(f"밝혀 낸 반지름 평균: {avg_radius:.4f}")
                print("\n반지름마다 밝혀 낸 정확도:")
                for r in radius_levels:
                    key = f'certified_acc_r={r}'
                    print(f"  r = {r}: {results[key]:.2%}")
                print("=" * 60)

            return results


    # 쓰는 보기
    if __name__ == "__main__":
        """
        아무렇게나 매끄럽게 하여 밝혀 낸 강건성을 보여 준다.
        """
        print("=" * 70)
        print("아무렇게나 매끄럽게 하여 밝혀 낸 강건성")
        print("=" * 70)
        print("\n이 글은 아무렇게나 매끄럽게 하기로 증명할 수 있는")
        print("강건성 보장을 보여 준다.")
        print("\n붙임말: 자료 얹기와 모형 잔손질에 utils.py이 있어야 한다.")
        print("\n알림: 밝히기는 셈이 아주 비싸다!")
        print("=" * 70)
    ```


**출력:**

```
======================================================================
아무렇게나 매끄럽게 하여 밝혀 낸 강건성
======================================================================

이 글은 아무렇게나 매끄럽게 하기로 증명할 수 있는
강건성 보장을 보여 준다.

붙임말: 자료 얹기와 모형 잔손질에 utils.py이 있어야 한다.

알림: 밝히기는 셈이 아주 비싸다!
======================================================================
```

## 2. 논의

`predict` 는 Cohen 등(2019)의 두 단계를 따른다. 먼저 `max(100, n // 10)` 개의 잡음 샘플로 다수결 클래스 $c_A$를 **고르고**, 그와 따로 뽑은 `n` 개의 샘플로 그 클래스의 확률을 **추정한다**. 같은 샘플로 고르고 추정하면 우연히 많이 나온 클래스를 골라 놓고 그 확률을 과대평가하게 되므로, 두 단계의 샘플을 나눈다. 추정한 $p_A$의 하한과 다음 클래스 $p_B$의 상한으로 반지름 $R = \tfrac{\sigma}{2}\left(\Phi^{-1}(p_A) - \Phi^{-1}(p_B)\right)$를 계산하고, 하한이 상한을 넘지 못하면 반지름 0, 곧 인증 실패를 돌려준다.

두 가지를 짚어 둔다. 첫째, 코드 주석은 클로퍼-피어슨(Clopper-Pearson) 구간이라고 적지만 `_lower_confidence_bound` 가 계산하는 `binom.ppf(alpha/2, n, count/n) / n` 은 추정값 $\hat{p}$를 참값으로 놓은 이항 분포의 분위수이며, 클로퍼-피어슨 하한(베타 분포의 분위수)과 다르다. 이 값으로는 "확률 $1-\alpha$로 보장"이라는 말이 엄밀히 서지 않는다. 둘째, `certify_batch` 가 보고하는 인증 정확도는 예측이 맞고 반지름이 $r$ 이상인 샘플의 비율이므로, 같은 테스트 집합에서 잰 클린 정확도보다 클 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 구현에서 선택한 것 셋을 들고, 각각 왜 적대적 강건성에 알맞은지 설명하여라.

</div>

??? success "풀이"
    이 구현에서 눈여겨볼 선택은 셋이다. (1) **고르기와 추정을 다른 샘플로 한다** -- 다수결 클래스를 고른 샘플로 그 확률까지 추정하면 선택 편향 탓에 $p_A$가 부풀려져 인증 반지름이 실제보다 커진다. (2) **신뢰 하한·상한과 기권** -- $p_A$에는 하한을, $p_B$에는 상한을 넣어 반지름을 보수적으로 잡고, 하한이 상한을 넘지 못하면 반지름 0을 돌려준다. 보장이 깨질 확률을 $\alpha$로 묶는 장치다. (3) **잡음 샘플을 배치로 묶어 분류한다** -- `n = 10000` 처럼 샘플이 많아야 구간이 좁아지는데, 한 장씩 분류하면 너무 느리고 한꺼번에 넣으면 메모리가 모자라므로 `batch_size` 단위로 나눈다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주 함수나 클래스에 추가하여라. 올바르지 않은 입력에는 알아보기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `predict` 메서드의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 추가한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아보기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 강건하게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 생길 수 있는 실패 양상 둘을 들고, 각각 어떻게 탐지하고 고칠지 설명하여라.

</div>

??? success "풀이"
    흔한 실패 양상은 이렇다. (1) **기반 분류기가 잡음을 견디지 못한다** -- 클린 이미지로만 학습한 분류기에 $\sigma = 0.25$ 잡음을 더하면 예측이 거의 무작위가 되어 $p_A$가 낮고, 대부분의 샘플이 반지름 0(기권)으로 끝난다. 잡음을 더한 입력에서 기반 분류기의 정확도를 재어 진단하고, 학습할 때부터 같은 $\sigma$의 가우스 잡음으로 데이터 어그멘테이션을 해 고친다. (2) **샘플 수가 모자라 반지름이 작게 나온다** -- $n$이 작으면 신뢰 구간이 넓어 $p_A$의 하한이 낮아지고, 클로퍼-피어슨 하한을 쓰면 모든 샘플이 맞게 분류되어도 하한이 1보다 작아 반지름에 상한이 생긴다. $n$을 바꿔 가며 평균 인증 반지름을 그려 진단하고, 계산 비용을 감수하고 $n$을 늘려 고친다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
인증 가능한 방어 구현을 검사하는 포괄적인 테스트 함수를 작성하여라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 조건을 시험하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 시험 함수를 작성한다.
    ```python
    def test_randomizedsmoothing():
        model = RandomizedSmoothing(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    구조가 엔드투엔드 학습을 지원하는지 알려면 그래디언트 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 인증 가능한 방어

랜덤 스무딩은 가우스 잡음 아래 다수결로 분류기를 매끄럽게 만들고, 다수결의 압도 정도로 $\ell_2$ 인증 반지름을 계산한다.

핵심 클래스는 `RandomizedSmoothing`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
