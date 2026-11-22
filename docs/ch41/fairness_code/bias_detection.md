# 바이어스 탐지

딥러닝에서의 바이어스 탐지와 측정. 여기서 **바이어스**는 모델의 예측이 성별·인종 같은 보호 속성에 따라 어느 집단에 체계적으로 유리하거나 불리하게 치우치는 것을 말한다.

머신러닝 시스템에서 공정성을 지키는 일은 윤리적으로도 마땅하고 실무적으로도 중요한 문제다. 이 꾸러미는 모델 예측의 바이어스를 탐지하고 측정하는 기법을 보이며, 이론의 공정성 지표(통계적 패리티 차이, 이질적 영향 비, 기회 균등 차이, 균등화 오즈 차이)를 실제 코드로 연결해 준다.

## 1. 코드

```python
"""
딥러닝에서의 치우침 탐지와 재기
이 꾸러미는 기계 러닝 모형의 치우침을 알아내고 재는 연장을 준다.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

# ========================================================================
# 메인
# ========================================================================


class BiasDetector:
    """모형의 미루어 봄에서 치우침을 알아내고 잰다."""

    def __init__(self, sensitive_attributes: List[str]):
        """
        치우침 알아내개의 첫자리를 잡는다.

        Args:
            sensitive_attributes: 예민한 됨됨이 이름 목록(보기로 'gender', 'race')
        """
        self.sensitive_attributes = sensitive_attributes
        self.metrics = {}

    def statistical_parity_difference(
        self, 
        y_pred: np.ndarray, 
        sensitive_attr: np.ndarray
    ) -> float:
        """
        통계적 패리티 차이를 셈한다.

        통계적 패리티: 모든 무리 a, b에 대해 P(Y=1|A=a) = P(Y=1|A=b)

        Args:
            y_pred: 두 값 미루어 봄
            sensitive_attr: 예민한 됨됨이 값

        Returns:
            무리 사이 양수 미루어 봄 비율의 차이
        """
        groups = np.unique(sensitive_attr)
        if len(groups) != 2:
            raise ValueError("이 구현은 두 값 예민한 됨됨이만 받친다")

        group_0_mask = sensitive_attr == groups[0]
        group_1_mask = sensitive_attr == groups[1]

        rate_0 = np.mean(y_pred[group_0_mask])
        rate_1 = np.mean(y_pred[group_1_mask])

        return abs(rate_0 - rate_1)

    def disparate_impact_ratio(
        self, 
        y_pred: np.ndarray, 
        sensitive_attr: np.ndarray
    ) -> float:
        """
        달리 미침 비를 셈한다.

        달리 미침: 모든 무리에 대해 min(P(Y=1|A=a) / P(Y=1|A=b))
        비가 0.8 아래면 흔히 탈이 있다고 여긴다(80% 규칙)

        Args:
            y_pred: 두 값 미루어 봄
            sensitive_attr: 예민한 됨됨이 값

        Returns:
            양수 미루어 봄 비율의 비
        """
        groups = np.unique(sensitive_attr)
        if len(groups) != 2:
            raise ValueError("이 구현은 두 값 예민한 됨됨이만 받친다")

        group_0_mask = sensitive_attr == groups[0]
        group_1_mask = sensitive_attr == groups[1]

        rate_0 = np.mean(y_pred[group_0_mask])
        rate_1 = np.mean(y_pred[group_1_mask])

        # 0으로 나누지 않게 한다
        if rate_1 == 0:
            return float('inf')

        return min(rate_0 / rate_1, rate_1 / rate_0)

    def equal_opportunity_difference(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> float:
        """
        고른 틈 차이를 셈한다.

        고른 틈: 무리 사이에 TPR이 같아야 한다
        TPR = P(Y_pred=1|Y_true=1, A=a)

        Args:
            y_true: 참 이름표
            y_pred: 미루어 본 이름표
            sensitive_attr: 예민한 됨됨이 값

        Returns:
            무리 사이 참 양수 비율의 차이
        """
        groups = np.unique(sensitive_attr)
        if len(groups) != 2:
            raise ValueError("이 구현은 두 값 예민한 됨됨이만 받친다")

        tpr_list = []
        for group in groups:
            group_mask = sensitive_attr == group
            positive_mask = y_true == 1
            combined_mask = group_mask & positive_mask

            if np.sum(combined_mask) == 0:
                tpr_list.append(0.0)
            else:
                tpr = np.sum((y_pred == 1) & combined_mask) / np.sum(combined_mask)
                tpr_list.append(tpr)

        return abs(tpr_list[0] - tpr_list[1])

    def equalized_odds_difference(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Tuple[float, float]:
        """
        고른 승산 차이를 셈한다.

        고른 승산: 무리 사이에 TPR과 FPR이 모두 같아야 한다

        Args:
            y_true: 참 이름표
            y_pred: 미루어 본 이름표
            sensitive_attr: 예민한 됨됨이 값

        Returns:
            (TPR 차이, FPR 차이) 짝
        """
        groups = np.unique(sensitive_attr)
        if len(groups) != 2:
            raise ValueError("이 구현은 두 값 예민한 됨됨이만 받친다")

        tpr_list = []
        fpr_list = []

        for group in groups:
            group_mask = sensitive_attr == group

            # 참 양수 비율
            positive_mask = y_true == 1
            combined_mask = group_mask & positive_mask
            if np.sum(combined_mask) == 0:
                tpr = 0.0
            else:
                tpr = np.sum((y_pred == 1) & combined_mask) / np.sum(combined_mask)
            tpr_list.append(tpr)

            # 거짓 양수 비율
            negative_mask = y_true == 0
            combined_mask = group_mask & negative_mask
            if np.sum(combined_mask) == 0:
                fpr = 0.0
            else:
                fpr = np.sum((y_pred == 1) & combined_mask) / np.sum(combined_mask)
            fpr_list.append(fpr)

        return abs(tpr_list[0] - tpr_list[1]), abs(fpr_list[0] - fpr_list[1])

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray,
        attr_name: str
    ) -> Dict[str, float]:
        """
        주어진 예민한 됨됨이에 대해 온 치우침 자를 셈한다.

        Args:
            y_true: 참 이름표
            y_pred: 미루어 본 이름표
            sensitive_attr: 예민한 됨됨이 값
            attr_name: 예민한 됨됨이의 이름

        Returns:
            자 이름과 값의 사전
        """
        metrics = {}

        metrics[f'{attr_name}_statistical_parity_diff'] = \
            self.statistical_parity_difference(y_pred, sensitive_attr)

        metrics[f'{attr_name}_disparate_impact_ratio'] = \
            self.disparate_impact_ratio(y_pred, sensitive_attr)

        metrics[f'{attr_name}_equal_opportunity_diff'] = \
            self.equal_opportunity_difference(y_true, y_pred, sensitive_attr)

        tpr_diff, fpr_diff = self.equalized_odds_difference(y_true, y_pred, sensitive_attr)
        metrics[f'{attr_name}_equalized_odds_tpr_diff'] = tpr_diff
        metrics[f'{attr_name}_equalized_odds_fpr_diff'] = fpr_diff

        return metrics

    def generate_bias_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attrs_dict: Dict[str, np.ndarray]
    ) -> str:
        """
        두루 갖춘 치우침 알림을 만든다.

        Args:
            y_true: 참 이름표
            y_pred: 미루어 본 이름표
            sensitive_attrs_dict: 됨됨이 이름을 값에 이어 주는 사전

        Returns:
            형태 갖춘 알림 글자열
        """
        report = "=" * 60 + "\n"
        report += "치우침과 공정성 알림\n"
        report += "=" * 60 + "\n\n"

        for attr_name, attr_values in sensitive_attrs_dict.items():
            report += f"\n{attr_name.upper()}\n"
            report += "-" * 60 + "\n"

            metrics = self.compute_all_metrics(y_true, y_pred, attr_values, attr_name)

            for metric_name, value in metrics.items():
                report += f"{metric_name}: {value:.4f}\n"

            # 읽는 법을 더한다
            report += "\n읽는 법:\n"

            spd = metrics[f'{attr_name}_statistical_parity_diff']
            if spd < 0.1:
                report += "✓ 통계적 패리티: 치우침 적음\n"
            elif spd < 0.2:
                report += "⚠ 통계적 패리티: 치우침 가운데\n"
            else:
                report += "✗ 통계적 패리티: 치우침 큼\n"

            di = metrics[f'{attr_name}_disparate_impact_ratio']
            if di >= 0.8:
                report += "✓ 달리 미침: 받아들일 만함 (>= 0.8)\n"
            else:
                report += "✗ 달리 미침: 탈이 있음 (< 0.8)\n"

            report += "\n"

        return report


def example_usage():
    """BiasDetector을 쓰는 보기."""
    np.random.seed(42)

    # 자료를 흉내낸다
    n_samples = 1000

    # 치우친 지어낸 자료를 만든다
    gender = np.random.randint(0, 2, n_samples)  # 0: 남, 1: 여

    # 치우친 미루어 봄: 남자 쪽 양수 미루어 봄 비율이 높다
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.where(
        gender == 0,
        np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),  # 남자는 70%이 양수
        np.random.choice([0, 1], n_samples, p=[0.6, 0.4])   # 여자는 40%이 양수
    )

    # 치우침을 알아낸다
    detector = BiasDetector(['gender'])

    sensitive_attrs = {'gender': gender}
    report = detector.generate_bias_report(y_true, y_pred, sensitive_attrs)

    print(report)


if __name__ == "__main__":
    example_usage()
```

**출력:**

```
============================================================
치우침과 공정성 알림
============================================================


GENDER
------------------------------------------------------------
gender_statistical_parity_diff: 0.2639
gender_disparate_impact_ratio: 0.6117
gender_equal_opportunity_diff: 0.2722
gender_equalized_odds_tpr_diff: 0.2722
gender_equalized_odds_fpr_diff: 0.2563

읽는 법:
✗ 통계적 패리티: 치우침 큼
✗ 달리 미침: 탈이 있음 (< 0.8)
```

## 2. 논의

이 구현은 NumPy만으로 네 가지 집단 공정성 지표를 계산한다. 모델을 트레인하지 않으며, 예측은 남성에게 70%, 여성에게 40%의 비율로 양성을 주도록 지어낸 것이다. 그래서 출력의 통계적 패리티 차이 0.2639와 이질적 영향 비 0.6117은 일부러 심어 둔 차이(0.3과 $0.4/0.7 \approx 0.57$)를 표본 1,000개로 잰 값이다. 정답 레이블이 예측과 무관한 무작위 값이므로 TPR 차이(0.2722)와 FPR 차이(0.2563)도 패리티 차이와 비슷하게 나온다.

지표마다 함수가 따로 있어 하나씩 이해하기 쉽고, 실제 모델의 예측과 보호 속성 배열을 넣으면 그대로 쓸 수 있다. 다만 모든 함수가 이진 보호 속성만 받으며, 집단이 둘이 아니면 오류를 낸다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 그렇게 했는지와 그 한계를 설명하라.

</div>

??? success "풀이"
    (1) 차이 지표를 절댓값으로 돌려준다 -- 0.1, 0.2 같은 임계값과 바로 견줄 수 있지만, 어느 집단이 불리한지는 잃는다. (2) 이질적 영향 비를 두 방향의 비 가운데 작은 쪽으로 잡는다 -- 집단의 순서와 상관없이 80% 규칙을 적용할 수 있다. (3) 보호 속성을 이진으로 한정한다 -- 식이 단순해지지만 집단이 셋 이상이면 쓸 수 없다. 덧붙여, 어떤 집단에 정답 양성 표본이 하나도 없으면 TPR을 0으로 두는데, 이는 실제 차이를 부풀리거나 가릴 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`BiasDetector`의 지표 함수들에 입력 검증을 더하여라. 배열의 형태와 값이 기대한 대로인지 확인하고, 올바르지 않은 입력에는 알아보기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    지표 함수마다 앞부분에서 같은 검사를 하도록 도우미 함수 하나를 두는 것이 깔끔하다. 확인할 것은 넷이다. (1) `y_pred`, `y_true`, `sensitive_attr`가 모두 1차원 배열이고 길이가 같은가. 길이가 다르면 불리언 마스크 인덱싱이 바로 오류를 내거나, 브로드캐스팅으로 엉뚱한 값을 낸다. (2) 배열이 비어 있지 않은가. 빈 배열의 평균은 경고와 함께 `nan`이 되어 보고서에 그대로 찍힌다. (3) `y_pred`와 `y_true`의 값이 0과 1뿐인가. 확률을 실수로 넘기면 `y_pred == 1`인 표본이 거의 없어 TPR과 FPR이 0에 가깝게 나오는데, 오류 없이 그럴듯한 수가 나오므로 가장 위험하다. (4) 각 집단에 표본이 있고, TPR·FPR을 계산할 때는 그 집단에 정답 양성·음성 표본이 있는가. 검사가 실패하면 어느 배열의 무엇이 기대와 다른지(예: 「`y_pred`에 0과 1이 아닌 값 0.73이 있다」)를 메시지에 적는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 실패할 만한 유형 둘을 들고, 저마다 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    (1) **이질적 영향 비가 거꾸로 판정된다.** `disparate_impact_ratio`는 집단 1의 양성 비율이 0이면 `inf`를 돌려준다. 그런데 보고서는 `di >= 0.8`이면 「받아들일 만함」이라 적으므로, 한 집단이 양성을 하나도 받지 못한 가장 심한 바이어스가 합격으로 찍힌다. 집단 0의 비율이 0일 때도 0으로 나누기 경고가 난다. 양성 비율이 0인 집단을 만들어 넣는 테스트로 탐지하고, 한쪽만 0이면 0을, 둘 다 0이면 `nan`과 경고를 돌려주도록 고친다. (2) **표본이 없는 칸을 0으로 메운다.** 어떤 집단에 정답 양성 표본이 없으면 TPR을 0으로 두므로, 다른 집단의 TPR이 그대로 「차이」로 보고된다. 실제로는 잴 수 없는 값이다. 집단과 정답 레이블의 칸마다 표본 수를 함께 보고해 탐지하고, 표본이 없거나 너무 적은 칸은 `nan`으로 두고 경고하도록 고친다. 두 실패 모두 오류 없이 그럴듯한 수를 내므로, 지표 값만 보고는 알아챌 수 없다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
바이어스 탐지 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_biasdetector():
        model = BiasDetector(...)
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
    위 틀은 뉴럴 네트워크용이라 `BiasDetector`에는 그대로 맞지 않는다. 이 클래스에서 시험할 것은 이렇다. 빈 배열과 원소 하나짜리 배열은 집단이 둘이 아니므로 `ValueError`가 나야 한다. 두 집단의 예측이 똑같으면 패리티 차이는 0, 이질적 영향 비는 1이어야 한다. 한 집단의 양성 비율이 0이면 이질적 영향 비가 0이어야 하는데, 그 집단이 집단 1이면 지금 구현은 `inf`를 내므로 이 테스트가 실패한다(연습문제 3). 표본이 아주 많을 때는 지어낸 비율(0.7과 0.4)에 대해 패리티 차이가 0.3에, 비가 $0.4/0.7$에 다가가는지 본다.

## 정리하며

**다룬 것** — 바이어스 탐지

이 구현은 NumPy 코드로 집단 공정성 지표 네 가지를 계산하고, 지어낸 바이어스가 지표에 어떻게 드러나는지 보여 준다.

핵심 클래스는 `BiasDetector`이며, 앞의 연습문제 4개로 그 설계와 약점을 스스로 따져 볼 수 있다.
