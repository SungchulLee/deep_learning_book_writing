# 바이어스 탐지

딥러닝에서의 바이어스 탐지와 재기. 이 꾸러미는 머신러닝 모델의 바이어스를 탐지하고 재는 도구를 준다.

머신러닝 시스템에서 공정성을 지키는 일은 윤리적으로도 마땅하고 실무적으로도 중요한 문제다. 이 꾸러미는 딥러닝 모델의 바이어스를 탐지하고, 측정하고, 완화하는 기법을 보이며, 이론의 공정성 지표를 실제 코드로 연결해 준다.

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

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 딥러닝의 핵심 아이디어를 보여 준다. 모듈로 나눈 구조 덕에 부분마다 따로 이해하고 다른 과제나 데이터에 맞추어 고치기 쉽다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 머신러닝 과제에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 딥러닝에 알맞은지 설명하라.

</div>

??? success "풀이"
    설계 선택은 구현마다 다르나 흔히 이런 것이 있다. (1) 활성화 함수 선택 -- ReLU 계열은 그래디언트가 소실되지 않아 학습이 빠르다. (2) 정규화 전략 -- 배치 정규화가 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 -- 있으면 스킵 경로를 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 트레이드오프를 드러낸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 데이터 타입이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아보기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 해당 함수)의 앞부분에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 더한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아보기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기 안전하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 실패할 만한 유형 둘을 들고, 저마다 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 유형은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 모니터링해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 수정(잔차 연결, 정규화 레이어)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 증강)나 모델 축소로 고친다. 학습 지표와 검증 지표를 늘 함께 모니터링해 이를 일찍 잡아야 한다.

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
    구조가 엔드투엔드 학습을 지원하는지 알려면 그래디언트 흐름을 테스트하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 바이어스 탐지

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 딥러닝의 핵심 아이디어를 보여 준다.

핵심 클래스는 `BiasDetector`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
