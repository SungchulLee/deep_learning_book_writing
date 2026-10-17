# 바이어스 탐지

딥러닝에서의 바이어스 알아내기와 재기. 이 꾸러미는 머신러닝 모델의 바이어스를 알아내고 재는 연장을 준다.

머신러닝 구조에서 공정성을 지키는 일은 윤리로도 마땅하고 참으로도 걸린 문제다. 이 꾸러미는 딥러닝 모델의 바이어스를 알아내고, 재고, 눅이는 재주를 보이며, 이론의 공정성 잣대를 손에 잡히는 코드로 이어 준다.

## 1. 코드

```python
"""
깊은 러닝에서의 치우침 알아내기와 재기
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
        통계 고름 차이를 셈한다.

        통계 고름: 모든 무리 a, b에 대해 P(Y=1|A=a) = P(Y=1|A=b)

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
            꼴 갖춘 알림 글자열
        """
        report = "=" * 60 + "\n"
        report += "치우침과 고름 알림\n"
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
                report += "✓ 통계 고름: 치우침 적음\n"
            elif spd < 0.2:
                report += "⚠ 통계 고름: 치우침 가운데\n"
            else:
                report += "✗ 통계 고름: 치우침 큼\n"

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
치우침과 고름 알림
============================================================


GENDER
------------------------------------------------------------
gender_statistical_parity_diff: 0.2639
gender_disparate_impact_ratio: 0.6117
gender_equal_opportunity_diff: 0.2722
gender_equalized_odds_tpr_diff: 0.2722
gender_equalized_odds_fpr_diff: 0.2563

읽는 법:
✗ 통계 고름: 치우침 큼
✗ 달리 미침: 탈이 있음 (< 0.8)
```

## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 딥러닝의 핵심이 되는 생각을 보여 준다. 조각으로 나눈 구조 덕에 부분마다 따로 살피고 다른 일이나 데이터에 맞추어 고치기 쉽다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 구조의 클래스, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 머신러닝 과제에 대한 감이 몸에 붙는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 딥러닝에 알맞은지 밝혀라.

</div>

??? success "풀이"
    설계 판단은 구현마다 다르나 흔히 이런 것이 있다. (1) 엑티베이션 함수 고르기 -- ReLU 클래스는 그래디언트가 잦아들지 않아 학습이 빠르다. (2) 고르게 하는 꾀 -- 배치 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 학습을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 고른 것마다 나타내는 힘, 계산 비용, 학습의 든든함 사이의 트레이드오프를 드러낸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 꼴과 데이터 클래스가 바라는 대로인지 살피는 입력 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아듣기 쉬운 어긋남 알림을 띄워라.

</div>

??? success "풀이"
    `forward` 방법(또는 알맞은 함수)의 소개에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 디버깅을 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

</div>

??? success "풀이"
    흔히 무너지는 결은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 그래디언트 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 구조 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 학습 로스는 줄어드는데 살핌 로스가 오르면 짚어낸다. 정규화(드롭아웃, 짐 줄이기, 데이터 늘리기)나 모델 크기 줄이기로 고친다. 학습과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
바이어스 알아내기 구현을 살피는 두루 갖춘 시험 함수를 써라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 끝자락 값(0, 아주 큰 수)이 든 입력 같은 가장자리 자리를 시험하여라.

</div>

??? success "풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
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
    구조가 끝에서 끝까지 학습을 받치는지 알려면 그래디언트 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 바이어스 알아내기

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 딥러닝의 핵심이 되는 생각을 보여 준다.

핵심 클래스는 `BiasDetector`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
