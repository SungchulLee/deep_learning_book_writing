# 고름 자

고름 자와 따지기. 기계 배움 모형의 고름을 따지는 두루 갖춘 고름 자.

기계 배움 얼개에서 고름을 지키는 일은 윤리로도 마땅하고 참으로도 걸린 문제다. 이 꾸러미는 깊은 배움 모형의 치우침을 알아내고, 재고, 눅이는 재주를 보이며, 이론의 고름 잣대를 손에 잡히는 코드로 이어 준다.

## 1. 코드

```python
"""
고름 자와 따지기
기계 배움 모형의 고름을 따지는 두루 갖춘 고름 자.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, accuracy_score

# ========================================================================
# 메인
# ========================================================================


class FairnessMetrics:
    """기계 배움 모형을 위한 두루 갖춘 고름 자."""

    @staticmethod
    def demographic_parity(
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict[str, float]:
        """
        인구 고름 자를 셈한다.

        인구 고름(통계 고름):
        P(Y_pred=1|A=0) = P(Y_pred=1|A=1)

        Args:
            y_pred: 미루어 본 이름표
            sensitive_attr: 예민한 됨됨이(두 값)

        Returns:
            인구 고름 자를 담은 사전
        """
        groups = np.unique(sensitive_attr)
        positive_rates = {}

        for group in groups:
            mask = sensitive_attr == group
            positive_rates[f'group_{group}'] = np.mean(y_pred[mask])

        max_rate = max(positive_rates.values())
        min_rate = min(positive_rates.values())

        return {
            'positive_rates': positive_rates,
            'demographic_parity_difference': max_rate - min_rate,
            'demographic_parity_ratio': min_rate / max_rate if max_rate > 0 else 0
        }

    @staticmethod
    def equal_opportunity(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict[str, float]:
        """
        고른 틈 자를 셈한다.

        고른 틈: 무리 사이에 TPR이 같아야 한다
        TPR = P(Y_pred=1|Y_true=1, A=a)

        Args:
            y_true: 참 이름표
            y_pred: 미루어 본 이름표
            sensitive_attr: 예민한 됨됨이

        Returns:
            고른 틈 자를 담은 사전
        """
        groups = np.unique(sensitive_attr)
        tpr_dict = {}

        for group in groups:
            mask = (sensitive_attr == group) & (y_true == 1)
            if np.sum(mask) > 0:
                tpr = np.sum((y_pred == 1) & mask) / np.sum(mask)
                tpr_dict[f'tpr_group_{group}'] = tpr
            else:
                tpr_dict[f'tpr_group_{group}'] = 0.0

        tpr_values = list(tpr_dict.values())

        return {
            'true_positive_rates': tpr_dict,
            'equal_opportunity_difference': max(tpr_values) - min(tpr_values)
        }

    @staticmethod
    def equalized_odds(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict[str, float]:
        """
        고른 승산 자를 셈한다.

        고른 승산: 무리 사이에 TPR과 FPR이 모두 같아야 한다

        Args:
            y_true: 참 이름표
            y_pred: 미루어 본 이름표
            sensitive_attr: 예민한 됨됨이

        Returns:
            고른 승산 자를 담은 사전
        """
        groups = np.unique(sensitive_attr)
        tpr_dict = {}
        fpr_dict = {}

        for group in groups:
            group_mask = sensitive_attr == group

            # TPR
            pos_mask = group_mask & (y_true == 1)
            if np.sum(pos_mask) > 0:
                tpr = np.sum((y_pred == 1) & pos_mask) / np.sum(pos_mask)
            else:
                tpr = 0.0
            tpr_dict[f'tpr_group_{group}'] = tpr

            # FPR
            neg_mask = group_mask & (y_true == 0)
            if np.sum(neg_mask) > 0:
                fpr = np.sum((y_pred == 1) & neg_mask) / np.sum(neg_mask)
            else:
                fpr = 0.0
            fpr_dict[f'fpr_group_{group}'] = fpr

        tpr_values = list(tpr_dict.values())
        fpr_values = list(fpr_dict.values())

        return {
            'true_positive_rates': tpr_dict,
            'false_positive_rates': fpr_dict,
            'tpr_difference': max(tpr_values) - min(tpr_values),
            'fpr_difference': max(fpr_values) - min(fpr_values),
            'average_odds_difference': (max(tpr_values) - min(tpr_values) + 
                                       max(fpr_values) - min(fpr_values)) / 2
        }

    @staticmethod
    def predictive_parity(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Dict[str, float]:
        """
        미루어 봄 고름 자를 셈한다.

        미루어 봄 고름(결과 따지기):
        PPV(맞힘)가 무리 사이에 같아야 한다
        PPV = P(Y_true=1|Y_pred=1, A=a)

        Args:
            y_true: 참 이름표
            y_pred: 미루어 본 이름표
            sensitive_attr: 예민한 됨됨이

        Returns:
            미루어 봄 고름 자를 담은 사전
        """
        groups = np.unique(sensitive_attr)
        ppv_dict = {}

        for group in groups:
            mask = (sensitive_attr == group) & (y_pred == 1)
            if np.sum(mask) > 0:
                ppv = np.sum((y_true == 1) & mask) / np.sum(mask)
                ppv_dict[f'ppv_group_{group}'] = ppv
            else:
                ppv_dict[f'ppv_group_{group}'] = 0.0

        ppv_values = list(ppv_dict.values())

        return {
            'positive_predictive_values': ppv_dict,
            'predictive_parity_difference': max(ppv_values) - min(ppv_values)
        }

    @staticmethod
    def calibration_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sensitive_attr: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Dict]:
        """
        무리마다 눈금 맞음 자를 셈한다.

        눈금이 잘 맞은 모형: 점수가 s인 미루어 봄 가운데
        대략 s 몫이 양수여야 한다.

        Args:
            y_true: 참 이름표
            y_pred_proba: 미루어 본 낌새
            sensitive_attr: 예민한 됨됨이
            n_bins: 눈금 맞음에 쓸 통의 수

        Returns:
            무리마다의 눈금 맞음 자
        """
        groups = np.unique(sensitive_attr)
        calibration_dict = {}

        for group in groups:
            mask = sensitive_attr == group
            y_true_group = y_true[mask]
            y_proba_group = y_pred_proba[mask]

            # 통을 만든다
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_proba_group, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            bin_true_prob = []
            bin_pred_prob = []
            bin_counts = []

            for i in range(n_bins):
                bin_mask = bin_indices == i
                if np.sum(bin_mask) > 0:
                    bin_true_prob.append(np.mean(y_true_group[bin_mask]))
                    bin_pred_prob.append(np.mean(y_proba_group[bin_mask]))
                    bin_counts.append(np.sum(bin_mask))
                else:
                    bin_true_prob.append(0)
                    bin_pred_prob.append(0)
                    bin_counts.append(0)

            # 바라는 눈금 맞음 어긋남(ECE)
            ece = 0
            total_samples = len(y_true_group)
            for i in range(n_bins):
                if bin_counts[i] > 0:
                    ece += (bin_counts[i] / total_samples) * abs(bin_true_prob[i] - bin_pred_prob[i])

            calibration_dict[f'group_{group}'] = {
                'expected_calibration_error': ece,
                'bin_true_probabilities': bin_true_prob,
                'bin_predicted_probabilities': bin_pred_prob,
                'bin_counts': bin_counts
            }

        return calibration_dict

    @staticmethod
    def group_fairness_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        아우른 무리 고름 점수를 셈한다.

        고름 자 여럿을 점수 하나로 아우른다.
        점수가 낮을수록 더 고르다.

        Args:
            y_true: 참 이름표
            y_pred: 미루어 본 이름표
            sensitive_attr: 예민한 됨됨이
            weights: 자마다의 짐(없어도 된다)

        Returns:
            아우른 고름 점수
        """
        if weights is None:
            weights = {
                'demographic_parity': 1.0,
                'equal_opportunity': 1.0,
                'equalized_odds': 1.0,
                'predictive_parity': 1.0
            }

        metrics = FairnessMetrics()

        # 온 자를 얻는다
        dp = metrics.demographic_parity(y_pred, sensitive_attr)
        eo = metrics.equal_opportunity(y_true, y_pred, sensitive_attr)
        eq = metrics.equalized_odds(y_true, y_pred, sensitive_attr)
        pp = metrics.predictive_parity(y_true, y_pred, sensitive_attr)

        # 짐 실은 점수를 셈한다
        score = 0
        score += weights['demographic_parity'] * dp['demographic_parity_difference']
        score += weights['equal_opportunity'] * eo['equal_opportunity_difference']
        score += weights['equalized_odds'] * eq['average_odds_difference']
        score += weights['predictive_parity'] * pp['predictive_parity_difference']

        total_weight = sum(weights.values())
        return score / total_weight if total_weight > 0 else score


def comprehensive_fairness_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    sensitive_attrs: Dict[str, np.ndarray]
) -> str:
    """
    두루 갖춘 고름 따지기를 한다.

    Args:
        y_true: 참 이름표
        y_pred: 미루어 본 이름표
        y_pred_proba: 미루어 본 낌새(없어도 된다)
        sensitive_attrs: 예민한 됨됨이의 사전

    Returns:
        꼴 갖춘 따짐 알림
    """
    metrics = FairnessMetrics()
    report = []

    report.append("=" * 80)
    report.append("두루 갖춘 고름 따지기")
    report.append("=" * 80)

    for attr_name, attr_values in sensitive_attrs.items():
        report.append(f"\n{'=' * 80}")
        report.append(f"예민한 됨됨이: {attr_name.upper()}")
        report.append(f"{'=' * 80}\n")

        # 인구 고름
        report.append("1. 인구 고름")
        report.append("-" * 40)
        dp = metrics.demographic_parity(y_pred, attr_values)
        for key, value in dp.items():
            report.append(f"   {key}: {value}")
        report.append("")

        # 고른 틈
        report.append("2. 고른 틈")
        report.append("-" * 40)
        eo = metrics.equal_opportunity(y_true, y_pred, attr_values)
        for key, value in eo.items():
            report.append(f"   {key}: {value}")
        report.append("")

        # 고른 승산
        report.append("3. 고른 승산")
        report.append("-" * 40)
        eq = metrics.equalized_odds(y_true, y_pred, attr_values)
        for key, value in eq.items():
            report.append(f"   {key}: {value}")
        report.append("")

        # 미루어 봄 고름
        report.append("4. 미루어 봄 고름")
        report.append("-" * 40)
        pp = metrics.predictive_parity(y_true, y_pred, attr_values)
        for key, value in pp.items():
            report.append(f"   {key}: {value}")
        report.append("")

        # 눈금 맞음(낌새가 주어졌을 때)
        if y_pred_proba is not None:
            report.append("5. 눈금 맞음")
            report.append("-" * 40)
            cal = metrics.calibration_metrics(y_true, y_pred_proba, attr_values)
            for group, cal_metrics in cal.items():
                report.append(f"   {group}: ECE = {cal_metrics['expected_calibration_error']:.4f}")
            report.append("")

        # 아우른 점수
        report.append("6. 아우른 고름 점수")
        report.append("-" * 40)
        score = metrics.group_fairness_score(y_true, y_pred, attr_values)
        report.append(f"   점수: {score:.4f} (낮을수록 좋다)")
        report.append("")

    return "\n".join(report)


if __name__ == "__main__":
    # 쓰는 보기
    np.random.seed(42)

    n_samples = 1000
    gender = np.random.randint(0, 2, n_samples)
    y_true = np.random.randint(0, 2, n_samples)

    # 치우친 미루어 봄
    y_pred = np.where(
        gender == 0,
        np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    )

    y_pred_proba = np.random.rand(n_samples)

    report = comprehensive_fairness_evaluation(
        y_true, y_pred, y_pred_proba,
        {'gender': gender}
    )

    print(report)
```

**출력:**

```
================================================================================
두루 갖춘 고름 따지기
================================================================================

================================================================================
예민한 됨됨이: GENDER
================================================================================

1. 인구 고름
----------------------------------------
   positive_rates: {'group_0': 0.6795918367346939, 'group_1': 0.41568627450980394}
   demographic_parity_difference: 0.26390556222489
   demographic_parity_ratio: 0.6116704940234352

2. 고른 틈
----------------------------------------
   true_positive_rates: {'tpr_group_0': 0.6796536796536796, 'tpr_group_1': 0.4074074074074074}
   equal_opportunity_difference: 0.27224627224627224

3. 고른 승산
----------------------------------------
   true_positive_rates: {'tpr_group_0': 0.6796536796536796, 'tpr_group_1': 0.4074074074074074}
   false_positive_rates: {'fpr_group_0': 0.6795366795366795, 'fpr_group_1': 0.4232209737827715}
   tpr_difference: 0.27224627224627224
   fpr_difference: 0.256315705753908
   average_odds_difference: 0.2642809890000901

4. 미루어 봄 고름
----------------------------------------
   positive_predictive_values: {'ppv_group_0': 0.47147147147147145, 'ppv_group_1': 0.4669811320754717}
   predictive_parity_difference: 0.004490339395999743

5. 눈금 맞음
----------------------------------------
   group_0: ECE = 0.2341
   group_1: ECE = 0.2404

6. 아우른 고름 점수
----------------------------------------
   점수: 0.2012 (낮을수록 좋다)
```

## 2. 논의

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 깊은 배움의 고갱이가 되는 생각을 보여 준다. 조각으로 나눈 얼개 덕에 부분마다 따로 살피고 다른 일이나 자료에 맞추어 고치기 쉽다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 얼개의 갈래, 여러 자료를 바꿔 가며 해 보면 이해가 깊어지고 기계 배움 일감에 대한 감이 몸에 붙는다.

## 연습문제

**연습문제 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 깊은 배움에 알맞은지 밝혀라.

??? success "연습문제 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 고르게 하는 꾀 -- 묶음 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 나타내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 드러낸다.

---

**연습문제 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**연습문제 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 살핌 잃음이 오르면 짚어낸다. 정칙화(드롭아웃, 짐 줄이기, 자료 늘리기)나 모형 크기 줄이기로 고친다. 익힘과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**연습문제 4.**
고름 자 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_fairnessmetrics():
        model = FairnessMetrics(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 묶음
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 기울기 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    얼개가 끝에서 끝까지 익히기를 받치는지 알려면 기울기 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 고름 자

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 깊은 배움의 고갱이가 되는 생각을 보여 준다.

고갱이 갈래는 `FairnessMetrics`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
