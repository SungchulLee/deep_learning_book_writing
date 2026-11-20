# 공정성 지표

공정성 지표와 평가. 머신러닝 모델의 공정성을 평가하는 포괄적인 공정성 지표.

머신러닝 시스템에서 공정성을 지키는 일은 윤리적으로도 마땅하고 실무적으로도 중요한 문제다. 이 꾸러미는 딥러닝 모델의 바이어스를 탐지하고, 측정하고, 완화하는 기법을 보이며, 이론의 공정성 지표를 실제 코드로 연결해 준다.

## 1. 코드

??? note "코드 (401줄)"

    ```python
    """
    공정성 지표와 따지기
    기계 러닝 모형의 공정성을 따지는 두루 갖춘 공정성 지표.
    """

    import numpy as np
    from typing import Dict, List, Optional, Tuple
    from sklearn.metrics import confusion_matrix, accuracy_score

    # ========================================================================
    # 메인
    # ========================================================================


    class FairnessMetrics:
        """기계 러닝 모형을 위한 두루 갖춘 공정성 지표."""

        @staticmethod
        def demographic_parity(
            y_pred: np.ndarray,
            sensitive_attr: np.ndarray
        ) -> Dict[str, float]:
            """
            인구 공정성 지표를 셈한다.

            인구 공정성(통계적 패리티):
            P(Y_pred=1|A=0) = P(Y_pred=1|A=1)

            Args:
                y_pred: 미루어 본 이름표
                sensitive_attr: 예민한 됨됨이(두 값)

            Returns:
                인구 공정성 지표를 담은 사전
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
            미루어 봄 공정성 지표를 셈한다.

            미루어 봄 공정성(결과 따지기):
            PPV(맞힘)가 무리 사이에 같아야 한다
            PPV = P(Y_true=1|Y_pred=1, A=a)

            Args:
                y_true: 참 이름표
                y_pred: 미루어 본 이름표
                sensitive_attr: 예민한 됨됨이

            Returns:
                미루어 봄 공정성 지표를 담은 사전
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
                y_pred_proba: 미루어 본 확률
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
            아우른 무리 공정성 점수를 셈한다.

            공정성 지표 여럿을 점수 하나로 아우른다.
            점수가 낮을수록 더 고르다.

            Args:
                y_true: 참 이름표
                y_pred: 미루어 본 이름표
                sensitive_attr: 예민한 됨됨이
                weights: 자마다의 짐(없어도 된다)

            Returns:
                아우른 공정성 점수
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
        두루 갖춘 공정성 따지기를 한다.

        Args:
            y_true: 참 이름표
            y_pred: 미루어 본 이름표
            y_pred_proba: 미루어 본 확률(없어도 된다)
            sensitive_attrs: 예민한 됨됨이의 사전

        Returns:
            형태 갖춘 따짐 알림
        """
        metrics = FairnessMetrics()
        report = []

        report.append("=" * 80)
        report.append("두루 갖춘 공정성 따지기")
        report.append("=" * 80)

        for attr_name, attr_values in sensitive_attrs.items():
            report.append(f"\n{'=' * 80}")
            report.append(f"예민한 됨됨이: {attr_name.upper()}")
            report.append(f"{'=' * 80}\n")

            # 인구 공정성
            report.append("1. 인구 공정성")
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

            # 미루어 봄 공정성
            report.append("4. 미루어 봄 공정성")
            report.append("-" * 40)
            pp = metrics.predictive_parity(y_true, y_pred, attr_values)
            for key, value in pp.items():
                report.append(f"   {key}: {value}")
            report.append("")

            # 눈금 맞음(확률이 주어졌을 때)
            if y_pred_proba is not None:
                report.append("5. 눈금 맞음")
                report.append("-" * 40)
                cal = metrics.calibration_metrics(y_true, y_pred_proba, attr_values)
                for group, cal_metrics in cal.items():
                    report.append(f"   {group}: ECE = {cal_metrics['expected_calibration_error']:.4f}")
                report.append("")

            # 아우른 점수
            report.append("6. 아우른 공정성 점수")
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
두루 갖춘 공정성 따지기
================================================================================

================================================================================
예민한 됨됨이: GENDER
================================================================================

1. 인구 공정성
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

4. 미루어 봄 공정성
----------------------------------------
   positive_predictive_values: {'ppv_group_0': 0.47147147147147145, 'ppv_group_1': 0.4669811320754717}
   predictive_parity_difference: 0.004490339395999743

5. 눈금 맞음
----------------------------------------
   group_0: ECE = 0.2341
   group_1: ECE = 0.2404

6. 아우른 공정성 점수
----------------------------------------
   점수: 0.2012 (낮을수록 좋다)
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
공정성 지표 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_fairnessmetrics():
        model = FairnessMetrics(...)
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

**다룬 것** — 공정성 지표

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 딥러닝의 핵심 아이디어를 보여 준다.

핵심 클래스는 `FairnessMetrics`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
