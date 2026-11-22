# 공정성 지표

공정성 지표와 평가. 머신러닝 모델의 공정성을 여러 각도에서 재는 지표들을 한데 모은다.

머신러닝 시스템에서 공정성을 지키는 일은 윤리적으로도 마땅하고 실무적으로도 중요한 문제다. 이 꾸러미는 집단 공정성 지표 다섯 가지 — 인구통계적 동등성, 기회 균등, 균등화 오즈, 예측 동등성(양성으로 예측된 사람 가운데 실제 양성의 비율인 PPV가 집단 사이에 같은가), 캘리브레이션(예측 확률이 실제 비율과 맞는가) — 를 NumPy로 계산하고, 이를 가중 평균한 종합 점수를 보고서로 묶는다. 바이어스를 완화하지는 않는다.

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

이 구현은 NumPy만으로 지표를 계산하며 모델을 트레인하지 않는다. 예제의 예측은 남성에게 70%, 여성에게 40%의 비율로 양성을 주도록 지어냈고, 정답 레이블과 예측 확률은 둘 다 다른 것과 무관한 난수다. 시드와 생성 방식이 바이어스 탐지 쪽과 같으므로 패리티 차이 0.2639, 비 0.6117, TPR 차이 0.2722, FPR 차이 0.2563이 그 쪽과 똑같이 나온다.

눈여겨볼 것은 지표끼리 엇갈린다는 점이다. 인구통계적 동등성과 균등화 오즈는 0.26~0.27의 큰 차이를 보이는데, 예측 동등성의 차이는 0.0045로 거의 0이다. 정답 레이블이 예측과 무관하므로 양성으로 예측된 사람 가운데 실제 양성의 비율이 두 집단 모두 기저율 근처(0.471, 0.467)에 머물기 때문이다. 곧 「공정한가」의 답은 어느 지표를 보느냐에 따라 달라지며, 이 장 개요의 불가능성 정리가 말하는 긴장이 작은 예제에서도 드러난다. 캘리브레이션 오차(ECE)가 두 집단 모두 0.23~0.24로 큰 것은 예측 확률이 레이블과 무관한 균등 난수라서다.

종합 점수 0.2012는 네 차이 값(0.2639, 0.2722, 0.2643, 0.0045)의 평균이다. 다만 기회 균등의 차이(TPR 차이)가 균등화 오즈의 평균 오즈 차이에도 들어 있어 TPR 차이가 두 번 세어지고, 예측 동등성이 잘 맞으면 다른 지표의 큰 위반이 평균에 묻힌다. 점수 하나보다는 지표별 값을 함께 보고하는 편이 안전하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 그렇게 했는지와 그 한계를 설명하라.

</div>

??? success "풀이"
    (1) 차이를 「최댓값 − 최솟값」으로 잡는다 -- 바이어스 탐지 쪽과 달리 집단이 셋 이상이어도 돌아가지만, 가장 벌어진 두 집단만 보고 나머지 집단의 분포는 버린다. (2) 지표마다 집단별 비율을 사전으로 함께 돌려준다 -- 차이 값만 주면 어느 집단이 불리한지 알 수 없는데, 이 설계로 그것이 보인다. (3) 여러 지표를 가중 평균한 종합 점수를 둔다 -- 모델을 비교하기 쉽지만, 논의에서 본 것처럼 TPR 차이를 두 번 세고 한 지표의 큰 위반이 다른 지표의 작은 값에 묻힌다. 덧붙여, 표본이 없는 칸의 TPR·FPR·PPV를 0으로 두므로 잴 수 없는 값이 차이로 보고될 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`FairnessMetrics`의 지표 함수들에 입력 검증을 더하여라. 배열의 형태와 값이 기대한 대로인지 확인하고, 올바르지 않은 입력에는 알아보기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    도우미 함수 하나를 두어 모든 지표 함수가 앞부분에서 부르게 한다. 확인할 것은 이렇다. (1) 모든 배열이 1차원이고 길이가 같은가. (2) 배열이 비어 있지 않고 집단이 둘 이상인가. 집단이 하나뿐이면 차이가 늘 0으로 나와 「공정하다」로 잘못 읽힌다. (3) `y_true`와 `y_pred`의 값이 0과 1뿐인가. (4) `y_pred_proba`가 0 이상 1 이하인가. 범위를 벗어난 점수는 `np.clip` 때문에 끝 통으로 조용히 몰려 ECE를 왜곡한다. 실패하면 어느 배열의 무엇이 기대와 다른지를 메시지에 적는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 실패할 만한 유형 둘을 들고, 저마다 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    (1) **빈 칸을 0으로 메운다.** 어떤 집단이 양성 예측을 하나도 받지 못하면 그 집단의 PPV를 0으로 두므로, 실제로는 정의되지 않는 값이 큰 예측 동등성 위반으로 보고된다. TPR·FPR도 같다. 칸마다 표본 수를 함께 출력해 탐지하고, 빈 칸은 `nan`으로 두어 차이 계산에서 빼고 경고하도록 고친다. (2) **작은 집단에서 지표가 흔들린다.** 집단의 표본이 수십 개뿐이면 비율의 표준오차가 커서, 예측이 공정해도 차이가 0.1을 쉽게 넘는다. ECE는 통이 10개라 더 심하다. 부트스트랩으로 차이의 신뢰구간을 함께 보고해 탐지하고, 구간이 0을 포함하면 「위반」이라 단정하지 않는다. 작은 집단에서는 통 수도 줄인다.

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
    위 틀은 뉴럴 네트워크용이라 `FairnessMetrics`에는 그대로 맞지 않는다. 이 클래스에서 시험할 것은 이렇다. 두 집단의 예측이 똑같으면 모든 차이가 0이고 비가 1인지, 완벽한 예측($\hat{y} = y$)이면 TPR이 모두 1, FPR이 모두 0인지, 확률이 완벽히 캘리브레이션된 합성 데이터에서 ECE가 0에 가까운지 본다. 경계 사례로는 빈 배열, 집단이 하나뿐인 배열, 한 집단이 양성 예측을 하나도 받지 못한 배열을 넣어, 오류가 나거나 `nan`이 나와야 할 곳에서 그럴듯한 수가 나오지 않는지 확인한다.

## 정리하며

**다룬 것** — 공정성 지표

이 구현은 NumPy 코드로 집단 공정성 지표 다섯 가지와 종합 점수를 계산하고, 같은 예측이 어떤 지표로는 크게 불공정하고 어떤 지표로는 공정해 보일 수 있음을 보여 준다.

핵심 클래스는 `FairnessMetrics`이며, 앞의 연습문제 4개로 그 설계와 약점을 스스로 따져 볼 수 있다.
