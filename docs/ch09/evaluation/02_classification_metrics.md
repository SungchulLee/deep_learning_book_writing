# 분류 지표

분류 지표는 오류와 확률 예측의 서로 다른 면을 나타낸다. 클래스의 의미와 비용, 불균형과 목표 운영 조건을 먼저 정한 뒤 지표를 선택한다.

## 1. 코드

```python
"""
분류 지표
======================

분류 모델을 평가하는 지표를 두루 다룬다.

다루는 지표:
- 정확도
- 정밀도, 재현율, F1 점수
- 혼동 행렬
- ROC-AUC, PR-AUC
- 다중 클래스 지표
"""

import numpy as np
from sklearn.metrics import (

# ========================================================================
# 메인
# ========================================================================
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss
)


class ClassificationMetrics:
    """
    분류 지표를 두루 계산하는 도구
    """
    
    def __init__(self, y_true, y_pred, y_pred_proba=None):
        """
        참 레이블과 예측으로 초기화
        
        인수:
            y_true: 참 레이블
            y_pred: 예측 레이블
            y_pred_proba: 예측 확률 (선택 사항, 일부 지표에 필요)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_pred_proba = np.array(y_pred_proba) if y_pred_proba is not None else None
    
    def accuracy(self):
        """
        정확도 계산: (TP + TN) / (TP + TN + FP + FN)
        
        가장 알맞은 곳: 균형 잡힌 데이터셋
        한계: 불균형한 데이터셋에서는 오해를 부른다
        """
        acc = accuracy_score(self.y_true, self.y_pred)
        return acc
    
    def precision(self, average='binary'):
        """
        정밀도 계산: TP / (TP + FP)
        
        해석: 양성이라 예측한 것 가운데 얼마나 맞았는가?
        쓸 때: 거짓 양성의 비용이 클 때
        
        인수:
            average: 다중 클래스에는 'binary', 'micro', 'macro', 'weighted'
        """
        return precision_score(self.y_true, self.y_pred, average=average, zero_division=0)
    
    def recall(self, average='binary'):
        """
        재현율(민감도) 계산: TP / (TP + FN)
        
        해석: 실제 양성 가운데 얼마나 잡아냈는가?
        쓸 때: 거짓 음성의 비용이 클 때 (예: 질병 진단)
        
        인수:
            average: 다중 클래스에는 'binary', 'micro', 'macro', 'weighted'
        """
        return recall_score(self.y_true, self.y_pred, average=average, zero_division=0)
    
    def f1(self, average='binary'):
        """
        F1 점수 계산: 2 * (정밀도 * 재현율) / (정밀도 + 재현율)
        
        해석: 정밀도와 재현율의 조화평균
        쓸 때: 정밀도와 재현율의 균형이 필요할 때
        
        인수:
            average: 다중 클래스에는 'binary', 'micro', 'macro', 'weighted'
        """
        return f1_score(self.y_true, self.y_pred, average=average, zero_division=0)
    
    def confusion_matrix_detailed(self):
        """
        자세한 해석과 함께 혼동 행렬 생성
        
        반환값:
            혼동 행렬과 파생 지표를 담은 사전
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            return {
                'confusion_matrix': cm,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
            }
        else:
            return {
                'confusion_matrix': cm,
                'note': 'Multi-class confusion matrix'
            }
    
    def roc_auc(self, average='macro'):
        """
        ROC-AUC 점수 계산
        
        해석: 무작위로 고른 양성 예를 무작위로 고른 음성 예보다
                       높게 매길 확률
        범위: 0.5(무작위)부터 1.0(완벽)까지
        쓸 때: 모델이 클래스를 가르는 능력을 평가할 때
        
        필요 조건: y_pred_proba가 주어져야 한다
        """
        if self.y_pred_proba is None:
            return "ROC-AUC requires predicted probabilities"
        
        try:
            # 이진 분류용
            if len(np.unique(self.y_true)) == 2:
                return roc_auc_score(self.y_true, self.y_pred_proba)
            # 다중 클래스용
            else:
                return roc_auc_score(self.y_true, self.y_pred_proba, 
                                   average=average, multi_class='ovr')
        except Exception as e:
            return f"Error calculating ROC-AUC: {str(e)}"
    
    def average_precision(self):
        """
        평균 정밀도 계산 (정밀도-재현율 곡선 아래 넓이)
        
        ROC-AUC보다 나은 경우: 불균형한 데이터셋
        해석: 정밀도-재현율 곡선의 요약
        
        필요 조건: y_pred_proba가 주어져야 한다
        """
        if self.y_pred_proba is None:
            return "Average Precision requires predicted probabilities"
        
        return average_precision_score(self.y_true, self.y_pred_proba)
    
    def matthews_correlation_coefficient(self):
        """
        매슈스 상관계수(MCC) 계산
        
        범위: -1(완전히 어긋남)부터 +1(완벽한 예측)까지
        장점: 불균형한 데이터셋에서도 잘 통한다
        해석: 관측값과 예측값의 상관
        """
        return matthews_corrcoef(self.y_true, self.y_pred)
    
    def cohen_kappa(self):
        """
        코헨의 카파 계산
        
        범위: -1부터 1까지 (1이 완전한 일치)
        해석: 우연을 고려한, 예측과 참값의
                       일치도
        """
        return cohen_kappa_score(self.y_true, self.y_pred)
    
    def log_loss_score(self):
        """
        로그 로스(교차 엔트로피 로스) 계산
        
        범위: 0(완벽)부터 무한대까지
        쓸 때: 레이블만이 아니라 예측 확률을 평가할 때
        
        필요 조건: y_pred_proba가 주어져야 한다
        """
        if self.y_pred_proba is None:
            return "Log Loss requires predicted probabilities"
        
        return log_loss(self.y_true, self.y_pred_proba)
    
    def classification_report_detailed(self):
        """
        종합적인 분류 보고서 생성
        """
        return classification_report(self.y_true, self.y_pred)
    
    def full_evaluation_report(self):
        """
        모든 지표를 담은 완전한 평가 보고서 생성
        """
        report = {
            'Accuracy': self.accuracy(),
            'Precision': self.precision(),
            'Recall': self.recall(),
            'F1 Score': self.f1(),
            'MCC': self.matthews_correlation_coefficient(),
            'Cohen Kappa': self.cohen_kappa(),
        }
        
        if self.y_pred_proba is not None:
            report['ROC-AUC'] = self.roc_auc()
            report['Average Precision'] = self.average_precision()
            report['Log Loss'] = self.log_loss_score()
        
        cm_details = self.confusion_matrix_detailed()
        report['Confusion Matrix Details'] = cm_details
        
        return report


def metric_selection_guide():
    """
    알맞은 지표를 고르는 안내
    """
    guide = """
    지표 선택 안내
    ======================
    
    균형 잡힌 데이터셋:
        → 정확도, F1 점수
    
    불균형한 데이터셋:
        → 정밀도-재현율 AUC, F1 점수, MCC
        → 정확도만 보아서는 안 된다!
    
    거짓 양성의 비용이 클 때 (예: 스팸 탐지):
        → 정밀도
    
    거짓 음성의 비용이 클 때 (예: 질병 진단):
        → 재현율 (민감도)
    
    균형이 필요할 때:
        → F1 점수
    
    모델을 견줄 때:
        → ROC-AUC, 교차 검증 점수
    
    확률의 보정이 중요할 때:
        → 로그 로스, 브라이어 점수
    
    다중 클래스 문제:
        → 매크로 평균 지표 (모든 클래스를 똑같이 다룬다)
        → 가중 평균 지표 (클래스의 빈도로 가중한다)
    """
    print(guide)


# 사용 예
if __name__ == "__main__":
    print("=" * 60)
    print("CLASSIFICATION METRICS DEMONSTRATION")
    print("=" * 60)
    
    # 예제 1: 이진 분류
    print("\n1. BINARY CLASSIFICATION EXAMPLE")
    print("-" * 40)
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.7, 0.3, 0.6, 0.85, 0.15])
    
    metrics = ClassificationMetrics(y_true, y_pred, y_pred_proba)
    report = metrics.full_evaluation_report()
    
    for metric_name, value in report.items():
        if metric_name != 'Confusion Matrix Details':
            print(f"{metric_name}: {value}")
    
    print("\nConfusion Matrix Details:")
    for key, value in report['Confusion Matrix Details'].items():
        if key != 'confusion_matrix':
            print(f"  {key}: {value}")
    
    # 지표 선택 안내
    print("\n2. METRIC SELECTION GUIDE")
    print("-" * 40)
    metric_selection_guide()
```

**출력:**

```
============================================================
CLASSIFICATION METRICS DEMONSTRATION
============================================================

1. BINARY CLASSIFICATION EXAMPLE
----------------------------------------
Accuracy: 0.8
Precision: 0.8
Recall: 0.8
F1 Score: 0.8
MCC: 0.6
Cohen Kappa: 0.6
ROC-AUC: 0.9600000000000001
Average Precision: 0.9666666666666666
Log Loss: 0.3527977344565397

Confusion Matrix Details:
  true_negatives: 4
  false_positives: 1
  false_negatives: 1
  true_positives: 4
  specificity: 0.8
  sensitivity: 0.8
  false_positive_rate: 0.2
  false_negative_rate: 0.2

2. METRIC SELECTION GUIDE
----------------------------------------

    지표 선택 안내
    ======================
    
    균형 잡힌 데이터셋:
        → 정확도, F1 점수
    
    불균형한 데이터셋:
        → 정밀도-재현율 AUC, F1 점수, MCC
        → 정확도만 보아서는 안 된다!
    
    거짓 양성의 비용이 클 때 (예: 스팸 탐지):
        → 정밀도
    
    거짓 음성의 비용이 클 때 (예: 질병 진단):
        → 재현율 (민감도)
    
    균형이 필요할 때:
        → F1 점수
    
    모델을 견줄 때:
        → ROC-AUC, 교차 검증 점수
    
    확률의 보정이 중요할 때:
        → 로그 로스, 브라이어 점수
    
    다중 클래스 문제:
        → 매크로 평균 지표 (모든 클래스를 똑같이 다룬다)
        → 가중 평균 지표 (클래스의 빈도로 가중한다)
    
```

## 2. 논의

정확도·정밀도·재현율·F1·MCC는 예측 클래스를 사용한다. ROC-AUC와 평균 정밀도는 점수의 순위에 기반하며, 특정 임계값 하나를 고르지 않고 여러 수준을 요약한다. 점수의 순위 품질과 확률 보정은 별개이므로 필요하면 로그로스 등도 확인한다.

불균형 데이터에서 높은 정확도는 다수 클래스만 예측해도 얻을 수 있다. 정밀도와 PR 지표는 양성 비율에 영향을 받고, F1은 참음성의 성능이나 모든 비용을 직접 반영하지 않는다. 어떤 지표도 모든 과제의 기본 정답은 아니다.

다중 클래스에서는 macro, micro, weighted 집계와 클래스별 표본 수를 명시한다. 양성 클래스와 확률 열의 순서도 맞춰야 한다. 한 클래스만 있는 평가 분할에서는 ROC-AUC 같은 지표가 정의되지 않을 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
정확도가 높아도 양성을 놓치는 예를 설명하라.

</div>

??? success "연습문제 1 풀이"
    양성이 1%인데 항상 음성으로 예측하면 정확도는 99%이고 양성 재현율은 0이다. 목표 비용과 클래스별 성능을 함께 평가한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
ROC-AUC가 높은 모델의 확률이 잘 보정되었다고 할 수 있는가?

</div>

??? success "연습문제 2 풀이"
    아니다. 순위를 유지하는 점수 변환은 ROC-AUC를 유지할 수 있지만 확률값과 로그로스·보정은 바뀐다. 순위, 임계값의 운영 성능과 확률 품질을 따로 확인한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
macro F1과 weighted F1의 차이를 설명하라.

</div>

??? success "연습문제 3 풀이"
    macro는 클래스별 F1을 같은 비중으로 평균내고 weighted는 클래스 표본 수로 가중한다. 소수 클래스의 낮은 성능이 가려질 수 있으므로 클래스별 지표도 함께 보고한다.

## 정리하며

분류 지표는 예측 클래스, 순위와 확률 품질을 구분해 선택한다. 양성 클래스와 집계 방식을 명시하고 목표 분포·비용에서 해석한다.
