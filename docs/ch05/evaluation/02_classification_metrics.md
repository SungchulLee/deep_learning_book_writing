# 분류 지표

분류 지표는 모델 성능의 서로 다른 면을 수치로 나타낸다. 불균형한 데이터셋에서는 정확도만으로는 오해를 살 수 있으므로, 실무자는 정밀도, 재현율, F1 점수, ROC-AUC, 매슈스 상관계수를 함께 보아 전체 그림을 얻는다. 어떤 지표가 알맞은지는 거짓 양성과 거짓 음성의 상대적인 비용에 달렸다.

## 코드

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
        로그 손실(교차 엔트로피 손실) 계산
        
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
        → 로그 손실, 브라이어 점수
    
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
    metric_selection_guide()```

## 논의

전체 평가 보고서는 문턱값에 따라 달라지는 지표(정확도, 정밀도, 재현율, F1), 문턱값과 무관한 지표(ROC-AUC, 평균 정밀도), 상관에 기반한 지표(MCC, 코헨의 카파)를 함께 담는다. 각각이 모델의 품질을 다른 각도에서 보여 준다.

불균형한 데이터셋에서 정확도는 오해를 부른다. 언제나 다수 클래스를 내놓는 모델도 저절로 높은 정확도를 얻기 때문이다. 정밀도-재현율 곡선과 MCC는 클래스 분포를 고려하므로 더 많은 것을 알려 준다. F1 점수는 정밀도와 재현율의 조화평균으로 둘의 균형을 잡는다.

지표 선택 안내는 흔한 상황을 알맞은 지표에 이어 준다. 거짓 양성에 민감한 응용(스팸 탐지)은 정밀도를, 거짓 음성에 민감한 응용(질병 진단)은 재현율을 앞세우고, 범용 응용에는 F1이나 MCC를 쓴다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

