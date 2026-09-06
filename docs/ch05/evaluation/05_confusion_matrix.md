# 혼동 행렬

혼동 행렬은 참 양성, 참 음성, 거짓 양성, 거짓 음성의 개수를 보여 분류 예측을 자세히 쪼개 준다. 행이나 열로 정규화하면 클래스별 재현율과 정밀도가 드러나므로, 모델이 어떤 클래스를 헷갈리는지 진단하는 데 더없이 쓸모 있다.

## 코드

```python
"""
혼동 행렬과 시각화
===================================

혼동 행렬과 그 시각화를 두루 다룬다.

다루는 주제:
- 이진 분류의 혼동 행렬
- 다중 클래스의 혼동 행렬
- 정규화한 혼동 행렬
- 혼동 행렬에서 지표 이끌어 내기
- 시각화 기법
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ========================================================================
# 메인
# ========================================================================


class ConfusionMatrixAnalyzer:
    """
    혼동 행렬의 종합 분석과 시각화
    """
    
    def __init__(self, y_true, y_pred, labels=None, class_names=None):
        """
        참 레이블과 예측으로 초기화
        
        인수:
            y_true: 참 레이블
            y_pred: 예측 레이블
            labels: 포함할 레이블 값의 목록 (선택 사항)
            class_names: 보여 줄 이름 (선택 사항)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.labels = labels
        self.class_names = class_names
        self.cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    def get_basic_metrics_binary(self):
        """
        이진 혼동 행렬에서 기본 지표 계산
        
        이진 분류에만 쓴다 (2x2 행렬)
        
        반환값:
            TP, TN, FP, FN과 파생 지표를 담은 사전
        """
        if self.cm.shape != (2, 2):
            return "This method is for binary classification only"
        
        tn, fp, fn, tp = self.cm.ravel()
        
        # 파생 지표 계산
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 오류율
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 거짓 양성률
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # 거짓 음성률
        
        # 예측도
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # 양성 예측도 (정밀도)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 음성 예측도
        
        return {
            'True Positives (TP)': tp,
            'True Negatives (TN)': tn,
            'False Positives (FP)': fp,
            'False Negatives (FN)': fn,
            'Accuracy': accuracy,
            'Precision (PPV)': precision,
            'Recall (Sensitivity/TPR)': recall,
            'Specificity (TNR)': specificity,
            'F1 Score': f1,
            'False Positive Rate (FPR)': fpr,
            'False Negative Rate (FNR)': fnr,
            'Negative Predictive Value (NPV)': npv
        }
    
    def get_normalized_cm(self, normalize='true'):
        """
        정규화한 혼동 행렬 얻기
        
        인수:
            normalize: 'true', 'pred', 또는 'all'
                - 'true': 참 레이블로 정규화 (행 방향) - 재현율을 보인다
                - 'pred': 예측으로 정규화 (열 방향) - 정밀도를 보인다
                - 'all': 전체 표본으로 정규화
        
        반환값:
            정규화한 혼동 행렬
        """
        if normalize == 'true':
            # 행으로 정규화 (참 레이블 기준)
            cm_norm = self.cm.astype('float') / self.cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            # 열로 정규화 (예측 기준)
            cm_norm = self.cm.astype('float') / self.cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            # 전체로 정규화
            cm_norm = self.cm.astype('float') / self.cm.sum()
        else:
            raise ValueError("normalize must be 'true', 'pred', or 'all'")
        
        # NaN을 0으로 바꾸기 (0으로 나눈 경우에 대비)
        cm_norm = np.nan_to_num(cm_norm)
        
        return cm_norm
    
    def plot_confusion_matrix(self, normalize=None, figsize=(8, 6), 
                            cmap='Blues', save_path=None):
        """
        matplotlib으로 혼동 행렬 그리기
        
        인수:
            normalize: None, 'true', 'pred', 또는 'all'
            figsize: 도형의 크기 튜플
            cmap: 색지도의 이름
            save_path: 도형을 저장할 경로 (선택 사항)
        """
        if normalize:
            cm_to_plot = self.get_normalized_cm(normalize)
            title = f'Confusion Matrix (Normalized by {normalize})'
            fmt = '.2f'
        else:
            cm_to_plot = self.cm
            title = 'Confusion Matrix (Counts)'
            fmt = 'd'
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm_to_plot, annot=True, fmt=fmt, cmap=cmap,
                   xticklabels=self.class_names or 'auto',
                   yticklabels=self.class_names or 'auto',
                   cbar=True)
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return plt.gcf()
    
    def plot_multiple_normalizations(self, figsize=(15, 5), save_path=None):
        """
        정규화 방식을 달리한 혼동 행렬을 나란히 그리기
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        normalizations = [None, 'true', 'pred']
        titles = ['Counts', 'Normalized by True Label', 'Normalized by Prediction']
        
        for ax, norm, title in zip(axes, normalizations, titles):
            if norm:
                cm_to_plot = self.get_normalized_cm(norm)
                fmt = '.2f'
            else:
                cm_to_plot = self.cm
                fmt = 'd'
            
            sns.heatmap(cm_to_plot, annot=True, fmt=fmt, cmap='Blues',
                       xticklabels=self.class_names or 'auto',
                       yticklabels=self.class_names or 'auto',
                       ax=ax, cbar=True)
            
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multiple confusion matrices saved to {save_path}")
        
        return fig
    
    def analyze_multiclass_performance(self):
        """
        다중 클래스 분류의 클래스별 성능 분석
        
        반환값:
            클래스별 지표를 담은 사전
        """
        n_classes = self.cm.shape[0]
        
        per_class_metrics = {}
        
        for i in range(n_classes):
            class_name = self.class_names[i] if self.class_names else f"Class {i}"
            
            # 이 클래스의 참 양성
            tp = self.cm[i, i]
            
            # 거짓 양성 (이 클래스로 예측했지만 실제로는 아님)
            fp = self.cm[:, i].sum() - tp
            
            # 거짓 음성 (실제로는 이 클래스인데 예측하지 못함)
            fn = self.cm[i, :].sum() - tp
            
            # 참 음성 (이 클래스도 아니고 이 클래스로 예측하지도 않음)
            tn = self.cm.sum() - (tp + fp + fn)
            
            # 지표를 계산한다
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[class_name] = {
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Support': self.cm[i, :].sum()
            }
        
        return per_class_metrics
    
    def print_analysis(self):
        """
        혼동 행렬의 종합 분석 출력
        """
        print("=" * 60)
        print("CONFUSION MATRIX ANALYSIS")
        print("=" * 60)
        
        print("\nConfusion Matrix (Counts):")
        print(self.cm)
        
        if self.cm.shape == (2, 2):
            print("\n" + "-" * 60)
            print("BINARY CLASSIFICATION METRICS")
            print("-" * 60)
            
            metrics = self.get_basic_metrics_binary()
            
            print("\nBasic Counts:")
            for key in ['True Positives (TP)', 'True Negatives (TN)', 
                       'False Positives (FP)', 'False Negatives (FN)']:
                print(f"  {key}: {metrics[key]}")
            
            print("\nPerformance Metrics:")
            for key in ['Accuracy', 'Precision (PPV)', 'Recall (Sensitivity/TPR)', 
                       'Specificity (TNR)', 'F1 Score']:
                print(f"  {key}: {metrics[key]:.4f}")
            
            print("\nError Rates:")
            for key in ['False Positive Rate (FPR)', 'False Negative Rate (FNR)']:
                print(f"  {key}: {metrics[key]:.4f}")
            
            print("\nPredictive Values:")
            for key in ['Precision (PPV)', 'Negative Predictive Value (NPV)']:
                if 'Precision' in key:
                    print(f"  Positive {key}: {metrics[key]:.4f}")
                else:
                    print(f"  {key}: {metrics[key]:.4f}")
        
        else:
            print("\n" + "-" * 60)
            print("MULTI-CLASS CLASSIFICATION METRICS")
            print("-" * 60)
            
            per_class = self.analyze_multiclass_performance()
            
            print("\nPer-Class Performance:")
            for class_name, metrics in per_class.items():
                print(f"\n{class_name}:")
                for metric_name, value in metrics.items():
                    if metric_name != 'Support':
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {value}")


def confusion_matrix_interpretation_guide():
    """
    혼동 행렬을 해석하는 안내
    """
    guide = """
    혼동 행렬 해석 안내
    =====================================
    
    이진 분류 (2x2 행렬):
    
                    예측
                    음성    양성
    실제  음성     TN     FP
            양성     FN     TP
    
    핵심 용어:
    ----------
    TP (참 양성): 양성을 옳게 예측
    TN (참 음성): 음성을 옳게 예측
    FP (거짓 양성): 양성으로 잘못 예측 (제1종 오류)
    FN (거짓 음성): 음성으로 잘못 예측 (제2종 오류)
    
    파생 지표:
    ---------------
    정확도 = (TP + TN) / 전체
        → 전체적으로 얼마나 맞았는가
    
    정밀도 = TP / (TP + FP)
        → 양성이라 예측한 것 가운데 얼마나 맞았는가?
        → 정밀도가 높으면 헛경보가 적다
    
    재현율 = TP / (TP + FN)
        → 실제 양성 가운데 얼마나 잡아냈는가?
        → 재현율이 높으면 놓치는 경우가 적다
    
    특이도 = TN / (TN + FP)
        → 실제 음성 가운데 얼마나 옳게 가려냈는가?
    
    F1 점수 = 2 * (정밀도 × 재현율) / (정밀도 + 재현율)
        → 정밀도와 재현율의 조화평균
    
    정규화:
    =============
    
    참 레이블로 정규화 (행):
        → 클래스별 재현율을 보인다
        → "실제 X 가운데 몇 %를 Y로 예측했는가?"
    
    예측으로 정규화 (열):
        → 클래스별 정밀도를 보인다
        → "X로 예측한 것 가운데 몇 %가 실제로 Y였는가?"
    
    다중 클래스:
    ===========
    - 대각선 = 옳은 예측
    - 대각선 밖 = 클래스 사이의 혼동
    - 양상을 살피라: 어떤 클래스끼리 헷갈리는가?
    
    실용적인 조언:
    ==============
    1. 개수와 정규화한 것을 언제나 함께 보라
    2. 불균형한 데이터에서는 날 개수가 오해를 부를 수 있다
    3. 어떤 종류의 오류가 가장 흔한지 확인하라
    4. 오류 종류마다의 비용을 고려하라
    5. 대각선이 밝아야 한다 (옳은 예측이 많다는 뜻)
    """
    print(guide)


# 사용 예
if __name__ == "__main__":
    print("=" * 60)
    print("CONFUSION MATRIX DEMONSTRATION")
    print("=" * 60)
    
    # 예제 1: 이진 분류
    print("\n1. BINARY CLASSIFICATION EXAMPLE")
    print("-" * 60)
    
    y_true_binary = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
    y_pred_binary = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0])
    
    cm_binary = ConfusionMatrixAnalyzer(
        y_true_binary, y_pred_binary,
        class_names=['Negative', 'Positive']
    )
    cm_binary.print_analysis()
    
    # 예제 2: 다중 클래스 분류
    print("\n\n2. MULTI-CLASS CLASSIFICATION EXAMPLE")
    print("-" * 60)
    
    y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred_multi = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2, 1, 1, 2])
    
    cm_multi = ConfusionMatrixAnalyzer(
        y_true_multi, y_pred_multi,
        class_names=['Class A', 'Class B', 'Class C']
    )
    cm_multi.print_analysis()
    
    # 해석 안내
    print("\n\n3. INTERPRETATION GUIDE")
    print("-" * 60)
    confusion_matrix_interpretation_guide()
    
    print("\n" + "=" * 60)
    print("Note: Run with matplotlib backend to see visualizations")
    print("=" * 60)```

## 논의

이진 혼동 행렬에는 네 항목이 있다. 참 양성(TP), 참 음성(TN), 거짓 양성(FP, 제1종 오류), 거짓 음성(FN, 제2종 오류)이다. 표준적인 분류 지표는 모두 이 네 값에서 나온다. 정확도 = (TP+TN)/전체, 정밀도 = TP/(TP+FP), 재현율 = TP/(TP+FN), 특이도 = TN/(TN+FP)이다.

정규화는 서로 보완하는 관점을 준다. 행 정규화(참 레이블 기준)는 클래스별 재현율, 곧 실제 양성 가운데 얼마를 찾아냈는지를 보여 준다. 열 정규화(예측 기준)는 정밀도, 곧 양성이라 예측한 것 가운데 얼마가 맞았는지를 보여 준다. 온전히 이해하려면 둘 다 필요하다.

다중 클래스 문제에서 혼동 행렬은 모델이 어떤 클래스끼리 헷갈리는지 드러낸다. 대각선 밖의 항목은 조직적인 오분류를 뜻하며, 클래스별 데이터 증강, 특화된 특징, 계층적 분류 전략으로 다룰 수 있다.

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

