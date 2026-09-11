"""evaluation_metrics_07_evaluation_metrics — ch20/sequence_labeling/07_evaluation_metrics.md 의 코드를
모듈로 쓸 수 있게 옮겨 놓은 것이다.
"""

"""
이름 알아보기 값매김 잣대
======================

이름 알아보기 체계를 두루 살피는 값매김 잣대.

잣대:
- 정밀도, 재현율, F1(토막 수준과 것 수준)
- 빡빡한 짝짓기와 느슨한 짝짓기
- 것 갈래별 잣대

지은이: 배움 목적
날짜: 2025
"""

from typing import List, Dict, Tuple
from collections import defaultdict

# ========================================================================
# 메인
# ========================================================================


class NERMetrics:
    """이름 알아보기의 값매김 잣대."""
    
    @staticmethod
    def compute_metrics(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
        """
        토막 수준의 정밀도, 재현율, F1 셈하기.
        
        인수:
            y_true: 참 레이블
            y_pred: 예측 레이블
            
        반환값:
            정밀도, 재현율, F1 점수를 담은 사전
        """
        tp = 0  # 참양성
        fp = 0  # 헛양성
        fn = 0  # 헛음성
        
        for true_seq, pred_seq in zip(y_true, y_pred):
            for true_label, pred_label in zip(true_seq, pred_seq):
                if true_label != "O":
                    if pred_label == true_label:
                        tp += 1
                    else:
                        fn += 1
                        if pred_label != "O":
                            fp += 1
                elif pred_label != "O":
                    fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    @staticmethod
    def entity_level_f1(true_entities: List[Tuple], pred_entities: List[Tuple]) -> Dict:
        """
        것 수준 F1 점수 셈하기.
        
        인수:
            true_entities: (text, type, start, end) 튜플의 목록
            pred_entities: (text, type, start, end) 튜플의 목록
            
        반환값:
            것 수준 잣대를 담은 사전
        """
        true_set = set((e[1], e[2], e[3]) for e in true_entities)  # (type, start, end)
        pred_set = set((e[1], e[2], e[3]) for e in pred_entities)
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }


if __name__ == "__main__":
    # 예
    y_true = [["B-PER", "I-PER", "O", "B-ORG"]]
    y_pred = [["B-PER", "I-PER", "O", "B-ORG"]]
    
    metrics = NERMetrics.compute_metrics(y_true, y_pred)
    print(f"F1 Score: {metrics['f1']:.3f}")
