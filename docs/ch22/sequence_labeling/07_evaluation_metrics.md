# NER 평가 예제: 토큰 태그와 개체 span

이 예제에는 토큰 태그 기준 지표와 개체 구간 기준 지표가 따로 있다. 둘은 같은 정밀도·재현율·F1 이름을 쓰더라도 세는 대상이 다르다. 뉴럴 넷트웍 학습·텍스트 생성은 포함하지 않으며 기존 시연은 같은 정답·예측 태그를 비교한다.

## 1. 코드와 기존 출력

```python
"""
이름 알아보기 밸류 잣대
======================

이름 알아보기 체계를 두루 살피는 밸류 잣대.

잣대:
- 정밀도, 재현율, F1(토막 수준과 것 수준)
- 빡빡한 짝짓기와 느슨한 짝짓기
- 것 클래스별 잣대

지은이: 러닝 목적
날짜: 2025
"""

from typing import List, Dict, Tuple
from collections import defaultdict

# ========================================================================
# 메인
# ========================================================================


class NERMetrics:
    """이름 알아보기의 밸류 잣대."""
    
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
```

**기존 출력:**

```
F1 Score: 1.000
```

## 2. compute_metrics가 세는 대상

이 함수는 각 토큰의 O를 음성, B/I와 유형을 포함한 나머지 문자열을 양성 태그로 취급한다. 정답이 비O이고 예측 문자열까지 같아야TP다. 비O 태그끼리 달라지면FN과FP를 각각 더하며 O→비O는FP, 비O→O는FN이다. O→O는 세지 않는다.

따라서 **토큰 태그의 micro 지표**이며 개체 경계·전체 유형이 맞는 span F1이 아니다. B-PER과I-PER을 같은PER로 합치는 지표도 아니다. 여러 토큰짜리 개체는 여러 번 기여하므로 긴 이름이 더 큰 비중을 받을 수 있다. 정답과 예측이 모두O인 경우 함수는0을 반환하며 전체 정확도100%와 다른 정의다.

바깥 문장 목록과 안쪽 태그 목록을 모두zip하므로 길이가 다르면 남은 문장·토큰을 조용히 버린다. 누락을 자동으로FN으로 세지 않으며 BIO 유효성 검사도 없다. 지표 호출 전에 표본·토큰 대응을 검증해야 한다.

## 3. entity_level_f1의 구간 계약

입력 원소는 `(text,type,start,end)` 형태를 가정하고 text를 제외한(type,start,end)을 집합으로 비교한다. span의 경계와 유형이 일치해야TP다. start·end가 토큰인지 문자인지 함수가 정해 주지 않으므로 호출하는 두 자료에서 같은 기준을 사용해야 한다.

문서·문장ID가 없어 여러 문장의 로컬offset을 한 목록에 합치면 서로 다른 언급이 같은집합원소로 합쳐질 수 있다. 표본별로 집계하거나 ID를 포함한 대응을 정의해야 한다. 집합은 중복도 제거하므로 중복 예측을 별도FP로 세는 다른 계약과는 다르다. 원문 문자열은 비교하지 않지만 같은위치가 같은표본을 가리켜야 의미가 있다.

예측·정답이 모두비면P·R·F1을0으로 두는 규칙이다. 부분겹침·유형만일치·macro·support가중F1은 구현하지 않는다. 더 자세한 집계는 [NER 평가 절](ner_evaluation.md)을 참고한다.

## 4. 결과를 보고하는 방법

시연의 완전 일치 F1은 함수 호출 형식을 보여 줄 뿐 실제 모델 성능이 아니다. 평가 단위·BIO 체계·무효 태그 폴리시·패딩·빈 경우·문서 구분과 micro/macro를 기록한다. 같은 문서의 문장이나 중복 자료가 분할 사이에 섞이는지도 확인한다. 토큰 지표와 span 지표를 같은 성능표의 동일 척도로 바꾸어 인용하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
정답 B-LOC I-LOC I-LOC를 B-LOC I-LOC O로 예측하면 토큰과 정확 span 지표는 어떻게 다른가?

</div>

??? success "연습문제 1 풀이"

    토큰 기준으로 두 양성 태그가 맞고 하나를 놓쳐TP2·FN1이다. 정확 span 기준에서는 짧은 예측 하나가FP이고 원래 개체 하나가FN이라TP0이다. 부분 토큰 정답을 개체의 부분 점수로 해석하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
문장·태그 길이와 개체 tuple에서 어떤 계약이 필요한가?

</div>

??? success "연습문제 2 풀이"

    두 목록의 표본·토큰 수와 순서가 같아야 한다. tuple은 text·type·start·end를 가정하며 양쪽 offset의 단위·문서가 일치해야 한다. float32 뉴럴 넷트웍 입력 검사는 이 문자열·구간 계약을 대신하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
여러 문장의 로컬 offset을 entity_level_f1에 한꺼번에 넣으면 어떤 문제가 생길 수 있는가?

</div>

??? success "연습문제 3 풀이"

    문장 ID가 없어 같은 유형·같은 숫자 범위의 서로 다른 언급이 집합에서 합쳐질 수 있다. 표본별 TP·FP·FN을 누적하거나 ID를 포함해 구분해야 한다. 중복 예측을 제거할지 별도 오탐으로 셀지도 명시한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 NERMetrics 시험 조각이 적절한가? 알맞은 경계 사례를 설명하라.

</div>

??? success "연습문제 4 풀이"

    NERMetrics는 정적 계산 함수이고 PyTorch forward·parameters·역전파를 제공하지 않는다. 미정의 입력을 쓰는 아래 조각은 완성 시험이 아니다. 완전 일치·경계/유형 오류·모두 O·빈 집합·길이 불일치·다른 문장의 동일 offset과 중복 예측을 명세로 확인한다.

    ```python
    def test_nermetrics():
        model = NERMetrics(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```


## 정리하며

토큰 태그 micro와 정확 개체 span F1은 세는 단위가 다르다. 입력 길이·offset·문서 ID·중복·빈 경우의 폴리시를 맞추고 현재 함수가 구현하지 않는 부분 일치·macro 평가를 구분한다.
