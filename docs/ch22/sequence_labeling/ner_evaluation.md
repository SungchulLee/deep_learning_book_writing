# 개체명 인식 평가: 정확 span과 집계 기준

NER의 정확 일치 평가는 예측과 정답의 경계·유형이 모두 맞는 언급을 센다. 문자열이 같아도 다른 문장·위치의 언급은 별개다. 토큰 정확도·부분 겹침·정확 span F1은 서로 다른 지표이므로 평가 단위·매칭·빈 경우와 집계를 명시해야 한다.

## 1. TP·FP·FN과 정확 일치

같은 표본 안에서 개체를 (유형,start,end)의 반열린 토큰 구간으로 표현할 수 있다. 정확한 예측은TP, 맞지 않는 예측은FP, 누락된 정답은FN이다. New York City를 New York만 예측하면 정확 일치에서는FP하나·FN하나이며 부분 점수를 주지 않는다.

$$P=\frac{TP}{TP+FP},\quad R=\frac{TP}{TP+FN},\quad F_1=\frac{2TP}{2TP+FP+FN}.$$

분모0의 처리는 명시적인 약속이 필요하다. 예측·정답이 모두 비었을 때0·1·미정 중 어떤 값을 사용할지는 지표·라이브러리 설정에 따른다. 이런 표본을 문장별 평균하는 것과 전체 개체를 누적하는 micro 지표는 다르다.

## 2. 추출·집계 코드의 범위

```python
from typing import List, Dict, Set
from collections import defaultdict
from dataclasses import dataclass

@dataclass(frozen=True)
class Entity:
    entity_type: str
    start: int
    end: int

def extract_entities(tags: List[str]) -> Set[Entity]:
    """IOB2 이름표에서 것 뽑기."""
    entities = set()
    current = None
    
    for i, tag in enumerate(tags):
        if tag == 'O':
            if current:
                entities.add(Entity(current[0], current[1], i))
                current = None
        elif tag.startswith('B-'):
            if current:
                entities.add(Entity(current[0], current[1], i))
            current = (tag[2:], i)
        elif tag.startswith('I-') and current and current[0] == tag[2:]:
            pass  # 것 이어 가기
        else:
            if current:
                entities.add(Entity(current[0], current[1], i))
            current = None
    
    if current:
        entities.add(Entity(current[0], current[1], len(tags)))
    return entities

def compute_ner_metrics(
    pred_tags: List[List[str]],
    gold_tags: List[List[str]],
    average: str = 'micro'
) -> Dict[str, float]:
    """이름 알아보기 F1 잣대 셈하기."""
    tp_per_type = defaultdict(int)
    fp_per_type = defaultdict(int)
    fn_per_type = defaultdict(int)
    
    for pred, gold in zip(pred_tags, gold_tags):
        pred_ents = extract_entities(pred)
        gold_ents = extract_entities(gold)
        
        for e in pred_ents:
            if e in gold_ents:
                tp_per_type[e.entity_type] += 1
            else:
                fp_per_type[e.entity_type] += 1
        
        for e in gold_ents:
            if e not in pred_ents:
                fn_per_type[e.entity_type] += 1
    
    if average == 'micro':
        tp = sum(tp_per_type.values())
        fp = sum(fp_per_type.values())
        fn = sum(fn_per_type.values())
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        return {'precision': p, 'recall': r, 'f1': f1}
    
    # 거시 평균
    all_types = set(tp_per_type) | set(fp_per_type) | set(fn_per_type)
    f1s = []
    for t in all_types:
        tp, fp, fn = tp_per_type[t], fp_per_type[t], fn_per_type[t]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
    
    return {'f1': sum(f1s) / len(f1s) if f1s else 0}
```

extract_entities는 IOB2 형태의 평면 span을 집합으로 반환한다. 토큰 offset이고 원문 문자 offset이 아니다. 잘못 시작한 I태그·유형이 바뀐I태그는 엄격한 오류 대신 열린 개체를 닫고 무시하는 폴리시를 사용한다. BIOES를 직접 처리하는 함수도 아니다. 입력을 검증할지, 오류 교정할지와 평가 도구의 관례를 맞춰야 한다.

compute_ner_metrics는 표본별 집합끼리 비교해 같은 위치 번호라도 다른 문장의 언급을 섞지 않는다. 다만 바깥 목록을 zip하므로 예측·정답 표본 수가 다르면 남은 표본을 조용히 버린다. 각 표본의 토큰 길이가 같은지도 검사하지 않는다. 평가 전에 두 축의 대응을 확인해야 한다. 집합은 같은 span의 중복을 제거하므로 중복 탐지 출력을 허용하는 다른 API의 집계와도 다를 수 있다.

micro는 모든 유형의TP·FP·FN을 합친 뒤P·R·F1을 계산하고 분모0이면0을 반환한다. micro F1은 유형별 F1의 단순 평균이나 정답 support 가중 평균이 아니다.

그 밖의 average 값은 모두 관측된 유형들의 F1 단순 평균으로 처리한다. 따라서 `weighted`를 넣어도 **weighted F1은 구현되지 않는다**. 알려지지 않은 문자열도 오류 없이 이 경로로 간다. macro 유형 집합은 예측 또는 정답에 나타난 유형이며 전체 등록 유형을 고정해 평균하는 폴리시와 다르다. macro 경로는 F1만 반환해 P·R도 제공하는 micro 반환값과 키가 다르다.

## 3. seqeval과 태그 체계

```python
from seqeval.metrics import classification_report, f1_score

# 표준 밸류
y_true = [['B-PER', 'I-PER', 'O', 'B-LOC']]
y_pred = [['B-PER', 'I-PER', 'O', 'B-LOC']]

print(classification_report(y_true, y_pred))
print(f"F1: {f1_score(y_true, y_pred):.4f}")
```

예제는 같은 태그 열을 비교하는 간단한 API 시연이다. 유효한 두 개체가 정확히 일치하므로 점수가 높게 나오는 것은 당연하며 실제 모델 성능이 아니다. [seqeval](https://github.com/chakki-works/seqeval)은 기본 모드와 strict·scheme 설정에서 무효 태그의 해석이 달라질 수 있다. 설치 버전·IOB1/IOB2/BIOES 등 체계·mode·zero_division을 고정해 보고한다.

같은 BIO 문자열이라도 strict 규칙과 관대한 오류 교정은 다른 span을 만들 수 있다. 라이브러리 이름만 같다고 재현 가능한 평가가 되는 것은 아니다. 토큰화·서브워드 복원·특수 토큰·패딩 제외·문장 경계도 함께 정의한다.

## 4. Macro·weighted와 부분 일치

Macro F1은 정한 유형별 F1을 같은 웨이트로 평균한다. Weighted F1은 보통 정답 개체 support의 비율로 가중하지만 그 정의를 명시해야 한다. 둘 다 micro F1과 일반적으로 같지 않다. 희귀 유형의 성능·표본 수와 유형별P·R을 함께 확인한다.

부분 겹침 점수를 쓰려면 겹침 기준·유형 일치·일대일 매칭·여러 겹침의 대응과 부분 점수 산식을 정의해야 한다. ‘유형만 맞음’도 어떤 예측과 어떤 정답을 대응시키는지가 필요하다. 이 페이지의 코드는 정확 span 비교만 구현하며 부분 일치·가중 평균은 제공하지 않는다. 표준 정확 span 결과를 별도로 보고해 서로 다른 평가를 혼동하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
정답 LOC [0,3)에 예측 LOC [0,2) 하나만 있으면 정확 일치의 TP·FP·FN은 얼마인가?

</div>

??? success "풀이"

    TP=0·FP=1·FN=1이다. 일부 토큰이 맞았더라도 정확 span 평가에는 부분 점수가 없다. 겹침 기반 지표는 별도의 매칭·점수 정의를 필요로 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
예측 문장이 하나 적을 때 현재 compute_ner_metrics는 이를 어떻게 처리하는가?

</div>

??? success "풀이"

    zip으로 짧은 쪽까지만 비교해 남은 정답 문장을 버릴 수 있다. 누락이 FN으로 자동 합산되지 않으므로 표본 수·순서와 각 문장의 토큰 길이를 먼저 검증해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
average=weighted는 실제 support 가중 F1을 계산하는가?

</div>

??? success "풀이"

    아니다. micro 이외는 관측 유형의 F1 단순 평균으로 가므로 weighted도 macro처럼 처리된다. 지원 설정과 유형 집합을 명시하고 미지원 문자열은 오류로 처리하는 계약이 필요하다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
자체 extract_entities와 seqeval 결과가 다를 수 있는 이유를 설명하라.

</div>

??? success "풀이"

    초기 I·유형 전환 I 등 무효 태그의 교정·거부 폴리시와 strict 설정·태그 체계가 다를 수 있다. 같은 토큰화·유효 위치·문장 구분·빈 경우 규칙을 맞추고 경계 사례의 예상 span을 명시해 비교해야 한다.



## 정리하며

NER 평가의 핵심은 동일 표본·좌표·태그 체계에서 경계·유형을 비교하는 것이다. micro·macro·weighted와 부분 일치는 서로 다르며 현재 코드는 정확 span의 micro와 제한된 macro만 구현한다. 표본 누락·무효 태그·패딩·빈 경우의 규칙을 고정해야 한다.
