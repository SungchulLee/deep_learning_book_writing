# NER 데이터셋의 변환과 분할

NERDatasetBuilder는 원문과 주석을 보관하고 공백 토큰의 IOB2 태그로 변환하거나 표본을 두 집합으로 나누는 작은 도구다. 주석 품질 검수·원문 offset 정렬·누수 방지·파일 형식 저장까지 완성된 데이터 구축 시스템은 아니다. 뉴럴 넷트웍 학습도 포함하지 않는다.

## 1. 코드와 기존 출력

```python
"""
이름 알아보기 자료 뭉치 만들기와 꼴 갖추기
====================================

이름 알아보기 자료 뭉치를 만들고 꼴을 갖추는 도구.

받치는 것:
- IOB 이름표 붙이기
- 자료 나누기
- 꼴 바꾸기

지은이: 러닝 목적
날짜: 2025
"""

from typing import List, Tuple, Dict
import random

# ========================================================================
# 메인
# ========================================================================


class NERDatasetBuilder:
    """제대로 꼴을 갖춘 이름 알아보기 자료 뭉치 세우기."""
    
    def __init__(self):
        """자료 뭉치 세우개 첫자리매김."""
        self.samples = []
    
    def add_sample(self, text: str, entities: List[Dict]):
        """
        자료 뭉치에 표본 하나 더하기.
        
        인수:
            text: 글월 문자열
            entities: 'text', 'type', 'start', 'end'를 담은 것 사전의 목록
        """
        self.samples.append({
            'text': text,
            'entities': entities
        })
    
    def to_iob_format(self) -> List[Tuple[List[str], List[str]]]:
        """
        자료 뭉치를 IOB 꼴로 바꾸기.
        
        반환값:
            (토막들, 이름표들) 튜플의 목록
        """
        iob_data = []
        
        for sample in self.samples:
            text = sample['text']
            entities = sample['entities']
            
            # 단순 토막내기
            tokens = text.split()
            labels = ['O'] * len(tokens)
            
            # 이름표 매기기(간추림 — 토막이 맞는다고 가정)
            for entity in entities:
                entity_tokens = entity['text'].split()
                # 토막 목록에서 것의 자리 찾기
                for i in range(len(tokens) - len(entity_tokens) + 1):
                    if tokens[i:i+len(entity_tokens)] == entity_tokens:
                        labels[i] = f"B-{entity['type']}"
                        for j in range(1, len(entity_tokens)):
                            labels[i+j] = f"I-{entity['type']}"
                        break
            
            iob_data.append((tokens, labels))
        
        return iob_data
    
    def train_test_split(self, test_size: float = 0.2, 
                        random_seed: int = 42) -> Tuple[List, List]:
        """
        자료 뭉치를 익힘 뭉치와 시험 뭉치로 나누기.
        
        인수:
            test_size: 시험 뭉치로 쓸 자료의 몫
            random_seed: 같은 결과를 얻기 위한 마구잡이 씨앗
            
        반환값:
            (익힘 표본, 시험 표본) 튜플
        """
        random.seed(random_seed)
        samples_copy = self.samples.copy()
        random.shuffle(samples_copy)
        
        split_idx = int(len(samples_copy) * (1 - test_size))
        train = samples_copy[:split_idx]
        test = samples_copy[split_idx:]
        
        return train, test


if __name__ == "__main__":
    # 예
    builder = NERDatasetBuilder()
    
    builder.add_sample(
        "Steve Jobs founded Apple Inc.",
        [
            {'text': 'Steve Jobs', 'type': 'PER', 'start': 0, 'end': 10},
            {'text': 'Apple Inc.', 'type': 'ORG', 'start': 19, 'end': 29}
        ]
    )
    
    iob_data = builder.to_iob_format()
    print(f"IOB format: {iob_data}")
```

**기존 출력:**

```
IOB format: [(['Steve', 'Jobs', 'founded', 'Apple', 'Inc.'], ['B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG'])]
```

## 2. 저장한 주석과 실제 변환 기준

add_sample은 text·entities를 그대로 보관하며 원문 범위·유형·중첩과 text[start:end] 일치를 검사하지 않는다. 주석 사전에 start·end가 있어도 to_iob_format에서는 사용하지 않는다. 실제 변환은 entity.text를 공백으로 나눈 단어 열이 원문의 공백 토큰과 같은지 찾는 방식이다.

같은 이름이 여러 번 나와도 첫 일치에서 break하므로 뒤의 정확한 주석 위치를 선택하지 못한다. 서로 다른 위치의 동일 문자열 주석을 여러 개 줘도 첫 구간만 반복해 태깅할 수 있다. 원문 offset을 유지한 토큰화와 주석 구간 정렬이 필요하다.

text.split()은 문장부호를 단어에 붙인 채 두므로 주석 표현과 구두점이 다르면 못 찾는다. 토큰이 여러 공백·줄바꿈으로 분리된 위치도 문자 offset으로 복원하지 않는다. 문자열 목록만으로 원문 좌표가 보존된다고 가정하지 않는다.

## 3. 겹침과 태그 유효성

일치한 첫 토큰에 B, 뒤 토큰에 I를 붙여 IOB2를 만든다. 중첩·겹침 주석이 있으면 나중에 처리한 주석이 일부 토큰을 덮어쓴다. 기존 큰 개체의 I 태그가 남으면 유형 불일치 등 무효 시퀀스가 될 수 있다. 처리 순서가 유형·경계 폴리시를 암묵적으로 결정하지 않도록 평면 선택·다층 표현 등을 명시해야 한다.

주석을 못 찾아도 오류 없이 O가 남는다. 빈 entity.text는 빈 단어 열이 되어 B를 잘못 부여하거나 빈 원문에서 인덱스 오류를 일으킬 수 있다. 유형 목록·빈 언급·토큰 경계와 변환 전후 개체 수·범위를 검증하는 절차가 필요하다.

## 4. train_test_split의 범위

함수는 표본 목록의 얕은 복사본을 섞고 floor(N(1−test_size)) 지점에서 나눈다. 반환은 토큰·태그 형식이 아니라 원문·entities 사전 목록이다. to_iob_format도 특정 분할을 받지 않고 builder 전체를 변환하므로 학습·시험 변환을 어떻게 연결할지 별도로 정해야 한다.

얕은 복사는 내부 사전·주석 리스트까지 독립 복제하지 않는다. 분할 뒤 수정이 원본에도 영향을 줄 수 있다. random.seed는 전역 난수 상태를 바꾸며 test_size의 0~1 범위나 작은 자료에서 빈 분할이 되는지를 검사하지 않는다.

같은 원문·관련 문서·중복 문장·같은 출처가 양쪽에 갈 수 있고 유형별 레이어화나 독립 검증 분할도 없다. 실제 목적의 일반화 단위로 문서·출처·시간 등을 분리하고 분할 버전·ID를 보관해야 한다. 테스트 자료로 어휘·사전·피처를 설계하지 않는다.

## 5. 데이터 구축과 시연의 구분

시연은 주석 두 개를 직접 지정한 한 문장을 변환해 출력한다. 모델의 인식 정확도·수렴·생성 능력이나 대규모 데이터 품질을 평가하지 않는다. 실제 구축에는 주석 지침·독립 검수·offset 검증·유효 태그·분할·저장 형식의 계약이 필요하다. 이 편집에서는 기존 코드와 출력을 유지하며 파일을 실행하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
주석의 start·end가 올바르면 변환 결과도 해당 위치를 사용하게 되는가?

</div>

??? success "연습문제 1 풀이"

    아니다. 이 함수는 offset을 사용하지 않고 주석 문자열의 첫 공백 토큰 일치를 찾는다. 같은 이름이 반복되면 다른 위치를 가리킬 수 있으므로 원문 offset과 토큰 span을 직접 정렬해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
데이터를 추가하기 전에 어떤 계약을 확인해야 하는가?

</div>

??? success "연습문제 2 풀이"

    유형·비어 있지 않은 언급, 0≤start<end≤원문 길이와 원문 구간의 문자열 일치, 중첩·겹침 폴리시를 확인한다. 변환 뒤에는 토큰·태그 길이·유효 IOB2와 원래 개체가 보존되는지도 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
train_test_split이 데이터 누수를 자동으로 막는가?

</div>

??? success "연습문제 3 풀이"

    표본을 섞을 뿐 관련 원문·출처·중복을 묶지 않아 누수가 가능하다. 분할 후 반환도 raw 사전이고 내부 객체는 얕게 공유한다. 그룹별 분리·독립 검증과 원문 ID·분할 버전, 변환 절차를 별도로 정해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 NERDatasetBuilder 시험 조각이 적절한가? 필요한 확인 항목을 설명하라.

</div>

??? success "연습문제 4 풀이"

    이 클래스에는 뉴럴 넷트웍 forward·parameters·역전파가 없다. 아래 미정의 입력 조각은 완성 시험이 아니다. 반복 이름·잘못된 offset·빈 언급·구두점·중첩 덮어쓰기·작은/빈 분할·비율 범위와 반환 자료의 계약을 확인해야 한다.

    ```python
    def test_nerdatasetbuilder():
        model = NERDatasetBuilder(...)
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

NER 데이터 구축은 원문·주석·토큰·태그·분할의 대응을 보존해야 한다. 현재 도구는 offset을 사용하지 않는 첫 일치 변환과 단순 표본 분할만 제공하므로 반복 언급·중첩·누수와 변환 범위를 별도로 검토한다.
