# 사전 학습 트랜스포머의 토큰 분류 head

이 템플릿은 사전 학습 인코더에서 토큰 피처를 얻고 새 선형 레이어로 NER 태그 로짓을 만든다. BERT의 언어 사전 학습과 NER head의 지도 학습은 다르다. 아래 코드는 새 head의 학습·정답 정렬·평가를 구현하지 않으므로 생성 직후의 태그를 의미 있는 NER 결과로 해석하면 안 된다.

## 1. 코드

```python
"""
미리 익힌 모델을 쓰는 변환기 바탕 이름 알아보기
===============================================

이름 알아보기에 BERT/RoBERTa 쓰기.

핵심 개념:
- 미리 익힌 변환기 곱게 다듬기
- 아래낱말 토막내기 다루기
- 토막 클래스 매기기

지은이: 러닝 목적
날짜: 2025
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict

# ========================================================================
# 메인
# ========================================================================


class TransformerNER(nn.Module):
    """
    미리 익힌 모델을 쓰는 변환기 바탕 이름 알아보기.
    
    구조:
    들임 → BERT/RoBERTa → 선형 → 소프트맥스 → 이름표
    """
    
    def __init__(self, model_name: str = 'bert-base-cased', num_labels: int = 9):
        """
        변환기 이름 알아보기 첫자리매김.
        
        인수:
            model_name: Hugging Face 모델 이름
            num_labels: 것 이름표의 개수
        """
        super(TransformerNER, self).__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # 미리 익힌 변환기 읽어 들이기
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 정칙화를 위한 드롭아웃
        self.dropout = nn.Dropout(0.1)
        
        # 분류 머리
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None):
        """
        피드포워드.
        
        인수:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        반환값:
            logits: [batch_size, seq_len, num_labels]
        """
        # 변환기의 내놓음 얻기
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 숨은 상태 얻기
        sequence_output = outputs[0]  # [배치, seq_len, hidden_size]
        
        # 떨구기 쓰기
        sequence_output = self.dropout(sequence_output)
        
        # 분류
        logits = self.classifier(sequence_output)  # [batch, seq_len, num_labels]
        
        return logits
    
    def predict(self, text: str) -> List[Tuple[str, str]]:
        """
        글 속의 것 어림하기.
        
        인수:
            text: 들임 글월 문자열
            
        반환값:
            (토막, 이름표) 짝의 목록
        """
        # 토큰으로 나누기
        encoding = self.tokenizer(text, return_tensors='pt', 
                                  padding=True, truncation=True)
        
        # 순전파
        with torch.no_grad():
            logits = self.forward(encoding['input_ids'], 
                                 encoding['attention_mask'])
        
        # 예측을 얻는다
        predictions = torch.argmax(logits, dim=-1)[0]
        
        # 토막으로 되돌려 대응시키기
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        return list(zip(tokens, predictions.tolist()))


if __name__ == "__main__":
    print("Transformer NER model template")
    print("Note: Requires transformers library and pre-trained models")
    print("Example: BERT, RoBERTa, DistilBERT for token classification")
```

## 2. 생성과 입력·출력 계약

기본 모델 이름은 bert-base-cased이고9개 레이블을 출력하도록 선언한다. 그러나 PER·ORG·LOC·MISC의BIO9개처럼 구체적인 id2label 매핑은 없다. 같은9클래스라도 순서·의미가 다르면 모델·정답·출력 해석이 맞지 않는다. tokenizer는 사전 학습 모델과 같은 이름에서 로드하지만 forward는 텍스트가 아니라 정수 input_ids와 attention_mask를 받는다.

출력 피처는 B×L×H이고 dropout 뒤 선형 head는 B×L×K 로짓을 반환한다. 기본 BERT의 H768에서 새 head는(768+1)×9=6921개 파라메터다. 전체 수는 로드한 인코더와 head를 합하며 모델 이름·설정이 바뀌면 달라진다. 여기서 사전 학습 웨이트를 다운로드·실행할 필요는 없다.

현재 import에는 List·Dict만 있고 predict의 반환 annotation에 사용하는 **Tuple이 없다**. 지연 annotation을 켜지 않은 일반 Python 환경에서는 클래스 정의 중 NameError가 날 수 있어 이 파일을 그대로 실행 가능한 템플릿으로 설명하지 않는다. 실제 반환도 `(토큰 문자열,정수 예측 ID)`의 목록이라 `Tuple[str,str]` 선언과 의미가 다르다.

## 3. 서브워드·특수 토큰·긴 입력

predict는 토큰화된 모든 위치의 argmax를 반환한다. 특수 토큰·padding을 제외하거나 서브워드를 단어·개체 span으로 합치는 처리가 없다. continuation 조각에 붙인 태그와 단어 레이블의 관계를 정의하고 tokenizer의 offset·word_ids를 이용해 원문 span으로 정렬해야 한다. CRF나 BIO 강제 제약도 없다.

truncation=True는 모델 한도를 넘는 텍스트를 자를 수 있다. 긴 문서 전체를 자동 처리하거나 잘린 개체를 복원하는 기능은 아니다. 창별 입력·겹침·경계·중복 합치기와 원문 offset을 명시해야 한다. 학습에도 특수·패딩·무시 서브워드를 loss의 유효 위치에 맞춰 처리한다.

## 4. 모델 모드와 장치

predict는 no_grad를 쓰지만 eval로 전환하지 않는다. 모델이 train 모드이면 인코더와 새 head의 dropout이 남는다. 생성된 토큰 텐서는 CPU이며 모델을 CUDA·MPS로 옮긴 경우 입력도 같은 장치에 맞춰야 한다. no_grad는 평가 모드나 장치 이동을 대신하지 않는다.

이 클래스의 dropout0.1은 인코더 출력 뒤 피처 dropout이다. 내부 softmax attention 웨이트 dropout을 새로 추가한 것이 아니다. 기존 사전 학습 인코더의 attention 설정과 학습 head의 정규화를 구분한다. 어떤 dropout도 자동으로 attention을 균등하게 하거나 NER 정확도를 보장하지 않는다.

## 5. 학습과 비교의 범위

토큰별 로짓에는 단일 레이블 교차 엔트로피 등을 연결할 수 있지만 이 코드에는 labels·로스·최적화가 없다. 사전 학습과 파인튜닝·평가 자료를 구분하고 정확 span 평가를 사용한다. 레이어 수를 임의로 바꾸는 것은 사전 학습 인코더의 구조·웨이트 호환성 문제를 포함하므로 generic ModuleList 반복만으로 처리할 수 없다.

길이L의 dense attention 쌍별 연산은 O(L²H), projection은 O(LH²)이며 전체 인코더의 MLP·레이어 수와 구현도 비용에 영향을 준다. 토큰 수·관찰 문서 범위·padding·사전 학습 조건을 맞춰 다른 NER 구조와 비교한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 BERT hidden_size768·num_labels9에서 새 분류 head의 파라메터를 계산하라.

</div>

??? success "연습문제 1 풀이"

    웨이트768×9와 bias9를 합해6921개다. 전체 모델 수는 사전 학습 인코더의 파라메터를 추가해야 한다. 모델 이름·설정에 따라 달라지며 tokenizer는 뉴럴 넷트웍 파라메터가 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
코드의 dropout이 attention 웨이트에 적용되는가? predict가 항상 결정적인가?

</div>

??? success "연습문제 2 풀이"

    현재 dropout은 인코더 출력 피처 뒤에 있다. predict의 no_grad는 eval 전환이 아니어서 train 모드이면 dropout이 남을 수 있다. 내부 attention dropout과 피처 dropout의 위치를 구분한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
predict의 토큰·정수 ID 목록을 곧바로 개체 span으로 볼 수 없는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"

    특수 토큰 제외·서브워드 정렬·id2label·BIO 검사와 원문 offset 복원이 없다. 잘린 문서의 뒤 구간도 처리하지 않는다. 출력 annotation의 미수입 Tuple과 실제 정수 반환도 현재 템플릿의 한계다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 조각으로 사전 학습 인코더 깊이를 바꿀 때 고려할 점을 설명하라.

</div>

??? success "연습문제 4 풀이"

    YourBlock이 정의되어 있지 않으며 인코더의 hidden 크기·레이어 구조·웨이트 이름·사전 학습 호환성을 맞춰야 한다. 라이브러리 설정과 모델 로드 폴리시를 정하고 새 구조를 평가해야 한다. 단순 모듈 등록이 사전 학습·NER 학습을 자동 완성하지 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

사전 학습 인코더의 토큰 피처와 새 NER head는 서로 다른 학습 단계다. 현재 템플릿의 import·타입·장치·모드 한계와 특수 토큰·서브워드·잘림을 확인하고, 태그 매핑·로스·원문 span 평가를 연결해야 한다.
