# 추출형 질의응답

추출형 질의응답은 질문과 문맥이 주어졌을 때 문맥의 연속 구간을 답으로 선택한다. 답이 문맥에 없으면 답변을 유보할 수 있어야 한다. 생성형 질의응답은 새 문자열을 만들 수 있지만, 두 방식 모두 질문·근거와 맞는 답인지 별도로 평가해야 한다.

## 1. 시작·끝 위치와 구간 제약

토큰 i부터 j까지 양 끝을 포함하는 구간의 길이는 j−i+1이다. 허용된 문맥 토큰에 대해서만 0≤i≤j, j−i+1≤L_max를 만족하는 구간을 비교한다.

$$\hat a=\arg\max_{(i,j)\in\mathcal A}\{s_{\mathrm{start},i}+s_{\mathrm{end},j}\}.$$

같은 입력에서 시작·끝 소프트맥스 확률의 곱을 최대화하는 것과 같은 점수 순서를 준다. 다만 이 곱이 모든 유효 구간 위에서 정규화되거나 실제 정답 확률로 보정된 것은 아니다. 시작과 끝의 argmax를 독립적으로 고르면 끝이 시작보다 앞설 수 있다.

## 2. 문맥 인코딩과 head

BERT의 개념적 입력 구성은 다음과 같다. 실제 구분 토큰·segment ID는 모델과 토크나이저에 따라 다르다.

```
[CLS] question tokens [SEP] context tokens [SEP]
```

각 위치의 은닉 표현에서 시작·끝 로짓을 계산한다. 질문·PAD·특수 토큰은 일반 답 후보에서 제외하고, 답 없음에 사용하는 위치는 별도로 처리한다. 토큰 구간을 원문 문자 구간으로 돌려주려면 offset mapping과 문맥 소속 정보가 필요하다.

## 3. 사전 학습된 QA 모델 예

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt",
                      max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)

    # 답 구간 풀어내기
    input_ids = inputs["input_ids"][0]
    answer_tokens = input_ids[start_idx:end_idx + 1]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# 예
context = """Tesla reported Q3 2024 revenue of \$25.18 billion,
up 8% year-over-year. Net income was \$2.17 billion."""

print(answer_question("What was Tesla's Q3 revenue?", context))
# "25.18 billion"
```

[roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)는 RoBERTa 기반 QA 체크포인트다. 예제의 단순 answer_question은 시작·끝을 독립 argmax하고 질문·특수 토큰 마스킹·최대 답 길이·답 없음 결정을 수행하지 않는다. 따라서 위의 제약 있는 구간 탐색이나 SQuAD 2.0 전체 추론을 구현한 함수가 아니다.

truncation=True는 긴 입력을 자를 뿐 자동으로 겹치는 여러 문맥 창을 만들지 않는다. 잘린 영역에 정답이 있으면 현재 함수로 복구할 수 없다. 입력도 CPU에서 만들어지므로 모델 장치와 맞춰야 한다. 코드에 적힌 회사·실적 수치와 출력 주석은 예시 문맥이며 실제 공시를 검증한 자료나 이번 편집의 실행 결과가 아니다.

## 4. 사용자 정의 head와 학습

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class ExtractiveQAModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)  # 시작, 끝

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        logits = self.qa_outputs(outputs.last_hidden_state)  # (B, L, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
```

ExtractiveQAModel은 encoder 뒤에 H→2 선형 레이어를 붙여 B×T의 시작·끝 로짓을 반환한다. 이 새 head는 QA 데이터로 학습하지 않은 초기 상태다. 답의 문자 offset을 subword 위치에 정렬하고 시작·끝 교차 엔트로피, 답 없음 위치, 잘린 답·padding의 폴리시를 구성해야 한다. encoder의 attention_mask만으로 QA 로스나 답 후보가 자동으로 제한되지 않는다.

학습된 QA 체크포인트를 불러오는 앞 예제와 새 head를 붙이는 이 예제는 학습 상태가 다르다. 모든 encoder가 BERT식 token_type_ids를 같은 방식으로 지원하는 것도 아니므로 선택한 모델의 인터페이스에 맞춘다.

## 5. 답 없음과 긴 문맥

[SQuAD 2.0](https://aclanthology.org/P18-2124/)은 문맥에서 답할 수 없는 질문을 포함한다. BERT식 추론에서는 null 위치의 시작·끝 로짓 합과 최선의 유효 구간 점수를 비교할 수 있다. 예를 들어 s_null−s_span>τ이면 답 없음을 선택한다. τ는 개발 자료에서 선택하며 점수 차를 보정된 확률로 오해하지 않는다.

긴 문맥은 질문·특수 토큰 길이를 제외한 문맥 예산으로 겹치는 창을 구성한다. 라이브러리의 stride가 겹침 토큰 수를 뜻하는지 이동 간격인지 확인한다. 창마다 원문 offset을 보존하고, 답 전체를 포함하지 못한 학습 창의 처리와 중복 답·null 점수의 창 간 집계 규칙을 정한다. 서로 다른 창의 소프트맥스 확률을 곧바로 공통 신뢰도로 간주하지 않는다.

## 6. 정확 일치와 토큰 F1

SQuAD 평가는 정규화한 문자열의 정확 일치(EM)와 답 토큰 중복 기반 F1을 사용한다. 이 토큰은 BERT subword ID가 아니라 평가 스크립트의 문자열 정규화·공백 분리 기준이다. 반복 토큰도 출현 수로 제한해 세고 정답 표현이 여러 개면 공식 폴리시에 따라 비교한다. 답 없음의 빈 답 처리도 포함해야 한다.

영어 SQuAD의 소문자화·관사·구두점 제거를 모든 언어의 일반 평가 규칙으로 적용하지 않는다. 한국어의 형태·띄어쓰기와 업무 요구에 맞는 원문·값 일치 기준을 별도로 명시할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
추출형과 생성형 질의응답의 답 공간과 근거 확인의 차이를 설명하라.

</div>

??? success "연습문제 1 풀이"

    추출형은 주어진 문맥의 연속 구간 중 선택하므로 답의 위치를 제시하기 쉽지만 답이 없거나 비연속 정보의 결합이 필요한 경우 제약이 있다. 생성형은 새 문자열·요약을 만들 수 있으나 원문에 없는 내용을 만들 위험도 있다. 어느 방식도 문맥과 질문의 일치·사실성을 자동으로 보장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
독립 시작·끝 argmax와 유효 구간의 공동 탐색이 다른 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    두 독립 최대 위치는 j<i이거나 질문·특수 토큰·과도하게 긴 구간일 수 있다. 공동 탐색은 문맥 후보와 i≤j, j−i+1≤L_max를 만족하는 쌍의 로짓 합을 비교한다. 본문의 단순 함수에는 이 제한과 답 없음·offset 복원이 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
정답 New York City, 예측 New York의 토큰 F1을 계산하고 EM과 구분하라.

</div>

??? success "연습문제 3 풀이"

    겹치는 토큰은 2개이고 예측 2개·정답 3개이므로 정밀도 1, 재현율 2/3, F1=0.8이다. 정규화 뒤 두 문자열이 다르므로 EM은 0이다. 반복 토큰과 여러 정답·빈 답은 공식 scorer 규칙을 따르며 이 토큰을 모델 subword와 혼동하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
답 없음 임계값과 문맥 창 구성이 필요한 이유를 설명하라.

</div>

??? success "연습문제 4 풀이"

    null 점수와 최선 구간의 차이 임계값은 답변과 유보의 균형을 정하므로 개발 자료에서 선택한다. 긴 문맥을 한 번 자르면 정답을 잃을 수 있어 겹치는 창과 원문 offset이 필요하다. 여러 창에서 중복 구간과 null 점수를 어떻게 모으는지도 명시해야 전체 문서의 일관된 답을 선택할 수 있다.



## 정리하며

추출형 QA는 시작·끝 로짓뿐 아니라 문맥 후보·길이·답 없음·원문 offset과 긴 문서의 창 처리를 포함한다. 단순 argmax 예제와 완성된 추론을 구분하고 학습 상태와 공식 평가 규칙을 확인한다.

**참고 문헌**

1. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*.
2. Rajpurkar, P., et al. (2018). Know What You Don't Know: Unanswerable Questions for SQuAD. *ACL*.
