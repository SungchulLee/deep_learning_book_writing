# 뽑아내는 물음 답하기
## 학습 목표

- 구간 뽑기를 물음 답하기의 세우기로 이해한다
- BERT 바탕 뽑아내는 물음 답하기를 짠다
- 답할 수 없는 물음을 다룬다

## 일 세우기

물음 $q$과 앞뒤 흐름 글월 $c$이 주어졌을 때 답 구간 $a = c_{i:j}$을 뽑아낸다. 여기서 $i$과 $j$은 비롯하는 토막 자리와 끝나는 토막 자리다.

$$\hat{a} = \arg\max_{(i,j): i \leq j \leq i + L_{\max}} P(\text{start}=i \mid q, c) \cdot P(\text{end}=j \mid q, c)$$

## 뽑아내는 물음 답하기를 위한 BERT

BERT(Devlin 외, 2019)는 물음과 맥락을 한 차례로 부호화한다:

```
[CLS] question tokens [SEP] context tokens [SEP]
```

선형 머리 둘이 맥락 토막에 대한 시작 자리와 끝 자리를 어림한다:

$$P(\text{start}=i) = \frac{\exp(\mathbf{w}_s^T \mathbf{h}_i)}{\sum_k \exp(\mathbf{w}_s^T \mathbf{h}_k)}$$

$$P(\text{end}=j) = \frac{\exp(\mathbf{w}_e^T \mathbf{h}_j)}{\sum_k \exp(\mathbf{w}_e^T \mathbf{h}_k)}$$

### 구현

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

### 맞춤 모델 짜기

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

## 답할 수 없는 물음 다루기

SQuAD 2.0에는 답할 수 없는 물음이 들어 있다. 맥락에 답이 없으면 모델이 답하지 않는 법을 배워야 한다.

### 답 없음 점수

가장 좋은 구간 점수를 ("답 없음"을 나타내는) `[CLS]` 점수와 견준다:

$$s_{\text{span}} = \max_{i,j} (s_{\text{start},i} + s_{\text{end},j})$$

$$s_{\text{null}} = s_{\text{start},[\text{CLS}]} + s_{\text{end},[\text{CLS}]}$$

$s_{\text{null}} > s_{\text{span}} + \tau$이면 "답할 수 없음"으로 미루어 본다. 여기서 $\tau$은 개발 묶음에서 잡은 문턱이다.

## 긴 글월 다루기

BERT의 토막 512개 한계 때문에 긴 글월은 미끄러지는 창으로 잘라야 한다:

1. 맥락을 겹치는 덩이로 쪼갠다(성큼 = 토막 128개)
2. 덩이마다 물음 답하기 모델을 돌린다
3. 모든 덩이에 걸쳐 믿음도가 가장 높은 답 구간을 고른다

## 참고 문헌

1. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*.
2. Rajpurkar, P., et al. (2018). Know What You Don't Know: Unanswerable Questions for SQuAD. *ACL*.

## 연습문제

**연습문제 1.**
뽑아내는 물음 답하기와 지어내는 물음 답하기의 차이를 밝혀라. 저마다 언제 쓰는 것이 좋은가?

??? success "연습문제 1 풀이"
    **뽑아내는 물음 답하기**는 글월 속의 잇닿은 구간을 답으로 가려낸다(보기로 SQuAD). 모델이 시작 자리와 끝 자리를 어림한다. **지어내는 물음 답하기**는 답을 자유로운 글로 내놓는데, 그 말이 글월에 그대로 나오지 않을 수 있다. 답이 맥락 안에 반드시 있는 경우(사실 물음, 읽고 이해하기)에는 확인이 쉽고 단순한 뽑아내는 방식이 낫다. 새로 지은 답, 여러 글월을 아우른 따짐, 또는 답에 녹여 내기나 다시 말하기가 필요할 때는 지어내는 방식이 낫다.

---

**연습문제 2.**
BERT 바탕 뽑아내는 물음 답하기 모델이 답 구간을 어떻게 어림하는지 설명하여라.

??? success "연습문제 2 풀이"
    들임은 `[CLS] question [SEP] context [SEP]` 꼴로 만든다. BERT은 모든 토막에 대해 숨은 상태 $h_1, \ldots, h_n$을 낸다. 선형 켜 둘이 비롯함과 끝남의 로짓을 미루어 본다. $s_i = w_s^\top h_i$과 $e_i = w_e^\top h_i$이다. 미루어 본 구간은 $(i^*, j^*)$이며 $i^* = \arg\max_i s_i$, $j^* = \arg\max_{j \geq i^*} e_j$이다. 익힐 때는 비롯하는 자리와 끝나는 자리에 각각 엇결 엔트로피 잃음을 쓴다. 자신 점수는 $\text{softmax}(s_{i^*}) \cdot \text{softmax}(e_{j^*})$이다.

---

**연습문제 3.**
SQuAD 값매김에 쓰이는 F1 잣대는 무엇인가? 일부만 맞는 경우를 어떻게 다루는가?

??? success "연습문제 3 풀이"
    SQuAD F1은 미루어 본 것과 참값을 토막 자루로 본다. 정밀도 = (겹치는 토막) / (미루어 본 토막), 재현율 = (겹치는 토막) / (참 토막)이고 $F_1 = 2PR/(P+R)$이다. 이러면 반쯤 맞은 것도 다룰 수 있다. 참값이 "New York City"인데 "New York"으로 미루어 보면 0이 아니라 $F_1 = 2 \cdot (2/2) \cdot (2/3) / (1 + 2/3) = 0.8$을 얻는다. 딱 맞음(EM)은 더 깐깐한 자로, 잣대를 맞춘 뒤(소문자로 바꾸고 관사와 문장 부호를 없앤 뒤) 글자열이 딱 같을 때만 1을 준다.

---

**연습문제 4.**
물음 답하기 체계는 주어진 맥락으로 답할 수 없는 물음임을 어떻게 정할 수 있는가?

??? success "연습문제 4 풀이"
    SQuAD 2.0에서는 글월 안에 답이 없는 물음도 있다. 모델은 "답 없음" 점수를 배우는데, 흔히 `[CLS]` 토막을 비롯함이자 끝남으로 미루어 본 점수(빈 구간)다. 빈 구간 점수가 가장 좋은 빈 구간 아닌 점수보다 문턱 $\tau$만큼 크면 모델은 "답할 수 없음"으로 미루어 본다. 문턱 $\tau$은 답 없음 미루어 봄의 정밀도와 재현율을 저울질하도록 개발 묶음에서 잡는다. 아니면 구간을 뽑기 앞에 따로 둘 가름개를 두어 답할 수 있는지 미루어 볼 수도 있다.
