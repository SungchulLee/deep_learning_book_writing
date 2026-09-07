# 사건 뽑기
## 학습 목표

- 사건 방아쇠 찾기와 딸린 것 뽑기를 이해한다
- 월 수준 사건 뽑기와 글월 수준 사건 뽑기를 가린다
- 사건 뽑기를 금융 글에 쓴다

## 일의 정의

사건 뽑기는 글에서 **짜임 있는 사건 기록**을 가려낸다. 사건마다 다음으로 이루어진다:

- **방아쇠**: 사건이 일어남을 가장 또렷이 드러내는 낱말
- **사건 갈래**: 미리 정해 둔 갈래 체계의 갈래
- **딸린 것**: 몫이 정해진 참여자와 속성

### 예

들임: *"애플이 2014년 5월 28일에 비츠 일렉트로닉스를 약 \$30억에 사들였다."*

| 조각 | 값 | 몫 |
|-----------|-------|------|
| 방아쇠 | acquired | -- |
| 사건 갈래 | 인수 | -- |
| 딸린 것 | Apple | 사는 쪽 |
| 딸린 것 | Beats Electronics | 대상 |
| 인자 | \$30억 | 값 |
| 딸린 것 | 2014년 5월 28일 | 날짜 |

## 두 단계 물길

### 1단계: 방아쇠 찾기

사건 방아쇠 낱말을 가려내고 그 사건 갈래를 매긴다. 이는 토막 수준의 갈래 매기기 일이다:

$$P(\text{type}_i \mid x_i, \mathbf{x}) = \text{softmax}(\mathbf{W}_t \mathbf{h}_i + \mathbf{b}_t)$$

여기서 $\mathbf{h}_i$은 앞뒤 흐름을 담은 토막 $i$의 나타냄이다.

### 2단계: 딸린 것 뽑기

찾아낸 방아쇠마다 딸린 것의 구간을 가려내고 몫을 매긴다:

$$P(\text{role} \mid x_{i:j}, \text{trigger}, \mathbf{x}) = \text{softmax}(\mathbf{W}_a [\mathbf{h}_{i:j}; \mathbf{h}_{\text{trigger}}] + \mathbf{b}_a)$$

## BERT 바탕 사건 뽑기

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class EventExtractor(nn.Module):
    def __init__(self, model_name, num_event_types, num_roles, hidden_dim=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.trigger_classifier = nn.Linear(hidden_dim, num_event_types)
        self.argument_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_roles),
        )

    def forward(self, input_ids, attention_mask, trigger_idx=None):
        h = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state

        # 방아쇠 찾기: 토막마다 갈래 매기기
        trigger_logits = self.trigger_classifier(h)  # (B, L, num_event_types)

        # 딸린 것 뽑기: 방아쇠 자리에 조건을 건다
        if trigger_idx is not None:
            trigger_h = h[torch.arange(h.size(0)), trigger_idx]  # (B, H)
            trigger_expanded = trigger_h.unsqueeze(1).expand_as(h)
            combined = torch.cat([h, trigger_expanded], dim=-1)
            arg_logits = self.argument_classifier(combined)  # (B, L, num_roles)
            return trigger_logits, arg_logits

        return trigger_logits, None
```

## 글월 수준 사건 뽑기

월 수준 모델은 여러 월에 흩어진 딸린 것을 놓친다:

*"애플이 월요일에 큰 거래를 알렸다. 그 기술 거인은 비츠 일렉트로닉스에 \$30억을 치를 것이다. 팀 쿡은 이를 훌륭한 인수라고 불렀다."*

인수 사건에 딸린 것이 월 셋에 걸쳐 있다. 글월 수준 모델은 월을 넘나드는 눈길이나 같은 것 가리키기로 흩어진 딸린 것을 잇는다.

## 금융 사건 갈래

| 사건 갈래 | 방아쇠 보기 | 핵심 딸린 것 |
|------------|-----------------|---------------|
| 실적 | reported, posted | 회사, 매출, 주당순이익, 기간 |
| 인수·합병 | acquired, merged | 사는 쪽, 대상, 값, 날짜 |
| 상장 | went public, listed | 회사, 거래소, 값, 주식 수 |
| 파산 | filed, defaulted | 회사, 조항, 날짜, 부채 |
| 임원 바뀜 | appointed, resigned | 사람, 직책, 회사 |
| 배당 | declared, distributed | 회사, 금액, 배당락일 |
| 주식 나눔 | split, divided | 회사, 비율, 기준일 |

## 평가

- **방아쇠 가려내기**: 방아쇠 구간 찾기의 F1
- **방아쇠 갈래 매기기**: 방아쇠 구간 + 사건 갈래의 F1
- **딸린 것 가려내기**: 딸린 것 구간 찾기의 F1
- **딸린 것 갈래 매기기**: 딸린 것 구간 + 몫 매김의 F1

ACE 2005 잣대 결과(어림값):

| 모델 | 방아쇠 F1 | 딸린 것 F1 |
|-------|-----------|------------|
| DMCNN(2015) | 67.6 | 45.7 |
| JMEE(2018) | 73.7 | 51.1 |
| OneIE(2020) | 74.7 | 56.8 |
| DEGREE(2022) | 76.3 | 58.2 |

## 참고 문헌

1. Chen, Y., et al. (2015). Event Extraction via Dynamic Multi-Pooling CNNs. *ACL*.
2. Lin, Y., et al. (2020). A Joint Neural Model for IE with Global Features. *ACL*.
3. Hsu, I., et al. (2022). DEGREE: A Data-Efficient Generation-Based Event Extraction Model. *NAACL*.

## 연습문제

**연습문제 1.**
닫힌 앎 뽑기와 열린 앎 뽑기의 차이를 밝혀라.

??? success "연습문제 1 풀이"
    **닫힌 앎 뽑기**는 이름표 붙인 자료로 익힌 살펴 배운 모델을 써서 미리 정해 둔 틀(보기로 "born-in", "works-for")의 관계를 뽑는다. 아는 관계 갈래에서는 정밀도가 높지만 새 관계는 찾아내지 못한다. **열린 앎 뽑기**는 미리 정한 틀 없이 월의 짜임 무늬나 배운 뽑개로 아무 (주어, 관계, 목적어) 세 쌍이나 뽑는다. 새 관계를 찾아낼 수 있지만 잡음이 섞이거나 겹치거나 다듬어지지 않은 것이 나올 수 있다. 닫힌 앎 뽑기는 앎 곳간 채우기에, 열린 앎 뽑기는 큰 말뭉치를 캐어 보는 데 알맞다.

---

**연습문제 2.**
같은 것 가리키기 풀기란 무엇이며 뒤따르는 자연어 일에 왜 중요한가?

??? success "연습문제 2 풀이"
    같은 것 가리키기 풀기는 글에서 같은 것을 가리키는 모든 표현(언급)을 가려내 무리로 묶는다. 보기로 "Alice went to the store. She bought milk"에서 "Alice"와 "She"는 같은 것을 가리킨다. 이는 다음에 결정적이다. (1) 앎 뽑기(월을 넘나들며 같은 것에 대한 사실 잇기), (2) 물음 답하기(물음과 글월의 대이름씨 풀기), (3) 간추리기(겹침 피하기), (4) 대화 체계(차례를 넘나들며 것 좇기).

---

**연습문제 3.**
사건 뽑기 체계의 핵심 조각을 설명하여라. 사건 뽑기가 관계 뽑기보다 어려운 까닭은 무엇인가?

??? success "연습문제 3 풀이"
    사건 뽑기는 다음을 가려낸다. (1) **사건 방아쇠**(사건을 가리키는 낱말, 보기로 "attacked"), (2) **사건 갈래** 매기기, (3) **딸린 것의 몫**(누가 누구에게 무엇을 어디서 언제 했는가). 관계 뽑기보다 어려운 까닭은 이렇다. 곧 사건마다 딸린 것의 수가 다르고 그 몫이 복잡하며, 방아쇠가 아리송할 수 있고(같은 낱말이 맥락에 따라 다른 사건 갈래를 일으킨다), 딸린 것이 여러 월에 걸칠 수 있으며, 월 하나에 겹치는 사건이 여럿 들어 있을 수 있다.

---

**연습문제 4.**
달림 뜯어 읽기를 써서 단순한 규칙 바탕 열린 앎 뽑기 체계를 꾸며라. 그 한계는 무엇인가?

??? success "연습문제 4 풀이"
    규칙 바탕 열린 앎 뽑기 체계는 달림 나무에서 주어-움직씨-목적어 무늬를 가려내 세 쌍을 뽑는다. (1) 주된 움직씨(뿌리)를 찾고, (2) 이름씨 주어(nsubj)를 주어로 뽑고, (3) 직접 목적어(dobj)를 목적어로 뽑고, (4) (주어, 움직씨, 목적어) 세 쌍을 만든다. **한계**: (1) 이름씨로 바꾼 표현("Obama's visit to China")의 관계를 놓친다. (2) 마디가 여럿인 복잡한 월을 다루지 못한다. (3) 특별한 규칙 없이는 입음꼴에서 어그러진다. (4) 앞가지가 붙은 월에서는 질 낮은 세 쌍을 낸다. (5) 같은 관계의 서로 다른 겉모습을 하나로 다듬지 못한다.
