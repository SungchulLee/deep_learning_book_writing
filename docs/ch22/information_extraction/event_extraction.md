# 이벤트 추출

이벤트 추출은 텍스트에서 사건 유형과 참여자·시간·금액 같은 역할별 정보를 구조화한다. 여기서 이벤트는 하나의 실제 사건을 뜻하며 단순히 단어를 샘플링하는 작업이 아니다. 문장에 사건을 나타내는 표현이 있어도 부정·계획·가정·인용을 구분해야 한다.

## 1. 트리거와 인자

트리거(trigger)는 사건을 나타내는 텍스트 구간이고 인자(argument)는 사건에 참여하는 대상이나 속성이다. 역할은 사건 유형의 스키마에 따라 정한다. 예를 들어 가상의 문장 ‘A사는 5월 28일 B사를 30억 달러에 인수할 계획이라고 밝혔다’에서는 ‘인수’가 트리거, A사·B사가 인수자·대상, 30억 달러가 금액 후보다. 계획 발표를 인수 완료로 바꾸어 기록하면 안 된다.

트리거는 반드시 동사 한 토큰일 필요가 없고 한 사건에 여러 인자가 있거나 문서 여러 문장에 정보가 흩어질 수 있다. 트리거를 요구하는 주석과 트리거 없이 슬롯을 채우는 문서 수준 주석도 구분한다.

## 2. 두 단계 모델

먼저 토큰의 문맥 표현 h_i로 트리거 유형을 예측하고, 선택한 트리거를 조건으로 인자 역할을 예측할 수 있다. 트리거가 아닌 토큰과 역할이 없는 토큰의 클래스를 포함하고 다중 토큰 구간·겹치는 사건을 어떻게 표현할지 정해야 한다.

$$P(t_i\mid x)=\operatorname{softmax}(W_t h_i+b_t).$$

인자 예측에서는 후보 구간의 표현과 트리거 표현을 결합할 수 있다. 다만 아래 구현은 구간 전체가 아니라 각 토큰의 표현을 결합하는 단순화된 head다.

## 3. BERT 기반 점수 계산

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

encoder 출력은 B×T×H, trigger_logits는 B×T×num_event_types다. trigger_idx를 제공하면 배치 항목마다 한 트리거 토큰을 골라 모든 위치에 결합하고 B×T×num_roles를 반환한다. trigger_idx가 없으면 인자 로짓은 None이다. 자동으로 트리거를 골라 모든 사건을 추출하는 시스템은 아니다.

hidden_dim은 불러온 encoder의 실제 hidden size와 같아야 한다. trigger_idx는 배치별 유효 정수 위치이며 장치도 맞춰야 한다. 구간 구성·subword 정렬·특수 토큰 및 padding 손실 제외·학습 손실·추론은 구현되어 있지 않다. attention_mask를 encoder에 전달하는 것만으로 분류 손실의 padding이 제외되지는 않는다.

## 4. 문서 수준 사건과 유형 설계

문장 밖의 인자를 모으려면 동일 사건인지, 동일 개체인지, 시간·부정·상태가 일치하는지 판단해야 한다. 상호참조와 문장 간 인코딩이 도움이 될 수 있지만 모든 문서 정보를 한 사건으로 합쳐서는 안 된다. 같은 회사에 대한 여러 인수·실적 사건을 별도로 식별해야 한다.

| 사건 유형 예 | 역할 예 | 구분할 문맥 |
|-------------|---------|-------------|
| 실적 발표 | 회사·기간·매출·주당순이익 | 실제 수치와 전망 |
| 인수·합병 | 인수자·대상·금액·날짜 | 계획·승인·완료·철회 |
| 임원 변경 | 사람·회사·직책·효력일 | 선임과 사임 |
| 배당 | 회사·금액·통화·관련 날짜 | 발표일·기준일·지급일 |
| 주식 분할 | 회사·분할 비율·효력일 | 정방향 분할과 역분할 |

단어 하나만으로 사건 유형을 확정하지 않는다. filed는 제출 일반을 뜻할 수 있고 default는 채무불이행을 나타낼 수 있어 파산 신청과 같은 유형으로 무조건 묶을 수 없다. 역할·유형은 사용하는 자료의 주석 정의를 따른다.

## 5. 평가

트리거 탐지는 문서와 구간, 트리거 분류는 여기에 사건 유형을 추가해 비교한다. 인자 평가도 문서·구간뿐 아니라 어떤 사건에 속하는지와 역할을 정의해야 한다. 인자 구간·유형만 같다고 다른 사건의 예측까지 정답으로 세면 안 된다.

정답 트리거를 주고 인자만 평가한 F1과 예측 트리거부터 시작한 전체 F1은 다르다. ACE 등 자료의 분할·전처리·주석·매칭 조건을 맞추지 않은 수치로 모델을 순위화하지 않는다. 본문에는 학습·벤치마크 실행이 없으므로 기존의 근거 없는 비교 수치 대신 평가 조건과 오류 유형을 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
트리거·사건 유형·인자 역할을 예를 들어 구분하라.

</div>

??? success "연습문제 1 풀이"

    ‘A사가 B사를 인수했다’에서 인수 표현은 트리거, 인수는 사건 유형, A사와 B사는 각각 인수자·대상 역할의 인자다. 같은 회사 이름이라도 사건에서 맡는 역할은 다를 수 있다. 트리거·인자의 구간과 실제 사건 상태는 주석 정책에 맞춰 기록한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
EventExtractor가 trigger_idx 없이 반환하는 값과 배치당 여러 트리거를 다루기 위해 필요한 단계를 설명하라.

</div>

??? success "연습문제 2 풀이"

    트리거 토큰별 로짓과 None을 반환한다. 인자 head를 사용하려면 각 후보 트리거를 선택해 조건화하고 사건별 인자를 모아야 한다. 현재 trigger_idx는 배치 항목마다 하나의 위치만 고르므로 모든 트리거·인자 관계를 한 번에 구성하지 않는다. 손실·학습·구간 디코딩도 별도로 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
인수 계획 발표와 거래 완료를 하나의 레코드로 잘못 합쳤을 때 생기는 오류와 방지 정보를 설명하라.

</div>

??? success "연습문제 3 풀이"

    참여 회사가 같아도 발표·승인·완료·철회는 상태와 시점이 다르다. 문서 출처·발표일·사건 시간·조건·부정·상태를 남기고 사건 동일성을 판단해야 한다. 상호참조로 회사 언급을 연결하는 것만으로 사건 동일성까지 결정되지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
정답 트리거 기반 평가와 전체 시스템의 인자 F1을 직접 비교할 수 없는 이유를 설명하라.

</div>

??? success "연습문제 4 풀이"

    정답 트리거를 주면 트리거 누락·오탐을 제거한 조건에서 인자 성능을 잰다. 전체 시스템에서는 그 오류도 인자 추출에 영향을 준다. 예측 사건·인자를 문서·트리거·역할과 함께 매칭하고 같은 분할·전처리·평가 조건에서 비교해야 한다.



## 정리하며

이벤트 추출은 사건의 참여자·역할과 상태를 문맥에 맞춰 구조화한다. 토큰 분류 head와 전체 추출 시스템을 구분하고, 사건별 구간·역할·트리거 조건을 맞춘 평가로 성능을 판단한다.

**참고 문헌**

1. Chen, Y., et al. (2015). Event Extraction via Dynamic Multi-Pooling CNNs. *ACL*.
2. Lin, Y., et al. (2020). A Joint Neural Model for IE with Global Features. *ACL*.
3. Hsu, I., et al. (2022). DEGREE: A Data-Efficient Generation-Based Event Extraction Model. *NAACL*.
