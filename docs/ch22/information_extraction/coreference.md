# 상호참조 해결

상호참조 해결(coreference resolution)은 문서에서 같은 대상을 가리키는 언급을 묶는 작업이다. 여기서 언급은 고유명사·명사구·대명사 등의 텍스트 구간이다. 동일한 문자열이라도 다른 대상을 가리킬 수 있고, 다른 표현이라도 문맥에서 같은 대상을 뜻할 수 있다.

## 1. 언급과 클러스터

예를 들어 ‘A사가 실적을 발표했다. 그 회사는 매출이 늘었다고 밝혔다. 대표 김민수는 자신이 전망을 설명하겠다고 말했다’에서 {A사, 그 회사}와 {대표 김민수, 자신}은 각각 가능한 클러스터다. 실제 연결은 문맥·화자·주석 기준을 확인해야 한다. 모든 주석 자료가 단독 언급이나 모든 대명사를 같은 방식으로 포함하지는 않는다.

개체명 인식만으로 일반 명사구와 대명사를 모두 찾을 수는 없다. 반대로 명사구를 모두 후보로 삼으면 실제 개체 언급이 아닌 구간도 많이 생긴다. 언급 탐지와 동일 개체 연결을 구분해서 평가할 필요가 있다.

## 2. 선행 언급 선택

언급 쌍 모델은 두 구간이 같은 대상을 가리킬 점수를 계산한 뒤 일관된 클러스터를 만들어야 한다. 쌍별 판단만으로 추이성이 자동으로 보장되지는 않는다. 언급 순위 모델은 언급 i의 앞선 후보 j<i와 더미 ε를 함께 비교한다.

$$P(a_i=j)=\frac{\exp s(i,j)}{\exp s(i,\epsilon)+\sum_{k<i}\exp s(i,k)},\qquad j<i,$$

더미의 확률도 같은 분모에 exp s(i,ε)를 분자로 사용한다. ε는 선행 언급이 없다는 선택이며 반드시 새로운 현실 개체가 실제로 존재한다는 판정은 아니다. 자기 자신 i를 선행 후보로 넣지 않는다.

## 3. 구간 기반 신경 모델

[Lee 등(2017)](https://aclanthology.org/D17-1018/)은 문서의 구간 후보를 만들고 경계·어텐션으로 얻은 내부 표현·너비 특징으로 언급 표현을 구성한다. 언급 점수와 선행 쌍 점수를 함께 사용하며 후보를 가지치기한다. 다음 코드는 이 전체 모델 중 점수 계산 부분만 보여 준다.

```python
import torch
import torch.nn as nn

class CorefScorer(nn.Module):
    """간추린 같은 것 가리키기 점수 매기기 단원."""
    def __init__(self, hidden_dim=768, ffnn_dim=1000):
        super().__init__()
        span_dim = hidden_dim * 3 + 20  # 시작, 끝, 머리 어텐션, 너비 특징
        self.mention_score = nn.Sequential(
            nn.Linear(span_dim, ffnn_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(ffnn_dim, 1),
        )
        pair_dim = span_dim * 3 + 20  # g_i, g_j, g_i*g_j, 거리 특징
        self.antecedent_score = nn.Sequential(
            nn.Linear(pair_dim, ffnn_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(ffnn_dim, 1),
        )

    def forward(self, span_i, span_j, pair_features):
        s_m_i = self.mention_score(span_i)
        s_m_j = self.mention_score(span_j)
        pair_repr = torch.cat([span_i, span_j, span_i * span_j, pair_features], -1)
        s_a = self.antecedent_score(pair_repr)
        return s_m_i + s_m_j + s_a
```

hidden_dim=768이면 span_dim=3×768+20=2324, pair_dim=3×2324+20=6992다. 두 span 입력의 마지막 차원은 2324, pair_features는 20이어야 한다. 이 클래스는 실제 인코더·구간 생성·너비 및 쌍 특징·더미·후보 마스킹·클러스터 추론을 만들지 않는다. 출력은 주어진 두 구간의 점수이며 완성된 상호참조 시스템이 아니다.

## 4. 여러 정답 선행 언급의 학습

같은 정답 클러스터의 앞선 언급은 여러 개일 수 있으므로 하나만 정답으로 강제하지 않고 확률을 합한다. 후보 집합 C(i)는 앞선 후보와 ε, 정답 집합 Y(i)는 그중 올바른 선행 언급들이다. 올바른 앞선 후보가 없을 때는 ε를 정답에 넣는 정책이 필요하다.

$$\mathcal{L}=-\sum_i\log\frac{\sum_{j\in Y(i)}\exp s(i,j)}{\sum_{j\in C(i)}\exp s(i,j)}.$$

가지치기로 정답이 빠졌을 때의 처리를 정의하고 logsumexp로 안정적으로 계산해야 한다. 위 CorefScorer는 이 손실이나 가지치기를 구현하지 않는다. 정답 언급을 제공받는 설정과 언급부터 예측하는 설정도 구분한다.

## 5. 평가와 문서 응용

MUC는 클러스터 연결·분할, B³는 언급별 정밀도·재현율, CEAF는 클러스터 간 일대일 정렬에 초점을 둔다. CoNLL 점수는 통상 MUC·B³·CEAF_e 각각의 F1 평균이다. 언급 집합·단독 언급 처리·공식 scorer 설정을 맞춰 비교해야 하며 단순 쌍 정확도와 같지 않다.

긴 보고서와 대화에서 대명사·회사 언급을 연결하면 문장 간 사실을 모으는 데 도움이 된다. 다만 지주회사와 자회사, 발언자와 인용 대상, 같은 이름의 다른 인물을 구분해야 한다. Alphabet과 Google 같은 관련 조직을 항상 같은 개체라고 합치거나 모든 대명사를 가장 가까운 명사에 연결하는 규칙은 적절하지 않다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
상호참조 클러스터와 개체명 유형 레이블의 차이를 설명하라.

</div>

??? success "연습문제 1 풀이"

    유형은 PERSON·ORG처럼 언급이 어떤 종류인지 나타낸다. 클러스터는 특정 문서에서 어떤 언급들이 같은 대상을 가리키는지 나타낸다. 같은 ORG 유형이라도 다른 회사일 수 있고, 대명사는 이름이 아니어도 회사 언급과 같은 클러스터에 속할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
선행 후보의 범위와 더미 선택의 확률을 설명하라.

</div>

??? success "연습문제 2 풀이"

    언급 i보다 앞선 j<i와 더미 ε를 후보로 둔다. 분모는 이 모든 후보의 지수 점수 합이며 더미도 같은 소프트맥스에 포함한다. 자기 자신을 앞선 후보로 넣지 않는다. 더미는 적절한 선행 후보가 없다는 선택으로 단독 언급·비언급 등 주석 정책과 함께 해석해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
한 언급에 올바른 선행 언급이 여러 개 있을 때 주변화 목적의 의미를 설명하라.

</div>

??? success "연습문제 3 풀이"

    정답 클러스터의 여러 선행 후보에 할당한 확률을 합해 그 합의 음의 로그를 최소화한다. 특정 하나를 강제 정답으로 선택할 필요가 없다. 후보 가지치기로 정답이 모두 사라지거나 실제 선행 언급이 없는 경우 더미를 포함하는 처리 규칙을 명시해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
CorefScorer가 문서 원문만으로 클러스터를 반환할 수 없는 이유와 필요한 추가 단계를 설명하라.

</div>

??? success "연습문제 4 풀이"

    입력은 이미 구성한 구간 표현 둘과 쌍 특징이다. 문서 인코더·구간 후보/특징·후보 가지치기·선행 마스킹·더미 점수·학습 손실·연결 및 클러스터 구성은 별도로 필요하다. 최종 평가는 주석 기준을 맞춘 클러스터 지표와 언급 탐지 오류를 함께 살핀다.



## 정리하며

상호참조는 언급의 유형이 아니라 동일한 지시 대상의 연결을 다룬다. 선행 후보·더미·정답 주변화와 클러스터 평가를 구분하고, 점수 계산 모듈을 전체 시스템으로 설명하지 않는다.

**참고 문헌**

1. Lee, K., et al. (2017). End-to-End Neural Coreference Resolution. *EMNLP*.
2. Joshi, M., et al. (2020). SpanBERT: Improving Pre-Training by Representing and Predicting Spans. *TACL*.
3. Wu, W., et al. (2020). CorefQA: Coreference Resolution as Query-Based Span Prediction. *ACL*.
