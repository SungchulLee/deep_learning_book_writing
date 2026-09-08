# 같은 것 가리키기 풀기

---

## 1. 학습 목표

- 언급 찾기와 앞선 말 잇기를 이해한다
- 끝에서 끝까지의 신경 같은 것 가리키기 풀기를 짠다
- 글월 수준 앎 뽑기에서 같은 것 가리키기의 몫을 헤아린다

---

## 2. 일의 정의

같은 것 가리키기 풀기는 글월 안에서 같은 실제 것을 가리키는 모든 **언급**을 **무리**로 묶는다.

### 예

*"**애플**이 **그** 분기 실적을 알렸다. **그 기술 거인**은 매출 \$90B을 밝혔다. **최고경영자 팀 쿡**은 **그 회사**가 기대를 넘어섰다고 말했다."*

무리 1: {Apple, its, The tech giant, the company}
무리 2: {CEO Tim Cook}

---

## 3. 언급의 갈래

| 갈래 | 보기 | 찾는 법 |
|------|---------|-----------|
| 고유 이름씨 | "Goldman Sachs" | 이름 알아보기 |
| 이름씨 마디 | "the company", "the deal" | 이름씨 마디 덩이 짓기 |
| 대이름씨 | "it", "they", "his" | 품사 붙이기 |

---

## 4. 고전적인 방식

### 언급 짝 모델

언급 짝마다 따로 점수를 매긴다:

$$s(m_i, m_j) = \mathbf{w}^T \phi(m_i, m_j)$$

특징으로는 글자열 맞음, 거리, 성/수 일치, 뜻의 닮음 등이 있다.

### 언급 매김 모델

언급마다 앞선 모든 언급과 "새 것"이라는 선택지를 함께 매긴다:

$$P(a_j \mid m_i) = \frac{\exp(s(m_i, m_j))}{\sum_{k \leq i} \exp(s(m_i, m_k)) + \exp(s_{\text{new}}(m_i))}$$

---

## 5. 끝에서 끝까지의 신경 같은 것 가리키기(Lee 외, 2017)

요즘 판을 잡은 방식은 언급 찾기와 같은 것 가리키기 잇기를 함께 한다.

### 구조

1. **구간 늘어놓기**: 길이 $L$까지의 모든 구간을 헤아린다
2. **구간 나타내기**: $\mathbf{g}_i = [\mathbf{h}_{\text{start}}; \mathbf{h}_{\text{end}}; \hat{\mathbf{h}}_i; \phi(i)]$이며 $\hat{\mathbf{h}}_i$은 눈길 짐을 실은 머리 낱말 나타냄이고 $\phi(i)$은 구간 너비를 담는다
3. **언급 점수**: $s_m(i) = \text{FFNN}_m(\mathbf{g}_i)$
4. **앞선 말 점수**: $s_a(i, j) = \text{FFNN}_a([\mathbf{g}_i; \mathbf{g}_j; \mathbf{g}_i \circ \mathbf{g}_j; \phi(i,j)])$
5. **짝 점수**: $s(i, j) = s_m(i) + s_m(j) + s_a(i, j)$

### 학습 목표

맞는 앞선 말 모두에 대해 주변화한다:

$$\mathcal{L} = -\sum_{i=1}^{N} \log \frac{\sum_{j \in \mathcal{Y}(i)} \exp(s(i, j))}{\sum_{j' \in \mathcal{C}(i)} \exp(s(i, j'))}$$

여기서 $\mathcal{Y}(i)$은 옳은 앞선 말의 묶음이고 $\mathcal{C}(i)$은 모든 후보에 허깨비 "새 개체" 앞선 말을 더한 것이다.

```python
import torch
import torch.nn as nn

class CorefScorer(nn.Module):
    """간추린 같은 것 가리키기 점수 매기기 단원."""
    def __init__(self, hidden_dim=768, ffnn_dim=1000):
        super().__init__()
        span_dim = hidden_dim * 3 + 20  # 시작, 끝, 머리 눈길, 너비 특징
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

---

## 6. 평가 지표

| 잣대 | 초점 |
|--------|-------|
| MUC | 짝 이음 F1 |
| B세제곱 | 것 수준 정밀도/재현율 |
| CEAF | 가장 좋은 무리 맞추기 |
| CoNLL | MUC, B세제곱, CEAF의 평균 |

---

## 7. 금융에서의 쓰임

같은 것 가리키기는 글월 수준의 금융 앎 뽑기에 꼭 필요하다. 곧 10-K 보고서(100쪽 이상)에 걸쳐 것의 언급을 좇고, 실적 발표 녹취록의 대이름씨를 말하는 이에게 잇고, 회사의 다른 이름(보기로 "Alphabet" = "Google" = "the search giant")을 풀어낸다.

---

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

## 정리하며

이 마당은 학습 목표、일의 정의、언급의 갈래、고전적인 방식을 차례로 짚었다.

**참고 문헌**

1. Lee, K., et al. (2017). End-to-End Neural Coreference Resolution. *EMNLP*.
2. Joshi, M., et al. (2020). SpanBERT: Improving Pre-Training by Representing and Predicting Spans. *TACL*.
3. Wu, W., et al. (2020). CorefQA: Coreference Resolution as Query-Based Span Prediction. *ACL*.
