# 밀집 검색(Dense Retrieval)

---

## 1. 학습 목표

- 빽빽한 찾기의 두 인코더 구조를 이해한다
- 주요 빽빽한 찾기 모델을 견준다
- 찾기를 위한 맞대어 학습을 짠다

---

## 2. 두 인코더 구조

빽빽한 찾기는 따로 있는 인코더로 물음과 텍스트를 함께 쓰는 임베딩 공간에 부호화한다:

$$\text{score}(q, d) = \text{sim}(E_q(q), E_d(d))$$

여기서 $E_q$과 $E_d$은 인코더 네트워크이고 $\text{sim}$은 흔히 코사인 닮음이나 점곱이다.

찾을 때에는 텍스트 임베딩을 미리 셈해 두고 어림 가장 가까운 이웃(ANN)으로 효율적으로 찾도록 색인한다.

---

## 3. 주요 모델

| 모델 | 임베딩 차원 | 학습 데이터 | 핵심 특징 |
|-------|--------------|---------------|-------------|
| DPR | 768 | NQ, TriviaQA | 두 개의 BERT 인코더 |
| E5 | 1024 | 갖가지 웹 짝 | 시킴에 맞춰 다듬음 |
| BGE | 1024 | C-MTEB | 여러 말 |
| GTE | 1024 | 갖가지 | 두루 쓰기에 좋음 |
| OpenAI text-embedding-3 | 3072 | 비공개 | 품질 높음, API |
| Cohere embed-v3 | 1024 | 비공개 | 찾기에 맞춤 |

---

## 4. 맞대어 학습

빽빽한 찾개는 맞대어 손실(InfoNCE)로 학습된이다:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{d^- \in \mathcal{N}} \exp(\text{sim}(q, d^-) / \tau)}$$

여기서 $d^+$은 맞는(걸리는) 텍스트이고 $\mathcal{N}$은 아닌 텍스트이며 $\tau$은 온도다.

### 어려운 음의 보기 캐기

물음과 비슷하지만 맞닿지 않는 텍스트인 **어려운 음성**을 쓰면 성능이 크게 나아진다:

```python
def mine_hard_negatives(query_embedding, document_embeddings, positive_idx, k=10):
    """어려운 아닌 보기를 찾는다: 비슷하나 관련 없는 문서."""
    similarities = query_embedding @ document_embeddings.T
    # 맞는 문서를 뺀다
    similarities[positive_idx] = -float('inf')
    # 가장 비슷하면서 관련 없는 문서 위 k개
    hard_neg_indices = similarities.argsort(descending=True)[:k]
    return hard_neg_indices
```

---

## 5. 구현

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 미리 익힌 빽빽한 찾개를 불러온다
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# 문서를 부호화한다(한 번만 하고 목록에 담는다)
documents = [
    "AAPL reported Q3 revenue of $81.8B, beating estimates by \$1.2B",
    "The Federal Reserve held rates steady at 5.25-5.50%",
    "NVDA guidance implies 170% YoY data center revenue growth",
]
doc_embeddings = model.encode(documents, normalize_embeddings=True)

# 물음을 부호화하고 찾아온다
query = "Which company beat revenue estimates?"
query_embedding = model.encode([query], normalize_embeddings=True)

# 코사인 닮음(묻힘은 고르게 되어 있다)
scores = query_embedding @ doc_embeddings.T
top_idx = scores[0].argsort()[::-1]
print(f"Most relevant: {documents[top_idx[0]]}")
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
찾아 붙여 생성(RAG) 물길을 밝혀라. 대규모 언어 모델만 쓸 때와 비교하여 어떤 문제를 푸는가?

</div>

??? success "연습문제 1 풀이"
    찾아 붙여 생성은 찾개와 생성기를 아우른다. (1) 물음이 주어지면 찾개가 앎 곳간에서 맞닿는 텍스트를 찾는다. (2) 찾아온 텍스트를 물음과 이어 붙여 대규모 언어 모델의 맥락으로 준다. (3) 대규모 언어 모델이 찾아온 증거에 뿌리내린 답을 만든다. 이는 다음을 푼다. (1) **앎의 끊긴 때** — 다시 익히지 않고 앎 곳간을 고칠 수 있다. (2) **헛것 지어내기** — 찾아온 텍스트에 뿌리내려 지어냄이 줄어든다. (3) **분야 맞춤** — 분야 텍스트를 더하면 곱게 다듬지 않고도 특화된 질의응답을 할 수 있다. (4) **확인할 수 있음** — 답을 출처 텍스트까지 되짚을 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
성긴 찾기(BM25)와 빽빽한 찾기(임베딩 기초)를 비교하여라. 저마다 언제 뛰어난가?

</div>

??? success "연습문제 2 풀이"
    **BM25**(성김): 단어 잦기와 역 텍스트 잦기로 단어를 짝짓는다. 핵심 단어를 딱 맞추는 데 뛰어나고 익힐 필요가 없으며 뒤집은 색인으로 빠르다. 뜻의 닮음(비슷한 말, 바꿔 말하기)에는 어그러진다. **빽빽한 찾기**: 물음과 텍스트를 빽빽한 벡터로 부호화하고 코사인 닮음으로 짝짓는다. 뜻을 담아내지만 학습 데이터와 임베딩 계산이 필요하다. **BM25는** 분야 용어와 학습 데이터가 귀할 때 **뛰어나다**. **빽빽한 찾기는** 자연어 물음과 뜻으로 짝짓는 데 **뛰어나다**. 둘을 섞은 방식(BM25 + 빽빽한 찾기)이 어느 한쪽보다 나은 경우가 많다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
실전 찾아 붙여 생성 체계를 세울 때의 핵심 어려움은 무엇인가?

</div>

??? success "연습문제 3 풀이"
    핵심 어려움: (1) **덩이 짓기 전략**: 텍스트를 알맞은 크기의 찾을 수 있는 덩이로 쪼개야 한다. 너무 작으면 맥락을 잃고 너무 크면 잡음이 는다. (2) **임베딩의 좋음**: 두루 쓰는 임베딩이 분야의 뜻을 담아내지 못할 수 있다. (3) **찾기의 좋음**: 맞닿지 않거나 어긋나는 텍스트가 생성기를 잘못 이끌 수 있다. (4) **맥락 창의 한계**: 찾아온 텍스트가 대규모 언어 모델의 맥락 길이를 넘을 수 있다. (5) **늦음**: 찾기가 생성 물길에 늦음을 더한다. (6) **새로움**: 앎 곳간이 바뀌면 색인을 고쳐야 한다. (7) **평가**: 끝에서 끝까지의 좋음을 재려면 찾기와 생성을 모두 값매김해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
찾아 붙여 생성 물길에서 다시 매기기가 찾기의 좋음을 어떻게 낫게 하는지 밝혀라.

</div>

??? success "연습문제 4 풀이"
    처음 찾기(BM25나 빽빽한 찾기)는 빠르지만 어림이며 텍스트 $k$개의 후보 모음을 돌려준다. 그다음 **다시 매개**(대개 엇갈린 인코더)가 (물음, 텍스트) 짝마다 함께 점수를 매겨, 두 인코더 찾기가 놓치는 결이 고운 주고받음을 담아낸다. 엇갈린 인코더는 더 정확하지만 느리므로(벡터 한 번 찾기 대신 순방향 $O(k)$번) 상위 $k$개 후보에만 쓴다. 다시 매긴 상위 $n$개 텍스트($n < k$)을 생성기에 넘긴다. 이 두 단계 방식은 처음 찾기의 빠르기와 크로스 어텐션 점수 매기기의 정확도를 아우른다.

## 정리하며

이 마당은 학습 목표, 두 인코더 구조, 주요 모델, 맞대어 학습을 차례로 짚었다.

**참고 문헌**

1. Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain QA." *EMNLP*.
2. Wang, L., et al. (2022). "Text Embeddings by Weakly-Supervised Contrastive Pre-training." *arXiv*.
