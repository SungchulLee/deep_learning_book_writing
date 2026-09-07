# 빽빽한 찾기
## 학습 목표

- 빽빽한 찾기의 두 부호기 얼개를 이해한다
- 주요 빽빽한 찾기 모델을 견준다
- 찾기를 위한 맞대어 익히기를 짠다

## 두 부호기 얼개

빽빽한 찾기는 따로 있는 부호기로 물음과 글월을 함께 쓰는 묻힘 공간에 부호화한다:

$$\text{score}(q, d) = \text{sim}(E_q(q), E_d(d))$$

여기서 $E_q$과 $E_d$은 부호기 그물이고 $\text{sim}$은 흔히 코사인 닮음이나 점곱이다.

찾을 때에는 글월 묻힘을 미리 셈해 두고 어림 가장 가까운 이웃(ANN)으로 효율적으로 찾도록 색인한다.

## 주요 모델

| 모델 | 묻힘 차원 | 익힘 자료 | 핵심 특징 |
|-------|--------------|---------------|-------------|
| DPR | 768 | NQ, TriviaQA | 두 개의 BERT 부호기 |
| E5 | 1024 | 갖가지 웹 짝 | 시킴에 맞춰 다듬음 |
| BGE | 1024 | C-MTEB | 여러 말 |
| GTE | 1024 | 갖가지 | 두루 쓰기에 좋음 |
| OpenAI text-embedding-3 | 3072 | 비공개 | 품질 높음, API |
| Cohere embed-v3 | 1024 | 비공개 | 찾기에 맞춤 |

## 맞대어 익히기

빽빽한 찾개는 맞대어 손실(InfoNCE)로 익힌다:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{d^- \in \mathcal{N}} \exp(\text{sim}(q, d^-) / \tau)}$$

여기서 $d^+$은 맞는(걸리는) 글월이고 $\mathcal{N}$은 아닌 글월이며 $\tau$은 온도다.

### 어려운 음의 보기 캐기

물음과 비슷하지만 맞닿지 않는 글월인 **어려운 음성**을 쓰면 성능이 크게 나아진다:

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

## 구현

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

## 참고 문헌

1. Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain QA." *EMNLP*.
2. Wang, L., et al. (2022). "Text Embeddings by Weakly-Supervised Contrastive Pre-training." *arXiv*.

## 연습문제

**연습문제 1.**
찾아 붙여 만들어 내기(RAG) 물길을 밝혀라. 큰 말 모델만 쓸 때와 견주어 어떤 문제를 푸는가?

??? success "연습문제 1 풀이"
    찾아 붙여 만들어 내기는 찾개와 만들개를 아우른다. (1) 물음이 주어지면 찾개가 앎 곳간에서 맞닿는 글월을 찾는다. (2) 찾아온 글월을 물음과 이어 붙여 큰 말 모델의 맥락으로 준다. (3) 큰 말 모델이 찾아온 증거에 뿌리내린 답을 만든다. 이는 다음을 푼다. (1) **앎의 끊긴 때** — 다시 익히지 않고 앎 곳간을 고칠 수 있다. (2) **헛것 지어내기** — 찾아온 글월에 뿌리내려 지어냄이 줄어든다. (3) **분야 맞춤** — 분야 글월을 더하면 곱게 다듬지 않고도 특화된 물음 답하기를 할 수 있다. (4) **확인할 수 있음** — 답을 출처 글월까지 되짚을 수 있다.

---

**연습문제 2.**
성긴 찾기(BM25)와 빽빽한 찾기(묻힘 바탕)를 견주어라. 저마다 언제 뛰어난가?

??? success "연습문제 2 풀이"
    **BM25**(성김): 낱말 잦기와 거꿀 글월 잦기로 낱말을 짝짓는다. 핵심 낱말을 딱 맞추는 데 뛰어나고 익힐 필요가 없으며 뒤집은 색인으로 빠르다. 뜻의 닮음(비슷한 말, 바꿔 말하기)에는 어그러진다. **빽빽한 찾기**: 물음과 글월을 빽빽한 벡터로 부호화하고 코사인 닮음으로 짝짓는다. 뜻을 담아내지만 익힘 자료와 묻힘 셈하기가 필요하다. **BM25는** 분야 용어와 익힘 자료가 귀할 때 **뛰어나다**. **빽빽한 찾기는** 자연어 물음과 뜻으로 짝짓는 데 **뛰어나다**. 둘을 섞은 방식(BM25 + 빽빽한 찾기)이 어느 한쪽보다 나은 경우가 많다.

---

**연습문제 3.**
실전 찾아 붙여 만들어 내기 체계를 세울 때의 핵심 어려움은 무엇인가?

??? success "연습문제 3 풀이"
    핵심 어려움: (1) **덩이 짓기 전략**: 글월을 알맞은 크기의 찾을 수 있는 덩이로 쪼개야 한다. 너무 작으면 맥락을 잃고 너무 크면 잡음이 는다. (2) **묻힘의 좋음**: 두루 쓰는 묻힘이 분야의 뜻을 담아내지 못할 수 있다. (3) **찾기의 좋음**: 맞닿지 않거나 어긋나는 글월이 만들개를 잘못 이끌 수 있다. (4) **맥락 창의 한계**: 찾아온 글월이 큰 말 모델의 맥락 길이를 넘을 수 있다. (5) **늦음**: 찾기가 만들어 내기 물길에 늦음을 더한다. (6) **새로움**: 앎 곳간이 바뀌면 색인을 고쳐야 한다. (7) **값매김**: 끝에서 끝까지의 좋음을 재려면 찾기와 만들어 내기를 모두 값매김해야 한다.

---

**연습문제 4.**
찾아 붙여 만들어 내기 물길에서 다시 매기기가 찾기의 좋음을 어떻게 낫게 하는지 밝혀라.

??? success "연습문제 4 풀이"
    처음 찾기(BM25나 빽빽한 찾기)는 빠르지만 어림이며 글월 $k$개의 후보 모음을 돌려준다. 그다음 **다시 매개**(대개 엇갈린 부호기)가 (물음, 글월) 짝마다 함께 점수를 매겨, 두 부호기 찾기가 놓치는 결이 고운 주고받음을 담아낸다. 엇갈린 부호기는 더 정확하지만 느리므로(벡터 한 번 찾기 대신 앞먹임 $O(k)$번) 상위 $k$개 후보에만 쓴다. 다시 매긴 상위 $n$개 글월($n < k$)을 만들개에 넘긴다. 이 두 단계 방식은 처음 찾기의 빠르기와 엇갈린 눈길 점수 매기기의 정확도를 아우른다.
