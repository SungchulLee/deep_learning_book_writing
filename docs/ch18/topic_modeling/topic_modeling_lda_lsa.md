# 주제 나타내기 LDA LSA

주제 나타내기 LDA LSA.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 주제 모델 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 1. 코드

```python
"""주제 나타내기 LDA LSA."""
# ---
# title: "주제 나타내기: LDA와 LSA"
# description: "숨은 디리클레 나눔으로 하는 살펴보지 않는 주제 찾기
#               과 숨은 뜻 살피기 — 순수 파이썬 + PyTorch 짜기"
# ---
#
# 주제 나타내기는 이름표 붙인 자료 없이 글월 모음에서 숨은 주제를
# 찾아낸다. 바탕이 되는 방식 둘:
#
#   - LSA(숨은 뜻 살피기): 낱말-글월 행렬의 특잇값 쪼개기
#   - LDA(숨은 디리클레 나눔): 베이즈 지어내기 모델
#
#   1부 – 잘라 낸 특잇값 쪼개기로 하는 LSA(numpy/sklearn)
#   2부 – gensim으로 하는 LDA
#   3부 – 맨바닥부터 짠 LDA(접힌 기브스 표집)
#   4부 – 신경 주제 모델(PyTorch 변분 자기부호기 바탕)
#   5부 – 금융 글월의 주제 나타내기
#
# 바탕: O'Reilly "Practical NLP" 7장

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from typing import List, Dict, Tuple


# =====================================================================
# 1부 – 잘라 낸 특잇값 쪼개기로 하는 LSA
# =====================================================================
print("=" * 60)
print("Part 1: Latent Semantic Analysis (LSA)")
print("=" * 60)

# LSA = 낱말-글월 행렬에 특잇값 쪼개기를 쓴다.
# 잘라 낸 특잇값 쪼개기는 가장 중요한 숨은 차원을 담아내며,
# 이는 추상적인 "주제"에 맞대응된다.
#
# 수식:  X ≈ U_k Σ_k V_k^T
#   X:     낱말-글월 행렬 (V × D)
#   U_k:   낱말-주제 행렬 (V × k)
#   Σ_k:   주제의 세기 (k × k 대각)
#   V_k^T: 주제-글월 행렬 (k × D)

# 보기 글월
documents = [
    "The Federal Reserve raised interest rates by 25 basis points",
    "GDP growth slowed to 2.1 percent in the third quarter",
    "Inflation remains above the central bank target of two percent",
    "Treasury yields rose sharply after the employment report",
    "The unemployment rate fell to 3.5 percent a new low",
    "Apple reported record quarterly revenue driven by iPhone sales",
    "Tesla deliveries exceeded analyst expectations this quarter",
    "Microsoft cloud revenue grew 29 percent year over year",
    "Amazon announced a stock split and buyback program",
    "Google parent Alphabet beat earnings estimates for the quarter",
    "The S&P 500 index reached a new all time high today",
    "Oil prices surged after OPEC announced production cuts",
]

# 낱말-글월 행렬 세우기(단순 낱말 자루)
stopwords = {"the", "a", "an", "to", "of", "in", "by", "and", "for", "is", "was", "this"}


def build_bow(docs: List[str], stopwords: set) -> Tuple[np.ndarray, List[str]]:
    """낱말 자루 방식의 낱말-글월 행렬 세우기."""
    # 어휘 만들기
    vocab = {}
    for doc in docs:
        for word in doc.lower().split():
            if word not in stopwords and len(word) > 2:
                if word not in vocab:
                    vocab[word] = len(vocab)

    vocab_list = sorted(vocab.keys(), key=lambda w: vocab[w])

    # 행렬 세우기
    X = np.zeros((len(vocab), len(docs)))
    for j, doc in enumerate(docs):
        for word in doc.lower().split():
            if word in vocab:
                X[vocab[word], j] += 1

    return X, vocab_list


X, vocab_list = build_bow(documents, stopwords)
print(f"  Term-document matrix: {X.shape} (vocab × docs)")

# TF-IDF 무게 주기
tf = X / X.sum(axis=0, keepdims=True).clip(min=1)
idf = np.log(X.shape[1] / (1 + (X > 0).sum(axis=1, keepdims=True)))
X_tfidf = tf * idf

# 잘라 낸 특잇값 쪼개기
n_topics = 3
U, S, Vt = np.linalg.svd(X_tfidf, full_matrices=False)
U_k = U[:, :n_topics]
S_k = S[:n_topics]
Vt_k = Vt[:n_topics, :]

print(f"\n  Top words per topic (LSA, {n_topics} topics):")
for topic_idx in range(n_topics):
    # 상위 낱말 = U_k[:, topic_idx]에서 절댓값이 가장 큰 것
    top_word_idx = np.argsort(np.abs(U_k[:, topic_idx]))[-5:][::-1]
    words = [(vocab_list[i], U_k[i, topic_idx]) for i in top_word_idx]
    word_str = ", ".join(f"{w}({v:.3f})" for w, v in words)
    print(f"    Topic {topic_idx}: {word_str}")

# 글월-주제 매김
doc_topics = Vt_k.T  # (D × k)
print(f"\n  Document-topic matrix shape: {doc_topics.shape}")
for i, doc in enumerate(documents[:4]):
    topic = np.argmax(np.abs(doc_topics[i]))
    print(f"    Doc {i} → Topic {topic}: {doc[:50]}...")
print()


# =====================================================================
# 2부 – Gensim으로 하는 LDA
# =====================================================================
print("=" * 60)
print("Part 2: LDA with Gensim")
print("=" * 60)

print("""
  from gensim.models import LdaModel
  from gensim.corpora import Dictionary
  from nltk.tokenize import word_tokenize
  from nltk.corpus import stopwords
  import nltk

  nltk.download('stopwords')
  stops = set(stopwords.words('english'))

  # 앞손질: 토막내기, 소문자로, 불용어 없애기
  def preprocess(text):
      tokens = word_tokenize(text.lower())
      return [t for t in tokens if t.isalpha() and t not in stops]

  texts = [preprocess(doc) for doc in documents]

  # 사전과 말뭉치 만들기
  dictionary = Dictionary(texts)
  dictionary.filter_extremes(no_below=2, no_above=0.5)
  corpus = [dictionary.doc2bow(text) for text in texts]

  # LDA 익히기
  lda = LdaModel(
      corpus=corpus,
      id2word=dictionary.id2token,
      num_topics=5,
      iterations=400,
      passes=10,
      alpha='auto',         # 글월-주제 앞확률 배우기
      eta='auto',           # 주제-낱말 앞확률 배우기
      random_state=42,
  )

  # 주제 찍기
  for idx in range(5):
      print(f"Topic {idx}: {lda.print_topic(idx, num_words=8)}")

  # 새 글월의 주제 분포 얻기
  new_doc = preprocess("The central bank cut interest rates")
  bow = dictionary.doc2bow(new_doc)
  topic_dist = lda[bow]
  # → [(0, 0.72), (2, 0.15), (4, 0.13)]
""")


# =====================================================================
# 3부 – 맨바닥부터 짠 LDA(접힌 기브스 표집)
# =====================================================================
print("=" * 60)
print("Part 3: LDA From Scratch (Gibbs Sampling)")
print("=" * 60)

# LDA의 지어내는 과정:
#   글월 d마다:
#     주제 분포 θ_d ~ Dirichlet(α)을 뽑는다
#     d의 낱말 자리 i마다:
#       주제 z_{d,i} ~ Categorical(θ_d)을 뽑는다
#       낱말 w_{d,i} ~ Categorical(φ_{z_{d,i}})을 뽑는다
#
# 접힌 기브스 표집으로 미룸:
#   P(z_i = k | z_{-i}, w) ∝ (n_{d,k} + α) × (n_{k,w} + β) / (n_{k,·} + Vβ)


class LDAGibbs:
    """접힌 기브스 표집으로 하는 LDA.

    인수:
        n_topics: 주제의 개수
        alpha:    글월-주제 분포의 디리클레 앞확률
        beta:     주제-낱말 분포의 디리클레 앞확률
        n_iter:   기브스 표집 바퀴 수
    """

    def __init__(self, n_topics: int = 5, alpha: float = 0.1,
                 beta: float = 0.01, n_iter: int = 100):
        self.K = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter

    def fit(self, documents: List[List[int]], vocab_size: int):
        """기브스 표집으로 LDA 모델 맞추기.

        인수:
            documents: 글월의 목록. 글월마다 낱말 번호의 목록
            vocab_size: 낱말 곳간의 크기
        """
        self.V = vocab_size
        D = len(documents)
        K = self.K

        # 셈 행렬
        self.n_dk = np.zeros((D, K))       # 글월-주제 셈
        self.n_kv = np.zeros((K, self.V))  # 주제-낱말 셈
        self.n_k = np.zeros(K)             # 주제 합계

        # 첫자리매김: 낱말마다 마구잡이 주제 매김
        self.z = []  # 주제 매김
        for d, doc in enumerate(documents):
            doc_z = []
            for w in doc:
                k = np.random.randint(K)
                doc_z.append(k)
                self.n_dk[d, k] += 1
                self.n_kv[k, w] += 1
                self.n_k[k] += 1
            self.z.append(doc_z)

        # 기브스 표집 바퀴
        for iteration in range(self.n_iter):
            for d, doc in enumerate(documents):
                for i, w in enumerate(doc):
                    k_old = self.z[d][i]

                    # 지금 매김 없애기
                    self.n_dk[d, k_old] -= 1
                    self.n_kv[k_old, w] -= 1
                    self.n_k[k_old] -= 1

                    # 조건부 셈하기: P(z_i=k | 나머지)
                    p = (self.n_dk[d] + self.alpha) * \
                        (self.n_kv[:, w] + self.beta) / \
                        (self.n_k + self.V * self.beta)
                    p = p / p.sum()

                    # 새 주제 뽑기
                    k_new = np.random.choice(K, p=p)
                    self.z[d][i] = k_new

                    # 셈 고치기
                    self.n_dk[d, k_new] += 1
                    self.n_kv[k_new, w] += 1
                    self.n_k[k_new] += 1

        return self

    def get_topic_words(self, n_words: int = 10) -> List[List[Tuple[int, float]]]:
        """주제마다 상위 낱말 얻기."""
        topics = []
        for k in range(self.K):
            phi_k = (self.n_kv[k] + self.beta) / (self.n_k[k] + self.V * self.beta)
            top_idx = np.argsort(phi_k)[-n_words:][::-1]
            topics.append([(idx, phi_k[idx]) for idx in top_idx])
        return topics


# 데이터를 준비한다
all_tokens = []
word2id = {}
tokenized_docs = []
for doc in documents:
    tokens = []
    for word in doc.lower().split():
        if word not in stopwords and len(word) > 2:
            if word not in word2id:
                word2id[word] = len(word2id)
            tokens.append(word2id[word])
    tokenized_docs.append(tokens)

id2word = {v: k for k, v in word2id.items()}

# LDA 맞추기
np.random.seed(42)
lda = LDAGibbs(n_topics=3, alpha=0.1, beta=0.01, n_iter=50)
lda.fit(tokenized_docs, len(word2id))

print("  LDA topics (from scratch):")
for k, topic_words in enumerate(lda.get_topic_words(n_words=5)):
    words = ", ".join(f"{id2word[idx]}({prob:.3f})" for idx, prob in topic_words)
    print(f"    Topic {k}: {words}")
print()


# =====================================================================
# 4부 – 신경 주제 모델(PyTorch 변분 자기부호기 바탕)
# =====================================================================
print("=" * 60)
print("Part 4: Neural Topic Model (ProdLDA / ETM)")
print("=" * 60)

# 신경 주제 모델은 다음과 같은 변분 자기부호기를 쓴다:
#   - 부호기: 낱말 자루 → 주제 분포로 대응시킨다(매개변수 바꾸기 재주로)
#   - 풀개: 주제 분포에서 낱말 자루를 되살린다
#   - 손실 = 되살림 + KL 벌어짐
#
# ProdLDA는 디리클레 대신 로지스틱 정규 분포를 쓴다
# 기울기 바탕 가장 좋게 하기를 쉽게 하려고.


class NeuralTopicModel(nn.Module):
    """ProdLDA 방식 신경 주제 모델.

    로지스틱 정규 앞확률을 쓴 변분 자기부호기로 주제를 배운다.
    풀개 무게 행렬의 줄이 바로 주제이다.
    """

    def __init__(self, vocab_size: int, n_topics: int, hidden_dim: int = 64):
        super().__init__()
        # 부호기: 낱말 자루 → 숨은 층 → (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.mu_layer = nn.Linear(hidden_dim, n_topics)
        self.logvar_layer = nn.Linear(hidden_dim, n_topics)

        # 풀개: 주제 비율 → 낱말 자루 되살림
        self.decoder = nn.Linear(n_topics, vocab_size, bias=False)
        # decoder.weight: (vocab_size × n_topics)
        # 칸마다 = 한 주제의 낱말 분포

        self.bn = nn.BatchNorm1d(n_topics, affine=False)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # 주제에 소프트맥스를 씌운 뒤 되살리기
        theta = F.softmax(self.bn(z), dim=-1)
        return F.log_softmax(self.decoder(theta), dim=-1), theta

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon, theta = self.decode(z)
        return recon, mu, logvar, theta


def train_ntm(model, bow_matrix, n_epochs=100, lr=2e-3):
    """신경 주제 모델 익히기."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(bow_matrix, dtype=torch.float32)

    model.train()
    for epoch in range(n_epochs):
        recon, mu, logvar, theta = model(X)

        # 되살림 손실(음의 로그 가능도)
        recon_loss = -(X * recon).sum(dim=-1).mean()

        # KL 벌어짐(로지스틱 정규와 표준 정규)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        loss = recon_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}: recon={recon_loss:.3f}, KL={kl_loss:.3f}")

    return model


# 신경 모델을 위한 낱말 자루 행렬 세우기
bow_matrix = X.T  # (D × V)

n_topics = 3
ntm = NeuralTopicModel(len(vocab_list), n_topics, hidden_dim=32)
print("  Training Neural Topic Model:")
ntm = train_ntm(ntm, bow_matrix, n_epochs=100, lr=2e-3)

# 풀개 무게에서 주제 뽑기
ntm.eval()
topic_weights = ntm.decoder.weight.data.numpy()  # (V × K)
print(f"\n  Neural topics:")
for k in range(n_topics):
    top_idx = np.argsort(topic_weights[:, k])[-5:][::-1]
    words = ", ".join(f"{vocab_list[i]}({topic_weights[i, k]:.2f})" for i in top_idx)
    print(f"    Topic {k}: {words}")
print()


# =====================================================================
# 5부 – 금융 글월의 주제 나타내기
# =====================================================================
print("=" * 60)
print("Part 5: Financial Document Topic Analysis")
print("=" * 60)

print("""
  주제 나타내기의 금융 쓰임새:

  1. 실적 발표 살피기:
     - 분기별 주제 바뀜 좇기
     - 새 위험 요인이나 전략 전환 알아채기
     - 경쟁사끼리 주제 분포 견주기

  2. SEC 보고서 살피기:
     - 10-K 보고서에서 위험 요인 주제 뽑기
     - 경영진 논의 절의 주제 흘러감 지켜보기
     - 규정 지킴을 위해 이상한 주제 분포 표시하기

  3. 뉴스 갈래 짓기:
     - 금융 뉴스를 주제(거시, 업종, 기업)로 무리 짓기
     - 때에 따른 주제 눈길 좇기(뜨는 주제)
     - 주제 바탕 거래 신호 세우기

  4. 연구 보고서 살피기:
     - 분석 대상 전체에 걸친 분석 주제 간추리기
     - 다수 의견과 반대 의견 가려내기
     - 주제 마음결을 값 움직임에 잇기

  보기: 주제 바탕 거래 신호
  ─────────────────────────────────────
  때 t의 글월 d마다:
    1. 주제 분포 θ_d = LDA(d)을 셈한다
    2. 주제 k마다 주제 마음결 s_k을 셈한다
    3. 신호 = Σ_k θ_dk × s_k  (무게를 준 마음결)

  금융 말뭉치에서 흔히 떠오르는 주제:
    - 거시/금리:     "fed", "rates", "inflation", "gdp"
    - 실적:        "revenue", "eps", "guidance", "beat"
    - 인수·합병:             "acquisition", "merger", "deal", "bid"
    - 위험/규제: "compliance", "fine", "investigation"
    - 기술:      "cloud", "ai", "platform", "growth"
""")

print("Done.")


if __name__ == "__main__":
    pass
```

## 2. 논의

여기 짠 것은 함께 어울려 온전한 주제 모델 얼개를 이루는 클래스 2개(`LDAGibbs`, `NeuralTopicModel`)를 정한다. 클래스마다 뚜렷한 조각 하나를 감싸므로 코드가 단원별로 나뉘고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 셈 그래프를 정한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
붙박이 첫자리매김일 때 `LDAGibbs`의 배울 수 있는 매개변수 전체 개수를 셈하여라. 무게와 치우침을 모두 넣어 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
어텐션 가중치 뒤에(값과 곱하기 전에) 드롭아웃 층을 추가하라. 학습 중에는 드롭아웃 비율 0.1을 쓴다. 어텐션 드롭아웃이 정칙화에 도움이 되는 이유를 설명하라.

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`이다. 어텐션 드롭아웃은 학습 중에 일부 어텐션 가중치를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 관계에 지나치게 기대지 않게 한다. 이는 모델이 어텐션을 더 고르게 분산시키고 더 견고한 표현을 배우도록 북돋우며, 표준 드롭아웃이 뉴런의 공적응을 막는 것과 비슷하다.

---

**연습문제 3.**
자기 어텐션의 계산 복잡도를 열의 길이 $n$과 모델 차원 $d$의 함수로 설명하라. 이것이 왜 긴 열에 대해 Longformer나 Linformer 같은 구조의 동기가 되는가?

??? success "연습문제 3 풀이"
    표준 자기 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도가 $O(n^2 d)$이고 어텐션 가중치에 $O(n^2)$의 메모리가 든다. 열이 길면(예: $n = 4096$) 감당하기 어려워진다. Longformer는 국소적인 미끄럼창 어텐션($w$이 창 크기일 때 $O(n \cdot w \cdot d)$)과 선택된 토큰에 대한 희소한 전역 어텐션을 결합한다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 조금 내주고 긴 입력에서의 실용적인 효율을 얻는다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `LDAGibbs`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = LDAGibbs(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 주제 나타내기 LDA LSA

여기 짠 것은 함께 어울려 온전한 주제 모델 얼개를 이루는 클래스 2개(`LDAGibbs`, `NeuralTopicModel`)를 정한다.

고갱이 갈래는 `LDAGibbs`, `NeuralTopicModel`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
