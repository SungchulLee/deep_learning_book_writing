# 토픽 모델링: LSA와 LDA

토픽 모델링은 정답 주제 라벨 없이 문서의 단어 사용 패턴을 요약하는 방법이다. LSA는 행렬의 저차원 구조, LDA는 문서별 토픽 혼합과 토픽별 단어 분포를 추정한다. 뉴럴 토픽 모델은 또 다른 잠재 표현·트레인 목적을 사용한다. 사람이 붙인 토픽 이름은 해석이며 모델이 자동으로 얻은 확정적인 의미 분류가 아니다.

## 1. 코드와 출력

```python
"""토픽 모델링 LDA LSA."""
# ---
# title: "토픽 모델링: LDA와 LSA"
# description: "잠재 디리클레 할당으로 하는 비지도 토픽 발견
#               과 잠재 의미 분석 — 순수 파이썬 + PyTorch 구현"
# ---
#
# 토픽 모델링은 라벨 붙인 데이터 없이 문서 모음에서 잠재 토픽을
# 찾아낸다. 바탕이 되는 방식 둘:
#
#   - LSA(잠재 의미 분석): 단어-문서 행렬의 특잇값 분해
#   - LDA(잠재 디리클레 할당): 베이즈 생성 모델
#
#   1부 – 절단 특잇값 분해로 하는 LSA(numpy/sklearn)
#   2부 – gensim으로 하는 LDA
#   3부 – 밑바닥부터 짠 LDA(붕괴 깁스 샘플링, collapsed Gibbs)
#   4부 – 뉴럴 토픽 모델(PyTorch 변분 오토인코더 기반)
#   5부 – 금융 텍스트의 토픽 모델링
#
# 바탕: O'Reilly "Practical NLP" 7장

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from typing import List, Dict, Tuple
# 무작위로 뽑는 값이 아래에 나온다. 시드를 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)


# =====================================================================
# 1부 – 절단 특잇값 분해로 하는 LSA
# =====================================================================
print("=" * 60)
print("Part 1: Latent Semantic Analysis (LSA)")
print("=" * 60)

# LSA = 단어-문서 행렬에 특잇값 분해를 쓴다.
# 절단 특잇값 분해는 가장 중요한 잠재 차원을 담아내며,
# 이는 추상적인 "토픽"에 대응된다.
#
# 수식:  X ≈ U_k Σ_k V_k^T
#   X:     단어-문서 행렬 (V × D)
#   U_k:   단어-토픽 행렬 (V × k)
#   Σ_k:   토픽의 세기 (k × k 대각)
#   V_k^T: 토픽-문서 행렬 (k × D)

# 보기 문서
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

# 단어-문서 행렬 세우기(단순 BoW)
stopwords = {"the", "a", "an", "to", "of", "in", "by", "and", "for", "is", "was", "this"}


def build_bow(docs: List[str], stopwords: set) -> Tuple[np.ndarray, List[str]]:
    """BoW 방식의 단어-문서 행렬 세우기."""
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

# TF-IDF 가중
tf = X / X.sum(axis=0, keepdims=True).clip(min=1)
idf = np.log(X.shape[1] / (1 + (X > 0).sum(axis=1, keepdims=True)))
X_tfidf = tf * idf

# 절단 특잇값 분해
n_topics = 3
U, S, Vt = np.linalg.svd(X_tfidf, full_matrices=False)
U_k = U[:, :n_topics]
S_k = S[:n_topics]
Vt_k = Vt[:n_topics, :]

print(f"\n  Top words per topic (LSA, {n_topics} topics):")
for topic_idx in range(n_topics):
    # 상위 단어 = U_k[:, topic_idx]에서 절댓값이 가장 큰 것
    top_word_idx = np.argsort(np.abs(U_k[:, topic_idx]))[-5:][::-1]
    words = [(vocab_list[i], U_k[i, topic_idx]) for i in top_word_idx]
    word_str = ", ".join(f"{w}({v:.3f})" for w, v in words)
    print(f"    Topic {topic_idx}: {word_str}")

# 문서-토픽 할당
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

  # 전처리: 토큰화, 소문자로, 불용어 없애기
  def preprocess(text):
      tokens = word_tokenize(text.lower())
      return [t for t in tokens if t.isalpha() and t not in stops]

  texts = [preprocess(doc) for doc in documents]

  # 사전과 말뭉치 만들기
  dictionary = Dictionary(texts)
  dictionary.filter_extremes(no_below=2, no_above=0.5)
  corpus = [dictionary.doc2bow(text) for text in texts]

  # LDA 트레인
  lda = LdaModel(
      corpus=corpus,
      id2word=dictionary.id2token,
      num_topics=5,
      iterations=400,
      passes=10,
      alpha='auto',         # 문서-토픽 사전분포 학습
      eta='auto',           # 토픽-단어 사전분포 학습
      random_state=42,
  )

  # 토픽 출력
  for idx in range(5):
      print(f"Topic {idx}: {lda.print_topic(idx, num_words=8)}")

  # 새 문서의 토픽 분포 얻기
  new_doc = preprocess("The central bank cut interest rates")
  bow = dictionary.doc2bow(new_doc)
  topic_dist = lda[bow]
  # → [(0, 0.72), (2, 0.15), (4, 0.13)]
""")


# =====================================================================
# 3부 – 밑바닥부터 짠 LDA(붕괴 깁스 샘플링)
# =====================================================================
print("=" * 60)
print("Part 3: LDA From Scratch (Gibbs Sampling)")
print("=" * 60)

# LDA의 생성 과정:
#   문서 d마다:
#     토픽 분포 θ_d ~ Dirichlet(α)을 뽑는다
#     d의 단어 자리 i마다:
#       토픽 z_{d,i} ~ Categorical(θ_d)을 뽑는다
#       단어 w_{d,i} ~ Categorical(φ_{z_{d,i}})을 뽑는다
#
# 붕괴 깁스 샘플링으로 추론:
#   P(z_i = k | z_{-i}, w) ∝ (n_{d,k} + α) × (n_{k,w} + β) / (n_{k,·} + Vβ)


class LDAGibbs:
    """붕괴 깁스 샘플링으로 하는 LDA.

    인수:
        n_topics: 토픽의 개수
        alpha:    문서-토픽 분포의 디리클레 사전분포
        beta:     토픽-단어 분포의 디리클레 사전분포
        n_iter:   깁스 샘플링 반복 횟수
    """

    def __init__(self, n_topics: int = 5, alpha: float = 0.1,
                 beta: float = 0.01, n_iter: int = 100):
        self.K = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter

    def fit(self, documents: List[List[int]], vocab_size: int):
        """깁스 샘플링으로 LDA 모델 맞추기.

        인수:
            documents: 문서의 목록. 문서마다 단어 번호의 목록
            vocab_size: 어휘의 크기
        """
        self.V = vocab_size
        D = len(documents)
        K = self.K

        # 빈도 행렬
        self.n_dk = np.zeros((D, K))       # 문서-토픽 빈도
        self.n_kv = np.zeros((K, self.V))  # 토픽-단어 빈도
        self.n_k = np.zeros(K)             # 토픽 합계

        # 초기화: 단어마다 무작위 토픽 할당
        self.z = []  # 토픽 할당
        for d, doc in enumerate(documents):
            doc_z = []
            for w in doc:
                k = np.random.randint(K)
                doc_z.append(k)
                self.n_dk[d, k] += 1
                self.n_kv[k, w] += 1
                self.n_k[k] += 1
            self.z.append(doc_z)

        # 깁스 샘플링 반복
        for iteration in range(self.n_iter):
            for d, doc in enumerate(documents):
                for i, w in enumerate(doc):
                    k_old = self.z[d][i]

                    # 현재 할당 빼기
                    self.n_dk[d, k_old] -= 1
                    self.n_kv[k_old, w] -= 1
                    self.n_k[k_old] -= 1

                    # 조건부 확률 계산: P(z_i=k | 나머지)
                    p = (self.n_dk[d] + self.alpha) * \
                        (self.n_kv[:, w] + self.beta) / \
                        (self.n_k + self.V * self.beta)
                    p = p / p.sum()

                    # 새 토픽 뽑기
                    k_new = np.random.choice(K, p=p)
                    self.z[d][i] = k_new

                    # 빈도 갱신
                    self.n_dk[d, k_new] += 1
                    self.n_kv[k_new, w] += 1
                    self.n_k[k_new] += 1

        return self

    def get_topic_words(self, n_words: int = 10) -> List[List[Tuple[int, float]]]:
        """토픽마다 상위 단어 얻기."""
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
# 4부 – 뉴럴 토픽 모델(PyTorch 변분 오토인코더 기반)
# =====================================================================
print("=" * 60)
print("Part 4: Neural Topic Model (ProdLDA / ETM)")
print("=" * 60)

# 뉴럴 토픽 모델은 다음과 같은 변분 오토인코더를 쓴다:
#   - 인코더: BoW → 토픽 분포로 대응시킨다(리파라메리제이션 기법으로)
#   - 디코더: 토픽 분포에서 BoW를 복원한다
#   - 로스 = 복원 + KL 발산
#
# ProdLDA는 디리클레 대신 로지스틱 정규 분포를 쓴다
# 그래디언트 기반 최적화를 쉽게 하려고.


class NeuralTopicModel(nn.Module):
    """ProdLDA 방식 뉴럴 토픽 모델.

    로지스틱 정규 사전분포를 쓴 변분 오토인코더로 토픽을 배운다.
    디코더 웨이트 행렬의 열이 바로 토픽이다.
    """

    def __init__(self, vocab_size: int, n_topics: int, hidden_dim: int = 64):
        super().__init__()
        # 인코더: BoW → 은닉 레이어 → (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.mu_layer = nn.Linear(hidden_dim, n_topics)
        self.logvar_layer = nn.Linear(hidden_dim, n_topics)

        # 디코더: 토픽 비율 → BoW 복원
        self.decoder = nn.Linear(n_topics, vocab_size, bias=False)
        # decoder.weight: (vocab_size × n_topics)
        # 열마다 = 한 토픽의 단어 분포

        self.bn = nn.BatchNorm1d(n_topics, affine=False)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # 토픽에 소프트맥스를 씌운 뒤 복원
        theta = F.softmax(self.bn(z), dim=-1)
        return F.log_softmax(self.decoder(theta), dim=-1), theta

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon, theta = self.decode(z)
        return recon, mu, logvar, theta


def train_ntm(model, bow_matrix, n_epochs=100, lr=2e-3):
    """뉴럴 토픽 모델 트레인."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(bow_matrix, dtype=torch.float32)

    model.train()
    for epoch in range(n_epochs):
        recon, mu, logvar, theta = model(X)

        # 복원 로스(음의 로그 가능도)
        recon_loss = -(X * recon).sum(dim=-1).mean()

        # KL 발산(로지스틱 정규와 표준 정규)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        loss = recon_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}: recon={recon_loss:.3f}, KL={kl_loss:.3f}")

    return model


# 뉴럴 모델을 위한 BoW 행렬 세우기
bow_matrix = X.T  # (D × V)

n_topics = 3
ntm = NeuralTopicModel(len(vocab_list), n_topics, hidden_dim=32)
print("  Training Neural Topic Model:")
ntm = train_ntm(ntm, bow_matrix, n_epochs=100, lr=2e-3)

# 디코더 웨이트에서 토픽 뽑기
ntm.eval()
topic_weights = ntm.decoder.weight.data.numpy()  # (V × K)
print(f"\n  Neural topics:")
for k in range(n_topics):
    top_idx = np.argsort(topic_weights[:, k])[-5:][::-1]
    words = ", ".join(f"{vocab_list[i]}({topic_weights[i, k]:.2f})" for i in top_idx)
    print(f"    Topic {k}: {words}")
print()


# =====================================================================
# 5부 – 금융 텍스트의 토픽 모델링
# =====================================================================
print("=" * 60)
print("Part 5: Financial Document Topic Analysis")
print("=" * 60)

print("""
  토픽 모델링의 금융 쓰임새:

  1. 실적 발표 분석:
     - 분기별 토픽 변화 추적
     - 새 위험 요인이나 전략 전환 탐지
     - 경쟁사끼리 토픽 분포 견주기

  2. SEC 보고서 분석:
     - 10-K 보고서에서 위험 요인 토픽 추출
     - 경영진 논의 절의 토픽 변화 지켜보기
     - 규정 준수를 위해 이상한 토픽 분포 표시하기

  3. 뉴스 군집화:
     - 금융 뉴스를 토픽(거시, 업종, 기업)으로 군집화하기
     - 시간에 따른 토픽 주목도 추적(뜨는 토픽)
     - 토픽 기반 거래 신호 세우기

  4. 연구 보고서 분석:
     - 분석 대상 전체에 걸친 분석 토픽 요약하기
     - 다수 의견과 반대 의견 가려내기
     - 토픽 감성을 주가 움직임에 잇기

  보기: 토픽 기반 거래 신호
  ─────────────────────────────────────
  시점 t의 문서 d마다:
    1. 토픽 분포 θ_d = LDA(d)을 계산한다
    2. 토픽 k마다 토픽 감성 s_k을 계산한다
    3. 신호 = Σ_k θ_dk × s_k  (가중 감성)

  금융 말뭉치에서 흔히 떠오르는 토픽:
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

**출력:**

```
============================================================
Part 1: Latent Semantic Analysis (LSA)
============================================================
  Term-document matrix: (78, 12) (vocab × docs)

  Top words per topic (LSA, 3 topics):
    Topic 0: buyback(-0.397), program(-0.397), split(-0.397), stock(-0.397), amazon(-0.397)
    Topic 1: quarter(-0.372), expectations(-0.368), tesla(-0.368), deliveries(-0.368), exceeded(-0.368)
    Topic 2: year(-0.579), revenue(-0.307), microsoft(-0.289), grew(-0.289), cloud(-0.289)

  Document-topic matrix shape: (12, 3)
    Doc 0 → Topic 0: The Federal Reserve raised interest rates by 25 ba...
    Doc 1 → Topic 1: GDP growth slowed to 2.1 percent in the third quar...
    Doc 2 → Topic 2: Inflation remains above the central bank target of...
    Doc 3 → Topic 0: Treasury yields rose sharply after the employment ...

============================================================
Part 2: LDA with Gensim
============================================================

  from gensim.models import LdaModel
  from gensim.corpora import Dictionary
  from nltk.tokenize import word_tokenize
  from nltk.corpus import stopwords
  import nltk

  nltk.download('stopwords')
  stops = set(stopwords.words('english'))

  # 전처리: 토큰화, 소문자로, 불용어 없애기
  def preprocess(text):
      tokens = word_tokenize(text.lower())
      return [t for t in tokens if t.isalpha() and t not in stops]

  texts = [preprocess(doc) for doc in documents]

  # 사전과 말뭉치 만들기
  dictionary = Dictionary(texts)
  dictionary.filter_extremes(no_below=2, no_above=0.5)
  corpus = [dictionary.doc2bow(text) for text in texts]

  # LDA 트레인
  lda = LdaModel(
      corpus=corpus,
      id2word=dictionary.id2token,
      num_topics=5,
      iterations=400,
      passes=10,
      alpha='auto',         # 문서-토픽 사전분포 학습
      eta='auto',           # 토픽-단어 사전분포 학습
      random_state=42,
  )

  # 토픽 출력
  for idx in range(5):
      print(f"Topic {idx}: {lda.print_topic(idx, num_words=8)}")

  # 새 문서의 토픽 분포 얻기
  new_doc = preprocess("The central bank cut interest rates")
  bow = dictionary.doc2bow(new_doc)
  topic_dist = lda[bow]
  # → [(0, 0.72), (2, 0.15), (4, 0.13)]

============================================================
Part 3: LDA From Scratch (Gibbs Sampling)
============================================================
  LDA topics (from scratch):
    Topic 0: after(0.065), announced(0.065), cuts(0.033), sales(0.033), split(0.033)
    Topic 1: federal(0.041), s&p(0.041), apple(0.041), employment(0.041), sharply(0.041)
    Topic 2: percent(0.115), quarter(0.087), revenue(0.058), year(0.058), grew(0.029)

============================================================
Part 4: Neural Topic Model (ProdLDA / ETM)
============================================================
  Training Neural Topic Model:
    Epoch 25: recon=32.046, KL=0.022
    Epoch 50: recon=31.957, KL=0.041
    Epoch 75: recon=31.903, KL=0.044
    Epoch 100: recon=31.817, KL=0.048

  Neural topics:
    Topic 0: announced(0.64), raised(0.53), rates(0.46), stock(0.44), reserve(0.44)
    Topic 1: quarter(0.48), prices(0.44), bank(0.44), 500(0.42), percent(0.38)
    Topic 2: quarter(0.72), new(0.59), slowed(0.49), announced(0.47), estimates(0.43)

============================================================
Part 5: Financial Document Topic Analysis
============================================================

  토픽 모델링의 금융 쓰임새:

  1. 실적 발표 분석:
     - 분기별 토픽 변화 추적
     - 새 위험 요인이나 전략 전환 탐지
     - 경쟁사끼리 토픽 분포 견주기

  2. SEC 보고서 분석:
     - 10-K 보고서에서 위험 요인 토픽 추출
     - 경영진 논의 절의 토픽 변화 지켜보기
     - 규정 준수를 위해 이상한 토픽 분포 표시하기

  3. 뉴스 군집화:
     - 금융 뉴스를 토픽(거시, 업종, 기업)으로 군집화하기
     - 시간에 따른 토픽 주목도 추적(뜨는 토픽)
     - 토픽 기반 거래 신호 세우기

  4. 연구 보고서 분석:
     - 분석 대상 전체에 걸친 분석 토픽 요약하기
     - 다수 의견과 반대 의견 가려내기
     - 토픽 감성을 주가 움직임에 잇기

  보기: 토픽 기반 거래 신호
  ─────────────────────────────────────
  시점 t의 문서 d마다:
    1. 토픽 분포 θ_d = LDA(d)을 계산한다
    2. 토픽 k마다 토픽 감성 s_k을 계산한다
    3. 신호 = Σ_k θ_dk × s_k  (가중 감성)

  금융 말뭉치에서 흔히 떠오르는 토픽:
    - 거시/금리:     "fed", "rates", "inflation", "gdp"
    - 실적:        "revenue", "eps", "guidance", "beat"
    - 인수·합병:             "acquisition", "merger", "deal", "bid"
    - 위험/규제: "compliance", "fine", "investigation"
    - 기술:      "cloud", "ai", "platform", "growth"

Done.
```

## 2. LSA의 행렬과 부호

build_bow는 어휘×문서 빈도 행렬 X를 만든다. 소문자 공백 분리와 작은 불용어 목록을 사용하므로 구두점·숫자·표기 차이가 그대로 남을 수 있다. 코드의 TF는 문서 내 상대 빈도이고 IDF는 log[D/(1+df)]다. 이는 여러 가능한 정의 중 하나이며 모든 문서에 나온 단어는 음의 IDF도 가질 수 있다.

SVD로 X_tfidf≈U_k diag(S_k)V_kᵀ를 얻는다. 코드의 doc_topics=V_kᵀ의 전치에는 특잇값 배율이 없다. 흔히 쓰는 문서 좌표 V_k diag(S_k)와 다르므로 절댓값 argmax로 정한 토픽도 같은 결과를 보장하지 않는다. LSA 성분은 음수·양수 값을 가지며 확률 분포가 아니다.

특이벡터는 부호를 함께 뒤집어도 같은 복원을 만들 수 있다. 출력에서 세 성분의 상위 단어 값이 모두 음수인 것도 이 부호 선택의 결과일 뿐이다. 또 Doc 0(연준 금리 기사)이 상위 단어가 amazon·buyback인 Topic 0으로 배정된 것은 이 절댓값 argmax 규칙이 토픽 이름과 맞는 배정을 보장하지 않는다는 보기다. 절댓값이 큰 단어를 고르면 양·음 방향을 섞을 수 있으므로 하나의 일관된 의미 토픽이라고 단정하지 않는다. SVD 성분 수는 행렬 차원과 목적에 맞춰 정하고 복원·검색·해석 성능을 확인한다.

## 3. LDA와 깁스 샘플링

[LDA](https://www.jmlr.org/papers/v3/blei03a.html)는 문서별 토픽 혼합과 토픽별 단어 분포를 갖는 확률 모형이다. 위 코드의 LDAGibbs는 각 단어 출현의 토픽을 반복 갱신하는 붕괴 깁스 샘플링(collapsed Gibbs sampling) 방식이다. 현재 단어의 할당을 빈도에서 먼저 빼고 다음 비례식으로 샘플링한다.

$$P(z_i=k\mid z_{-i},w)\propto(n_{dk}^{-i}+\alpha)\frac{n_{kw}^{-i}+\beta}{n_k^{-i}+V\beta}.$$

문서 측 분모는 k에 공통이므로 정규화 과정에서 생략할 수 있다. α·β는 양수이고 토픽 수·어휘 크기·단어 ID 범위가 유효해야 한다. get_topic_words는 (n_kw+β)/(n_k+Vβ)의 상위 단어를 보여 준다. 토픽 ID는 순열을 바꾸어도 같은 모형이므로 시드별 번호를 그대로 의미상 대응시키지 않는다.

50회 반복했다고 수렴이 입증되지는 않는다. 여러 초기화·혼합·토픽 안정성을 검토해야 하며 현재 코드는 번인(burn-in, 초기 표본을 버리는 구간) 이후 여러 표본 평균이나 새 문서 추론을 구현하지 않는다. Gensim 부분은 실행 코드가 아니라 안내 문자열이다. 빈 문서·필터 후 빈 어휘와 새 문서의 미등록 단어 정책도 필요하다.

## 4. 뉴럴 토픽 모델의 목적

NeuralTopicModel은 문서 BoW를 인코더에 넣어 μ·logvar를 얻고 그 Gaussian에서 잠재 표현 z를 뽑는다. BatchNorm과 softmax로 토픽 웨이트 θ를 만든 뒤 선형 디코더의 log_softmax로 단어 분포를 얻는다. 로스는 빈도 가중 복원 로스인 음의 로그가능도(가능도는 likelihood, 우도라고도 한다)와 표준 정규분포에 대한 Gaussian KL의 합이다.

이는 LDA의 Dirichlet 혼합을 그대로 구현한 모델도, 코드 제목에 나온 ProdLDA·ETM의 완전한 재현도 아니다. 특히 ETM의 단어·토픽 임베딩 분해는 없다. 트레인 시 BatchNorm이 문서 간 통계를 공유하고 KL은 그 이전 z의 Gaussian에 적용되므로 해당 구성의 확률적 가정을 구분해서 읽는다.

θ는 합 1이지만 디코더의 개별 웨이트는 확률이 아니다. top 단어로 출력하는 값도 로짓 방향의 트레인 웨이트다. eval에서도 reparameterize는 난수를 계속 뽑으므로 forward가 자동으로 결정적이지 않다. BatchNorm의 트레인에는 충분한 배치가 필요하며 단일 문서 배치와 빈 데이터를 별도로 처리해야 한다.

## 5. 평가와 문서 활용

토픽 수·전처리·사전분포·초기화는 결과에 영향을 준다. 토픽 일관성(coherence)·안정성·사람의 해석 가능성과 독립 문서의 예측 평가를 함께 볼 수 있다. 높은 가능도가 사람이 읽기 좋은 토픽을 보장하지 않고, 단순 상위 단어 목록도 완성된 분류 정확도가 아니다.

현재 시연은 12개 예시 문서에 LSA·깁스 추정·뉴럴 모델 트레인을 적용하도록 작성되어 있으며 독립 시험 평가나 시기별 분석은 없다. 출력에 나온 금융 문장은 사실 확인용 데이터가 아니다. 토픽 가중 감성을 거래 신호로 쓰는 부분도 안내일 뿐 트레인·검증된 경제적 성과가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
LSA 성분·LDA 토픽 단어 분포·뉴럴 디코더 웨이트를 모두 같은 확률로 해석할 수 없는 이유를 설명하라.

</div>

??? success "풀이"

    LSA 특이벡터는 음·양 값을 가지며 부호도 바꿀 수 있다. LDA의 평활화된(smoothed) 토픽 단어 분포는 고정 토픽에서 어휘 전체 합이 1이다. 뉴럴 디코더 웨이트는 단어 로짓을 만드는 계수이고 그 자체로 정규화된 확률이 아니다. θ와 단어 softmax의 확률 역할도 구분한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
깁스 갱신에서 현재 단어의 빈도를 먼저 빼는 이유와 생략한 분모를 설명하라.

</div>

??? success "풀이"

    조건부 분포는 다른 토큰의 할당을 고정한 상태에서 현재 토큰의 새 토픽을 구한다. 현재 할당을 남기면 자기 관측을 중복 반영한다. 문서 측 분모 n_d^−i+Kα는 k에 공통이므로 비례확률 정규화에서 생략할 수 있지만 토픽별 n_k^−i+Vβ는 남겨야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
LSA 문서 좌표에서 특잇값을 포함하는 방식과 현재 코드의 차이를 설명하라.

</div>

??? success "풀이"

    코드는 V_k를 문서 표현으로 사용하고 특잇값의 상대 크기를 반영하지 않는다. V_k diag(S_k)는 복원의 성분별 스케일을 포함한다. 따라서 두 표현에서 성분 크기·거리·argmax가 같다고 가정하지 않는다. 어느 표현을 쓰는지 명시하고 해당 목적에서 평가한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 예가 LDAGibbs의 깊이를 늘리는 유효한 구현인지 검토하고 뉴럴 모델과 구분하라.

</div>

??? success "풀이"

    LDAGibbs는 nn.Module이나 레이어 기반 네트워크가 아니라 빈도·토픽 할당의 샘플러다. num_layers·YourBlock·dim으로 레이어를 추가하는 아래 코드는 적용되지 않는 자리표시자다. 깁스의 반복 횟수·토픽 수와 뉴럴 모델의 인코더 깊이는 서로 다른 설정이다. 토픽 수를 바꾸면 결과 토픽의 의미·순열 대응과 안정성을 다시 검토해야 한다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

LSA의 저차원 성분, LDA의 확률 분포, 뉴럴 모델의 잠재 변수와 디코더 웨이트는 서로 다르다. 행렬 방향·스케일·샘플링·평가 조건을 맞추고 작은 데이터의 토픽 해석을 검증된 사실이나 성과로 확대하지 않는다.
