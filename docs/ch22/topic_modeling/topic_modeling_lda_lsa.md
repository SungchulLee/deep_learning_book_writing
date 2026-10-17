# 토픽 모델링: LSA와 LDA

토픽 모델링은 정답 주제 레이블 없이 문서의 단어 사용 패턴을 요약하는 방법이다. LSA는 행렬의 저차원 구조, LDA는 문서별 토픽 혼합과 토픽별 단어 분포를 추정한다. 뉴럴 토픽 모델은 또 다른 잠재 표현·학습 목적을 사용한다. 사람이 붙인 토픽 이름은 해석이며 모델이 자동으로 얻은 확정적인 의미 분류가 아니다.

## 1. 코드와 기존 출력

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
    #   4부 – 뉴럴 주제 모델(PyTorch 변분 오토인코더 바탕)
    #   5부 – 금융 글월의 주제 나타내기
    #
    # 바탕: O'Reilly "Practical NLP" 7장

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from collections import Counter
    from typing import List, Dict, Tuple
    # 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
    # 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
    torch.manual_seed(0)


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
    # 4부 – 뉴럴 주제 모델(PyTorch 변분 오토인코더 바탕)
    # =====================================================================
    print("=" * 60)
    print("Part 4: Neural Topic Model (ProdLDA / ETM)")
    print("=" * 60)

    # 뉴럴 주제 모델은 다음과 같은 변분 오토인코더를 쓴다:
    #   - 인코더: 낱말 자루 → 주제 분포로 대응시킨다(파라메터 바꾸기 재주로)
    #   - 디코더: 주제 분포에서 낱말 자루를 되살린다
    #   - 로스 = 되살림 + KL 벌어짐
    #
    # ProdLDA는 디리클레 대신 로지스틱 정규 분포를 쓴다
    # 그래디언트 바탕 가장 좋게 하기를 쉽게 하려고.


    class NeuralTopicModel(nn.Module):
        """ProdLDA 방식 뉴럴 주제 모델.

        로지스틱 정규 앞확률을 쓴 변분 오토인코더로 주제를 배운다.
        디코더 무게 행렬의 줄이 바로 주제이다.
        """

        def __init__(self, vocab_size: int, n_topics: int, hidden_dim: int = 64):
            super().__init__()
            # 인코더: 낱말 자루 → 숨은 레이어 → (mu, logvar)
            self.encoder = nn.Sequential(
                nn.Linear(vocab_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.mu_layer = nn.Linear(hidden_dim, n_topics)
            self.logvar_layer = nn.Linear(hidden_dim, n_topics)

            # 디코더: 주제 비율 → 낱말 자루 되살림
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
        """뉴럴 주제 모델 익히기."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        X = torch.tensor(bow_matrix, dtype=torch.float32)

        model.train()
        for epoch in range(n_epochs):
            recon, mu, logvar, theta = model(X)

            # 되살림 로스(음의 로그 가능도)
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


    # 뉴럴 모델을 위한 낱말 자루 행렬 세우기
    bow_matrix = X.T  # (D × V)

    n_topics = 3
    ntm = NeuralTopicModel(len(vocab_list), n_topics, hidden_dim=32)
    print("  Training Neural Topic Model:")
    ntm = train_ntm(ntm, bow_matrix, n_epochs=100, lr=2e-3)

    # 디코더 무게에서 주제 뽑기
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

      3. 뉴스 클래스 짓기:
         - 금융 뉴스를 주제(거시, 업종, 기업)로 무리 짓기
         - 때에 따른 주제 어텐션 좇기(뜨는 주제)
         - 주제 바탕 거래 신호 세우기

      4. 연구 보고서 살피기:
         - 분석 대상 전체에 걸친 분석 주제 간추리기
         - 다수 의견과 반대 의견 가려내기
         - 주제 마음결을 값 액션에 잇기

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

**기존 출력:**

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

      주제 나타내기의 금융 쓰임새:

      1. 실적 발표 살피기:
         - 분기별 주제 바뀜 좇기
         - 새 위험 요인이나 전략 전환 알아채기
         - 경쟁사끼리 주제 분포 견주기

      2. SEC 보고서 살피기:
         - 10-K 보고서에서 위험 요인 주제 뽑기
         - 경영진 논의 절의 주제 흘러감 지켜보기
         - 규정 지킴을 위해 이상한 주제 분포 표시하기

      3. 뉴스 클래스 짓기:
         - 금융 뉴스를 주제(거시, 업종, 기업)로 무리 짓기
         - 때에 따른 주제 어텐션 좇기(뜨는 주제)
         - 주제 바탕 거래 신호 세우기

      4. 연구 보고서 살피기:
         - 분석 대상 전체에 걸친 분석 주제 간추리기
         - 다수 의견과 반대 의견 가려내기
         - 주제 마음결을 값 액션에 잇기

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

    Done.
    ```

## 2. LSA의 행렬과 부호

build_bow는 어휘×문서 빈도 행렬 X를 만든다. 소문자 공백 분리와 작은 불용어 목록을 사용하므로 구두점·숫자·표기 차이가 그대로 남을 수 있다. 코드의 TF는 문서 내 상대 빈도이고 IDF는 log[D/(1+df)]다. 이는 여러 가능한 정의 중 하나이며 모든 문서에 나온 단어는 음의 IDF도 가질 수 있다.

SVD로 X_tfidf≈U_k diag(S_k)V_kᵀ를 얻는다. 코드의 doc_topics=V_kᵀ의 전치에는 특이값 배율이 없다. 흔히 쓰는 문서 좌표 V_k diag(S_k)와 다르므로 절댓값 argmax로 정한 주제도 같은 결과를 보장하지 않는다. LSA 성분은 음수·양수 값을 가지며 확률 분포가 아니다.

특이벡터는 부호를 함께 뒤집어도 같은 재구성을 만들 수 있다. 절댓값이 큰 단어를 고르면 양·음 방향을 섞을 수 있으므로 하나의 일관된 의미 주제라고 단정하지 않는다. SVD 성분 수는 행렬 차원과 목적에 맞춰 정하고 재구성·검색·해석 성능을 확인한다.

## 3. LDA와 깁스 샘플링

[LDA](https://www.jmlr.org/papers/v3/blei03a.html)는 문서별 토픽 혼합과 토픽별 단어 분포를 갖는 확률 모형이다. 아래 LDAGibbs는 각 단어 출현의 토픽을 반복 갱신하는 collapsed Gibbs 방식이다. 현재 단어의 할당을 빈도에서 먼저 빼고 다음 비례식으로 샘플링한다.

$$P(z_i=k\mid z_{-i},w)\propto(n_{dk}^{-i}+\alpha)\frac{n_{kw}^{-i}+\beta}{n_k^{-i}+V\beta}.$$

문서 측 분모는 k에 공통이므로 정규화 과정에서 생략할 수 있다. α·β는 양수이고 토픽 수·어휘 크기·단어 ID 범위가 유효해야 한다. get_topic_words는 (n_kw+β)/(n_k+Vβ)의 상위 단어를 보여 준다. 토픽 ID는 순열을 바꾸어도 같은 모형이므로 시드별 번호를 그대로 의미상 대응시키지 않는다.

50회 반복했다고 수렴이 입증되지는 않는다. 여러 초기화·혼합·토픽 안정성을 검토해야 하며 현재 코드는 burn-in 이후 여러 표본 평균이나 새 문서 추론을 구현하지 않는다. Gensim 부분은 실행 코드가 아니라 안내 문자열이다. 빈 문서·필터 후 빈 어휘와 새 문서의 미등록 단어 폴리시도 필요하다.

## 4. 뉴럴 토픽 모델의 목적

NeuralTopicModel은 문서 BoW를 encoder에 넣어 μ·logvar를 얻고 Gaussian 표본 z를 만든다. BatchNorm과 softmax로 토픽 웨이트 θ를 만든 뒤 선형 decoder의 log_softmax로 단어 분포를 얻는다. 로스는 빈도 가중 재구성 음의 로그우도와 표준 정규분포에 대한 Gaussian KL의 합이다.

이는 LDA의 Dirichlet 혼합을 그대로 구현한 모델도, 코드 제목에 나온 ProdLDA·ETM의 완전한 재현도 아니다. 특히 ETM의 단어·토픽 임베딩 분해는 없다. 학습 시 BatchNorm이 문서 간 통계를 공유하고 KL은 그 이전 z의 Gaussian에 적용되므로 해당 구성의 확률적 가정을 구분해서 읽는다.

θ는 합 1이지만 decoder의 개별 웨이트는 확률이 아니다. top 단어로 출력하는 값도 로짓 방향의 학습 웨이트다. eval에서도 reparameterize는 난수를 계속 뽑으므로 forward가 자동으로 결정적이지 않다. BatchNorm의 학습에는 충분한 배치가 필요하며 단일 문서 배치와 빈 자료를 별도로 처리해야 한다.

## 5. 평가와 문서 활용

토픽 수·전처리·사전분포·초기화는 결과에 영향을 준다. 토픽 coherence·안정성·사람의 해석 가능성과 독립 문서의 예측 평가를 함께 볼 수 있다. 높은 likelihood가 사람이 읽기 좋은 토픽을 보장하지 않고, 단순 상위 단어 목록도 완성된 분류 정확도가 아니다.

현재 시연은 12개 예시 문서에 LSA·깁스 추정·뉴럴 모델 학습을 적용하도록 작성되어 있으며 독립 시험 평가나 시기별 분석은 없다. 출력에 나온 금융 문장은 사실 확인용 자료가 아니다. 토픽 가중 감성을 거래 신호로 쓰는 부분도 안내일 뿐 학습·검증된 경제적 성과가 아니다. 이 편집에서는 코드를 실행하거나 그림을 생성하지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
LSA 성분·LDA 토픽 단어 분포·뉴럴 decoder 웨이트를 모두 같은 확률로 해석할 수 없는 이유를 설명하라.

</div>

??? success "풀이"

    LSA 특이벡터는 음·양 값을 가지며 부호도 바꿀 수 있다. LDA의 smoothed 토픽 단어 분포는 고정 토픽에서 어휘 전체 합이 1이다. 뉴럴 decoder 웨이트는 단어 로짓을 만드는 계수이고 그 자체로 정규화된 확률이 아니다. θ와 단어 softmax의 확률 역할도 구분한다.



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
LSA 문서 좌표에서 특이값을 포함하는 방식과 현재 코드의 차이를 설명하라.

</div>

??? success "풀이"

    코드는 V_k를 문서 표현으로 사용하고 특이값의 상대 크기를 반영하지 않는다. V_k diag(S_k)는 재구성의 성분별 스케일을 포함한다. 따라서 두 표현에서 성분 크기·거리·argmax가 같다고 가정하지 않는다. 어느 표현을 쓰는지 명시하고 해당 목적에서 평가한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 예가 LDAGibbs의 깊이를 늘리는 유효한 구현인지 검토하고 뉴럴 모델과 구분하라.

</div>

??? success "풀이"

    LDAGibbs는 nn.Module이나 레이어 기반 네트워크가 아니라 빈도·토픽 할당의 샘플러다. num_layers·YourBlock·dim으로 레이어를 추가하는 아래 코드는 적용되지 않는 자리표시자다. 깁스의 반복 횟수·토픽 수와 뉴럴 모델의 encoder 깊이는 서로 다른 설정이다. 토픽 수를 바꾸면 결과 토픽의 의미·순열 대응과 안정성을 다시 검토해야 한다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

LSA의 저차원 성분, LDA의 확률 분포, 뉴럴 모델의 잠재 변수와 decoder 웨이트는 서로 다르다. 행렬 방향·스케일·샘플링·평가 조건을 맞추고 작은 자료의 토픽 해석을 검증된 사실이나 성과로 확대하지 않는다.
