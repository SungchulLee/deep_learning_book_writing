"""Topic modeling lda lsa."""
# ---
# title: "Topic Modeling: LDA & LSA"
# description: "Unsupervised topic discovery with Latent Dirichlet Allocation
#               and Latent Semantic Analysis — pure Python + PyTorch implementations"
# ---
#
# Topic modeling discovers latent themes in document collections without
# labeled data.  Two foundational approaches:
#
#   - LSA (Latent Semantic Analysis): SVD on the term-document matrix
#   - LDA (Latent Dirichlet Allocation): Bayesian generative model
#
#   Part 1 – LSA via truncated SVD (numpy/sklearn)
#   Part 2 – LDA with gensim
#   Part 3 – LDA from scratch (collapsed Gibbs sampling)
#   Part 4 – Neural Topic Model (PyTorch VAE-based)
#   Part 5 – Topic modeling for financial documents
#
# Adapted from: O'Reilly "Practical NLP" Ch 7

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from typing import List, Dict, Tuple


# =====================================================================
# Part 1 – LSA via Truncated SVD
# =====================================================================
print("=" * 60)
print("Part 1: Latent Semantic Analysis (LSA)")
print("=" * 60)

# LSA = apply SVD to the term-document matrix.
# The truncated SVD captures the most important latent dimensions,
# which correspond to abstract "topics".
#
# Math:  X ≈ U_k Σ_k V_k^T
#   X:     term-document matrix (V × D)
#   U_k:   term-topic matrix (V × k)
#   Σ_k:   topic strengths (k × k diagonal)
#   V_k^T: topic-document matrix (k × D)

# Sample documents
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

# Build term-document matrix (simple BoW)
stopwords = {"the", "a", "an", "to", "of", "in", "by", "and", "for", "is", "was", "this"}


def build_bow(docs: List[str], stopwords: set) -> Tuple[np.ndarray, List[str]]:
    """Build a bag-of-words term-document matrix."""
    # Build vocabulary
    vocab = {}
    for doc in docs:
        for word in doc.lower().split():
            if word not in stopwords and len(word) > 2:
                if word not in vocab:
                    vocab[word] = len(vocab)

    vocab_list = sorted(vocab.keys(), key=lambda w: vocab[w])

    # Build matrix
    X = np.zeros((len(vocab), len(docs)))
    for j, doc in enumerate(docs):
        for word in doc.lower().split():
            if word in vocab:
                X[vocab[word], j] += 1

    return X, vocab_list


X, vocab_list = build_bow(documents, stopwords)
print(f"  Term-document matrix: {X.shape} (vocab × docs)")

# Apply TF-IDF weighting
tf = X / X.sum(axis=0, keepdims=True).clip(min=1)
idf = np.log(X.shape[1] / (1 + (X > 0).sum(axis=1, keepdims=True)))
X_tfidf = tf * idf

# Truncated SVD
n_topics = 3
U, S, Vt = np.linalg.svd(X_tfidf, full_matrices=False)
U_k = U[:, :n_topics]
S_k = S[:n_topics]
Vt_k = Vt[:n_topics, :]

print(f"\n  Top words per topic (LSA, {n_topics} topics):")
for topic_idx in range(n_topics):
    # Top words = highest absolute values in U_k[:, topic_idx]
    top_word_idx = np.argsort(np.abs(U_k[:, topic_idx]))[-5:][::-1]
    words = [(vocab_list[i], U_k[i, topic_idx]) for i in top_word_idx]
    word_str = ", ".join(f"{w}({v:.3f})" for w, v in words)
    print(f"    Topic {topic_idx}: {word_str}")

# Document-topic assignments
doc_topics = Vt_k.T  # (D × k)
print(f"\n  Document-topic matrix shape: {doc_topics.shape}")
for i, doc in enumerate(documents[:4]):
    topic = np.argmax(np.abs(doc_topics[i]))
    print(f"    Doc {i} → Topic {topic}: {doc[:50]}...")
print()


# =====================================================================
# Part 2 – LDA with Gensim
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

  # Preprocess: tokenize, lowercase, remove stopwords
  def preprocess(text):
      tokens = word_tokenize(text.lower())
      return [t for t in tokens if t.isalpha() and t not in stops]

  texts = [preprocess(doc) for doc in documents]

  # Create dictionary and corpus
  dictionary = Dictionary(texts)
  dictionary.filter_extremes(no_below=2, no_above=0.5)
  corpus = [dictionary.doc2bow(text) for text in texts]

  # Train LDA
  lda = LdaModel(
      corpus=corpus,
      id2word=dictionary.id2token,
      num_topics=5,
      iterations=400,
      passes=10,
      alpha='auto',         # learn doc-topic prior
      eta='auto',           # learn topic-word prior
      random_state=42,
  )

  # Print topics
  for idx in range(5):
      print(f"Topic {idx}: {lda.print_topic(idx, num_words=8)}")

  # Get topic distribution for a new document
  new_doc = preprocess("The central bank cut interest rates")
  bow = dictionary.doc2bow(new_doc)
  topic_dist = lda[bow]
  # → [(0, 0.72), (2, 0.15), (4, 0.13)]
""")


# =====================================================================
# Part 3 – LDA From Scratch (Collapsed Gibbs Sampling)
# =====================================================================
print("=" * 60)
print("Part 3: LDA From Scratch (Gibbs Sampling)")
print("=" * 60)

# LDA generative process:
#   For each document d:
#     Draw topic distribution θ_d ~ Dirichlet(α)
#     For each word position i in d:
#       Draw topic z_{d,i} ~ Categorical(θ_d)
#       Draw word w_{d,i} ~ Categorical(φ_{z_{d,i}})
#
# Inference via collapsed Gibbs sampling:
#   P(z_i = k | z_{-i}, w) ∝ (n_{d,k} + α) × (n_{k,w} + β) / (n_{k,·} + Vβ)


class LDAGibbs:
    """LDA via collapsed Gibbs sampling.

    Args:
        n_topics: Number of topics
        alpha:    Dirichlet prior for document-topic distribution
        beta:     Dirichlet prior for topic-word distribution
        n_iter:   Number of Gibbs sampling iterations
    """

    def __init__(self, n_topics: int = 5, alpha: float = 0.1,
                 beta: float = 0.01, n_iter: int = 100):
        self.K = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter

    def fit(self, documents: List[List[int]], vocab_size: int):
        """Fit LDA model using Gibbs sampling.

        Args:
            documents: List of documents, each a list of word indices
            vocab_size: Size of vocabulary
        """
        self.V = vocab_size
        D = len(documents)
        K = self.K

        # Count matrices
        self.n_dk = np.zeros((D, K))       # doc-topic counts
        self.n_kv = np.zeros((K, self.V))  # topic-word counts
        self.n_k = np.zeros(K)             # topic totals

        # Initialize: random topic assignment for each word
        self.z = []  # topic assignments
        for d, doc in enumerate(documents):
            doc_z = []
            for w in doc:
                k = np.random.randint(K)
                doc_z.append(k)
                self.n_dk[d, k] += 1
                self.n_kv[k, w] += 1
                self.n_k[k] += 1
            self.z.append(doc_z)

        # Gibbs sampling iterations
        for iteration in range(self.n_iter):
            for d, doc in enumerate(documents):
                for i, w in enumerate(doc):
                    k_old = self.z[d][i]

                    # Remove current assignment
                    self.n_dk[d, k_old] -= 1
                    self.n_kv[k_old, w] -= 1
                    self.n_k[k_old] -= 1

                    # Compute conditional: P(z_i=k | rest)
                    p = (self.n_dk[d] + self.alpha) * \
                        (self.n_kv[:, w] + self.beta) / \
                        (self.n_k + self.V * self.beta)
                    p = p / p.sum()

                    # Sample new topic
                    k_new = np.random.choice(K, p=p)
                    self.z[d][i] = k_new

                    # Update counts
                    self.n_dk[d, k_new] += 1
                    self.n_kv[k_new, w] += 1
                    self.n_k[k_new] += 1

        return self

    def get_topic_words(self, n_words: int = 10) -> List[List[Tuple[int, float]]]:
        """Get top words for each topic."""
        topics = []
        for k in range(self.K):
            phi_k = (self.n_kv[k] + self.beta) / (self.n_k[k] + self.V * self.beta)
            top_idx = np.argsort(phi_k)[-n_words:][::-1]
            topics.append([(idx, phi_k[idx]) for idx in top_idx])
        return topics


# Prepare data
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

# Fit LDA
np.random.seed(42)
lda = LDAGibbs(n_topics=3, alpha=0.1, beta=0.01, n_iter=50)
lda.fit(tokenized_docs, len(word2id))

print("  LDA topics (from scratch):")
for k, topic_words in enumerate(lda.get_topic_words(n_words=5)):
    words = ", ".join(f"{id2word[idx]}({prob:.3f})" for idx, prob in topic_words)
    print(f"    Topic {k}: {words}")
print()


# =====================================================================
# Part 4 – Neural Topic Model (PyTorch VAE-based)
# =====================================================================
print("=" * 60)
print("Part 4: Neural Topic Model (ProdLDA / ETM)")
print("=" * 60)

# Neural topic models use variational autoencoders where:
#   - Encoder: maps BoW → topic distribution (via reparameterization)
#   - Decoder: reconstructs BoW from topic distribution
#   - Loss = reconstruction + KL divergence
#
# ProdLDA uses a logistic-normal distribution instead of Dirichlet
# for easier gradient-based optimization.


class NeuralTopicModel(nn.Module):
    """ProdLDA-style neural topic model.

    Uses a VAE with logistic-normal prior to learn topics.
    The decoder weight matrix rows ARE the topics.
    """

    def __init__(self, vocab_size: int, n_topics: int, hidden_dim: int = 64):
        super().__init__()
        # Encoder: BoW → hidden → (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.mu_layer = nn.Linear(hidden_dim, n_topics)
        self.logvar_layer = nn.Linear(hidden_dim, n_topics)

        # Decoder: topic proportions → BoW reconstruction
        self.decoder = nn.Linear(n_topics, vocab_size, bias=False)
        # decoder.weight: (vocab_size × n_topics)
        # Each column = a topic's word distribution

        self.bn = nn.BatchNorm1d(n_topics, affine=False)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Softmax over topics, then reconstruct
        theta = F.softmax(self.bn(z), dim=-1)
        return F.log_softmax(self.decoder(theta), dim=-1), theta

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon, theta = self.decode(z)
        return recon, mu, logvar, theta


def train_ntm(model, bow_matrix, n_epochs=100, lr=2e-3):
    """Train neural topic model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(bow_matrix, dtype=torch.float32)

    model.train()
    for epoch in range(n_epochs):
        recon, mu, logvar, theta = model(X)

        # Reconstruction loss (negative log-likelihood)
        recon_loss = -(X * recon).sum(dim=-1).mean()

        # KL divergence (logistic-normal vs standard normal)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        loss = recon_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}: recon={recon_loss:.3f}, KL={kl_loss:.3f}")

    return model


# Build BoW matrix for neural model
bow_matrix = X.T  # (D × V)

n_topics = 3
ntm = NeuralTopicModel(len(vocab_list), n_topics, hidden_dim=32)
print("  Training Neural Topic Model:")
ntm = train_ntm(ntm, bow_matrix, n_epochs=100, lr=2e-3)

# Extract topics from decoder weights
ntm.eval()
topic_weights = ntm.decoder.weight.data.numpy()  # (V × K)
print(f"\n  Neural topics:")
for k in range(n_topics):
    top_idx = np.argsort(topic_weights[:, k])[-5:][::-1]
    words = ", ".join(f"{vocab_list[i]}({topic_weights[i, k]:.2f})" for i in top_idx)
    print(f"    Topic {k}: {words}")
print()


# =====================================================================
# Part 5 – Topic Modeling for Financial Documents
# =====================================================================
print("=" * 60)
print("Part 5: Financial Document Topic Analysis")
print("=" * 60)

print("""
  Financial applications of topic modeling:

  1. Earnings call analysis:
     - Track topic shifts across quarters
     - Detect new risk factors or strategic pivots
     - Compare topic distributions across competitors

  2. SEC filing analysis:
     - Extract risk factor themes from 10-K filings
     - Monitor topic drift in MD&A sections
     - Flag unusual topic distributions for compliance

  3. News categorization:
     - Cluster financial news by theme (macro, sector, company)
     - Track topic attention over time (trending themes)
     - Build topic-based trading signals

  4. Research report analysis:
     - Summarize analyst themes across coverage universe
     - Identify consensus vs contrarian viewpoints
     - Link topic sentiment to price movements

  Example: Topic-based trading signal
  ─────────────────────────────────────
  For each document d at time t:
    1. Compute topic distribution θ_d = LDA(d)
    2. Compute topic sentiment s_k for each topic k
    3. Signal = Σ_k θ_dk × s_k  (weighted sentiment)

  Topics that typically emerge from financial corpora:
    - Macro/rates:     "fed", "rates", "inflation", "gdp"
    - Earnings:        "revenue", "eps", "guidance", "beat"
    - M&A:             "acquisition", "merger", "deal", "bid"
    - Risk/regulation: "compliance", "fine", "investigation"
    - Technology:      "cloud", "ai", "platform", "growth"
""")

print("Done.")


if __name__ == "__main__":
    pass
