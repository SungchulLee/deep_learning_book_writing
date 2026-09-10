# 심화 실습

심화 실습: 임베딩 시각화. 예상 시간 35분

순차열 모형은 시간적이고 순서가 있는 데이터를 다루는 데 바탕이 된다. 이 구현은 낱말 임베딩의 핵심 착상을 다루며, 순환 계산과 학습된 표현이 시각 사이의 의존을 어떻게 붙잡는지 보인다.

## 1. 코드

```python
# ========================================================
# 04_embedding_visualization.py
# t-SNE를 쓰는 심화 임베딩 시각화
# ========================================================

"""
심화 실습: 임베딩 시각화

학습 목표:
- 고차원 임베딩을 2차원에 그리기
- 차원 축소에 PCA와 t-SNE 쓰기
- 임베딩의 무리 분석하기
- 의미 관계를 눈으로 이해하기

예상 시간: 35분

먼저 알아야 할 것:
- 중급 실습 마치기
- 임베딩이 무엇을 나타내는지 이해하기
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.data_loader import (
    load_text_from_file,
    simple_tokenize,
    build_vocabulary,
    create_cbow_dataset
)

print("=" * 70)
print("ADVANCED TUTORIAL: Embedding Visualization")
print("=" * 70)

# ========================================================
# 1부: CBOW 모델 학습시키기
# ========================================================

print("\n" + "=" * 70)
print("PART 1: Training CBOW Model for Visualization")
print("=" * 70)

# 데이터 불러와 준비하기
text = load_text_from_file('../data/sample_text.txt')
tokens = simple_tokenize(text, lowercase=True)
word_to_ix, ix_to_word = build_vocabulary(tokens, min_freq=1)
vocab_size = len(word_to_ix)

print(f"Vocabulary size: {vocab_size}")

# CBOW 데이터셋 만들기
WINDOW_SIZE = 2
cbow_data = create_cbow_dataset(tokens, WINDOW_SIZE, word_to_ix)


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context):
        embeds = torch.mean(self.embeddings(context), dim=1)
        return self.linear(embeds)


# 모델을 학습시킨다
EMBEDDING_DIM = 30  # 임베딩을 좋게 하려고 차원을 높임
model = CBOWModel(vocab_size, EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

print(f"\nTraining CBOW model with {EMBEDDING_DIM}D embeddings...")

EPOCHS = 150
BATCH_SIZE = 32

for epoch in range(EPOCHS):
    total_loss = 0
    for i in range(0, len(cbow_data), BATCH_SIZE):
        batch = cbow_data[i:i+BATCH_SIZE]
        if len(batch) == 0:
            continue
        
        contexts = torch.stack([item[0] for item in batch])
        targets = torch.cat([item[1] for item in batch])
        
        optimizer.zero_grad()
        outputs = model(contexts)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 30 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(cbow_data)*BATCH_SIZE:.4f}")

print("Training complete!\n")

# ========================================================
# 2부: 임베딩 뽑아내기
# ========================================================

print("=" * 70)
print("PART 2: Extracting Embeddings")
print("=" * 70)

# 모든 낱말 임베딩 얻기
embeddings = model.embeddings.weight.detach().cpu().numpy()
words = list(word_to_ix.keys())

print(f"\nEmbedding matrix shape: {embeddings.shape}")
print(f"  {len(words)} words × {EMBEDDING_DIM} dimensions")

# ========================================================
# 3부: PCA 시각화
# ========================================================

print("\n" + "=" * 70)
print("PART 3: PCA (Principal Component Analysis)")
print("=" * 70)

print("\nPCA finds the directions of maximum variance...")
print("Projects high-dimensional data to 2D")

# PCA 적용
pca = PCA(n_components=2, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)

print(f"\nExplained variance ratio:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"  Total: {pca.explained_variance_ratio_.sum():.4f}")

# PCA 그리기
fig, ax = plt.subplots(figsize=(14, 10))

# 모든 점 그리기
ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
          alpha=0.5, s=30, c='lightblue', edgecolors='black', linewidth=0.5)

# 모든 낱말에 이름 붙이기
for i, word in enumerate(words):
    ax.annotate(word, 
               (embeddings_pca[i, 0], embeddings_pca[i, 1]),
               fontsize=8, alpha=0.8,
               xytext=(2, 2), textcoords='offset points')

ax.set_xlabel('First Principal Component', fontsize=12)
ax.set_ylabel('Second Principal Component', fontsize=12)
ax.set_title(f'Word Embeddings Visualization (PCA)\n{vocab_size} words from Shakespeare/Poetry corpus', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========================================================
# 4부: t-SNE 시각화
# ========================================================

print("\n" + "=" * 70)
print("PART 4: t-SNE (t-Distributed Stochastic Neighbor Embedding)")
print("=" * 70)

print("\nt-SNE is better at preserving local structure...")
print("Groups similar words together more clearly")
print("(This may take a moment...)\n")

# t-SNE 적용
perplexity = min(30, len(words) - 1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
           n_iter=1000, learning_rate=200)
embeddings_tsne = tsne.fit_transform(embeddings)

print("t-SNE complete!\n")

# t-SNE 그리기
fig, ax = plt.subplots(figsize=(14, 10))

# 낱말의 길이로 색 입히기 (보기 좋으라고)
word_lengths = [len(w) for w in words]
scatter = ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                    c=word_lengths, cmap='viridis', 
                    alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

# 낱말에 이름 붙이기
for i, word in enumerate(words):
    ax.annotate(word,
               (embeddings_tsne[i, 0], embeddings_tsne[i, 1]),
               fontsize=9, alpha=0.85, fontweight='bold',
               xytext=(3, 3), textcoords='offset points')

ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.set_title(f'Word Embeddings Visualization (t-SNE)\nColors indicate word length', 
            fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Word Length', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========================================================
# 5부: PCA와 t-SNE 견주기
# ========================================================

print("\n" + "=" * 70)
print("PART 5: PCA vs t-SNE Comparison")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# PCA 그림
ax1.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
           alpha=0.6, s=40, c='blue')
for i, word in enumerate(words[:50]):  # 보기 좋게 앞의 50개만 보임
    ax1.annotate(word, (embeddings_pca[i, 0], embeddings_pca[i, 1]),
                fontsize=8, alpha=0.7)
ax1.set_xlabel('PC1', fontsize=11)
ax1.set_ylabel('PC2', fontsize=11)
ax1.set_title('PCA: Linear dimensionality reduction', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# t-SNE 그림
ax2.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
           alpha=0.6, s=40, c='red')
for i, word in enumerate(words[:50]):  # 보기 좋게 앞의 50개만 보임
    ax2.annotate(word, (embeddings_tsne[i, 0], embeddings_tsne[i, 1]),
                fontsize=8, alpha=0.7)
ax2.set_xlabel('t-SNE Dim 1', fontsize=11)
ax2.set_ylabel('t-SNE Dim 2', fontsize=11)
ax2.set_title('t-SNE: Non-linear, preserves local structure', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================================================
# 6부: 의미 분석
# ========================================================

print("\n" + "=" * 70)
print("PART 6: Analyzing Semantic Clusters")
print("=" * 70)

# t-SNE 공간에서 비슷한 낱말의 무리 찾기
print("\nLooking for semantic clusters in the visualization...")

# t-SNE 공간에서 쌍마다의 거리 계산
from scipy.spatial.distance import cdist
distances_tsne = cdist(embeddings_tsne, embeddings_tsne, 'euclidean')

# 낱말마다 t-SNE 공간에서 가장 가까운 이웃 찾기
print("\nNearest neighbors in embedding space (t-SNE visualization):")
interesting_words = ["beauty", "thy", "love", "shall", "eyes"]

for word in interesting_words:
    if word not in word_to_ix:
        continue
    
    idx = word_to_ix[word]
    # 가장 가까운 이웃 얻기 (자기 자신은 뺀다)
    distances_from_word = distances_tsne[idx].copy()
    distances_from_word[idx] = float('inf')
    nearest_indices = np.argsort(distances_from_word)[:5]
    
    print(f"\n{word}:")
    for i, near_idx in enumerate(nearest_indices):
        near_word = ix_to_word[near_idx]
        dist = distances_tsne[idx, near_idx]
        print(f"  {i+1}. {near_word} (distance: {dist:.2f})")

# ========================================================
# 7부: 대화식 분석 요령
# ========================================================

print("\n" + "=" * 70)
print("PART 7: Interpretation Guide")
print("=" * 70)

print("""
그림을 해석하는 법:

PCA (주성분 분석):
-----------------------------------
✓ 선형 변환이다
✓ 전체적인 구조를 지킨다
✓ 설명된 분산이 정보를 얼마나 지켰는지 알려 준다
✓ 축에 수학적인 뜻이 있다 (주성분)
✗ 복잡한 관계는 담지 못할 수 있다

t-SNE (t-분포 확률적 이웃 임베딩):
---------------------------------------------------
✓ 비선형 변환이다
✓ 지역적인 구조를 아주 잘 지킨다
✓ 뚜렷한 무리를 만든다
✓ 비슷한 낱말이 가까이 나타난다
✗ 무리 사이의 거리는 뜻이 없다
✗ 돌릴 때마다 배치가 달라질 수 있다
✗ 계산이 비싸다

무엇을 볼 것인가:
----------------
1. 무리: 관련된 낱말의 모임
2. 이상점: 드물거나 특이한 낱말
3. 가까움: 비슷한 낱말은 가까워야 한다
4. 무늬: 의미적이거나 문법적인 묶임

해석의 예:
----------------------
- 대명사(thy, thine, thee)가 함께 모일 수 있다
- 추상적인 개념(beauty, truth, love)이 서로 가까울 수 있다
- 흔한 낱말과 드문 낱말이 서로 다른 무늬를 보인다
""")

# ========================================================
# 핵심 요점
# ========================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. 차원 축소:
   - 고차원 임베딩을 그리려면 꼭 필요하다
   - PCA: 빠르고 선형이며 전체 구조를 지킨다
   - t-SNE: 느리고 비선형이며 무리를 더 잘 만든다

2. 시각화에서 얻는 통찰:
   - 비슷한 낱말이 함께 모인다
   - 의미 관계가 눈에 보인다
   - 임베딩의 품질을 눈으로 확인할 수 있다

3. 실전에서의 쓰임:
   - 임베딩의 품질을 살펴 문제 찾기
   - 모델이 무엇을 배웠는지 이해하기
   - 기술을 모르는 청중에게 결과 보이기
   - 편향이나 문제 찾아내기

4. 한계:
   - 2차원 사영은 정보를 잃는다
   - 그림은 실제 임베딩 공간과 같지 않다
   - 정량적 도구가 아니라 정성적 도구로 쓰라

5. 모범 관행:
   - 발표에는 t-SNE를 쓰라
   - 빠른 확인에는 PCA를 쓰라
   - 정량적인 지표로도 늘 확인하라
   - 여러 무작위 씨앗으로 해 보라

축하한다! 이제 임베딩을 그리고 해석할 수 있다!
""")

print("=" * 70)
print("END OF TUTORIAL")
print("=" * 70)


if __name__ == "__main__":
    pass
```

## 2. 논의

`CBOWModel` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로 PyTorch의 자동 미분이 학습 중 기울기 계산을 알아서 처리한다. 이런 모듈식 설계 덕분에 부품 하나하나를 고치거나 모델을 더 큰 파이프라인에 넣기 쉽다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화를 쓴 `CBOWModel`의 학습 가능한 매개변수 총수를 계산하라. 가중치와 편향을 모두 넣어 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
층이나 블록의 수를 설정할 수 있도록 `CBOWModel`을 확장하라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 훑는다. (보통의 파이썬 리스트가 아니라) `nn.ModuleList`을 써야 PyTorch가 최적화를 위해 모든 매개변수를 등록한다. `for n in [2, 4, 8]: model = CBOWModel(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — 심화 실습

`CBOWModel` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 감싼다.

핵심 클래스는 `CBOWModel`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
