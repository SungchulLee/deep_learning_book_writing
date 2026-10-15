# 임베딩 시각화

학습한 단어 벡터를 PCA와 t-SNE로 2차원에 배치하는 예제다. 두 방법이 보존하려는 정보와 그림에서 읽을 수 있는 내용을 구분한다. 보기 좋은 군집이 나왔다는 사실만으로 의미 관계나 과제 성능이 검증되는 것은 아니다.

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

모델은 문맥 임베딩의 평균으로 중심 단어를 예측하는 CBOW다. 임베딩 차원은 30이고 옵티마이저는 이미 Adam이다. 학습 후 임베딩 가중치를 분리해 NumPy 배열로 변환하므로, 이후 차원 축소는 모델의 역전파 과정에 포함되지 않는다.

본문은 제공되지 않은 `utils.data_loader`와 `../data/sample_text.txt`에 의존한다. 또한 `words=list(word_to_ix.keys())`가 임베딩 행 순서와 같다고 가정한다. 사전 삽입 순서가 인덱스 순서와 다르면 그림에 잘못된 이름이 붙으므로, `ix_to_word`를 통해 각 행의 이름을 명시적으로 대응시켜야 한다.

PCA는 중심화된 데이터의 분산을 크게 설명하는 선형 축을 찾는다. 설명된 분산 비율은 원래 벡터의 분산을 얼마나 담는지를 뜻하며, 의미 정보나 분류 성능의 보존 비율은 아니다. 2차원 투영은 거리와 이웃 관계를 손실할 수 있다.

t-SNE는 고차원 이웃 관계를 낮은 차원의 확률적 배치로 표현한다. 군집의 간격·크기나 축 방향을 원래 공간의 의미 거리로 읽으면 안 된다. 결과는 perplexity, 초기화와 최적화에 영향을 받는다. 코드의 이웃 검색은 t-SNE 좌표의 유클리드 거리이므로 원래 30차원 임베딩의 코사인 이웃과 같다고 볼 수 없다. 색은 단어 길이를 나타내며 의미 범주를 뜻하지 않는다.

PCA의 2차원 출력에는 표본 수와 특징 수가 각각 2 이상 필요하고, t-SNE의 perplexity는 표본 수보다 작아야 한다. `n_iter`는 예전 API 이름이며 현재 [scikit-learn 문서](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)에서는 `max_iter`를 사용한다. 제시된 코드는 라이브러리 버전 확인이 필요하다. 학습 손실도 마지막 배치 크기를 반영하지 않은 집계이므로 정확한 표본 평균과 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
어휘 크기가 $V$이고 임베딩 차원이 30일 때 CBOWModel의 매개변수 수를 계산하라.

</div>

??? success "연습문제 1 풀이"
    임베딩은 $30V$개, 출력 선형층은 가중치 $30V$개와 편향 $V$개이므로 총 $61V$개다. 문맥 평균과 차원 축소는 이 모델의 학습 매개변수를 추가하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 코드는 이미 Adam을 사용한다. SGD와 옵티마이저 효과를 비교하려면 어떤 조건을 맞춰야 하는가?

</div>

??? success "연습문제 2 풀이"
    같은 초기 가중치, 학습 데이터와 순서, 배치 크기와 갱신 예산을 사용한다. 각 옵티마이저의 학습률은 공통 검증 기준으로 선택한다. 시각화가 더 보기 좋다는 이유로 옵티마이저가 더 낫다고 결론 내리지 말고 원래 공간의 평가 지표와 검증 손실도 비교한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
t-SNE 이웃과 원래 임베딩 이웃을 혼동하는 문제, 단어 라벨이 잘못 대응하는 문제를 각각 어떻게 진단하는가?

</div>

??? success "연습문제 3 풀이"
    원래 벡터의 코사인 이웃과 2차원 좌표의 유클리드 이웃을 따로 계산해 비교한다. 라벨은 사전 순서를 추측하지 않고 각 행 인덱스를 ix_to_word로 확인한다. 차원 축소에 의해 생긴 차이와 잘못된 인덱스 대응을 구분해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력층과 출력층은 과제의 차원을 유지하고, 은닉층 사이에 필요한 비선형성을 넣어야 한다. 선형층만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

PCA의 설명된 분산과 t-SNE의 군집은 임베딩 품질 자체를 뜻하지 않는다. 원래 공간의 이웃과 과제 지표를 함께 확인하고, 단어 라벨의 대응과 차원 축소의 설정·버전 조건을 점검해야 한다.
