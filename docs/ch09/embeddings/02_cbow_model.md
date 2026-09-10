# CBOW 모델

연속 낱말 주머니(CBOW) 모델은 둘레 문맥 낱말이 주어졌을 때 가운데 낱말을 예측한다. 앞선 낱말만 쓰는 N-그램 모형과 달리 CBOW는 표적의 양쪽 낱말을 모두 써서 양방향 문맥을 붙잡는다. 이 실습은 CBOW를 PyTorch로 밑바닥부터 구현하고 텍스트 말뭉치로 학습시킨 뒤 유사도 검색으로 학습된 임베딩을 살펴본다. CBOW는 Word2Vec의 두 핵심 구조 가운데 하나이며 요즘의 문맥 임베딩을 이해하는 바탕이 된다.

## 1. 코드

```python
# ========================================================
# 02_cbow_model.py
# 연속 낱말 주머니(CBOW) 모델
# ========================================================

"""
중급 실습 2: 연속 낱말 주머니 (CBOW)

학습 목표:
- CBOW의 구조 이해하기
- CBOW를 밑바닥부터 구현하기
- N-그램 모형과 견주기
- 문맥 창의 개념 익히기

예상 시간: 30분
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.data_loader import (
    load_text_from_file, simple_tokenize, build_vocabulary,
    create_cbow_dataset, print_corpus_stats, print_dataset_stats
)

# ========================================================
# 데이터 준비
# ========================================================

text = load_text_from_file('../data/sample_text.txt')
tokens = simple_tokenize(text, lowercase=True)
word_to_ix, ix_to_word = build_vocabulary(tokens, min_freq=1)
vocab_size = len(word_to_ix)

WINDOW_SIZE = 2
cbow_data = create_cbow_dataset(tokens, WINDOW_SIZE, word_to_ix)

# ========================================================
# CBOW 모델의 구조
# ========================================================

class CBOWModel(nn.Module):
    """
    연속 낱말 주머니(CBOW) 모델
    
    구조:
        1. 임베딩 층: 문맥 낱말마다 벡터로 보낸다
        2. 평균 풀링: 모든 문맥 임베딩을 평균한다
        3. 선형층: 어휘 크기로 사영한다
    """
    
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context_words):
        embeds = self.embeddings(context_words)
        mean_embeds = torch.mean(embeds, dim=1)
        out = self.linear(mean_embeds)
        return out

# ========================================================
# 학습
# ========================================================

EMBEDDING_DIM = 50
model = CBOWModel(vocab_size, EMBEDDING_DIM)
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 32

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

losses = []
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
    avg_loss = total_loss / (len(cbow_data) / BATCH_SIZE)
    losses.append(avg_loss)

# ========================================================
# 임베딩 분석
# ========================================================

def find_similar_words(word, model, word_to_ix, ix_to_word, top_k=5):
    if word not in word_to_ix:
        return []
    word_idx = word_to_ix[word]
    word_embedding = model.embeddings.weight[word_idx]
    similarities = F.cosine_similarity(
        word_embedding.unsqueeze(0), model.embeddings.weight, dim=1
    )
    similarities[word_idx] = -1
    top_sim, top_indices = torch.topk(similarities, k=top_k)
    return [(ix_to_word[idx.item()], sim.item()) for sim, idx in zip(top_sim, top_indices)]


if __name__ == "__main__":
    pass
```

## 2. 논의

CBOW 구조는 우아하리만치 간단하다. 문맥 낱말을 저마다 임베딩하고, 그 임베딩을 평균하고, 결과를 선형층에 넣어 가운데 낱말을 예측한다. 평균 풀링 단계가 "낱말 주머니"에 해당한다. 문맥 창 안의 낱말 순서를 버리고 문맥을 순서 없는 모임으로 다루는 것이다. 이렇게 단순화해도 CBOW는 놀랍도록 쓸모 있는 임베딩을 배우는데, 문맥 벡터의 평균이 가운데 자리에 어떤 낱말이 와야 하는지에 대한 강한 신호를 주기 때문이다.

문맥 창의 크기는 핵심 초매개변수이다. 창이 작으면(양쪽에 2~3낱말) 문법적 관계(문장에서 서로 바꿔 쓸 수 있는 낱말)를 담은 임베딩이 나오는 편이고, 창이 크면(5~10낱말) 더 넓은 주제적·의미적 관계를 담는다. 학습은 교차 엔트로피 손실로 평균 문맥 임베딩이 주어졌을 때 옳은 가운데 낱말의 확률을 최대로 하며, 보통의 SGD나 Adam으로 최적화한다.

학습이 끝나면 임베딩 행렬에는 낱말을 유사도에 따라 정돈한 표현이 담긴다. `find_similar_words` 함수는 코사인 유사도로 이 공간을 질의하여 관련된 낱말의 무리를 드러낸다. 이를테면 셰익스피어 소네트 말뭉치에서는 "beauty", "youth", "fair" 같은 낱말이 함께 모이는 편이다. 이 임베딩은 뒤따르는 자연어 처리 과제의 입력 특징으로 바로 쓰거나 더 복잡한 모델의 초기값으로 쓸 수 있다.

## 연습문제

**연습문제 1.**
문장 "the quick brown fox jumps over the lazy dog"과 창 크기 2가 주어졌을 때 CBOW의 (문맥, 표적) 학습 쌍을 모두 열거하라. 쌍이 몇 개 나오는가?

??? success "연습문제 1 풀이"
    자리 $i$의 낱말마다 문맥은 거리 2 안의 낱말들(자기 자신은 빼고)이다.

    | 표적 | 문맥 |
    |--------|---------|
    | the | [quick, brown] |
    | quick | [the, brown, fox] |
    | brown | [the, quick, fox, jumps] |
    | fox | [quick, brown, jumps, over] |
    | jumps | [brown, fox, over, the] |
    | over | [fox, jumps, the, lazy] |
    | the | [jumps, over, lazy, dog] |
    | lazy | [over, the, dog] |
    | dog | [the, lazy] |

    학습 쌍이 9개 나온다. 가장자리 낱말은 문맥이 작고(2~3낱말) 안쪽 낱말은 문맥이 4낱말로 꽉 찬다.

---

**연습문제 2.**
CBOW가 문맥 임베딩을 이어 붙이지 않고 평균 풀링을 쓰는 까닭을 설명하라. 대신 이어 붙이면 구조가 어떻게 달라지는가?

??? success "연습문제 2 풀이"
    평균 풀링은 문맥 낱말이 몇 개든 크기가 고정된 출력을 내는데, 가장자리 자리는 안쪽 자리보다 문맥 낱말이 적으므로 이 점이 중요하다. 이어 붙이려면 문맥 낱말의 수가 고정되어야 하고 크기가 $2w \times d$인 벡터가 나오므로($w$은 창 크기, $d$은 임베딩 차원) 뒤따르는 선형층이 훨씬 커진다. 매개변수가 $d \times V$개가 아니라 $(2w \times d) \times V$개가 된다. 또 평균 풀링은 모델이 문맥 낱말의 순서에 무관해지게 하는데("주머니"라는 성질) 이는 일부러 고른 설계이다. N-그램 모형은 언제나 앞선 낱말이 꼭 $n-1$개이고 그 순서를 지키므로 이어 붙이기를 쓴다.

---

**연습문제 3.**
CBOW 모델이 단순 평균 대신 가중 평균을 쓰도록 고쳐, 표적에 가까운 낱말에 더 큰 가중치를 주어라. 가운데에서 거리가 $d$인 문맥 낱말의 가중치를 $1/d$으로 하는 거리 기반 가중 방식을 구현하라. 모델을 학습시키고 손실 곡선을 표준 CBOW와 견주어라.

??? success "연습문제 3 풀이"
    ```python
    class WeightedCBOW(nn.Module):
        def __init__(self, vocab_size, embedding_dim, window_size):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear = nn.Linear(embedding_dim, vocab_size)
            # 가중치: window_size=2일 때 1/1, 1/2 -> [0.5, 1.0, 1.0, 0.5]
            distances = list(range(window_size, 0, -1)) + list(range(1, window_size + 1))
            weights = torch.tensor([1.0 / d for d in distances])
            self.register_buffer('weights', weights / weights.sum())
        
        def forward(self, context_words):
            embeds = self.embeddings(context_words)  # (배치, ctx_size, emb_dim)
            # 거리 기반 가중치 적용
            w = self.weights[:embeds.size(1)].unsqueeze(0).unsqueeze(-1)
            weighted = (embeds * w).sum(dim=1)
            return self.linear(weighted)
    ```
    
    가까운 문맥 낱말이 가운데 낱말을 더 잘 예측하므로 가중 판본이 대체로 조금 더 빨리 수렴하고 손실도 약간 낮다. 다만 표준 평균이 이미 잘 통하므로 개선의 폭은 작을 때가 많다.

## 정리하며

**다룬 것** — CBOW 모델

CBOW 구조는 우아하리만치 간단하다.

핵심 클래스는 `CBOWModel`, `WeightedCBOW`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
