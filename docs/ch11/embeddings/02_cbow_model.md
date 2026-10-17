# CBOW 모델

CBOW(Continuous Bag of Words)는 주변 단어의 임베딩을 결합해 중심 단어를 예측하는 Word2Vec 구조다. 이 예제는 문맥 임베딩의 평균을 선형 레이어에 전달하고 교차 엔트로피로 학습하는 방식을 보여 준다. 문맥의 순서를 구분하지 않는 설계와 데이터 준비 방식의 관계를 살펴본다.

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
        1. 임베딩 레이어: 문맥 낱말마다 벡터로 보낸다
        2. 평균 풀링: 모든 문맥 임베딩을 평균한다
        3. 선형 레이어: 어휘 크기로 사영한다
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

`CBOWModel`은 형태가 `(B, C)`인 문맥 단어 ID를 받아 `(B, C, d)`의 임베딩을 만들고, 문맥 축을 평균해 `(B, d)`로 줄인다. 선형 레이어의 출력은 `(B, V)`의 로짓이다. 같은 단어들을 같은 횟수로 포함한다면 순서를 바꾸어도 평균은 같다. 평균은 문맥 길이가 달라도 고정 차원 벡터를 만들지만, 배치로 쌓으려면 각 예제의 길이를 맞추거나 패딩과 마스크를 처리해야 한다.

학습 코드는 `utils.data_loader`와 `../data/sample_text.txt`에 의존한다. 해당 파일이 이 장의 저장소 경로에 제공되어 있지 않으므로 본문만으로 데이터 생성기의 가장자리 처리 방식이나 실제 말뭉치를 확인할 수 없다. 특정 단어들이 이 예제의 학습 후 이웃으로 모인다고 단정할 근거도 없다.

문맥 창은 사용 가능한 정보와 학습 쌍의 분포를 바꾼다. 좁은 창과 넓은 창에서 얻는 관계가 달라질 수 있지만, 창 크기만으로 문법·의미 관계의 품질이 결정되지는 않는다. 데이터 규모, 토큰화와 학습 목표도 함께 영향을 준다.

기록된 `avg_loss`는 배치 평균 로스의 합을 `len(cbow_data) / BATCH_SIZE`로 나눈 값이다. 마지막 배치가 작으면 표본별 평균 로스와 다르다. 정확한 에포크 평균을 구하려면 각 배치 로스에 배치 크기를 곱해 합산한 뒤 전체 예제 수로 나눈다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
문장 "the quick brown fox jumps over the lazy dog"에서 창 크기 2의 CBOW 학습 쌍을 열거하라. 가장자리에서는 범위를 벗어난 단어만 제외하는 규칙을 가정한다. 모든 위치에서 문맥 단어가 정확히 4개인 경우만 허용하면 쌍의 수가 어떻게 달라지는가?

</div>

??? success "풀이"
    자리 $i$의 단어마다 문맥은 거리 2 안의 단어들(자기 자신은 빼고)이다.

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

    이 규칙에서는 학습 쌍이 9개다. 가장자리 문맥 길이는 2~3이며, 안쪽에서는 4다. 완전한 문맥만 허용하면 중심 위치는 brown부터 두 번째 the까지여서 5개다. 본문의 외부 데이터 생성기가 어느 규칙을 쓰는지는 제공된 코드만으로 확인할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
문맥 임베딩의 평균과 이어 붙이기를 비교하라. 문맥 단어 수가 $2w$, 임베딩 차원이 $d$, 어휘 크기가 $V$일 때 출력 선형 레이어의 파라메터 수는 각각 얼마인가?

</div>

??? success "풀이"
    평균은 순서를 구분하지 않는 $d$차원 벡터이고, 이어 붙이기는 위치가 구별되는 $2wd$차원 벡터다. 바이어스를 포함한 출력 선형 레이어의 파라메터는 각각 $dV+V$, $2wdV+V$개다. 가변 길이 문맥을 이어 붙이려면 패딩 등 길이를 맞추는 규칙이 필요하다. 평균도 패딩을 쓰면 패딩을 제외한 합과 개수로 계산해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
CBOW 모델이 단순 평균 대신 가중 평균을 쓰도록 고쳐, 표적에 가까운 단어에 더 큰 웨이트를 주어라. 가운데에서 거리가 $d$인 문맥 단어의 웨이트를 $1/d$으로 하는 거리 기반 가중 방식을 구현하라. 모델을 학습시키고 로스 곡선을 표준 CBOW와 비교하여라.

</div>

??? success "풀이"
    ```python
    class WeightedCBOW(nn.Module):
        def __init__(self, vocab_size, embedding_dim, window_size):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear = nn.Linear(embedding_dim, vocab_size)
            # 웨이트: window_size=2일 때 1/1, 1/2 -> [0.5, 1.0, 1.0, 0.5]
            distances = list(range(window_size, 0, -1)) + list(range(1, window_size + 1))
            weights = torch.tensor([1.0 / d for d in distances])
            self.register_buffer('weights', weights / weights.sum())
        
        def forward(self, context_words):
            embeds = self.embeddings(context_words)  # (배치, ctx_size, emb_dim)
            # 거리 기반 웨이트 적용
            w = self.weights[:embeds.size(1)].unsqueeze(0).unsqueeze(-1)
            weighted = (embeds * w).sum(dim=1)
            return self.linear(weighted)
    ```
    
    풀이의 웨이트는 왼쪽에서 오른쪽으로 나열된 완전한 문맥을 가정한다. 창 크기 2일 때 거리 순서는 `[2, 1, 1, 2]`다. 가장자리에서 문맥 길이가 줄어들면 웨이트의 앞부분만 잘라 쓰는 방식은 실제 거리와 맞지 않고 웨이트의 합도 1이 아닐 수 있다. 실제 위치에 대응하는 거리를 전달하고 남은 웨이트를 다시 정규화해야 한다. 가중 평균이 단순 평균보다 항상 빨리 수렴하거나 로스가 낮아지는 것은 아니며, 같은 초기 조건과 검증 데이터에서 비교해야 한다.

## 정리하며

CBOW는 문맥 임베딩을 순서와 무관하게 결합해 중심 단어를 예측한다. 구현에서는 문맥의 길이와 패딩, 거리 웨이트의 대응을 맞춰야 한다. 이 페이지의 데이터 의존성이 제공되지 않은 점과 로스 집계 방식을 고려해 예제를 해석한다.
