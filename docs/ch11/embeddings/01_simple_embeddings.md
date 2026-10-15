# 간단한 임베딩

단어 임베딩은 단어 ID를 조밀한 벡터로 바꾸는 표현이다. PyTorch의 `nn.Embedding`은 이 벡터들을 행렬에 저장하고, 입력 ID에 해당하는 행을 꺼낸다. 이 예제에서는 임베딩 층을 만들고 한 단어 또는 여러 단어의 벡터를 조회한 뒤 코사인 유사도를 계산한다. 학습 단계는 없으므로 여기서 얻은 유사도에 단어의 의미가 반영되었다고 해석할 수는 없다.

## 1. 코드

```python
# ========================================================
# 01_simple_embeddings.py
# 낱말 임베딩 소개
# ========================================================

"""
실습 1: 낱말 임베딩 소개

학습 목표:
- 낱말 임베딩이 무엇인지 이해하기
- PyTorch에서 임베딩 층을 만드는 법 익히기
- 낱말이 조밀한 벡터로 어떻게 옮겨지는지 보기
- 원-핫 부호화와 임베딩의 차이 이해하기

예상 시간: 15분
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 70)
print("TUTORIAL 1: Introduction to Word Embeddings")
print("=" * 70)

# ========================================================
# 1부: 원-핫 부호화의 문제
# ========================================================

vocabulary = ["cat", "dog", "bird", "fish", "lion"]
vocab_size = len(vocabulary)

word_to_ix = {word: i for i, word in enumerate(vocabulary)}

# 원-핫 부호화 예제
word = "cat"
word_idx = word_to_ix[word]
one_hot = torch.zeros(vocab_size)
one_hot[word_idx] = 1

# ========================================================
# 2부: 낱말 임베딩 만들기
# ========================================================

embedding_dim = 3
embeddings = nn.Embedding(vocab_size, embedding_dim)

# ========================================================
# 3부: 낱말 임베딩 얻기
# ========================================================

word_tensor = torch.tensor([word_to_ix["cat"]], dtype=torch.long)
cat_embedding = embeddings(word_tensor)

# ========================================================
# 4부: 여러 낱말
# ========================================================

words = ["cat", "dog", "lion"]
word_indices = [word_to_ix[w] for w in words]
word_tensor = torch.tensor(word_indices, dtype=torch.long)
batch_embeddings = embeddings(word_tensor)

# ========================================================
# 5부: 임베딩의 유사도
# ========================================================

cat_emb = embeddings.weight[word_to_ix["cat"]]
dog_emb = embeddings.weight[word_to_ix["dog"]]
fish_emb = embeddings.weight[word_to_ix["fish"]]

cos = nn.CosineSimilarity(dim=0)
cat_dog_sim = cos(cat_emb, dog_emb)
cat_fish_sim = cos(cat_emb, fish_emb)


if __name__ == "__main__":
    pass
```

**출력:**

```
======================================================================
TUTORIAL 1: Introduction to Word Embeddings
======================================================================
```

## 2. 논의

임베딩 가중치의 형태는 $(V,d)$이며, $V$는 어휘 크기, $d$는 임베딩 차원이다. 단어 ID 하나를 입력하면 길이 $d$의 벡터를 얻고, 형태가 $(B,L)$인 ID 텐서를 입력하면 출력 형태는 $(B,L,d)$가 된다. 임베딩 조회는 원핫 벡터와 가중치 행렬의 곱과 같지만, 원핫 벡터를 직접 만들 필요가 없다.

기본 설정의 `nn.Embedding` 가중치는 학습 가능한 매개변수다. 다만 이 코드에는 손실 계산, 역전파, 옵티마이저 갱신이 없으므로 초기값 그대로 남는다. `cat_dog_sim`과 `cat_fish_sim`은 계산되지만 출력되지는 않는다. 어느 값이 더 크더라도 동물 사이의 의미 관계를 학습한 결과가 아니다.

임베딩을 언어 모델이나 분류 모델의 일부로 학습시키면 목적 함수에 유용한 방향으로 벡터가 바뀐다. 비슷한 문맥의 단어가 가까워질 수 있지만, 특정 단어 쌍의 유사도 순서가 반드시 정해지는 것은 아니다. 코사인 유사도는 벡터의 방향을 비교하며, 모델이 학습한 모든 관계를 하나의 값으로 설명하지는 못한다.

임베딩 차원은 표현력, 데이터 규모, 메모리와 계산량을 함께 고려해 정한다. 50이나 300은 선택 가능한 예일 뿐 상한이 아니다. Word2Vec, GloVe, FastText 등의 사전 학습 벡터를 사용할 때도 언어, 어휘 범위, 학습 도메인과 현재 과제의 적합성을 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
단어 1000개의 어휘에 대해 차원이 100인 임베딩 층을 만들어라. 단어 인덱스 42와 43의 임베딩을 꺼내 유클리드 거리를 계산하라. 두 행을 똑같은 값으로 직접 맞춘 뒤 다시 해 보고 거리가 0인지 확인하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    emb = nn.Embedding(1000, 100)
    v42 = emb(torch.tensor([42]))
    v43 = emb(torch.tensor([43]))
    dist = torch.norm(v42 - v43).item()
    print(f"Distance before: {dist:.4f}")

    # 둘을 같은 값으로 두기
    with torch.no_grad():
        emb.weight[43] = emb.weight[42].clone()
    v42 = emb(torch.tensor([42]))
    v43 = emb(torch.tensor([43]))
    dist = torch.norm(v42 - v43).item()
    print(f"Distance after: {dist:.6f}")  # 0.000000
    ```

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`nn.Embedding`이 원핫 벡터에 가중치 행렬을 곱하는 것과 수학적으로 같은 까닭을 설명하라. 특정 단어 인덱스에 대해 이를 확인하는 코드를 작성하라.

</div>

??? success "연습문제 2 풀이"
    임베딩에서 인덱스 $i$를 찾는 일은 자리 $i$에 1이 있는 원핫 벡터 $e_i$에 대해 $e_i^\top W$를 계산하는 것과 같다. 이 행렬 곱이 $W$의 $i$번째 행을 골라낸다.
    
    ```python
    vocab_size, emb_dim = 5, 3
    emb = nn.Embedding(vocab_size, emb_dim)
    idx = 2

    # 방법 1: 임베딩 조회
    result1 = emb(torch.tensor([idx]))

    # 방법 2: 원-핫 행렬 곱
    one_hot = torch.zeros(1, vocab_size)
    one_hot[0, idx] = 1.0
    result2 = one_hot @ emb.weight

    print(torch.allclose(result1, result2))  # True
    ```

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
임베딩 층에서 주어진 단어와 코사인 유사도가 높은 이웃 $k$개를 찾는 함수를 구현하라. 아래의 학습 전 임베딩 예제와 학습된 임베딩을 구분해 해석하고, 자기 자신을 후보에서 제외하는 방법의 한계도 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    def find_neighbors(word_idx, embedding_layer, k=5):
        word_vec = embedding_layer.weight[word_idx]
        cos_sim = F.cosine_similarity(
            word_vec.unsqueeze(0), embedding_layer.weight, dim=1
        )
        cos_sim[word_idx] = -1  # 낱말 자신은 뺀다
        top_k = torch.topk(cos_sim, k)
        return top_k.indices.tolist(), top_k.values.tolist()

    emb = nn.Embedding(10, 50)
    indices, sims = find_neighbors(0, emb, k=3)
    for idx, sim in zip(indices, sims):
        print(f"  Word {idx}: similarity = {sim:.4f}")
    ```
    
    풀이의 임베딩은 무작위 초기값이며 학습하지 않았다. 따라서 반환된 이웃을 의미가 비슷한 단어로 해석할 수 없다. 별도로 학습한 임베딩을 넣어야 학습 후의 관계를 살펴볼 수 있다. `F`를 사용하려면 `torch.nn.functional`을 해당 이름으로 가져와야 한다.

    자기 자신의 점수를 $-1$로 바꾸는 방식은 다른 후보도 코사인 유사도가 $-1$이면 동점이 되어 자신을 완전히 제외하지 못할 수 있다. 후보에서 해당 인덱스를 제거하거나 그 점수를 음의 무한대로 마스킹하는 방식이 더 명확하다. $k$는 자신을 제외한 어휘 수 이하로 정해야 한다.

## 정리하며

임베딩은 단어 ID를 학습 가능한 벡터로 연결하는 조회표다. 벡터를 생성하거나 유사도를 계산하는 것만으로 의미 관계가 생기지는 않는다. 학습 목표와 데이터를 확인한 뒤 이웃 관계를 해석해야 한다.
