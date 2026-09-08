# 간단한 임베딩

낱말 임베딩은 의미의 관계를 담은 조밀하고 차원이 낮은 낱말 벡터 표현이다. 낱말마다 직교하는 벡터를 주어 모든 낱말이 똑같이 멀어지는 원-핫 부호화와 달리, 임베딩은 낱말을 유사도가 뜻을 갖는 연속 공간으로 보낸다. 이 실습은 PyTorch의 `nn.Embedding` 층을 소개하고, 그것이 학습 가능한 조회표 노릇을 함을 보이며, 코사인 유사도로 낱말의 관련성을 재는 법을 미리 보여 준다.

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

## 2. 논의

PyTorch의 임베딩 층은 본질적으로 모양이 $(V, d)$인 행렬이며, $V$은 어휘 크기, $d$은 임베딩 차원이다. 어떤 낱말의 임베딩을 찾는 일은 이 행렬의 한 행을 색인으로 꺼내는 것과 같다. 학습 전에는 임베딩 벡터가 무작위로 초기화되므로 낱말 사이의 유사도를 재 보아야 아무 뜻이 없다. 학습 중에는 다른 학습 가능한 매개변수처럼 임베딩 가중치도 역전파로 갱신되며, 벡터 공간이 차츰 정돈되어 의미가 가까운 낱말이 모이게 된다.

원-핫에서 조밀한 임베딩으로 넘어가면 근본적인 한계 몇 가지가 풀린다. 원-핫 벡터는 성기고 차원이 높으며(크기가 어휘와 같다) 어떤 낱말 쌍이든 똑같이 다르다고 본다. 조밀한 임베딩은 표현을 훨씬 적은 차원(보통 50에서 300)으로 눌러 담으면서 의미 구조를 담는다. 학습이 끝나면 "고양이"와 "개"의 코사인 유사도가 "고양이"와 "물고기"보다 높아야 한다. 이렇게 배운 기하 덕분에 뒤따르는 모델이 낱말을 저마다 고립된 기호로 다루지 않고 관련된 낱말에 걸쳐 일반화할 수 있다.

흔한 임베딩 차원은 작은 데이터셋의 50에서 큰 모델의 300까지이다. 이 선택은 표현력과 과적합의 위험 사이에서 균형을 잡는 일이다. Word2Vec, GloVe, FastText 같은 사전 학습 임베딩이 좋은 출발점을 주며, 자료가 적을 때는 밑바닥부터 학습하기보다 과제 데이터로 미세 조정하는 편이 나은 결과를 낼 때가 많다.

## 연습문제

**연습문제 1.**
낱말 1000개의 어휘에 대해 차원이 100인 임베딩 층을 만들어라. 낱말 색인 42와 43의 임베딩을 꺼내 유클리드 거리를 계산하라. 두 행을 똑같은 값으로 직접 맞춘 뒤 다시 해 보고 거리가 0인지 확인하라.

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

**연습문제 2.**
`nn.Embedding`이 원-핫 벡터에 가중치 행렬을 곱하는 것과 수학적으로 같은 까닭을 설명하라. 특정 낱말 색인에 대해 이를 확인하는 코드를 작성하라.

??? success "연습문제 2 풀이"
    임베딩에서 색인 $i$을 찾는 일은 자리 $i$에 1이 있는 원-핫 벡터 $e_i$에 대해 $e_i^\top W$을 계산하는 것과 같다. 이 행렬 곱이 $W$의 $i$번째 행을 골라낸다.
    
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

**연습문제 3.**
학습된 임베딩 층을 받아 코사인 유사도로 주어진 낱말의 가장 가까운 이웃 $k$개를 찾는 함수를 구현하라. 간단한 모델을 학습시킨 뒤 작은 어휘로 시험하라.

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
    
    학습 전에는 가장 가까운 이웃이 사실상 무작위이다. 텍스트 말뭉치로 학습한 뒤에는 의미가 가까운 낱말이 이웃으로 나와야 한다.

## 정리하며

**다룬 것** — 간단한 임베딩

PyTorch의 임베딩 층은 본질적으로 모양이 $(V, d)$인 행렬이며, $V$은 어휘 크기, $d$은 임베딩 차원이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
