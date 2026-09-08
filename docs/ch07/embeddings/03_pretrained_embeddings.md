# 사전 학습 임베딩

GloVe와 Word2Vec 같은 사전 학습 낱말 임베딩은 (낱말 수십억 개의) 거대한 텍스트 말뭉치로 학습한 좋은 품질의 벡터 표현을 제공한다. 작은 데이터셋으로 임베딩을 밑바닥부터 배우는 대신, 이 사전 학습 벡터를 불러와 고정된 특징으로 쓰거나 특정 과제에 맞추어 미세 조정할 수 있다. 이 방법은 대규모 사전 학습에서 담긴 의미 지식을 이름표가 붙은 데이터가 적은 분야로 옮겨 준다.

## 1. 코드

```python
"""사전 학습 임베딩."""

# ========================================================================
# 메인
# ========================================================================
# 사전 학습 임베딩 쓰기 (GloVe, Word2Vec)
# 있는 임베딩을 불러와 미세 조정하기
print("Pre-trained Embeddings - Load GloVe or Word2Vec embeddings")
print("See: https://nlp.stanford.edu/projects/glove/")


if __name__ == "__main__":
    pass
```

**출력:**

```
Pre-trained Embeddings - Load GloVe or Word2Vec embeddings
See: https://nlp.stanford.edu/projects/glove/
```

## 2. 논의

PyTorch에서 사전 학습 임베딩을 불러오는 일은 대체로 두 단계이다. 사전 학습 임베딩 파일(낱말 40만 개를 담은 100차원 GloVe 6B 따위)을 내려받고, 그 가중치 행렬로 `nn.Embedding` 층을 초기화한다. 사전 학습 어휘에 있는 낱말은 해당 벡터를 그대로 옮겨 적는다. 어휘 밖(OOV) 낱말은 무작위 벡터로 초기화하거나, 모든 사전 학습 벡터의 평균을 쓰거나, FastText 같은 하위 낱말 분해 방법을 쓰는 것이 흔한 전략이다.

사전 학습 임베딩을 얼릴지 미세 조정할지는 과제와 데이터의 양에 달려 있다. 얼리면(`requires_grad=False`) 사전 학습된 표현이 지켜지고 과적합을 막을 수 있어 뒤따르는 데이터셋이 작을 때 이롭다. 미세 조정하면 임베딩이 과제의 의미에 맞추어지지만, 학습률이 너무 높거나 데이터셋이 너무 작으면 파국적 망각의 위험이 있다. 흔한 절충은 작은 학습률로 미세 조정하거나, 먼저 임베딩을 얼린 채 학습한 뒤 풀어서 잠깐 미세 조정하는 두 단계 방법을 쓰는 것이다.

GloVe(전역 벡터) 알고리즘은 낱말 동시 출현 행렬을 분해하여 임베딩을 배우며, (LSA 같은) 전역 통계의 장점과 (Word2Vec 같은) 지역 문맥 창의 장점을 아우른다. 나온 벡터는 Word2Vec과 마찬가지로 덧셈적인 조합성을 보인다. 예를 들어 $\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$이다. 그러면서도 날 말뭉치를 훑는 대신 미리 계산한 동시 출현 횟수를 다루므로 학습이 더 효율적일 때가 많다.

## 연습문제

**연습문제 1.**
텍스트 파일에서 GloVe 임베딩을 불러와 `nn.Embedding` 층을 초기화하는 코드를 작성하라. 어휘 밖 낱말에는 알려진 모든 임베딩의 평균을 주어 처리하라.

??? success "연습문제 1 풀이"
    ```python
    import numpy as np
    import torch
    import torch.nn as nn

    def load_glove(filepath, vocab, emb_dim=100):
        glove = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vec = np.array(parts[1:], dtype=np.float32)
                glove[word] = vec
        
        weight = np.zeros((len(vocab), emb_dim))
        found = 0
        all_vecs = np.stack(list(glove.values()))
        mean_vec = all_vecs.mean(axis=0)
        
        for word, idx in vocab.items():
            if word in glove:
                weight[idx] = glove[word]
                found += 1
            else:
                weight[idx] = mean_vec
        
        embedding = nn.Embedding(len(vocab), emb_dim)
        embedding.weight = nn.Parameter(torch.from_numpy(weight).float())
        print(f"Loaded {found}/{len(vocab)} words from GloVe")
        return embedding
    ```

---

**연습문제 2.**
사전 학습 임베딩을 얼리는 것과 미세 조정하는 것의 맞바꿈을 설명하라. 어떤 상황에서 각각이 나은가?

??? success "연습문제 2 풀이"
    **얼릴 때**: 뒤따르는 데이터셋이 작아 과적합의 위험이 있을 때, 사전 학습 임베딩이 그 분야를 잘 덮을 때, 갱신할 매개변수를 줄여 학습을 빠르게 하고 싶을 때.
    
    **미세 조정할 때**: 뒤따르는 과제의 어휘가 사전 학습 말뭉치와 달리 그 분야에 특화되어 있을 때(의료나 법률 텍스트 따위), 과적합을 피할 만큼 데이터가 넉넉할 때, 과제에 특화된 의미 구별이 중요할 때(일반 텍스트에서는 문맥이 비슷하더라도 감성 분석에서는 "terrific"이 "terrible"과 멀어야 한다).
    
    **실용적인 절충**: 임베딩을 얼린 채 몇 세대 학습시켜 나머지 모델을 먼저 학습시킨 뒤, 임베딩을 풀고 (주 학습률의 10분의 1 같은) 낮은 학습률로 미세 조정한다.

---

**연습문제 3.**
무작위로 초기화한 임베딩과 GloVe 임베딩으로 낱말 쌍의 코사인 유사도를 견주어라. 낱말 쌍 다섯 개(king-queen, cat-dog, happy-sad, car-bicycle, hot-cold 따위)를 골라 GloVe는 뜻있는 관계를 담지만 무작위 임베딩은 그렇지 않음을 보여라.

??? success "연습문제 3 풀이"
    ```python
    import torchtext.vocab as vocab
    
    glove = vocab.GloVe(name='6B', dim=100)
    pairs = [("king", "queen"), ("cat", "dog"), ("happy", "sad"),
             ("car", "bicycle"), ("hot", "cold")]
    
    print("GloVe similarities:")
    for w1, w2 in pairs:
        sim = torch.nn.functional.cosine_similarity(
            glove[w1].unsqueeze(0), glove[w2].unsqueeze(0)
        ).item()
        print(f"  {w1}-{w2}: {sim:.4f}")
    
    # 견주기 위한 무작위 임베딩
    random_emb = nn.Embedding(len(glove.itos), 100)
    w2i = {w: i for i, w in enumerate(glove.itos)}
    
    print("\nRandom similarities:")
    for w1, w2 in pairs:
        v1 = random_emb.weight[w2i[w1]]
        v2 = random_emb.weight[w2i[w2]]
        sim = torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0)
        ).item()
        print(f"  {w1}-{w2}: {sim:.4f}")
    ```
    
    관련된 쌍(king-queen, cat-dog)의 GloVe 유사도는 (뜻있는 무늬 없이 0.0 언저리를 맴도는) 무작위 임베딩보다 훨씬 높다(보통 0.5~0.8).

## 정리하며

**다룬 것** — 사전 학습 임베딩

PyTorch에서 사전 학습 임베딩을 불러오는 일은 대체로 두 단계이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
