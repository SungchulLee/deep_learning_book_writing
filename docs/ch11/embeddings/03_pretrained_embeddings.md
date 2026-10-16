# 사전 학습 임베딩

사전 학습 임베딩은 다른 말뭉치에서 학습한 단어 벡터를 현재 과제에 가져오는 방법이다. 벡터를 고정해서 사용하거나 과제 데이터로 미세 조정할 수 있다. 본문의 코드는 안내 메시지만 출력하며, GloVe 파일을 읽고 임베딩 레이어를 초기화하는 예시는 연습문제에 제시되어 있다.

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

사전 학습 벡터를 사용할 때는 현재 어휘의 각 인덱스에 맞춰 행을 복사해야 한다. 벡터 파일의 행 순서와 현재 어휘의 인덱스가 같다고 가정하면 잘못된 단어에 벡터가 연결된다. 임베딩 차원, 토큰화와 대소문자 처리도 맞춰야 한다.

어휘 밖(OOV) 단어에는 무작위 벡터나 공통 미등록 단어 벡터를 줄 수 있다. 모든 사전 학습 벡터의 평균을 쓰면 처음에는 OOV 단어들의 표현이 같아진다. 이를 고정하면 그 단어들을 구분할 수 없다. FastText의 하위 단어 벡터를 쓰는 방법은 해당 모델과 하위 단어 정보가 필요하며, GloVe 파일만 읽어서 같은 처리를 할 수는 없다.

웨이트를 고정하면 갱신할 파라메터를 줄이고 초기 표현을 유지할 수 있지만, 나머지 모델의 과적합까지 막지는 않는다. 미세 조정은 과제에 적응할 여지를 주는 대신 러닝레이트와 데이터 규모에 따라 표현이 훼손될 수 있다. 고정 학습 후 일부 또는 전체를 해제하는 방식도 검증 성능으로 선택한다. 옵티마이저를 만들 때 고정 파라메터를 제외했다면 해제 후 갱신 대상에 추가해야 한다.

GloVe는 단어 동시 출현 횟수의 로그를 내적과 바이어스로 설명하는 가중 최소제곱 목적 함수를 사용한다. 단순한 행렬 분해나 모든 단어에 성립하는 의미 연산으로 이해해서는 안 된다. 벡터 유추의 성공 여부는 단어와 학습 데이터에 따라 다르며, 사전 학습 벡터에는 데이터의 바이어스도 반영될 수 있다. 원래 벡터와 설명은 [GloVe 프로젝트](https://nlp.stanford.edu/projects/glove/)에서 확인할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
텍스트 파일에서 GloVe 임베딩을 불러와 `nn.Embedding` 레이어를 초기화하는 코드를 작성하라. 어휘 밖 단어에는 알려진 모든 임베딩의 평균을 주어 처리하라.

</div>

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

    이 로더는 헤더 없는 단어별 벡터 파일을 가정한다. 빈 줄과 벡터 차원 불일치를 검사하지 않으며, 빈 파일에서는 평균 계산에 실패한다. `vocab`의 인덱스는 0부터 연속이어야 한다. 파일 전체를 사전에 저장하고 다시 배열로 쌓으므로 메모리 비용이 크다. 또한 반환된 임베딩은 학습 가능한 상태이므로 고정하려면 별도 설정이 필요하다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
사전 학습 임베딩을 고정하거나 미세 조정할 때의 장단점과 선택 기준을 설명하라.

</div>

??? success "연습문제 2 풀이"
    고정하면 초기 벡터를 유지하고 갱신 비용을 줄이지만 현재 과제에 맞춘 조정은 하지 못한다. 미세 조정은 도메인과 과제에 적응할 수 있으나 데이터가 적거나 러닝레이트가 크면 일반적인 관계를 잃을 수 있다. 같은 검증 데이터에서 고정, 작은 러닝레이트의 미세 조정, 단계적 해제를 비교한다. 감성이 반대인 단어가 분류에 유용하려면 반드시 코사인 거리도 멀어야 하는 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
무작위 초기화 임베딩과 GloVe 임베딩에서 단어 쌍 다섯 개의 코사인 유사도를 비교하라. 유사어와 반의어를 함께 넣고, 문맥의 유사성과 의미의 동일성을 구분해 해석하라. 특정 유사도 범위가 나온다고 미리 가정하지 마라.

</div>

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
    
    GloVe에서 반의어도 비슷한 문맥에 나타나 높은 코사인 유사도를 가질 수 있다. 높은 값은 같은 뜻이나 같은 감성을 보장하지 않는다. 무작위 임베딩의 유사도는 대칭적인 초기화에서 평균적으로 0 부근이지만 개별 쌍의 값은 달라진다. 다섯 쌍만으로 임베딩 전체의 품질을 평가할 수는 없다.

    풀이의 `torchtext` 사용은 설치 버전 호환성을 확인해야 한다. [TorchText 0.18 릴리스 안내](https://github.com/pytorch/text/releases/tag/v0.18.0)는 개발 중단을 명시하므로, 현재 환경에서 그대로 사용할 수 있다고 가정하지 않는다.

## 정리하며

사전 학습 임베딩을 가져올 때는 단어·인덱스 대응, 어휘 범위와 전처리를 맞춘다. 고정과 미세 조정의 효과는 검증 데이터에서 비교하며, 코사인 유사도를 의미의 동일성으로 해석하지 않는다.
