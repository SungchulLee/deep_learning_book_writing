# Skip-gram 모델

Skip-gram 모델은 Word2Vec 틀에서 CBOW를 보완한다. CBOW가 문맥에서 가운데 낱말을 예측하는 반면 Skip-gram은 거꾸로 가운데 낱말에서 문맥 낱말을 예측한다. 가운데 낱말마다 (문맥 자리마다 하나씩) 여러 학습 예제를 만들므로 드문 낱말의 표현을 배우는 데 뛰어나다. Skip-gram은 많은 실용적인 임베딩 시스템의 바탕이며 두 Word2Vec 판본 가운데 더 널리 쓰인다.

## 1. 코드

```python
# ========================================================
# 03_skipgram_model.py
# Skip-gram 모델 (CBOW를 보완한다)
# ========================================================

"""
중급 실습 3: Skip-gram 모델

이 실습은 Skip-gram의 구조를 다룬다.
- 가운데 낱말에서 문맥 낱말 예측하기
- CBOW를 보완한다
- 드문 낱말에서 성능이 낫다
- Word2Vec의 나머지 반쪽이다

아직 미완성: 온전한 구현은 곧 나온다!
지금은 CBOW 실습을 보고 입력과 출력을 뒤집어 생각하라.
"""

print("Skip-gram tutorial - Implementation template")
print("Key concept: center_word -> context_words")
print("\nRefer to 02_cbow_model.py and reverse the architecture!")


if __name__ == "__main__":
    pass
```

**출력:**

```
Skip-gram tutorial - Implementation template
Key concept: center_word -> context_words

Refer to 02_cbow_model.py and reverse the architecture!
```

## 2. 논의

Skip-gram의 목표는 가운데 낱말이 주어졌을 때 문맥 낱말을 볼 확률 $\sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t)$을 최대로 하는 것이며, $c$은 문맥 창의 크기, $T$은 말뭉치의 길이이다. (가운데, 문맥) 쌍마다 모델은 $p(w_O | w_I) = \frac{\exp(v'_{w_O} \cdot v_{w_I})}{\sum_{w=1}^{V} \exp(v'_w \cdot v_{w_I})}$을 계산하며, $v$과 $v'$은 각각 입력 임베딩 행렬과 출력 임베딩 행렬이다.

드문 낱말에서 Skip-gram이 CBOW보다 나은 핵심 이유는 학습 예제를 만드는 방식에 있다. 말뭉치에 한 번 나온 드문 낱말이 가운데로 쓰이면 (문맥 자리마다 하나씩) 학습 쌍 $2c$개가 나온다. CBOW에서는 그 낱말이 $2c$개의 문맥 낱말 가운데 하나로 나와 나머지와 평균되므로 기울기 신호가 묽어진다. 이런 비대칭 덕분에 Skip-gram에서는 드문 낱말의 임베딩이 문맥 창에 비례하는 집중된 기울기 갱신을 받아, 몇 번 나오지 않아도 더 나은 표현을 얻는다.

구조로 보면 Skip-gram 모델은 놀랍도록 간단하다. 가운데 낱말의 임베딩 층과 어휘 크기의 로짓으로 보내는 선형 출력층이 전부이다. 표준 형태에는 은닉층도 비선형도 없다. 표현력은 온전히 학습 목표에서 나오며, 그 목표가 임베딩에 분포 의미론을 담게 만든다. 실제로 Skip-gram은 거의 언제나 전체 소프트맥스 대신 음성 표본 추출로 학습시키는데, 큰 어휘에 대해 예제마다 소프트맥스를 계산하는 것은 감당할 수 없기 때문이다.

## 연습문제

**연습문제 1.**
입력 임베딩 층과 출력 선형층을 갖춘 기본 Skip-gram 모델 클래스를 PyTorch로 구현하라. 가운데 낱말의 색인을 받아 어휘에 대한 로짓을 돌려주는 forward 메서드를 작성하라.

??? success "연습문제 1 풀이"
    ```python
    import torch
    import torch.nn as nn

    class SkipGram(nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            super().__init__()
            self.center_embedding = nn.Embedding(vocab_size, embedding_dim)
            self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        def forward(self, center_word):
            """
            인수:
                center_word: 가운데 낱말 색인의 (배치 크기,) 텐서
            반환값:
                logits: (배치 크기, vocab_size)
            """
            embedded = self.center_embedding(center_word)  # (배치, emb_dim)
            logits = self.output_layer(embedded)            # (배치, vocab_size)
            return logits

    model = SkipGram(vocab_size=5000, embedding_dim=100)
    center = torch.tensor([42, 7, 100])
    output = model(center)
    print(f"Output shape: {output.shape}")  # (3, 5000)
    ```

---

**연습문제 2.**
창 크기 2로 문장 "I love deep learning very much"에 대해 Skip-gram 모델 학습에 쓰일 (가운데, 문맥) 쌍을 모두 열거하라. 쌍의 수를 CBOW가 만들 쌍의 수와 견주어라.

??? success "연습문제 2 풀이"
    Skip-gram 쌍 (가운데 -> 문맥):

    | 가운데 | 문맥 낱말 |
    |--------|--------------|
    | I | love, deep |
    | love | I, deep, learning |
    | deep | I, love, learning, very |
    | learning | love, deep, very, much |
    | very | deep, learning, much |
    | much | learning, very |

    개별 쌍: (I, love), (I, deep), (love, I), (love, deep), (love, learning), (deep, I), (deep, love), (deep, learning), (deep, very), (learning, love), (learning, deep), (learning, very), (learning, much), (very, deep), (very, learning), (very, much), (much, learning), (much, very).

    모두 18개의 Skip-gram 쌍이다. CBOW는 (가운데 낱말 자리마다 하나씩) 6개의 쌍을 만들지만 쌍마다 문맥이 여러 낱말이다. 기울기 갱신의 수는 같지만(Skip-gram은 세대마다 18번) CBOW는 문맥을 묶어 처리한다.

---

**연습문제 3.**
원래 Word2Vec 논문처럼 (가운데 낱말용과 문맥 낱말용으로) 임베딩 행렬 두 개를 쓰도록 Skip-gram 모델을 고쳐라. 학습이 끝난 뒤 가장 가까운 이웃 질의로 두 행렬에서 나온 임베딩의 품질을 견주어라.

??? success "연습문제 3 풀이"
    ```python
    class SkipGramDualEmb(nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            super().__init__()
            self.center_emb = nn.Embedding(vocab_size, embedding_dim)
            self.context_emb = nn.Embedding(vocab_size, embedding_dim)
        
        def forward(self, center, context):
            c = self.center_emb(center)    # (배치, emb_dim)
            ctx = self.context_emb(context) # (배치, emb_dim)
            return torch.sum(c * ctx, dim=1)  # 내적 점수

    # 학습이 끝나면 두 행렬 모두 쓸모 있는 임베딩을 담는다.
    # 흔한 관행: center_emb를 쓰거나 둘을 평균한다:
    # final_emb = (model.center_emb.weight + model.context_emb.weight) / 2
    ```
    
    원래 Word2Vec은 수치적인 까닭으로 행렬을 따로 둔다. 학습 뒤에는 대체로 가운데 임베딩 행렬을 최종 낱말 벡터로 쓰지만, 두 행렬을 평균하면 양쪽 관점의 정보를 아우르므로 유추와 유사도 표준 자료에서 조금 더 나은 결과가 나올 때가 많다.

## 정리하며

**다룬 것** — Skip-gram 모델

Skip-gram의 목표는 가운데 낱말이 주어졌을 때 문맥 낱말을 볼 확률 $\sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t)$을 최대로 하는 것이며, $c$은 문맥 창의 크기, $T$은 말뭉치의 길이이다.

핵심 클래스는 `SkipGram`, `SkipGramDualEmb`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
