# Skip-gram 모델

Skip-gram은 중심 단어로부터 주변 문맥 단어를 예측하는 Word2Vec 구조다. CBOW가 여러 문맥 단어를 묶어 중심 단어를 예측하는 것과 입력·정답의 역할이 다르다. 본문의 코드는 개념을 출력하는 템플릿이며, 모델 정의는 연습문제에 제시되어 있다. 학습 결과를 보여 주는 완성 예제는 아니다.

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

문맥 반경이 $c$일 때 Skip-gram은 유효한 각 중심·문맥 위치 쌍의 로그 확률을 더한 값을 최대화한다. 문장 경계를 넘지 않도록 쌍을 만들며, 가장자리 위치에서는 문맥 수가 $2c$보다 적을 수 있다. 이 합은 확률 자체가 아니라 로그 우도다.

전체 소프트맥스를 사용하는 기본 점수는 입력 벡터 $v_{w_I}$와 출력 벡터 $u_{w_O}$의 내적이다. 조건부 확률은 $p(w_O\mid w_I)=\exp(u_{w_O}^{\top}v_{w_I})/\sum_{w=1}^{V}\exp(u_w^{\top}v_{w_I})$다. 연습문제 1의 선형층은 출력 임베딩 역할의 가중치에 편향도 더하므로, 편향 없는 기본 식과 정확히 같은 구조는 아니다.

중심 단어 하나가 여러 예측 쌍에 참여하므로, Skip-gram은 CBOW와 다른 학습 신호를 받는다. 다만 드문 단어가 적게 등장해도 항상 좋은 임베딩을 얻는다는 뜻은 아니다. 쌍의 수가 늘어나는 것과 독립적인 관측 정보가 늘어나는 것도 구분해야 한다. 업데이트 횟수는 쌍의 수뿐 아니라 배치 구성에 따라 달라진다.

큰 어휘에서는 전체 소프트맥스의 비용이 커지므로 네거티브 샘플링이나 계층적 소프트맥스 등을 사용할 수 있다. 네거티브 샘플링은 같은 확률을 단순히 더 빠르게 계산하는 방식이 아니라 다른 목적 함수를 사용하는 학습 방식이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
입력 임베딩 층과 출력 선형층을 갖춘 기본 Skip-gram 모델 클래스를 PyTorch로 구현하라. 중심 단어의 인덱스을 받아 어휘에 대한 로짓을 돌려주는 forward 메서드를 작성하라.

</div>

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

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
창 크기 2로 문장 "I love deep learning very much"에 대해 Skip-gram 모델 학습에 쓰일 (가운데, 문맥) 쌍을 모두 열거하라. 쌍의 수를 CBOW가 만들 쌍의 수와 비교하여라.

</div>

??? success "연습문제 2 풀이"
    Skip-gram 쌍 (가운데 -> 문맥):

    | 가운데 | 문맥 단어 |
    |--------|--------------|
    | I | love, deep |
    | love | I, deep, learning |
    | deep | I, love, learning, very |
    | learning | love, deep, very, much |
    | very | deep, learning, much |
    | much | learning, very |

    개별 쌍: (I, love), (I, deep), (love, I), (love, deep), (love, learning), (deep, I), (deep, love), (deep, learning), (deep, very), (learning, love), (learning, deep), (learning, very), (learning, much), (very, deep), (very, learning), (very, much), (much, learning), (much, very).

    가장자리의 짧은 문맥을 허용하면 Skip-gram 쌍은 18개이고 CBOW 예제는 6개다. 예제마다 한 번씩 갱신하는 경우에는 각각 18회와 6회이며, 배치 학습에서는 배치 크기에 따라 달라진다. 완전한 문맥만 허용하면 중심 위치는 deep과 learning 두 곳이므로 Skip-gram 쌍은 8개, CBOW 예제는 2개다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
원래 Word2Vec 논문처럼 (중심 단어용과 문맥 단어용으로) 임베딩 행렬 두 개를 쓰도록 Skip-gram 모델을 고쳐라. 학습이 끝난 뒤 가장 가까운 이웃 질의로 두 행렬에서 나온 임베딩의 품질을 비교하여라.

</div>

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
    
    두 행렬은 단어가 중심 입력으로 쓰이는 역할과 문맥 정답으로 쓰이는 역할을 구분한다. 연습문제 1의 출력 선형층 가중치도 이미 출력 임베딩 역할을 한다. 위 클래스는 지정한 쌍의 내적 점수만 반환하므로, 전체 소프트맥스로 학습하려면 모든 문맥 후보의 점수를 계산해야 하고, 네거티브 샘플링을 쓰려면 양성·음성 쌍과 이진 손실을 추가해야 한다. 중심 벡터, 문맥 벡터, 두 벡터의 평균 중 어느 것이 더 유용한지는 평가 과제에 따라 확인한다.

## 정리하며

Skip-gram은 중심·문맥 쌍별 예측을 통해 단어 표현을 학습한다. 중심과 문맥 벡터의 역할, 문장 경계 처리, 배치별 갱신 횟수를 구분해야 한다. 이 페이지의 모델 조각에는 데이터 준비와 학습 루프가 포함되어 있지 않다.
