# Word2Vec 개요

Word2Vec은 주변 문맥을 예측하는 과제로 단어 벡터를 학습한다. 대표 구조는 문맥으로 중심 단어를 예측하는 CBOW와 중심 단어로 문맥 단어를 예측하는 Skip-gram이다. 본문 코드는 안내 문자열만 출력하며, 연습문제에 간단한 Skip-gram과 네거티브 샘플링 학습 예제가 있다.

## 1. 코드

```python
"""Word2Vec 전체."""

# ========================================================================
# 메인
# ========================================================================
# 완전한 Word2Vec 구현
# CBOW와 Skip-gram을 최적화와 함께 담았다
print("Word2Vec Full Implementation - Combines CBOW (Tutorial 02_intermediate/02) and Skip-gram")


if __name__ == "__main__":
    pass
```

**출력:**

```
Word2Vec Full Implementation - Combines CBOW (Tutorial 02_intermediate/02) and Skip-gram
```

## 2. 논의

분포 가설은 비슷한 문맥에서 쓰이는 단어가 관련된 의미를 가질 수 있다는 관점이다. Word2Vec은 이를 예측 문제로 바꾼다. CBOW는 문맥 벡터를 결합해 중심 단어를 예측하고, Skip-gram은 중심·문맥 쌍별로 예측한다. 두 구조의 품질과 학습 시간은 데이터와 설정에 따라 비교해야 한다.

연습문제 1은 작은 문장의 모든 유효한 중심·문맥 쌍을 만들고, 전체 소프트맥스를 포함한 교차 엔트로피로 학습한다. 문장 양 끝에서는 문맥 수가 줄어든다. `set`으로 만든 어휘의 순서는 실행 환경에 따라 달라질 수 있으므로 결과를 비교하거나 벡터를 저장할 때 단어·인덱스 사전도 함께 관리해야 한다.

계층적 소프트맥스는 트리 경로의 확률을 이용하고, 네거티브 샘플링은 관측 쌍과 잡음 쌍을 구분하는 목적 함수를 사용한다. 네거티브 샘플링을 전체 소프트맥스 확률의 단순 근사 계산으로 설명하면 두 목적 함수의 차이를 놓친다. 본문에는 빈도가 높은 단어를 일부 제외하는 서브샘플링이 구현되어 있지 않다.

유추와 이웃 검색은 벡터 관계를 살펴보는 방법이지만, 벡터 차가 모든 의미 관계를 정확히 보존하는 것은 아니다. 이 예제처럼 한 문장으로 학습한 결과를 일반적인 언어 표현의 품질로 해석하지 않는다. Word2Vec은 단어마다 고정 벡터를 두므로 문맥에 따라 같은 단어의 표현을 바꾸는 모델과도 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
임베딩 레이어와 선형 출력 레이어를 갖춘 최소한의 Skip-gram 모델을 PyTorch로 구현하라. 중심 단어의 인덱스을 받아 문맥 단어를 예측하는 어휘 전체에 대한 로짓을 내놓아라. 창 크기 2로 문장 "the cat sat on the mat"에 대해 학습시켜라.

</div>

??? success "풀이"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim

    sentence = "the cat sat on the mat".split()
    vocab = list(set(sentence))
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    # Skip-gram 쌍 (가운데, 문맥) 만들기
    window = 2
    pairs = []
    for i, word in enumerate(sentence):
        for j in range(max(0, i - window), min(len(sentence), i + window + 1)):
            if i != j:
                pairs.append((w2i[word], w2i[sentence[j]]))

    class SkipGram(nn.Module):
        def __init__(self, vocab_size, emb_dim):
            super().__init__()
            self.center_emb = nn.Embedding(vocab_size, emb_dim)
            self.output = nn.Linear(emb_dim, vocab_size)
        def forward(self, center):
            return self.output(self.center_emb(center))

    model = SkipGram(V, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(200):
        total_loss = 0
        for c, ctx in pairs:
            logits = model(torch.tensor([c]))
            loss = loss_fn(logits, torch.tensor([ctx]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(pairs):.4f}")
    ```

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
CBOW와 Skip-gram의 학습 단위가 어떻게 다른지 설명하고, 드문 단어의 표현에 관한 결론을 낼 때 주의할 점을 서술하라.

</div>

??? success "풀이"
    CBOW는 여러 문맥 단어를 결합한 예제 하나에서 중심 단어를 예측한다. Skip-gram은 중심·문맥 쌍 각각을 학습 예제로 쓴다. 중심 단어 한 번의 등장으로 여러 쌍을 만들 수 있지만 독립적인 정보가 그만큼 늘어나는 것은 아니다. 드문 단어의 품질은 문맥의 다양성, 잡음 분포와 학습량에도 달려 있으므로 모든 데이터에서 Skip-gram이 우세하다고 단정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
위 Skip-gram 모델에 네거티브 샘플링을 구현하라. 전체 소프트맥스 대신 양성 쌍마다 음성 단어 5개를 뽑고 이진 교차 엔트로피 로스를 써라. 학습 속도와 마지막 임베딩 품질을 소프트맥스 판본과 비교하여라.

</div>

??? success "풀이"
    ```python
    class SkipGramNS(nn.Module):
        def __init__(self, vocab_size, emb_dim):
            super().__init__()
            self.center_emb = nn.Embedding(vocab_size, emb_dim)
            self.context_emb = nn.Embedding(vocab_size, emb_dim)
        
        def forward(self, center, context, negatives):
            c = self.center_emb(center)          # (배치, emb)
            pos = self.context_emb(context)       # (배치, emb)
            neg = self.context_emb(negatives)     # (배치, num_neg, emb)
            
            pos_score = torch.sum(c * pos, dim=1)                    # (배치,)
            neg_score = torch.bmm(neg, c.unsqueeze(2)).squeeze(2)    # (배치, num_neg)
            
            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10).mean()
            neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-10).mean()
            return pos_loss + neg_loss

    model_ns = SkipGramNS(V, 10)
    optimizer_ns = optim.Adam(model_ns.parameters(), lr=0.01)
    num_neg = 5

    for epoch in range(200):
        total_loss = 0
        for c, ctx in pairs:
            negs = torch.randint(0, V, (1, num_neg))
            loss = model_ns(torch.tensor([c]), torch.tensor([ctx]), negs)
            optimizer_ns.zero_grad()
            loss.backward()
            optimizer_ns.step()
            total_loss += loss.item()
    ```
    
    풀이에서는 음성 항을 샘플 수 $k$에 대해 합하지 않고 평균한다. 따라서 표준 목적 함수의 음성 항 합에 비해 그 비중이 $1/k$로 줄어든다. 음성 샘플은 균등 분포에서 뽑으며 정답 단어와 중복될 수도 있다. 작은 상수를 더한 `log(sigmoid(...))` 계산은 극단적인 점수에서 로스를 바꿀 수 있으므로 안정적인 로짓 기반 표현과 구분해야 한다.

    이 문장의 어휘는 5개인데 양성 1개와 음성 5개를 계산하므로 점수 수가 전체 어휘보다 적지도 않다. 큰 어휘에서 가능한 계산량 절감이 이 작은 예제의 속도 향상을 보장하지는 않는다. 서로 다른 목적 함수의 최종 로스는 직접 비교하지 말고 공통 임베딩 평가 과제를 사용한다.

## 정리하며

Word2Vec은 단어와 문맥의 예측을 통해 고정 단어 벡터를 학습한다. CBOW와 Skip-gram의 학습 단위, 전체 소프트맥스와 네거티브 샘플링의 목적 함수를 구분한다. 본문의 안내 코드와 연습문제의 구현 범위를 나눠 읽어야 한다.
