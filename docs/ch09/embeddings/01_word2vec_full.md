# Word2Vec 전체

Word2Vec은 Mikolov 등이 2013년에 내놓은, 낱말 임베딩을 배우는 가장 영향력 있는 알고리즘 가운데 하나이다. 서로 보완하는 두 구조가 있다. 둘레 문맥에서 가운데 낱말을 예측하는 연속 낱말 주머니(CBOW)와, 가운데 낱말에서 문맥 낱말을 예측하는 Skip-gram이다. 이 모듈은 두 방법을 함께 담고 음성 표본 추출과 잦은 낱말의 부표본 추출 같은 실용적인 학습 최적화도 곁들인 길잡이 노릇을 한다.

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

Word2Vec 틀은 비슷한 문맥에 나타나는 낱말은 뜻도 비슷하다는 분포 가설을 이용해 낱말 임베딩을 배운다. CBOW는 문맥 낱말의 임베딩을 평균하여 가운데 낱말을 예측하므로 잦은 낱말에 효율적이고 효과적이다. Skip-gram은 방향을 뒤집어 가운데 낱말에서 문맥 낱말을 하나씩 따로 예측하는데, 학습 예제마다 낱말 하나에 집중하므로 드문 낱말의 표현을 더 잘 잡아낸다.

어휘가 큰 데이터로 Word2Vec을 학습시키려면 효율을 높여야 한다. 어휘 전체에 대한 원래의 소프트맥스는 감당할 수 없이 비싸므로, 실제 구현은 계층적 소프트맥스(어휘를 이진 트리로 정리한다)나 음성 표본 추출(표적 낱말을 무작위로 뽑은 몇 개의 "음성" 낱말과 견주어 전체 소프트맥스를 어림한다)을 쓴다. 음성 표본 추출은 문제를 여러 이진 분류 과제로 바꾸며, 간단하고 효과적이어서 더 널리 쓰인다.

Word2Vec 임베딩의 품질은 낱말 유추 과제("왕 - 남자 + 여자 = 여왕" 따위)와 가장 가까운 이웃 찾기로 평가할 수 있다. 학습된 벡터 공간은 놀라운 대수적 구조를 보인다. 공간의 방향이 의미적·문법적 관계에 대응한다. 이런 성질 덕분에 Word2Vec은 자연어 처리의 돌파구가 되었고, 임베딩을 고정된 것이 아니라 문맥에 따라 달라지게 넓힌 트랜스포머의 문맥 임베딩에도 바탕을 놓았다.

## 연습문제

**연습문제 1.**
임베딩 층과 선형 출력층을 갖춘 최소한의 Skip-gram 모델을 PyTorch로 구현하라. 가운데 낱말의 색인을 받아 문맥 낱말을 예측하는 어휘 전체에 대한 로짓을 내놓아라. 창 크기 2로 문장 "the cat sat on the mat"에 대해 학습시켜라.

??? success "연습문제 1 풀이"
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

**연습문제 2.**
CBOW와 Skip-gram의 학습 목표가 어떻게 다른지 설명하라. 드문 낱말에는 어느 쪽이 더 나을 것으로 보이며 그 까닭은 무엇인가?

??? success "연습문제 2 풀이"
    CBOW는 문맥 임베딩의 평균으로 가운데 낱말을 예측한다. 평균을 내면 낱말 하나하나의 몫이 뭉개지므로 CBOW는 통계적 신호가 강한 잦은 낱말에 유리하다. 반면 Skip-gram은 (가운데, 문맥) 쌍마다 별개의 학습 예제로 다룬다. 드문 낱말도 가운데로 나오면 (문맥 자리마다 하나씩) 여러 학습 쌍을 만들므로 빈도에 견주어 더 많은 기울기 갱신을 받는다. 실험적으로 Skip-gram이 드문 낱말의 임베딩을 더 잘 만들고, CBOW는 학습이 빠르며 잦은 낱말에 잘 통한다.

---

**연습문제 3.**
위 Skip-gram 모델에 음성 표본 추출을 구현하라. 전체 소프트맥스 대신 양성 쌍마다 음성 낱말 5개를 뽑고 이진 교차 엔트로피 손실을 써라. 학습 속도와 마지막 임베딩 품질을 소프트맥스 판본과 견주어라.

??? success "연습문제 3 풀이"
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
    
    음성 표본 추출은 어휘 전체에 대한 로짓을 계산하지 않으므로 더 빠르다. 품질은 음성 표본의 수에 달려 있는데, 작거나 중간 크기의 어휘에서는 대체로 5~20개면 충분하다.

## 정리하며

**다룬 것** — Word2Vec 전체

Word2Vec 틀은 비슷한 문맥에 나타나는 낱말은 뜻도 비슷하다는 분포 가설을 이용해 낱말 임베딩을 배운다.

핵심 클래스는 `SkipGram`, `SkipGramNS`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
