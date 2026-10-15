# N-그램 교차 엔트로피

N-그램 언어 모델은 앞선 $n-1$개 단어를 이용해 다음 단어를 예측한다. 이 예제는 두 단어를 문맥으로 쓰는 트라이그램 모델을 교차 엔트로피로 학습한다. 임베딩, 완전 연결 층, 학습 손실과 예측의 관계를 살펴본다.

## 1. 코드

```python
# ========================================================
# 02_ngram_cross_entropy.py
# CrossEntropyLoss를 쓰는 N-그램 언어 모형
# ========================================================

"""
실습 2: CrossEntropyLoss를 쓰는 N-그램 언어 모형

학습 목표:
- 완전한 n-그램 언어 모형 세우기
- 낱말 임베딩을 처음부터 끝까지 학습시키기
- CrossEntropyLoss 이해하기
- 학습의 진행을 그려 보기

예상 시간: 20분
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import word_embedding_ngram as ngr

# ========================================================
# 데이터와 모델
# ========================================================

trigrams = ngr.make_context_target_pairs(ngr.ARGS.test_sentence, ngr.ARGS.context_size)
model = ngr.NGramLanguageModeler()

# ========================================================
# 학습
# ========================================================

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=ngr.ARGS.lr)
losses = ngr.train(model, loss_function, optimizer, epochs=ngr.ARGS.epochs, verbose=True)

# ========================================================
# 예측하기
# ========================================================

test_contexts = [["thy", "beauty's"], ["deep", "sunken"], ["fair", "child"]]
ix_to_word = {idx: word for word, idx in ngr.ARGS.word_to_ix.items()}

for context in test_contexts:
    if all(word in ngr.ARGS.word_to_ix for word in context):
        context_idxs = ngr.prepare_sequence(context, ngr.ARGS.word_to_ix)
        with torch.no_grad():
            logits = model(context_idxs.unsqueeze(0))
            predicted_idx = torch.argmax(logits, dim=1).item()


if __name__ == "__main__":
    pass
```

## 2. 논의

모델은 두 문맥 단어의 임베딩을 이어 붙이고, 128차원 ReLU 은닉층을 거쳐 어휘 크기만큼의 로짓을 출력한다. 이어 붙이는 위치가 구별되므로 문맥 단어의 순서에 따라 다른 출력을 만들 수 있다. 다만 공통 데이터 함수는 문맥을 최근 단어부터 역순으로 구성하므로, 예측 입력도 같은 순서여야 한다. 본문의 수동 문맥 목록을 자연스러운 문장 순서로 해석하면 학습 입력의 규칙과 어긋날 수 있다.

정답 단어의 인덱스를 $y$라 하면 손실은 $-z_y+\log\sum_j\exp(z_j)$다. `CrossEntropyLoss`에는 확률이 아닌 로짓을 전달한다. 앞에서 소프트맥스를 적용하면 그 확률값을 다시 로짓으로 해석하게 되어 의도한 목적 함수와 달라진다.

공통 학습 함수는 작은 말뭉치의 모든 문맥·정답 쌍을 하나의 배치로 사용한다. 본문에서 만든 `trigrams` 변수를 직접 사용하는 방식은 아니며, 학습 함수 안에서 공통 데이터로 배치를 만든다. 문맥 크기를 바꿀 때는 모델의 입력 차원과 학습 데이터 생성 규칙을 함께 바꿔야 한다.

예측 코드는 모든 단어가 어휘에 있을 때만 실행하며, 예측 인덱스를 계산하지만 출력하지 않는다. 손실 그래프나 임베딩 유사도 분석도 포함되어 있지 않다. 학습 손실이 줄어드는 현상만으로 의미 관계를 충분히 학습했거나 새로운 문맥에 잘 일반화한다고 판단할 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
어휘 크기가 $V = 500$, 임베딩 차원이 $d = 10$, 문맥 크기가 2일 때 N-그램 모델의 학습 가능한 매개변수 총수를 계산하라(임베딩 층 + 은닉 단위 128개 층 + 출력층).

</div>

??? success "연습문제 1 풀이"

    - 임베딩 층: $V \times d = 500 \times 10 = 5{,}000$
    - 은닉층: 입력이 $2 \times 10 = 20$(이어 붙인 임베딩), 출력이 128. 매개변수: $20 \times 128 + 128 = 2{,}688$
    - 출력층: $128 \times 500 + 500 = 64{,}500$
    - 합계: 매개변수 $5{,}000 + 2{,}688 + 64{,}500 = 72{,}188$개

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
로짓 $z = [2.0, 5.0, 1.0]$과 정답 클래스 $y = 1$에 대해 교차 엔트로피 손실을 손으로 계산하라. 지수 취하기, 정규화, 로그 확률, 부호 바꾸기의 단계를 모두 보여라.

</div>

??? success "연습문제 2 풀이"

    1. 지수 취하기: $e^{2.0} = 7.389$, $e^{5.0} = 148.413$, $e^{1.0} = 2.718$
    2. 합: $7.389 + 148.413 + 2.718 = 158.520$
    3. 정답($y=1$)의 소프트맥스: $p_1 = 148.413 / 158.520 = 0.9362$
    4. 로그 확률: $\log(0.9362) = -0.0659$
    5. 부호 바꾸기: $\mathcal{L} = -(-0.0659) = 0.0659$
    
    확인해 보면 `F.cross_entropy(torch.tensor([[2.0, 5.0, 1.0]]), torch.tensor([1]))`이 약 0.0659를 돌려준다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
N-그램 모델을 4-그램(문맥 크기 3)으로 넓혀 트라이그램 모델과 성능을 비교하여라. 두 손실 곡선을 같은 그림에 그리고 문맥 길이와 모델의 복잡함 사이의 트레이드오프를 논하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    # context_size를 바꾸고 다시 학습
    # 4-그램 모형은 은닉층의 입력이 더 크다: 3 * emb_dim
    # 매개변수는 늘지만 문맥이 더 넓어진다
    
    class FourGramModel(nn.Module):
        def __init__(self, vocab_size, emb_dim, context_size=3):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, emb_dim)
            self.linear1 = nn.Linear(context_size * emb_dim, 128)
            self.linear2 = nn.Linear(128, vocab_size)
        
        def forward(self, inputs):
            embeds = self.embeddings(inputs).view(inputs.shape[0], -1)
            out = torch.relu(self.linear1(embeds))
            return self.linear2(out)
    ```
    
    아래 클래스는 문맥 크기 3을 받는 구조만 정의하며 학습과 비교를 완성하지는 않는다. 공통 학습 함수는 여전히 기본 문맥 크기로 배치를 만들기 때문에, 모델만 교체해서는 입력 차원이 맞지 않는다. 데이터 생성과 학습 루프도 함께 맞춰야 한다.

    문맥 크기를 2에서 3으로 늘리면 은닉층 가중치가 $128d$개 증가한다. 문장을 이어 붙인 길이 $T$의 토큰열에서는 예제가 $T-2$개에서 $T-3$개로 줄어든다. 두 모델은 예측 대상도 일부 다르므로 공통 평가 위치와 검증 데이터를 맞춰 비교한다. 더 긴 문맥이 항상 더 낮은 최종 손실이나 더 좋은 일반화를 보장하지는 않는다.

## 정리하며

N-그램 모델은 문맥의 임베딩을 이어 붙여 다음 단어의 로짓을 계산한다. 문맥 길이뿐 아니라 단어의 나열 순서도 학습과 예측에서 같아야 한다. 모델 구조를 바꿀 때는 공통 학습 함수의 데이터 생성 방식도 확인한다.
