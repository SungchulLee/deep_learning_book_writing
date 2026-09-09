# N-그램 교차 엔트로피

N-그램 언어 모형은 앞선 $n-1$개의 낱말로 다음 낱말을 예측하며, 낱말 임베딩을 배우기에 가장 단순하면서도 가르침이 많은 틀 가운데 하나이다. 이 실습은 `nn.CrossEntropyLoss`으로 완전한 트라이그램 모형을 PyTorch로 세우고 텍스트 말뭉치로 학습시킨 뒤 나온 임베딩을 살펴본다. 문맥-표적 쌍 만들기에서 학습, 예측, 유사도 분석까지의 전 과정을 보인다.

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

N-그램 언어 모형의 구조는 세 부분으로 이루어진다. 문맥 낱말마다 조밀한 벡터로 보내는 임베딩 층, 이어 붙인 문맥 임베딩을 처리하는 ReLU 활성화 은닉층, 그리고 어휘에 대한 로짓을 내놓는 출력층이다. (CBOW처럼 평균하지 않고) 문맥 임베딩을 이어 붙이므로 낱말의 순서 정보가 지켜진다. 모델이 "cat sat"과 "sat cat"이 서로 다른 예측을 낸다는 것을 배울 수 있다.

CrossEntropyLoss은 이 다중 부류 분류 문제에 자연스러운 선택이다. 로그 소프트맥스와 음의 로그 가능도를 수치적으로 안정된 하나의 연산 $\mathcal{L} = -\log \frac{\exp(z_y)}{\sum_j \exp(z_j)}$으로 합치며, $z_y$은 옳은 부류 $y$의 로짓이다. 이 손실을 최소로 하는 것은 옳은 다음 낱말에 매기는 확률을 최대로 하는 것과 같다. 이 손실은 날 로짓을 입력으로 받는다. CrossEntropyLoss을 쓰면서 모델 안에서 소프트맥스를 적용하는 것은 소프트맥스를 두 번 쓰게 되는 흔한 실수이다.

학습 과정에서 몇 가지 중요한 무늬가 드러난다. 초반 세대에는 모델이 기본적인 낱말 빈도를 배우며 손실이 빠르게 떨어지고, 문맥의 무늬를 붙잡기 시작하면 더 완만해진다. 학습이 끝나면 임베딩 가중치에 의미가 비슷한 낱말끼리 벡터도 비슷한 표현이 담긴다. 모델은 문맥 낱말의 임베딩을 찾아 이어 붙이고 로짓 점수가 가장 높은 낱말을 골라 처음 보는 문맥에도 예측을 내놓는다.

## 연습문제

**연습문제 1.**
어휘 크기가 $V = 500$, 임베딩 차원이 $d = 10$, 문맥 크기가 2일 때 N-그램 모형의 학습 가능한 매개변수 총수를 계산하라(임베딩 층 + 은닉 단위 128개 층 + 출력층).

??? success "연습문제 1 풀이"

    - 임베딩 층: $V \times d = 500 \times 10 = 5{,}000$
    - 은닉층: 입력이 $2 \times 10 = 20$(이어 붙인 임베딩), 출력이 128. 매개변수: $20 \times 128 + 128 = 2{,}688$
    - 출력층: $128 \times 500 + 500 = 64{,}500$
    - 합계: 매개변수 $5{,}000 + 2{,}688 + 64{,}500 = 72{,}188$개

---

**연습문제 2.**
로짓 $z = [2.0, 5.0, 1.0]$과 표적 부류 $y = 1$에 대해 교차 엔트로피 손실을 손으로 계산하라. 지수 취하기, 정규화, 로그 확률, 부호 바꾸기의 단계를 모두 보여라.

??? success "연습문제 2 풀이"

    1. 지수 취하기: $e^{2.0} = 7.389$, $e^{5.0} = 148.413$, $e^{1.0} = 2.718$
    2. 합: $7.389 + 148.413 + 2.718 = 158.520$
    3. 표적($y=1$)의 소프트맥스: $p_1 = 148.413 / 158.520 = 0.9362$
    4. 로그 확률: $\log(0.9362) = -0.0659$
    5. 부호 바꾸기: $\mathcal{L} = -(-0.0659) = 0.0659$
    
    확인해 보면 `F.cross_entropy(torch.tensor([[2.0, 5.0, 1.0]]), torch.tensor([1]))`이 약 0.0659를 돌려준다.

---

**연습문제 3.**
N-그램 모형을 4-그램(문맥 크기 3)으로 넓혀 트라이그램 모형과 성능을 견주어라. 두 손실 곡선을 같은 그림에 그리고 문맥 길이와 모델의 복잡함 사이의 맞바꿈을 논하라.

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
    
    4-그램 모형은 쓸 문맥이 더 많으므로 대체로 마지막 손실이 더 낮다. 다만 매개변수가 더 많고(은닉층의 입력이 $2d$에서 $3d$으로 커진다) 같은 말뭉치에서 나오는 학습 예제가 더 적다(예제마다 앞선 낱말 3개가 필요하다). 말뭉치가 작으면 4-그램 모형이 더 쉽게 과적합하므로 규제를 쓰지 않는 한 트라이그램이 낫다.

## 정리하며

**다룬 것** — N-그램 교차 엔트로피

N-그램 언어 모형의 구조는 세 부분으로 이루어진다.

핵심 클래스는 `FourGramModel`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
