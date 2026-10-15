# 순방향 신경 언어 모델

[Bengio 등의 신경 확률 언어 모델](https://jmlr.org/papers/v3/bengio03a.html)은 단어 임베딩과 신경망을 함께 학습해 고정 길이 문맥의 다음 단어 확률을 추정한다. 관측된 n-그램마다 독립적인 빈도만 저장하는 방식과 달리 여러 문맥이 매개변수를 공유한다. 이 예제는 그 핵심인 임베딩 연결·은닉층·출력층을 보여 주는 축약 구조다.

## 1. 코드와 기존 출력

```python
"""
길잡이 04: 앞먹임 신경 말 모델
===============================================

신경 말 모델 구조(Bengio 외, 2003):
들임: 맥락 낱말 w_{t-n+1}, ..., w_{t-1}
내놓음: 다음 낱말 w_t에 대한 확률 분포

구조:
1. 묻힘 층: R^d의 C(w)
2. 숨은 층: h = tanh(W * concat(묻힘) + b)
3. 내놓는 층: P(w_t | 맥락) = softmax(U*h + c)

손실: 음의 로그 가능도(엇갈린 엔트로피)
L = -1/N sum log P(w_t | w_{t-n+1}, ..., w_{t-1})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import numpy as np
import random

# ========================================================================
# 메인
# ========================================================================


class Vocabulary:
    """신경 말 모델의 낱말 곳간 다스리기."""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.PAD_TOKEN = "<pad>"
        self.UNK_TOKEN = "<unk>"
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, self.END_TOKEN]
        for token in self.special_tokens:
            self._add_word(token)
    
    def _add_word(self, word: str) -> int:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self.word2idx[word]
    
    def build_from_corpus(self, corpus: List[str], min_freq: int = 1) -> None:
        word_counts = {}
        for sentence in corpus:
            for word in sentence.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1
        for word, count in word_counts.items():
            if count >= min_freq:
                self._add_word(word)
    
    def word_to_idx(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])
    
    def idx_to_word(self, idx: int) -> str:
        return self.idx2word[idx]
    
    def __len__(self) -> int:
        return len(self.word2idx)


class FeedforwardLanguageModel(nn.Module):
    """앞먹임 신경 말 모델(Bengio 외, 2003)."""
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 context_size: int, hidden_dim: int):
        super(FeedforwardLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(context)
        embeds_flat = embeds.view(embeds.size(0), -1)
        hidden = torch.tanh(self.fc1(embeds_flat))
        logits = self.fc2(hidden)
        return logits


if __name__ == "__main__":
    torch.manual_seed(42)
    
    corpus = ["the cat sat on the mat", "the dog sat on the log"] * 10
    vocab = Vocabulary()
    vocab.build_from_corpus(corpus)
    
    model = FeedforwardLanguageModel(len(vocab), embedding_dim=32,
                                      context_size=3, hidden_dim=64)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**기존 출력:**

```
Parameters: 7,275
```

## 2. 입력과 출력의 의미

입력은 B×c의 정수 단어 ID이며 c는 context_size다. 임베딩을 조회하면 B×c×d, 이어 붙이면 B×(cd)가 된다. 선형 층과 tanh를 거쳐 B×h의 은닉 표현을 만들고 출력층이 B×V의 로짓을 반환한다. forward 자체는 소프트맥스 확률을 반환하지 않는다. 다음 단어의 정수 ID와 CrossEntropyLoss를 사용하면 손실 내부에서 로그 소프트맥스가 적용된다.

각 창의 입력 단어와 그 바로 다음 목표 단어를 준비해야 한다. 문장 시작에 필요한 문맥 채우기, 종료 예측, PAD를 학습·평가에서 제외하는 정책도 따로 정해야 한다. Vocabulary가 특수 토큰을 등록한다는 사실만으로 그런 데이터 구성과 손실 마스킹까지 구현되는 것은 아니다.

## 3. 공유 표현과 일반화

같은 임베딩·은닉층을 여러 문맥에 사용하므로 한 조합에서 얻은 학습 신호가 다른 조합의 예측에도 영향을 준다. 비슷한 문맥 표현은 비슷한 출력을 낼 수 있지만 의미가 비슷한 단어가 반드시 가깝게 학습되거나 미관측 조합을 정확히 예측한다는 보장은 없다. 유한한 로짓의 수학적 소프트맥스는 어휘의 각 사건에 양의 확률을 주지만, 이는 올바른 일반화·확률 보정과 같은 뜻이 아니다.

어휘는 학습 자료로 만들고 검증·시험의 미등록 단어를 UNK로 매핑한다. build_from_corpus는 소문자화하지만 word_to_idx는 입력을 소문자로 바꾸지 않으므로 호출부의 전처리를 일치시켜야 한다. 이미 만든 Vocabulary를 다시 확장하면 기존 항목이 남고, 모델의 임베딩·출력 크기는 자동으로 늘어나지 않는다.

## 4. 고정 문맥과 실제 시연 범위

입력 창의 위치마다 fc1의 서로 다른 열이 쓰이지만 단어 임베딩 표는 모든 위치가 공유한다. 고정 창을 이동해 긴 문장을 처리할 수는 있어도 한 예측이 직접 보는 문맥은 c개로 제한된다. RNN은 상태를 순차적으로 전달하고 트랜스포머는 허용된 문맥 위치에 어텐션을 적용하는 다른 설계를 사용한다.

현재 main은 작은 말뭉치로 어휘를 만들고 모델의 매개변수 수만 출력한다. 학습 자료 구성·손실·옵티마이저·학습 루프와 퍼플렉서티 평가는 없다. Xavier 초기화는 적용하지만 기울기 클리핑은 구현하지 않는다. 따라서 기존 출력은 학습이나 생성 품질을 검증한 결과가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
V=10,000, 임베딩 차원 d=128, 입력 문맥 단어 수 c=4, 은닉 차원 h=256일 때 이 모델의 매개변수 수를 계산하라.

</div>

??? success "연습문제 1 풀이"

    임베딩 Vd=1,280,000개, fc1 가중치 cdh=131,072개와 편향 256개, fc2 가중치 hV=2,560,000개와 편향 10,000개다. 총합은 3,981,328개다. 여기서 c=4는 입력 단어 수이므로 다음 단어까지 포함한 n-그램 차수로는 5에 해당한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
빈도 MLE와 신경 모델이 미관측 n-그램에 주는 확률의 차이 및 임베딩 공유의 역할을 설명하라.

</div>

??? success "연습문제 2 풀이"

    평활화하지 않은 빈도 MLE는 미관측 조합에 0을 준다. 신경 모델은 공유 매개변수로 로짓을 계산하고 유한 로짓의 소프트맥스는 각 어휘 사건에 양의 확률을 준다. 비슷한 임베딩·문맥 표현을 통한 통계적 공유가 일반화를 도울 수 있지만 양의 확률만으로 예측의 정확성이나 의미적 유사성이 보장되지는 않는다. UNK와 자료 분리도 여전히 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 변형에서 드롭아웃이 적용되는 위치와 train·eval의 차이를 설명하고 비율을 비교할 평가 기준을 제안하라.

</div>

??? success "연습문제 3 풀이"

    이어 붙인 임베딩과 tanh 은닉 출력 뒤에 각각 드롭아웃을 적용한다. train 모드에서는 일부 활성화를 0으로 만들고 나머지를 재조정하며 eval에서는 비활성화한다. 비율 0.1·0.3·0.5는 같은 자료 분리·학습 조건에서 검증 토큰당 음의 로그우도나 퍼플렉서티로 선택할 수 있다. 큰 비율이 항상 과적합을 더 잘 해결하지는 않는다. 아래 코드는 구조만 정의하며 이 편집에서는 학습·비교를 수행하지 않는다.

    ```python
    class FeedforwardLMWithDropout(nn.Module):
        def __init__(self, vocab_size, embedding_dim, context_size, 
                     hidden_dim, dropout=0.2):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
        def forward(self, context):
            embeds = self.embeddings(context)
            embeds_flat = embeds.view(embeds.size(0), -1)
            embeds_flat = self.dropout1(embeds_flat)
            hidden = torch.tanh(self.fc1(embeds_flat))
            hidden = self.dropout2(hidden)
            return self.fc2(hidden)
    ```


## 정리하며

순방향 언어 모델은 고정 문맥의 임베딩을 연결해 다음 단어 로짓을 계산한다. 공유 표현은 희소한 조합의 일반화를 도울 수 있지만 데이터 구성·경계 처리·손실·독립 평가를 함께 설계해야 한다.
