# 앞먹임 신경 말 모델

Bengio 외(2003)가 내놓은 앞먹임 신경 말 모델은 띄엄띄엄한 n-그램 세기에서 이어진 낱말 나타냄으로 넘어가는 길목이었다. 낱말마다 흩뿌린 묻힘을 배우고 붙박이 크기의 맥락 창에서 신경망으로 다음 낱말을 어림함으로써, 이 얼개는 n-그램 모델의 자료 성김 문제를 넘어서고 함께 쓰는 묻힘 공간을 거쳐 본 적 없는 낱말 짝에도 두루 통한다.

## 1. 코드

```python
"""
길잡이 04: 앞먹임 신경 말 모델
===============================================

신경 말 모델 얼개(Bengio 외, 2003):
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

## 2. 논의

앞먹임 신경 말 모델은 n-그램 모델의 성긴 셈 표를, 낱말을 이어진 벡터 나타냄(묻힘)에 대응시키는 빽빽한 신경망으로 갈음한다. 뜻이 비슷한 낱말은 이 묻힘 공간에서 자연스레 뭉치므로 모델이 두루 통하게 된다. 곧 "the cat sat"이 그럴듯하다고 배웠다면 "cat"과 "dog"의 묻힘이 비슷하므로 "the dog sat"도 그럴듯하다고 미룰 수 있다. 흩뿌린 나타냄을 거친 이 두루 통함이, 낱말마다 서로 남남인 기호로 다루는 예로부터의 n-그램 방식보다 나은 근본 이점이다.

이 얼개는 $n-1$개 낱말의 붙박이 맥락 창을 다룬다. 곧 그 묻힘을 찾아 벡터 하나로 이어 붙이고, tanh 깨어남을 갖춘 숨은 층에 통과시킨 뒤 낱말 곳간 크기의 내놓음으로 내리쬔다. 그 내놓음에 소프트맥스를 씌우면 다음 낱말에 대한 확률 분포가 나온다. 익히기는 엇갈린 엔트로피 손실(음의 로그 가능도)을 쓰며 확률적 기울기 내려가기나 Adam으로 가장 좋게 한다. 무게 첫자리매김(자비에)과 기울기 자르기가 든든한 익히기에 중요하다.

이런 이점에도 앞먹임 모델은 n-그램 모델의 붙박이 맥락 창이라는 한계를 그대로 지닌다. 아무 길이의 차례도 다룰 수 없고 들임 창의 자리마다 매개변수를 나눠 쓰지도 않는다. 이 한계는 때 걸음을 가로질러 숨은 상태를 지니는 되돌이 신경망과, 스스로 눈길로 온 차례를 한꺼번에 다루는 변환기가 다룬다. 그럼에도 앞먹임 말 모델은 요즘 모든 말 모델을 떠받치는 핵심 생각, 곧 배운 묻힘과 신경망 확률 어림을 들여왔다.

## 연습문제

**연습문제 1.**
낱말 곳간 크기 $V = 10{,}000$, 묻힘 차원 $d = 128$, 맥락 크기 $n = 4$, 숨은 차원 $h = 256$인 앞먹임 말 모델의 매개변수 전체 개수를 셈하여라.

??? success "연습문제 1 풀이"

    - 담기 켜: $V \times d = 10{,}000 \times 128 = 1{,}280{,}000$
    - 숨은 켜 짐: $(n \times d) \times h = (4 \times 128) \times 256 = 512 \times 256 = 131{,}072$
    - 숨은 층 치우침: $h = 256$
    - 날임 켜 짐: $h \times V = 256 \times 10{,}000 = 2{,}560{,}000$
    - 내놓는 층 치우침: $V = 10{,}000$
    
    모두: $1{,}280{,}000 + 131{,}072 + 256 + 2{,}560{,}000 + 10{,}000 = 3{,}981{,}328$개의 매개변수.
    
    묻힘 층과 내놓는 층이 대부분을 차지하며, 둘 다 낱말 곳간 크기에 한 줄로 비례해 커진다.

---

**연습문제 2.**
앞먹임 말 모델이 예로부터의 n-그램 모델과 달리 본 적 없는 n-그램에 드러난 부드럽게 하기를 하지 않아도 되는 까닭을 밝혀라. 이어진 묻힘 공간의 어떤 성질이 이를 가능하게 하는가?

??? success "연습문제 2 풀이"
    예로부터의 n-그램 모델은 낱말마다 서로 남남인 기호인 띄엄띄엄한 나타냄을 쓴다. 본 적 없는 바이그램은 셈이 0이므로 확률도 0이다. 앞먹임 모델은 낱말을 함께 쓰는 묻힘 공간의 이어진 벡터에 대응시킨다. 특정한 낱말 짝을 익히는 동안 한 번도 보지 못했더라도 다음 까닭으로 0이 아닌 확률을 셈할 수 있다:
    
    1. 비슷한 낱말의 묻힘 벡터가 벡터 공간에서 가까이 있다.
    2. 신경망은 이어진 내놓음을 내는 매끄러운 함수이다.
    3. 맥락 낱말 $A$이 목표 낱말 $B$을 어림한다고 모델이 배웠고 낱말 $A'$의 묻힘이 $A$과 비슷하면, 맥락 $A'$은 비슷한 숨은 깨어남과 비슷한 내놓음 확률을 낸다.
    
    묻힘 공간의 기하를 거친 이 넌지시 이루어지는 두루 통함이 드러난 부드럽게 하기를 필요 없게 만들며, 이는 신경 말 모델의 큰 실전 이점 가운데 하나이다.

---

**연습문제 3.**
`FeedforwardLanguageModel` 클래스에 떨구기 벌주기를 더하여라. 묻힘을 이어 붙인 뒤와 숨은 층 뒤에 떨구기를 쓴다. 떨구기 비율 0.1, 0.3, 0.5로 시험하고 말 모델에 떨구기가 왜 중요한지 밝혀라.

??? success "연습문제 3 풀이"
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
    
    큰 묻힘 행렬과 내놓는 행렬은, 특히 말뭉치가 작을 때 지나치게 맞추어지기 쉬우므로 말 모델에 떨구기가 중요하다. 익히는 동안 깨어남을 마구잡이로 0으로 만들어, 신경 세포 사이의 특정한 함께 맞춤 무늬에 기대지 않는 더 든든한 나타냄을 배우게 한다.

## 정리하며

**다룬 것** — 앞먹임 신경 말 모델

앞먹임 신경 말 모델은 n-그램 모델의 성긴 셈 표를, 낱말을 이어진 벡터 나타냄(묻힘)에 대응시키는 빽빽한 신경망으로 갈음한다.

고갱이 갈래는 `Vocabulary`, `FeedforwardLanguageModel`, `FeedforwardLMWithDropout`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
