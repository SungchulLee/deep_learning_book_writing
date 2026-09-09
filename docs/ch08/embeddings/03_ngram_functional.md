# N-그램 함수형 API

PyTorch는 대부분의 연산에 서로 같은 두 API를 제공한다. 모듈 API(`nn.CrossEntropyLoss()` 따위)와 함수형 API(`F.cross_entropy()` 따위)이다. 이 실습은 N-그램 언어 모형을 예로 손실 계산의 함수형 방법을 살피며, 두 API가 똑같은 결과를 내놓음을 보이고 어느 쪽이 언제 알맞은지 논한다. 이 비교는 선택이 성능이 아니라 순전히 코드를 짜는 방식의 문제임을 뚜렷이 보여 준다.

## 1. 코드

```python
# ========================================================
# 03_ngram_functional.py
# F.cross_entropy(함수형 API)를 쓰는 N-그램 언어 모형
# ========================================================

"""
실습 3: 함수형 API(F.cross_entropy) 쓰기

학습 목표:
- nn.Module과 함수형 API의 차이 이해하기
- nn.CrossEntropyLoss() 대신 F.cross_entropy 쓰기
- 어느 쪽을 언제 쓸지 익히기
- 앞 실습과 결과 견주기

예상 시간: 10분
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import word_embedding_ngram as ngr

# ========================================================
# 모듈 API와 함수형 API
# ========================================================

logits = torch.tensor([[2.0, 5.0, 1.0], [1.0, 3.0, 2.0]])
targets = torch.tensor([1, 2])

# 방법 1: 모듈 API
loss_module = nn.CrossEntropyLoss()
loss_1 = loss_module(logits, targets)

# 방법 2: 함수형 API
loss_2 = F.cross_entropy(logits, targets)

# 둘은 똑같다
assert torch.allclose(loss_1, loss_2)

# ========================================================
# 함수형 API로 학습하기
# ========================================================

model = ngr.NGramLanguageModeler()
optimizer = optim.SGD(model.parameters(), lr=ngr.ARGS.lr)
losses = ngr.train(model, F.cross_entropy, optimizer, epochs=ngr.ARGS.epochs, verbose=True)


if __name__ == "__main__":
    pass
```

## 2. 논의

PyTorch의 모듈 API와 함수형 API는 같은 것의 두 얼굴이다. `nn.CrossEntropyLoss()`은 함수처럼 부를 수 있는 객체를 만들고, `F.cross_entropy()`은 그 자체로 함수이다. 속을 들여다보면 모듈 API의 `forward` 메서드가 함수형 판본을 부를 뿐이다. 여러 번 호출하는 동안 (부류 가중치나 `ignore_index` 같은) 설정을 담아 두어야 할 때는 모듈 방식이 쓸모 있고, 상태가 없는 간단한 계산에는 함수형 방식이 더 간결하다.

이 구별은 손실 함수를 넘어선다. PyTorch의 많은 연산에 두 형태가 있다. `nn.ReLU()`과 `F.relu()`, `nn.Dropout()`과 `F.dropout()`, `nn.BatchNorm1d()`과 `F.batch_norm()`이 그렇다. 관행은 학습 가능한 매개변수나 (배치 정규화의 이동 통계량 같은) 상태를 지닌 층에는 모듈을 쓰고, 상태가 없는 연산에는 함수형 호출을 쓰는 것이다. 손실 함수는 (가중치나 축약 방식 같은) 설정은 있지만 학습 가능한 매개변수는 없어 그 사이 어딘가에 있다.

실제로 두 방법 모두 똑같이 옳고 널리 쓰인다. 많은 실무자가 한결같음을 위해 모듈 API를 즐겨 쓴다. 모든 부품을 `nn.Module` 속성으로 정의하면 `model.named_modules()`으로 모델을 온전히 들여다볼 수 있다. 간결하고 곧바르다는 이유로 함수형 API를 좋아하는 이들도 있다. PyTorch 문서는 모델 구조를 세울 때는 모듈 API를, 빠른 실험이나 한 번뿐인 계산에는 함수형 API를 권한다.

## 연습문제

**연습문제 1.**
부류 가중치가 $w = [1.0, 2.0, 0.5]$일 때 `nn.CrossEntropyLoss(weight=w)(logits, targets)`과 `F.cross_entropy(logits, targets, weight=w)`이 같은 결과를 내놓는지 확인하라.

??? success "연습문제 1 풀이"
    ```python
    logits = torch.tensor([[2.0, 5.0, 1.0], [1.0, 3.0, 2.0]])
    targets = torch.tensor([1, 2])
    w = torch.tensor([1.0, 2.0, 0.5])

    loss_mod = nn.CrossEntropyLoss(weight=w)(logits, targets)
    loss_func = F.cross_entropy(logits, targets, weight=w)

    print(f"Module: {loss_mod.item():.6f}")
    print(f"Functional: {loss_func.item():.6f}")
    print(f"Equal: {torch.allclose(loss_mod, loss_func)}")  # True
    ```
    둘은 똑같은 결과를 낸다. weight 매개변수는 부류마다 손실에 이바지하는 정도를 조절하며, 불균형한 데이터셋을 다루는 데 쓸모가 있다.

---

**연습문제 2.**
언제나 (함수형 API가 아니라) 모듈 API를 써야 하는 PyTorch 연산 세 가지를 열거하고 그 까닭을 설명하라.

??? success "연습문제 2 풀이"

    1. **`nn.BatchNorm1d` / `nn.BatchNorm2d`**: 순전파 사이에 이어져야 하는 이동 평균과 분산 통계량을 지닌다. 함수형 `F.batch_norm`은 그 완충기를 직접 관리해야 해서 실수하기 쉽다.
    
    2. **`nn.Dropout`**: `F.dropout`도 되지만 `training` 깃발을 직접 넘겨야 한다. `nn.Dropout`은 `self.training`을 알아서 살펴 학습 중에는 드롭아웃을 적용하고 `model.eval()`으로 평가할 때는 끈다.
    
    3. **`nn.Linear` / `nn.Conv2d`**: 학습 가능한 가중치와 편향 매개변수를 지닌다. 직접 만든 매개변수로 `F.linear(x, weight, bias)`을 쓸 수도 있지만, `nn.Linear`은 매개변수 등록과 초기화와 직렬화를 알아서 해 준다.

---

**연습문제 3.**
(도우미 함수 `ngr.train` 없이) `F.cross_entropy`을 바로 쓰는 학습 반복문을 작성하라. 기울기 0으로 만들기, 순전파, 손실 계산, 역전파, 최적화기 걸음을 빠짐없이 넣어라. 10세대마다 손실을 출력하라.

??? success "연습문제 3 풀이"
    ```python
    model = ngr.NGramLanguageModeler()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    trigrams = ngr.make_context_target_pairs(ngr.ARGS.test_sentence, ngr.ARGS.context_size)

    for epoch in range(100):
        total_loss = 0
        for context, target in trigrams:
            context_idxs = ngr.prepare_sequence(context, ngr.ARGS.word_to_ix)
            target_idx = torch.tensor([ngr.ARGS.word_to_ix[target]])
            
            optimizer.zero_grad()
            logits = model(context_idxs.unsqueeze(0))
            loss = F.cross_entropy(logits, target_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(trigrams):.4f}")
    ```

## 정리하며

**다룬 것** — N-그램 함수형 API

PyTorch의 모듈 API와 함수형 API는 같은 것의 두 얼굴이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
