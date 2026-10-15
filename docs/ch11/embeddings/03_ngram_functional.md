# N-그램 함수형 API

`nn.CrossEntropyLoss`와 `F.cross_entropy`는 같은 교차 엔트로피를 모듈과 함수 형태로 제공한다. 이 예제는 같은 로짓과 정답에서 두 결과를 비교하고, 함수형 API를 공통 N-그램 학습 함수에 전달한다. 옵션을 같게 설정했을 때의 동등성과 상태 관리의 차이를 구분한다.

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

`nn.CrossEntropyLoss`는 호출 가능한 모듈 객체이며, 클래스 가중치나 `ignore_index`, `reduction` 등의 설정을 보관한다. `F.cross_entropy`는 호출할 때 해당 설정을 인수로 받는다. 본문처럼 입력과 설정이 같으면 같은 손실을 계산한다.

모듈과 함수형 API의 차이가 모든 연산에서 단순한 문법 차이로 끝나지는 않는다. `nn.Dropout`은 자신의 학습 모드를 따르지만 `F.dropout`에는 학습 여부를 명시적으로 전달해야 한다. 배치 정규화의 함수형 연산에는 이동 통계량과 학습 여부를 직접 관리해 전달한다.

학습 가능한 가중치와 상태를 가진 구성 요소는 모듈로 등록하면 장치 이동, 저장·복원, 옵티마이저에 전달할 매개변수 수집을 관리하기 편하다. 함수형 연산을 써도 매개변수를 `nn.Parameter` 등으로 올바르게 등록하면 학습할 수 있다. `named_modules()`는 등록된 하위 모듈 목록이며 함수형 호출까지 포함한 전체 계산 그래프는 아니다.

본문의 동등성 검사는 고정된 입력의 손실을 비교한다. 이후 학습된 모델과 다른 페이지에서 별도로 초기화한 모델의 결과까지 같다는 뜻은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
클래스 가중치가 $w = [1.0, 2.0, 0.5]$일 때 `nn.CrossEntropyLoss(weight=w)(logits, targets)`과 `F.cross_entropy(logits, targets, weight=w)`이 같은 결과를 내놓는지 확인하라.

</div>

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
    두 호출은 입력과 옵션이 같아 같은 결과를 낸다. 클래스 가중치는 정답 클래스에 따라 손실의 기여를 조절한다. 정수 클래스 정답과 기본 `mean` 집계에서는 가중 손실의 합을 정답 클래스 가중치의 합으로 나눈다. 가중치를 넣었다는 사실만으로 불균형 문제의 해결이 보장되지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
모듈 API를 사용하면 상태 관리가 편리한 연산 세 가지를 들고, 함수형 API로 구현할 때 직접 관리해야 할 항목을 설명하라.

</div>

??? success "연습문제 2 풀이"
    배치 정규화는 이동 평균·분산 버퍼와 학습 모드를 관리한다. 드롭아웃은 학습 여부를 관리하며 함수형 호출에서는 `training=self.training` 등을 전달해야 한다. 선형층이나 합성곱은 가중치·편향의 등록과 초기화가 필요하다. 함수형 구현도 가능하므로 모듈 사용이 반드시 필요한 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
(도우미 함수 `ngr.train` 없이) `F.cross_entropy`을 바로 쓰는 학습 반복문을 작성하라. 기울기 0으로 만들기, 순전파, 손실 계산, 역전파, 옵티마이저 갱신을 빠짐없이 넣어라. 10에포크마다 손실을 출력하라.

</div>

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

    이 풀이에서는 예제마다 한 번씩 갱신한다. 공통 `ngr.train`의 전체 배치 갱신과는 에포크당 갱신 횟수와 학습률이 다르므로, 손실 차이를 함수형 API의 효과로 해석하면 안 된다.

## 정리하며

교차 엔트로피의 두 API는 같은 입력과 옵션에서 같은 손실을 계산한다. API를 선택할 때는 설정과 상태를 어디에서 관리할지 고려한다. 학습 결과를 비교하려면 갱신 횟수와 배치 구성도 맞춰야 한다.
