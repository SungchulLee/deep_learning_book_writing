# 로스 함수 비교

다음 단어 예측은 문맥으로부터 어휘의 각 단어에 대한 점수를 구하는 다중 클래스 분류 문제다. 이 예제는 같은 구조의 N-그램 모델을 `nn.CrossEntropyLoss`, `F.cross_entropy`, 로그 소프트맥스와 `nn.NLLLoss`의 조합으로 각각 학습한다. 세 표현의 수학적 관계와 실제 학습 결과를 비교할 때 필요한 조건을 살펴본다.

## 1. 코드

```python
# ========================================================
# 01_loss_comparison.py
# 로스 함수 종합 비교
# ========================================================

"""
중급 실습 1: 로스 함수 종합 비교

학습 목표:
- 세 로스 함수를 나란히 견주기
- 수렴 방식 이해하기
- 성능의 차이 분석하기
- 결과를 그려 보고 해석하기

예상 시간: 20분
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils import word_embedding_ngram as ngr

print("=" * 70)
print("INTERMEDIATE TUTORIAL 1: Loss Function Comparison")
print("=" * 70)

# ========================================================
# 세 모델 모두 학습시키기
# ========================================================

print("\nTraining three models with different loss functions...")

# 모델 1: CrossEntropyLoss
model1 = ngr.NGramLanguageModeler()
loss_fn1 = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(model1.parameters(), lr=ngr.ARGS.lr)
losses1 = ngr.train(model1, loss_fn1, optimizer1, epochs=ngr.ARGS.epochs, verbose=False)

# 모델 2: F.cross_entropy
model2 = ngr.NGramLanguageModeler()
optimizer2 = optim.SGD(model2.parameters(), lr=ngr.ARGS.lr)
losses2 = ngr.train(model2, F.cross_entropy, optimizer2, epochs=ngr.ARGS.epochs, verbose=False)

# 모델 3: LogSoftmax와 함께 쓰는 NLLLoss
class NGramNLL(nn.Module):
    def __init__(self):
        super(NGramNLL, self).__init__()
        self.embeddings = nn.Embedding(ngr.ARGS.vocab_size, ngr.ARGS.embedding_dim)
        self.linear1 = nn.Linear(ngr.ARGS.context_size * ngr.ARGS.embedding_dim, 128)
        self.linear2 = nn.Linear(128, ngr.ARGS.vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((inputs.shape[0], -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        return F.log_softmax(out, dim=1)

model3 = NGramNLL()
loss_fn3 = nn.NLLLoss()
optimizer3 = optim.SGD(model3.parameters(), lr=ngr.ARGS.lr)
losses3 = ngr.train(model3, loss_fn3, optimizer3, epochs=ngr.ARGS.epochs, verbose=False)


if __name__ == "__main__":
    pass
```

**출력:**

```
======================================================================
INTERMEDIATE TUTORIAL 1: Loss Function Comparison
======================================================================

Training three models with different loss functions...
```

## 2. 논의

정수 클래스 레이블을 쓰고 웨이트, 무시할 인덱스, 로스 집계 방식 등의 설정을 맞추며 레이블 스무딩을 적용하지 않으면, 교차 엔트로피는 정답 클래스의 음의 로그 확률이다. 따라서 같은 로짓 $z$와 정답 $y$에 대해 `CrossEntropyLoss(z, y)`와 `NLLLoss(log_softmax(z), y)`는 같은 목적 함수를 계산한다. `F.cross_entropy`는 교차 엔트로피의 함수형 인터페이스다. 부동소수점 계산에서는 구현에 따른 작은 차이가 있을 수 있다.

`CrossEntropyLoss`에는 소프트맥스를 적용하지 않은 로짓을 넣는다. `NLLLoss`에는 로그 확률을 넣어야 한다. 입력의 의미를 혼동하면 같은 로스를 구현한 것이 아니다. 클래스 웨이트나 레이블 스무딩 등의 옵션이 달라져도 단순한 동등성 비교가 성립하지 않는다.

코드의 세 모델은 같은 구조지만 별도로 초기화된다. 공통 초기 웨이트를 복사하지 않으므로 로스 곡선의 차이를 로스 API의 차이로 돌릴 수 없다. API의 동등성을 비교하려면 초기 웨이트, 데이터와 순서, 옵티마이저 설정을 맞춰야 한다. 공통 학습 함수는 작은 말뭉치 전체를 한 배치로 사용하고 에포크마다 한 번 갱신한다.

본문 코드는 학습 로스를 `losses1`, `losses2`, `losses3`에 저장하지만 그래프나 요약 통계표를 만들지는 않는다. 제시된 출력만으로 수렴 속도나 최종 로스가 같다고 결론 내릴 수 없다. API 선택은 모델 출력이 로짓인지 로그 확률인지, 어떤 옵션이 필요한지에 맞춰 결정한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
로짓 텐서 `[[2.0, 1.0, 0.1]]`과 정답 `[0]`에 대해 `nn.CrossEntropyLoss`, `F.cross_entropy`, `nn.NLLLoss(F.log_softmax(...))`을 계산하여 수학적으로 같음을 확인하라. 셋 모두 같은 스칼라 로스 값을 내놓음을 보여라.

</div>

??? success "연습문제 1 풀이"
    ```python
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    target = torch.tensor([0])

    loss1 = nn.CrossEntropyLoss()(logits, target)
    loss2 = F.cross_entropy(logits, target)
    loss3 = nn.NLLLoss()(F.log_softmax(logits, dim=1), target)

    print(f"nn.CrossEntropyLoss: {loss1.item():.6f}")
    print(f"F.cross_entropy:     {loss2.item():.6f}")
    print(f"NLLLoss + LogSoftmax:{loss3.item():.6f}")
    ```
    셋 모두 약 $0.4170$이라는 같은 값을 출력한다. 이는 `CrossEntropyLoss(x, y) = NLLLoss(LogSoftmax(x), y)`임을 확인해 준다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`F.softmax` 뒤에 `torch.log`을 쓰는 것이 `F.log_softmax`을 바로 쓰는 것보다 수치적으로 나쁜 까닭을 설명하라. 로짓이 클 때 순진한 방법이 실패하는 예를 만들어라.

</div>

??? success "연습문제 2 풀이"
    `F.softmax` 자체도 큰 로짓에 대해 안정적인 계산을 사용한다. 여기서 문제는 매우 작은 확률이 먼저 0으로 언더플로된 뒤 `torch.log(0)`이 $-\infty$가 될 수 있다는 점이다. 아래의 유한한 로짓 예제에서 작은 클래스 확률에 이 문제가 생긴다.
    
    ```python
    x = torch.tensor([[1000.0, 1.0, 0.1]])
    naive = torch.log(F.softmax(x, dim=1))   # -inf나 nan이 들어갈 수 있다
    stable = F.log_softmax(x, dim=1)          # 수치적으로 올바르다
    print(f"Naive:  {naive}")
    print(f"Stable: {stable}")
    ```
    
    `F.log_softmax`는 확률을 먼저 저장한 뒤 로그를 취하지 않고, 로짓에서 안정적으로 계산한 로그 정규화 상수를 뺀다. 따라서 이 예제처럼 확률 표현에서 발생하는 언더플로를 피할 수 있다. 입력에 이미 무한대나 NaN이 있거나 표현 범위를 벗어나는 경우까지 모두 해결해 주는 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
레이블 스무딩(`nn.CrossEntropyLoss(label_smoothing=0.1)`)로 학습시킨 넷째 모델을 넣어 비교를 넓혀라. 그 로스 곡선을 나머지 셋과 함께 그리고, 레이블 스무딩이 수렴과 마지막 로스에 어떤 영향을 주는지 논하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    model4 = ngr.NGramLanguageModeler()
    loss_fn4 = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer4 = optim.SGD(model4.parameters(), lr=ngr.ARGS.lr)
    losses4 = ngr.train(model4, loss_fn4, optimizer4, epochs=ngr.ARGS.epochs, verbose=False)

    plt.figure(figsize=(10, 5))
    plt.plot(losses1, label='CrossEntropyLoss')
    plt.plot(losses4, label='CrossEntropyLoss (smoothing=0.1)')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Effect of Label Smoothing')
    plt.show()
    ```
    
    레이블 스무딩은 원핫 정답 분포와 균등 분포를 혼합하여 $q=(1-\epsilon)e_y+\epsilon/K$를 정답으로 사용한다. 여기서 $K$는 클래스 수이고 $\epsilon=0.1$이다. 교차 엔트로피의 이론적 최솟값은 이 정답 분포의 엔트로피이므로, 스무딩이 없는 로스와 수치만으로 성능을 비교하면 안 된다.

    코드의 넷째 모델도 별도로 초기화되므로 스무딩 효과만 분리한 비교는 아니다. 스무딩이 일반화나 수렴 속도에 주는 영향은 과제와 설정에 따라 달라진다. 초기 조건을 맞추고, 같은 검증 데이터에서 스무딩 없는 음의 로그 우도와 정확도 등을 함께 비교해야 한다.

## 정리하며

교차 엔트로피와 로그 소프트맥스·음의 로그 우도 조합은 입력과 옵션을 맞추면 같은 목적 함수를 표현한다. 별도로 초기화한 모델의 곡선은 API의 동등성을 직접 검증하지 못한다. 레이블 스무딩은 목적 함수 자체를 바꾸므로 공통 평가 기준이 필요하다.
