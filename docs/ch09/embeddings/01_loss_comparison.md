# 손실 함수 비교

손실 함수는 신경망 학습의 중심이며, 다음 낱말 예측 같은 분류 과제에 PyTorch는 서로 같은 값을 주는 여러 형태를 제공한다. 이 실습은 `nn.CrossEntropyLoss`, `F.cross_entropy`, (명시적인 로그 소프트맥스와 함께 쓰는) `nn.NLLLoss`으로 똑같은 N-그램 언어 모형 세 개를 학습시키고 수렴 거동을 견준다. 핵심은 셋이 수학적으로 같은 값을 계산하며 API 방식만 다르다는 것이다.

## 1. 코드

```python
# ========================================================
# 01_loss_comparison.py
# 손실 함수 종합 비교
# ========================================================

"""
중급 실습 1: 손실 함수 종합 비교

학습 목표:
- 세 손실 함수를 나란히 견주기
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

## 2. 논의

여기서 견주는 세 손실 형태는 모두 같은 수학적 목표, 곧 범주형 분포 아래에서 옳은 부류의 음의 로그 가능도를 구현한다. `nn.CrossEntropyLoss`이 가장 흔한 형태로, 날 로짓을 받아 안에서 로그 소프트맥스를 적용한 뒤 음의 로그 가능도를 계산한다. `F.cross_entropy`은 그 함수형 짝으로, 모듈 객체가 아니라 상태 없는 함수 호출일 뿐 값은 똑같다. `nn.NLLLoss` 판본은 모델이 `F.log_softmax`으로 로그 확률을 직접 내놓아야 하므로 $\text{CrossEntropy} = \text{LogSoftmax} + \text{NLLLoss}$이라는 두 단계 분해가 코드에 드러난다.

세 방법이 수학적으로 똑같으므로 학습 곡선도 사실상 서로 바꿔 놓아도 된다. 작은 차이는 모델마다 무작위 초기 상태가 달라서 생길 뿐이다. 셋 중 무엇을 쓸지는 순전히 코드를 어떻게 짜느냐의 문제이다. 대부분의 응용에서는 `nn.CrossEntropyLoss`을 즐겨 쓰는데, 수치적으로 안정적이고(로그 소프트맥스와 음의 로그 가능도를 합쳐 계산하므로 중간에 정밀도를 잃지 않는다) 모델 구조를 고칠 필요가 없기 때문이다.

이 실습의 시각화 부분은 손실 곡선, 앞쪽과 뒤쪽 세대를 확대한 그림, 방법 사이의 절대 차이, 요약 통계표를 그린다. 이 그림들은 세 방법의 수렴 속도와 마지막 손실, 전체 개선율이 거의 같음을 확인해 주며, 선택의 기준이 성능이 아니라 코드의 명료함이어야 함을 뒷받침한다.

## 연습문제

**연습문제 1.**
로짓 텐서 `[[2.0, 1.0, 0.1]]`과 표적 `[0]`에 대해 `nn.CrossEntropyLoss`, `F.cross_entropy`, `nn.NLLLoss(F.log_softmax(...))`을 계산하여 수학적으로 같음을 확인하라. 셋 모두 같은 스칼라 손실 값을 내놓음을 보여라.

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

**연습문제 2.**
`F.softmax` 뒤에 `torch.log`을 쓰는 것이 `F.log_softmax`을 바로 쓰는 것보다 수치적으로 나쁜 까닭을 설명하라. 로짓이 클 때 순진한 방법이 실패하는 예를 만들어라.

??? success "연습문제 2 풀이"
    순진한 계산 `torch.log(F.softmax(x, dim=1))`은 먼저 로짓에 지수를 취하고(값이 크면 넘칠 수 있다) 정규화한 뒤 로그를 취하는데(확률이 아주 작으면 $-\infty$이 나올 수 있다), 예를 들면 다음과 같다.
    
    ```python
    x = torch.tensor([[1000.0, 1.0, 0.1]])
    naive = torch.log(F.softmax(x, dim=1))   # -inf나 nan이 들어갈 수 있다
    stable = F.log_softmax(x, dim=1)          # 수치적으로 올바르다
    print(f"Naive:  {naive}")
    print(f"Stable: {stable}")
    ```
    
    `F.log_softmax`은 항등식 $\log(\text{softmax}(x_i)) = x_i - \log(\sum_j e^{x_j})$을 쓰고 지수를 취하기 전에 가장 큰 로짓을 빼므로(로그-합-지수 요령) 지수에서의 넘침과 로그에서의 밑넘침을 모두 피한다.

---

**연습문제 3.**
이름표 매끄럽게 하기(`nn.CrossEntropyLoss(label_smoothing=0.1)`)로 학습시킨 넷째 모델을 넣어 비교를 넓혀라. 그 손실 곡선을 나머지 셋과 함께 그리고, 이름표 매끄럽게 하기가 수렴과 마지막 손실에 어떤 영향을 주는지 논하라.

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
    
    이름표 매끄럽게 하기는 원-핫 표적 분포를 혼합 분포 $(1 - \epsilon) \cdot \text{원-핫} + \epsilon / K$으로 바꾼다. 여기서 $K$은 부류의 수이고 $\epsilon = 0.1$이다. (표적 분포의 엔트로피가 0이 아니므로) 매끄럽게 하면 마지막 손실이 더 높지만, 모델이 지나치게 확신하지 않게 되어 일반화가 나아질 때가 많다. 수렴 속도는 대체로 비슷하고 손실 곡선이 더 높은 값에서 평평해진다.

## 정리하며

**다룬 것** — 손실 함수 비교

여기서 견주는 세 손실 형태는 모두 같은 수학적 목표, 곧 범주형 분포 아래에서 옳은 부류의 음의 로그 가능도를 구현한다.

핵심 클래스는 `NGramNLL`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
