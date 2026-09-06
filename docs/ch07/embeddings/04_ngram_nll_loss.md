# 실습 4

실습 4: NLLLoss(음의 로그 가능도 손실) 이해하기. 예상 시간 15분

순차열 모형은 시간적이고 순서가 있는 데이터를 다루는 데 바탕이 된다. 이 구현은 낱말 임베딩의 핵심 착상을 다루며, 순환 계산과 학습된 표현이 시각 사이의 의존을 어떻게 붙잡는지 보인다.

## 코드

```python
# ========================================================
# 04_ngram_nll_loss.py
# NLLLoss를 쓰는 N-그램 언어 모형
# ========================================================

"""
실습 4: NLLLoss(음의 로그 가능도 손실) 이해하기

학습 목표:
- NLLLoss가 무엇인지 이해하기
- CrossEntropyLoss = LogSoftmax + NLLLoss라는 관계 익히기
- 로그 확률을 내놓도록 모델 고치기
- NLLLoss와 CrossEntropyLoss 견주기

예상 시간: 15분

NLLLoss란 무엇인가?
----------------
NLLLoss(음의 로그 가능도 손실)는 로그 확률이 표적과 얼마나 잘 맞는지를
잰다. CrossEntropyLoss와 달리 날 로짓이 아니라 로그 확률을 입력으로
받는다.

핵심 관계:
  CrossEntropyLoss(logits, target) = NLLLoss(log_softmax(logits), target)
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

print("=" * 70)
print("TUTORIAL 4: N-gram Model with NLLLoss")
print("=" * 70)

# ========================================================
# 1부: 관계 이해하기
# ========================================================

print("\n" + "=" * 70)
print("PART 1: CrossEntropyLoss vs NLLLoss")
print("=" * 70)

print("""
수학적인 관계:
------------------------------

CrossEntropyLoss는 연산 두 가지를 합친 것이다.
1. LogSoftmax: 로짓을 로그 확률로 바꾼다
2. NLLLoss: 음의 로그 가능도를 계산한다

식:
  CrossEntropyLoss(x, y) = NLLLoss(LogSoftmax(x), y)

왜 중요한가:
- CrossEntropyLoss: 입력은 로짓(날 점수)이다
- NLLLoss: 입력은 로그 확률이다

수학적으로 확인해 보자!
""")

# 시연
print("\n--- Demonstration ---")

# 예제 데이터 만들기
logits = torch.tensor([[2.0, 5.0, 1.0]])  # 날 점수
target = torch.tensor([1])  # 색인 1이 정답이다

print(f"Logits (raw scores): {logits[0]}")
print(f"Target: {target.item()}")

# 방법 1: CrossEntropyLoss를 바로 쓰기
ce_loss = F.cross_entropy(logits, target)
print(f"\nMethod 1 - CrossEntropyLoss directly:")
print(f"  Loss: {ce_loss:.6f}")

# 방법 2: LogSoftmax와 NLLLoss
log_probs = F.log_softmax(logits, dim=1)
nll_loss = F.nll_loss(log_probs, target)
print(f"\nMethod 2 - LogSoftmax + NLLLoss:")
print(f"  Log probabilities: {log_probs[0]}")
print(f"  Loss: {nll_loss:.6f}")

# 서로 같은지 확인
print(f"\nAre they equal? {torch.allclose(ce_loss, nll_loss)}")
print("Yes! CrossEntropyLoss = LogSoftmax + NLLLoss")

# 확률 보이기
probs = torch.softmax(logits, dim=1)
print(f"\nFor reference:")
print(f"  Softmax probabilities: {probs[0]}")
print(f"  Log softmax: {log_probs[0]}")
print(f"  Log(prob[target]): {log_probs[0, target.item()]:.6f}")
print(f"  Negative log likelihood: {-log_probs[0, target.item()]:.6f}")

# ========================================================
# 2부: NLLLoss를 위해 고친 모델
# ========================================================

print("\n" + "=" * 70)
print("PART 2: Modifying the Model for NLLLoss")
print("=" * 70)

print("""
NLLLoss를 쓰려면 모델의 순전파가 날 로짓 대신 로그 확률을 내놓도록
고쳐야 한다.

필요한 변경:
1. 마지막 층으로 LogSoftmax 더하기
2. CrossEntropyLoss 대신 NLLLoss 쓰기

고친 모델을 만들어 보자!
""")


class NGramLanguageModelerNLL(nn.Module):
    """
    NLLLoss와 함께 쓰려고 로그 확률을 내놓는 N-그램 모형.
    
    원래 모델과의 유일한 차이는 순전파에 log_softmax를 더한
    것뿐이다.
    """
    
    def __init__(self, vocab_size=None, embedding_dim=None, context_size=None):
        super(NGramLanguageModelerNLL, self).__init__()
        
        if vocab_size is None:
            vocab_size = ngr.ARGS.vocab_size
        if embedding_dim is None:
            embedding_dim = ngr.ARGS.embedding_dim
        if context_size is None:
            context_size = ngr.ARGS.context_size
        
        # 앞과 같은 구조
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        # 앞과 같은 순전파...
        embeds = self.embeddings(inputs)
        embeds = embeds.view((embeds.shape[0], -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        
        # 핵심 차이: 돌려주기 전에 log_softmax 적용
        # 로짓을 로그 확률로 바꾼다
        log_probs = F.log_softmax(out, dim=1)
        return log_probs  # 로짓이 아니라 로그 확률을 돌려준다


print("\nModified model created!")
print("Key change: forward() returns log probabilities using F.log_softmax()")

# ========================================================
# 3부: NLLLoss로 학습하기
# ========================================================

print("\n" + "=" * 70)
print("PART 3: Training with NLLLoss")
print("=" * 70)

# 고친 모델 만들기
model_nll = NGramLanguageModelerNLL()
loss_function = nn.NLLLoss()  # 이번에는 NLLLoss를 쓴다
optimizer = optim.SGD(model_nll.parameters(), lr=ngr.ARGS.lr)

print("\nTraining configuration:")
print(f"  Model: NGramLanguageModelerNLL (outputs log probabilities)")
print(f"  Loss function: nn.NLLLoss")
print(f"  Optimizer: SGD")
print(f"  Learning rate: {ngr.ARGS.lr}")
print(f"  Epochs: {ngr.ARGS.epochs}")

print("\nStarting training...\n")

# 모델을 학습시킨다
losses_nll = ngr.train(model_nll, loss_function, optimizer, epochs=ngr.ARGS.epochs, verbose=True)

# ========================================================
# 4부: 세 방법 모두 견주기
# ========================================================

print("\n" + "=" * 70)
print("PART 4: Comparing All Three Loss Functions")
print("=" * 70)

print("\nTraining comparison models...")

# CrossEntropyLoss(모듈 API)로 학습
print("Training model 1: nn.CrossEntropyLoss...")
model_ce = ngr.NGramLanguageModeler()
loss_ce = nn.CrossEntropyLoss()
optimizer_ce = optim.SGD(model_ce.parameters(), lr=ngr.ARGS.lr)
losses_ce = ngr.train(model_ce, loss_ce, optimizer_ce, epochs=ngr.ARGS.epochs, verbose=False)

# F.cross_entropy(함수형 API)로 학습
print("Training model 2: F.cross_entropy...")
model_func = ngr.NGramLanguageModeler()
optimizer_func = optim.SGD(model_func.parameters(), lr=ngr.ARGS.lr)
losses_func = ngr.train(model_func, F.cross_entropy, optimizer_func, epochs=ngr.ARGS.epochs, verbose=False)

# 셋 모두 그리기
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(losses_ce, label='nn.CrossEntropyLoss', linewidth=2, alpha=0.8)
ax.plot(losses_func, label='F.cross_entropy', linewidth=2, alpha=0.8)
ax.plot(losses_nll, label='nn.NLLLoss (with LogSoftmax)', linewidth=2, alpha=0.8)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Comparing Three Loss Function Approaches', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "-" * 70)
print("Final losses:")
print(f"  nn.CrossEntropyLoss:        {losses_ce[-1]:.6f}")
print(f"  F.cross_entropy:            {losses_func[-1]:.6f}")
print(f"  nn.NLLLoss (+ LogSoftmax):  {losses_nll[-1]:.6f}")
print("-" * 70)

print("\nAll three approaches give similar results!")
print("Small differences are due to random weight initialization.")

# ========================================================
# 5부: 언제 무엇을 쓸까
# ========================================================

print("\n" + "=" * 70)
print("PART 5: When to Use Each Loss Function")
print("=" * 70)

print("""
선택 길잡이:
--------------

CrossEntropyLoss를 쓸 때:
✓ 모델이 (소프트맥스 없이) 날 로짓을 내놓을 때
✓ PyTorch가 알아서 다 처리하기를 바랄 때
✓ 분류에서 가장 흔한 선택이다
✓ 처음 배우는 이에게 권한다

NLLLoss를 쓸 때:
✓ 다른 데 쓸 로그 확률이 필요할 때
✓ 모델이 이미 log_softmax를 적용할 때
✓ 확률 계산을 더 세밀하게 다루고 싶을 때
✓ 확률 분포를 직접 다룰 때

코드 견주기:
---------------

CrossEntropyLoss를 쓸 때:
```python
def forward(self, x):
    logits = self.layers(x)
    return logits  # No activation
    
loss = nn.CrossEntropyLoss()
output = model(input)
loss_value = loss(output, target)  # PyTorch applies softmax internally
```

With NLLLoss:
```python
def forward(self, x):
    logits = self.layers(x)
    return F.log_softmax(logits, dim=1)  # Apply log_softmax
    
loss = nn.NLLLoss()
output = model(input)  # Already log probabilities
loss_value = loss(output, target)  # Directly compute NLL
```

Key Insight:
-----------
Both approaches are equivalent:
  CrossEntropyLoss(logits, y) = NLLLoss(log_softmax(logits), y)

The choice is about where you apply the log_softmax:
- CrossEntropyLoss: Applied inside the loss function
- NLLLoss: Applied in your model's forward pass
""")

# ========================================================
# 6부: 수치적 안정성
# ========================================================

print("\n" + "=" * 70)
print("PART 6: Why LogSoftmax? (Numerical Stability)")
print("=" * 70)

print("""
Why use log_softmax instead of log(softmax)?
-------------------------------------------

log_softmax is numerically more stable!

Naive approach (unstable):
  probs = softmax(logits)
  log_probs = log(probs)  # 수치 문제가 생길 수 있다!

Better approach (stable):
  log_probs = log_softmax(logits)  # 수학적으로 다듬어졌다

Example issue with naive approach:
  softmax([1000, 1001]) → [~0, ~1]  
  log(~0) → -inf (problem!)
  
log_softmax handles this correctly using mathematical tricks.

Lesson: Always use log_softmax, never log(softmax)!
""")

# 불안정함 보이기
print("\n--- Demonstration of Numerical Issues ---")

large_logits = torch.tensor([[100.0, 101.0, 99.0]])

# 순진한 방법 (문제가 생길 수 있다)
probs_naive = F.softmax(large_logits, dim=1)
print(f"Softmax of large logits: {probs_naive[0]}")
print(f"Notice the very small probability: {probs_naive[0, 2]:.2e}")

log_probs_naive = torch.log(probs_naive)
print(f"Log of softmax: {log_probs_naive[0]}")
print(f"See the very negative number? {log_probs_naive[0, 2]:.2f}")

# 제대로 된 방법 (안정적이다)
log_probs_stable = F.log_softmax(large_logits, dim=1)
print(f"\nUsing log_softmax directly: {log_probs_stable[0]}")
print("Much more stable and accurate!")

# ========================================================
# 핵심 요점
# ========================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Mathematical relationship:
   CrossEntropyLoss = LogSoftmax + NLLLoss
   
2. Input requirements:
   - CrossEntropyLoss: Expects logits (raw scores)
   - NLLLoss: Expects log probabilities
   
3. When to use each:
   - CrossEntropyLoss: Default choice, easiest to use
   - NLLLoss: When you need log probabilities elsewhere
   
4. Numerical stability:
   - Always use log_softmax, never log(softmax)
   - PyTorch functions are optimized for stability
   
5. All approaches give same results:
   - Choose based on convenience and code structure
   - Performance is equivalent
   
6. Best practices:
   - For classification: Use CrossEntropyLoss
   - Keep model and loss consistent
   - Don't apply softmax in model if using CrossEntropyLoss

Congratulations!
---------------
You've completed all basic tutorials and understand:
✓ Word embeddings
✓ N-gram language models
✓ Three different loss function approaches
✓ The relationship between them

Ready for intermediate tutorials? Go to 02_intermediate/!
""")

print("=" * 70)
print("END OF TUTORIAL 4")
print("=" * 70)


if __name__ == "__main__":
    pass
```

## 논의

`NGramLanguageModelerNLL` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로 PyTorch의 자동 미분이 학습 중 기울기 계산을 알아서 처리한다. 이런 모듈식 설계 덕분에 부품 하나하나를 고치거나 모델을 더 큰 파이프라인에 넣기 쉽다.

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화를 쓴 `NGramLanguageModelerNLL`의 학습 가능한 매개변수 총수를 계산하라. 가중치와 편향을 모두 넣어 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
층이나 블록의 수를 설정할 수 있도록 `NGramLanguageModelerNLL`을 확장하라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 훑는다. (보통의 파이썬 리스트가 아니라) `nn.ModuleList`을 써야 PyTorch가 최적화를 위해 모든 매개변수를 등록한다. `for n in [2, 4, 8]: model = NGramLanguageModelerNLL(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
