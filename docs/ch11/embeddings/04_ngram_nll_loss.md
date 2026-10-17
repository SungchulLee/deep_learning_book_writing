# N-그램 모델과 음의 로그 우도 로스

`NLLLoss`는 정답 클래스의 로그 확률에 음수를 붙여 로스를 계산한다. 이 예제는 N-그램 모델의 출력에 `log_softmax`를 적용하고, 교차 엔트로피를 사용하는 두 API와 비교한다. 로짓과 로그 확률의 구분, 수치적으로 안정적인 계산, 비교 실험의 조건을 살펴본다.

## 1. 코드

```python
# ========================================================
# 04_ngram_nll_loss.py
# NLLLoss를 쓰는 N-그램 언어 모형
# ========================================================

"""
실습 4: NLLLoss(음의 로그 가능도 로스) 이해하기

학습 목표:
- NLLLoss가 무엇인지 이해하기
- CrossEntropyLoss = LogSoftmax + NLLLoss라는 관계 익히기
- 로그 확률을 내놓도록 모델 고치기
- NLLLoss와 CrossEntropyLoss 견주기

예상 시간: 15분

NLLLoss란 무엇인가?
----------------
NLLLoss(음의 로그 가능도 로스)는 로그 확률이 표적과 얼마나 잘 맞는지를
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

# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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
1. 마지막 레이어로 LogSoftmax 더하기
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
def forward(self, x):
    logits = self.layers(x)
    return logits  # 활성화 없음
    
loss = nn.CrossEntropyLoss()
output = model(input)
loss_value = loss(output, target)  # PyTorch가 속에서 소프트맥스를 건다

With NLLLoss:
def forward(self, x):
    logits = self.layers(x)
    return F.log_softmax(logits, dim=1)  # log_softmax을 건다
    
loss = nn.NLLLoss()
output = model(input)  # 이미 로그 확률이다
loss_value = loss(output, target)  # 음의 로그 가능도를 곧바로 셈한다

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

??? note "전체 출력 (229줄)"

    ```
    ======================================================================
    TUTORIAL 4: N-gram Model with NLLLoss
    ======================================================================

    ======================================================================
    PART 1: CrossEntropyLoss vs NLLLoss
    ======================================================================

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


    --- Demonstration ---
    Logits (raw scores): tensor([2., 5., 1.])
    Target: 1

    Method 1 - CrossEntropyLoss directly:
      Loss: 0.065884

    Method 2 - LogSoftmax + NLLLoss:
      Log probabilities: tensor([-3.0659, -0.0659, -4.0659])
      Loss: 0.065884

    Are they equal? True
    Yes! CrossEntropyLoss = LogSoftmax + NLLLoss

    For reference:
      Softmax probabilities: tensor([0.0466, 0.9362, 0.0171])
      Log softmax: tensor([-3.0659, -0.0659, -4.0659])
      Log(prob[target]): -0.065884
      Negative log likelihood: 0.065884

    ======================================================================
    PART 2: Modifying the Model for NLLLoss
    ======================================================================

    NLLLoss를 쓰려면 모델의 순전파가 날 로짓 대신 로그 확률을 내놓도록
    고쳐야 한다.

    필요한 변경:
    1. 마지막 레이어로 LogSoftmax 더하기
    2. CrossEntropyLoss 대신 NLLLoss 쓰기

    고친 모델을 만들어 보자!


    Modified model created!
    Key change: forward() returns log probabilities using F.log_softmax()

    ======================================================================
    PART 3: Training with NLLLoss
    ======================================================================

    Training configuration:
      Model: NGramLanguageModelerNLL (outputs log probabilities)
      Loss function: nn.NLLLoss
      Optimizer: SGD
      Learning rate: 0.001
      Epochs: 10

    Starting training...

      epoch   1/10  loss 4.0883
      epoch   2/10  loss 4.0879
      epoch   3/10  loss 4.0876
      epoch   4/10  loss 4.0872
      epoch   5/10  loss 4.0868
      epoch   6/10  loss 4.0865
      epoch   7/10  loss 4.0861
      epoch   8/10  loss 4.0858
      epoch   9/10  loss 4.0854
      epoch  10/10  loss 4.0851

    ======================================================================
    PART 4: Comparing All Three Loss Functions
    ======================================================================

    Training comparison models...
    Training model 1: nn.CrossEntropyLoss...
    Training model 2: F.cross_entropy...

    ----------------------------------------------------------------------
    Final losses:
      nn.CrossEntropyLoss:        4.043935
      F.cross_entropy:            4.132479
      nn.NLLLoss (+ LogSoftmax):  4.085099
    ----------------------------------------------------------------------

    All three approaches give similar results!
    Small differences are due to random weight initialization.

    ======================================================================
    PART 5: When to Use Each Loss Function
    ======================================================================

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
    def forward(self, x):
        logits = self.layers(x)
        return logits  # 활성화 없음
        
    loss = nn.CrossEntropyLoss()
    output = model(input)
    loss_value = loss(output, target)  # PyTorch가 속에서 소프트맥스를 건다

    With NLLLoss:
    def forward(self, x):
        logits = self.layers(x)
        return F.log_softmax(logits, dim=1)  # log_softmax을 건다
        
    loss = nn.NLLLoss()
    output = model(input)  # 이미 로그 확률이다
    loss_value = loss(output, target)  # 음의 로그 가능도를 곧바로 셈한다

    Key Insight:
    -----------
    Both approaches are equivalent:
      CrossEntropyLoss(logits, y) = NLLLoss(log_softmax(logits), y)

    The choice is about where you apply the log_softmax:
    - CrossEntropyLoss: Applied inside the loss function
    - NLLLoss: Applied in your model's forward pass


    ======================================================================
    PART 6: Why LogSoftmax? (Numerical Stability)
    ======================================================================

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


    --- Demonstration of Numerical Issues ---
    Softmax of large logits: tensor([0.2447, 0.6652, 0.0900])
    Notice the very small probability: 9.00e-02
    Log of softmax: tensor([-1.4076, -0.4076, -2.4076])
    See the very negative number? -2.41

    Using log_softmax directly: tensor([-1.4076, -0.4076, -2.4076])
    Much more stable and accurate!

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

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

    ======================================================================
    END OF TUTORIAL 4
    ======================================================================
    ```


## 2. 논의

`NGramLanguageModelerNLL`은 문맥 단어의 임베딩을 이어 붙이고 두 선형 레이어와 ReLU를 거친 뒤 로그 확률을 반환한다. `nn.NLLLoss`는 이 출력에서 정답 클래스의 값을 골라 음수를 취한다. 입력은 형태가 $(B,C)$인 정수 단어 ID 텐서이며 $C$는 문맥 크기다. 출력은 $(B,V)$의 로그 확률이다.

같은 로짓과 정수 정답에 기본 옵션을 사용하면 `cross_entropy`와 `nll_loss(log_softmax(...))`는 같은 로스다. 반면 이 페이지의 학습 비교는 세 모델을 따로 초기화한다. 그래프와 최종 로스의 차이는 초기 웨이트 등의 영향도 포함하므로, 로스 API 자체의 수렴 차이를 보여 주는 실험으로 해석할 수 없다.

코드와 출력의 수치 안정성 설명 중 `softmax([1000,1001])`가 약 `[0,1]`이라는 부분은 잘못된 예시다. 소프트맥스는 두 값에서 같은 상수를 빼도 변하지 않으므로 이 확률은 약 `[0.269,0.731]`이다. 문제는 로짓의 절댓값보다 차이가 매우 클 때 작은 확률이 0으로 언더플로되고, 그 뒤 로그를 취하면 음의 무한대가 된다는 점이다. `log_softmax`는 확률을 먼저 표현하지 않아 이 문제를 피한다.

여기에는 순환 상태를 갱신하는 RNN이 없다. 고정 길이 문맥을 입력하는 피드포워드 언어 모델이며, 짧은 학습 말뭉치의 로스 감소만으로 장기 의존성이나 일반적인 언어 이해 능력을 평가할 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
어휘 크기 $V$, 임베딩 차원 $d$, 문맥 크기 $C$일 때 이 모델의 파라메터 수를 구하라.

</div>

??? success "풀이"
    임베딩은 $Vd$개, 첫 선형 레이어는 $128Cd+128$개, 출력 레이어는 $128V+V$개다. 합계는 $Vd+128Cd+129V+128$이다. 기본 $C=2,d=10$에서는 $139V+2688$개다. ReLU와 로그 소프트맥스는 파라메터를 추가하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
모델 입력과 NLLLoss 정답에 필요한 형태·자료형·값의 범위를 설명하라.

</div>

??? success "풀이"
    모델 입력은 $(B,C)$ 형태의 정수 인덱스이며 각 값은 $0$ 이상 $V$ 미만이어야 한다. 이 예제에서는 `torch.long`을 사용한다. 정답은 $(B,)$ 형태의 long 텐서이며 클래스 범위에 있어야 한다. 로스에 전달하는 모델 출력은 $(B,V)$의 로그 확률이다. 입력을 float 이미지 텐서로 검사하는 것은 이 모델에 맞지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
로짓을 NLLLoss에 바로 넣는 경우와 학습 로스만으로 일반화를 판단하는 경우의 문제를 설명하라.

</div>

??? success "풀이"
    NLLLoss는 입력을 로그 확률로 해석하므로 정규화되지 않은 로짓을 넣으면 의도한 음의 로그 우도가 아니다. 출력에 log_softmax가 있는지 확인한다. 또한 작은 말뭉치의 학습 로스가 감소해도 처음 보는 문장에 대한 성능은 알 수 없으므로 분리된 검증 데이터에서 토큰별 음의 로그 우도 등을 평가한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

NLLLoss는 로그 확률을 입력으로 받는다. 모델 출력과 로스의 입력 규약을 맞추면 교차 엔트로피와 같은 목적 함수를 구현할 수 있다. API 비교에서는 초기 조건을 맞추고, 수치 안정성은 로짓의 차이와 확률의 표현 범위를 기준으로 설명해야 한다.
