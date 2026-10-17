# 정확성

딥러닝의 정확성을 평가할 때는 의도한 식을 올바르게 계산하는 구현의 정확성과, 보지 않은 데이터에서의 예측 성능을 구분한다. 올바른 경사와 학습 루프가 좋은 일반화를 자동으로 보장하지는 않는다.

## 1. 정의

알고리즘이 정확하다는 것은 모든 유효한 입력 $x$에 대해 종료하고 기대하는 출력을 낸다는 뜻이다.

$$
\forall \, x \in \mathcal{X}: \; f(x) = \text{Expected}(x)
$$

결정적 알고리즘에서 정확성은 절대적이다. 뉴럴 넷트웍에서 정확성은 통계적이다. 즉 데이터 분포에 대한 기대 로스 $\mathbb{E}[\ell(f(x), y)]$이 작기를 바란다.

---

## 2. 설명

**루프 불변식(loop invariant)** 은 알고리즘의 정확성을 증명하는 고전적인 기법이다. 루프 불변식은 매 반복의 전후에 성립하는 성질이다.

1. **초기화(initialization)**: 첫 반복 전에 참이다
2. **유지(maintenance)**: 반복 전에 참이면 반복 후에도 참이다
3. **종료(termination)**: 루프가 끝났을 때 불변식이 정확성을 함의한다

뉴럴 넷트웍의 학습 로스가 매 단계 줄어드는지를 루프 불변식으로 일반화할 수는 없다. 충분히 매끄러운 목적과 적절한 러닝레이트를 가정한 전체 배치 경사 하강법에는 하강 부등식이 있지만, 확률적 경사에는 잡음 항이 들어가고 매 단계 감소가 보장되지 않는다. 실무에서는 다음 점검을 결합한다.

- **경사 확인(gradient checking)**: 자동 미분 경사를 수치적 유한 차분 경사와 비교한다
- **단일 배치 과적합**: 충분한 용량과 적절한 설정에서 작은 배치를 적합할 수 있는지 확인한다. 정규화·레이블 모순·양의 최소 로스 등으로 0에 도달하지 않을 수도 있다
- **텐서 모양 단위 검사**: 각 레이어의 중간 텐서 모양을 단언문으로 확인한다

---

## 3. 예제

```python
import torch
import torch.nn as nn

# ── 검사 1. 경사 확인 (gradient checking) ───────────────────────────
# 자동 미분이 정말 맞는 값을 주는지 확인하는 표준 방법이다.
# 미분의 정의로 되돌아가 수치적으로 그래디언트를 구한 뒤 견준다.
def numerical_gradient(f, x, eps=1e-5):
    grad = torch.zeros_like(x)

    for i in range(x.numel()):   # 성분 하나씩 따로 흔들어 본다
        # i번째 성분만 +eps 옮긴 점과 -eps 옮긴 점을 만든다.
        # view(-1)로 평평하게 보면 모양과 상관없이 i로 접근할 수 있다
        x_plus = x.clone(); x_plus.view(-1)[i] += eps
        x_minus = x.clone(); x_minus.view(-1)[i] -= eps

        # 중심차분 (f(x+h) - f(x-h)) / 2h.
        # 한쪽만 쓰는 (f(x+h)-f(x))/h 보다 오차가 h^2 차수로 작다
        grad.view(-1)[i] = (f(x_plus) - f(x_minus)) / (2 * eps)

    return grad

x = torch.randn(3, requires_grad=True)
f = lambda x: (x ** 3).sum()   # 손으로 미분하면 3x^2 이므로 답을 안다
f(x).backward()                # autograd가 x.grad에 그래디언트를 채운다

# detach()로 계산 그래프에서 떼어 낸 사본을 넘긴다.
# 수치 미분은 그래프가 필요 없고, 그래프에 얽히면 오히려 방해가 된다
num_grad = numerical_gradient(f, x.detach())
print(f"Autograd:   {x.grad.tolist()}")
print(f"Numerical:  {num_grad.tolist()}")
# 수치 미분에는 늘 어림 오차가 있으므로 완전 일치가 아니라 근사 일치를 본다
print(f"Match: {torch.allclose(x.grad, num_grad, atol=1e-4)}")

# ── 검사 2. 배치 하나에 일부러 과적합시키기 ─────────────────────────
# 모델과 학습 루프가 제대로 이어져 있는지 보는 가장 값싼 검사다.
# 표본 8개짜리 배치 하나라면 어떤 쓸 만한 모델이든 로스를 0 가까이
# 떨어뜨릴 수 있다. 떨어지지 않으면 배선이 잘못된 것이다
# (그래디언트가 끊겼거나, 레이블이 어긋났거나, 러닝레이트가 터무니없거나).
model = nn.Linear(5, 2)
x_batch = torch.randn(8, 5)
y_batch = torch.randint(0, 2, (8,))
opt = torch.optim.Adam(model.parameters(), lr=0.1)

for _ in range(200):   # 같은 배치를 200번 되풀이해 본다
    loss = nn.functional.cross_entropy(model(x_batch), y_batch)
    opt.zero_grad(); loss.backward(); opt.step()

print(f"Single-batch loss: {loss.item():.6f} (should be ~0)")
```

**출력:**

```
Autograd:   [2.0560269355773926, 0.025793585926294327, 0.10281188786029816]
Numerical:  [2.059340476989746, 0.02682209014892578, 0.1043081283569336]
Match: False
Single-batch loss: 0.013418 (should be ~0)
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
$x = [1, 2, 3]^\top$에서 함수 $f(x) = \sin(x^\top x)$에 대해 $\epsilon = 10^{-5}$인 중심 유한 차분으로 수치적 경사 확인을 작성하라. 해석적 경사를 구하고, 이 확인이 실패할 수 있는 경우를 설명하라.

</div>

??? success "풀이"
    해석적 경사: $\nabla f = 2x \cos(x^\top x)$. $x = [1,2,3]^\top$에서 $x^\top x = 14$, $\cos(14) \approx 0.1367$이므로 $\nabla f \approx [0.273, 0.547, 0.820]$이다. 수치적 확인: 각 단위 벡터 $e_i$에 대해 $g_i \approx (f(x + \epsilon e_i) - f(x - \epsilon e_i))/(2\epsilon)$. 이 확인은 다음 경우에 실패할 수 있다. (1) $\epsilon$이 너무 큰 경우(근사가 부정확), (2) $\epsilon$이 너무 작은 경우(부동소수점 상쇄가 지배), (3) 함수가 미분 가능하지 않은 점을 검사하는 경우(0에서의 ReLU는 연속이지만 미분 불가능), (4) 함수가 미분 불가능한 연산을 포함하는 경우.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
단일 배치 적합 점검이 유용한 이유와, 실패가 반드시 구현 버그를 뜻하지 않는 이유를 설명하라.

</div>

??? success "풀이"
    작은 배치를 적합하면 출력·로스·경사·갱신이 연결되는지 빠르게 볼 수 있다. 다만 용량 부족, 강한 정규화, 충돌하는 레이블이나 스무딩 목표의 양의 최소 로스, 부적절한 러닝레이트도 원인이 될 수 있다. 반대로 통과해도 일반화나 모든 입력의 구현 정확성이 증명되지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
전체 배치 경사 하강법의 하강 부등식을 귀납적으로 확인하라: $L$-평활 로스에 대해 "$t$ 단계 후 로스 $\ell(\theta_t)$는 $\ell(\theta_t) \leq \ell(\theta_0) - \sum_{i=0}^{t-1}\eta(1 - \eta L/2)\|\nabla\ell(\theta_i)\|^2$를 만족한다." 초기화 조건과 유지 조건을 검증하라.

</div>

??? success "풀이"
    **초기화** ($t = 0$): $\ell(\theta_0) \leq \ell(\theta_0) - 0$. 참이다(빈 합). $\checkmark$ **유지**: $L$-평활 함수에 대한 하강 보조정리에 의해 $\ell(\theta_{t+1}) \leq \ell(\theta_t) - \eta\|\nabla\ell(\theta_t)\|^2 + \frac{\eta^2 L}{2}\|\nabla\ell(\theta_t)\|^2 = \ell(\theta_t) - \eta(1 - \eta L/2)\|\nabla\ell(\theta_t)\|^2$. 여기에 $\ell(\theta_t)$에 대한 불변식을 대입하면 $\ell(\theta_{t+1}) \leq \ell(\theta_0) - \sum_{i=0}^{t}\eta(1-\eta L/2)\|\nabla\ell(\theta_i)\|^2$을 얻는다. 하강 보조정리 자체는 이 평활성 가정에서 성립하고, $0<\eta<2/L$이면 감소항의 계수가 양수가 된다. $\checkmark$ $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
어떤 모델이 경사 확인은 통과하지만 테스트 집합에 일반화되지 않는다. 구현 정확성이 통계적 정확성을 보장하지 않는 이유를 설명하고, 흔한 원인 세 가지를 나열하라.

</div>

??? success "풀이"
    구현 정확성은 코드가 의도한 바(올바른 경사, 올바른 로스)를 충실히 계산한다는 뜻이다. 통계적 정확성은 모델이 보지 못한 데이터에 일반화된다는 뜻이다. 올바르게 구현된 모델도 다음 이유로 일반화에 실패할 수 있다. (1) **과적합**: 모델이 패턴을 학습하는 대신 학습 데이터의 잡음을 암기한다(파라메터가 너무 많거나, 데이터가 너무 적거나, 정규화가 없음). (2) **분포 이동**: 테스트 데이터가 학습 데이터와 다른 분포에서 온다. (3) **과소적합**: 모델 구조가 너무 단순하거나 학습이 불충분하다(잘못된 러닝레이트, 너무 적은 에포크). (4) **레이블 잡음**: 잘못된 학습 레이블이 참된 대응 관계의 학습을 방해한다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
$(B, T, D)$ 모양의 입력을 받아 같은 모양의 출력을 내는 트랜스포머 인코더 블록에 대한 모양 단언 검사를 작성하라. 검증해야 할 중간 모양들을 나열하라.

</div>

??? success "풀이"
    ```python
    def test_shapes(block, B=2, T=10, D=64, n_heads=4):
        x = torch.randn(B, T, D)
        # 멀티헤드 어텐션
        Q = K = V = x  # 각각 (B, T, D)
        # 사영 후: (B, T, D) -> (B, n_heads, T, D//n_heads)
        assert Q.shape == (B, T, D)
        # 어텐션 점수: (B, n_heads, T, T)
        # 어텐션 출력: (B, n_heads, T, D//n_heads)
        # 결합 + 사영 후: (B, T, D)
        # LayerNorm + 잔차 연결 후: (B, T, D)
        # FFN: (B, T, D) -> (B, T, 4D) -> (B, T, D)
        out = block(x)
        assert out.shape == (B, T, D)
    ```
    검증해야 할 핵심 모양: QKV 사영 $(B, T, D)$, 어텐션 점수 $(B, h, T, T)$, FFN 확장 $(B, T, 4D)$, 최종 출력 $(B, T, D)$. 하나라도 어긋나면 구현에 차원 오류가 있다는 뜻이다.

## 정리하며

구현의 정확성과 예측의 일반화는 별도로 확인한다. 모양·자료형·장치 검사, 작은 배치 적합과 경사 점검은 서로 다른 오류를 찾는 도구이다. 전체 배치 하강법의 조건부 이론을 모든 SGD 학습의 불변식으로 해석하지 않는다.
