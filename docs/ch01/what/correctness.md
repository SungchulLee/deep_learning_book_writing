# 정확성

정확성(correctness)이란 알고리즘이나 모델이 모든 유효한 입력에 대해 올바른 출력을 낸다는 뜻이다. 딥러닝에서 정확성은 구현 정확성(코드가 의도한 대로 동작하는가)과 통계적 정확성(모델이 학습 데이터를 넘어 일반화되는가)을 모두 포함한다.

## 정의

알고리즘이 정확하다는 것은 모든 유효한 입력 $x$에 대해 종료하고 기대하는 출력을 낸다는 뜻이다.

$$
\forall \, x \in \mathcal{X}: \; f(x) = \text{Expected}(x)
$$

결정적 알고리즘에서 정확성은 절대적이다. 신경망에서 정확성은 통계적이다. 즉 데이터 분포에 대한 기대 손실 $\mathbb{E}[\ell(f(x), y)]$이 작기를 바란다.

## 설명

**루프 불변식(loop invariant)** 은 알고리즘의 정확성을 증명하는 고전적인 기법이다. 루프 불변식은 매 반복의 전후에 성립하는 성질이다.

1. **초기화(initialization)**: 첫 반복 전에 참이다
2. **유지(maintenance)**: 반복 전에 참이면 반복 후에도 참이다
3. **종료(termination)**: 루프가 끝났을 때 불변식이 정확성을 함의한다

딥러닝에서 이에 대응하는 개념은 **학습 불변식** 이다. 적절한 학습률 아래에서 매 경사 단계마다 손실이 (기댓값의 의미에서) 단조 감소한다는 것이다. 실무에서 정확성을 검증하는 방법은 다음과 같다.

- **경사 확인(gradient checking)**: 자동 미분 경사를 수치적 유한 차분 경사와 비교한다
- **단일 배치 과적합**: 올바른 모델이라면 단일 배치에서 손실이 거의 0에 도달해야 하며, 이는 구조와 손실 함수가 함께 잘 동작함을 확인해 준다
- **텐서 모양 단위 검사**: 각 층의 중간 텐서 모양을 단언문으로 확인한다

## 예제

```python
import torch
import torch.nn as nn

# 경사 확인: 자동 미분의 정확성을 검증한다
def numerical_gradient(f, x, eps=1e-5):
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_plus = x.clone(); x_plus.view(-1)[i] += eps
        x_minus = x.clone(); x_minus.view(-1)[i] -= eps
        grad.view(-1)[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

x = torch.randn(3, requires_grad=True)
f = lambda x: (x ** 3).sum()
f(x).backward()

num_grad = numerical_gradient(f, x.detach())
print(f"Autograd:   {x.grad.tolist()}")
print(f"Numerical:  {num_grad.tolist()}")
print(f"Match: {torch.allclose(x.grad, num_grad, atol=1e-4)}")

# 단일 배치 과적합(정확성 검사)
model = nn.Linear(5, 2)
x_batch = torch.randn(8, 5)
y_batch = torch.randint(0, 2, (8,))
opt = torch.optim.Adam(model.parameters(), lr=0.1)
for _ in range(200):
    loss = nn.functional.cross_entropy(model(x_batch), y_batch)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"Single-batch loss: {loss.item():.6f} (should be ~0)")
```

## 연습문제

**연습문제 1.**
$x = [1, 2, 3]^\top$에서 함수 $f(x) = \sin(x^\top x)$에 대해 $\epsilon = 10^{-5}$인 중심 유한 차분으로 수치적 경사 확인을 작성하라. 해석적 경사를 구하고, 이 확인이 실패할 수 있는 경우를 설명하라.

??? success "연습문제 1 풀이"
    해석적 경사: $\nabla f = 2x \cos(x^\top x)$. $x = [1,2,3]^\top$에서 $x^\top x = 14$, $\cos(14) \approx 0.1367$이므로 $\nabla f \approx [0.273, 0.547, 0.820]$이다. 수치적 확인: 각 단위 벡터 $e_i$에 대해 $g_i \approx (f(x + \epsilon e_i) - f(x - \epsilon e_i))/(2\epsilon)$. 이 확인은 다음 경우에 실패할 수 있다. (1) $\epsilon$이 너무 큰 경우(근사가 부정확), (2) $\epsilon$이 너무 작은 경우(부동소수점 상쇄가 지배), (3) 함수에 불연속점이 있는 경우(0에서의 ReLU), (4) 함수가 미분 불가능한 연산을 포함하는 경우.

---

**연습문제 2.**
"단일 배치 과적합" 정확성 검사를 설명하라. 모델이 단일 배치에서 학습 손실을 거의 0으로 만들어야 하는 이유는 무엇이며, 실패는 무엇을 시사하는가?

??? success "연습문제 2 풀이"
    용량이 충분한 모델은 단일 배치(예: 표본 8개)를 암기하여 손실을 거의 0으로 만들 수 있어야 한다. 이 검사는 다음을 확인한다. (1) 순전파가 올바른 모양의 출력을 내는가, (2) 손실 함수가 출력 형식과 호환되는가, (3) 모든 매개변수로 경사가 흐르는가, (4) 최적화기가 매개변수를 올바르게 갱신하는가. 실패는 버그를 시사한다. 흔한 원인은 손실 함수 불일치(예: 로짓 대신 확률에 교차 엔트로피를 적용), 매개변수 고정(`requires_grad=True`를 빠뜨림), 잘못된 레이블 형식, 학습률이 0인 경우 등이다.

---

**연습문제 3.**
SGD 학습 루프의 루프 불변식을 정의하라: $L$-평활 손실에 대해 "$t$ 단계 후 손실 $\ell(\theta_t)$는 $\ell(\theta_t) \leq \ell(\theta_0) - \sum_{i=0}^{t-1}\eta(1 - \eta L/2)\|\nabla\ell(\theta_i)\|^2$를 만족한다." 초기화 조건과 유지 조건을 검증하라.

??? success "연습문제 3 풀이"
    **초기화** ($t = 0$): $\ell(\theta_0) \leq \ell(\theta_0) - 0$. 참이다(빈 합). $\checkmark$ **유지**: $L$-평활 함수에 대한 하강 보조정리에 의해 $\ell(\theta_{t+1}) \leq \ell(\theta_t) - \eta\|\nabla\ell(\theta_t)\|^2 + \frac{\eta^2 L}{2}\|\nabla\ell(\theta_t)\|^2 = \ell(\theta_t) - \eta(1 - \eta L/2)\|\nabla\ell(\theta_t)\|^2$. 여기에 $\ell(\theta_t)$에 대한 불변식을 대입하면 $\ell(\theta_{t+1}) \leq \ell(\theta_0) - \sum_{i=0}^{t}\eta(1-\eta L/2)\|\nabla\ell(\theta_i)\|^2$을 얻는다. 이는 $\eta < 2/L$일 때 성립하며, 이 조건이 각 항을 음이 아니게 만든다. $\checkmark$ $\square$

---

**연습문제 4.**
어떤 모델이 경사 확인은 통과하지만 테스트 집합에 일반화되지 않는다. 구현 정확성이 통계적 정확성을 보장하지 않는 이유를 설명하고, 흔한 원인 세 가지를 나열하라.

??? success "연습문제 4 풀이"
    구현 정확성은 코드가 의도한 바(올바른 경사, 올바른 손실)를 충실히 계산한다는 뜻이다. 통계적 정확성은 모델이 보지 못한 데이터에 일반화된다는 뜻이다. 올바르게 구현된 모델도 다음 이유로 일반화에 실패할 수 있다. (1) **과적합**: 모델이 패턴을 학습하는 대신 학습 데이터의 잡음을 암기한다(매개변수가 너무 많거나, 데이터가 너무 적거나, 정칙화가 없음). (2) **분포 이동**: 테스트 데이터가 학습 데이터와 다른 분포에서 온다. (3) **과소적합**: 모델 구조가 너무 단순하거나 학습이 불충분하다(잘못된 학습률, 너무 적은 에폭). (4) **레이블 잡음**: 잘못된 학습 레이블이 참된 대응 관계의 학습을 방해한다.

---

**연습문제 5.**
$(B, T, D)$ 모양의 입력을 받아 같은 모양의 출력을 내는 트랜스포머 인코더 블록에 대한 모양 단언 검사를 작성하라. 검증해야 할 중간 모양들을 나열하라.

??? success "연습문제 5 풀이"
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
