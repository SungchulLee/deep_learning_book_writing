# 경사 계산

<div class="leadbox" markdown>

**들어가는 이야기** — 곱하는 차례를 고를 수 있다

여러 단계를 거쳐 나온 결과를 미분하려면 단계마다의 **야코비 행렬**(출력 각 성분을 입력 각 성분으로 편미분해 모은 행렬)을 **차례로
곱한다.** 학교에서 배우는 사슬 규칙 그대로다.

그런데 곱셈은 어느 쪽부터 묶어도 답이 같다. 셋을 곱할 때 앞의 둘을 먼저
곱하든 뒤의 둘을 먼저 곱하든 결과는 하나다. **답은 같은데 드는 계산량은 다르다.**

자동 미분의 두 방식이 바로 이 차례의 선택이다. 앞에서부터 묶어 가는 **순방향
모드**(forward mode)와 뒤에서부터 묶어 오는 **역방향 모드**(reverse mode) — 역전파가 뒤엣것이다. 둘은 같은
수를 내놓는다.

같은 수를 내놓는데 이 쪽의 측정에서는 **한쪽이 60배 느리다.** 그리고 함수의
모양을 뒤집으면 **승부도 뒤집혀** 이번에는 48배 차이로 반대쪽이 이긴다.
파라메터가 수백만인 요즘 모델이 트레인 가능한 까닭이 이 선택 하나에 있다.

</div>

PyTorch의 autograd 엔진은 딥러닝에서 자동 미분의 토대이다. 텐서에 대한 연산을 기록하고 **역방향 모드 자동 미분**(역전파)으로 그래디언트를 계산하는 테이프 기반 체계를 제공한다. 이 쪽에서는 그래디언트 계산의 수학적 틀, 순방향 모드와 역방향 모드 자동 미분의 구분, 그리고 PyTorch의 벡터-야코비 곱(VJP) 기법이 사슬 규칙을 어떻게 효율적으로 구현하는지를 다룬다.

---

## 1. 학습 목표

이 쪽을 마치면 다음을 할 수 있게 된다.

1. `.backward()`와 `torch.autograd.grad`로 그래디언트를 계산한다
2. 야코비 행렬과 벡터-야코비 곱(VJP)의 틀을 이해한다
3. 순방향 모드(JVP)와 역방향 모드(VJP) 자동 미분을 구분한다
4. 뉴럴 네트워크 트레인에 역방향 모드가 선호되는 이유를 설명한다
5. 두 모드 모두로 전체 야코비 행렬을 계산한다

---

## 2. 수학적 기초

<div class="defn" markdown>

### 정의 1. 야코비 행렬 { .dfn }

미분 가능한 $f : \mathbb{R}^n \to \mathbb{R}^m$에 대해 모든 1차 편도함수를 모은 행렬

$$
J_f(x) = \frac{\partial f}{\partial x} = \begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

을 $f$의 **야코비 행렬**이라 한다. $(i,j)$칸은 $j$번째 입력을 조금 흔들었을 때 $i$번째 출력이 얼마나 따라 움직이는지다.

</div>

<div class="thmbox" markdown>

### 정리 1. 사슬 규칙 — 합성의 야코비 행렬은 야코비 행렬의 곱이다 { .thm }

$f = f_L \circ f_{L-1} \circ \cdots \circ f_1$이고 각 $f_\ell$이 미분 가능하면

$$
J_f(x) = J_L \, J_{L-1} \cdots J_1
$$

이다. 여기서 $z_0 = x$, $z_\ell = f_\ell(z_{\ell-1})$로 두고 $J_\ell = J_{f_\ell}(z_{\ell-1})$, 곧 $f_\ell$을 **그 자리의 입력**에서 잰 야코비 행렬이다.

</div>

??? proof "증명"

    $L = 2$에서 보이면 나머지는 귀납이다. $f = f_2 \circ f_1$이라 두고, 값에 이름을 붙여 $z = f_1(x)$, $y = f_2(z)$라 하고 성분으로 적는다. 다변수 사슬 규칙에서

    $$
    \frac{\partial y_i}{\partial x_j} = \sum_{k} \frac{\partial y_i}{\partial z_k}\,\frac{\partial z_k}{\partial x_j}
    $$

    이다. 오른쪽은 $(i,k)$칸이 $\partial y_i / \partial z_k$인 행렬과 $(k,j)$칸이 $\partial z_k / \partial x_j$인 행렬의 **곱의 $(i,j)$칸**과 정확히 같은 형태다. 곧 $J_f = J_2 J_1$이다.

    $L-1$개까지 성립한다고 하면 $g = f_{L-1}\circ\cdots\circ f_1$에 대해 $J_g = J_{L-1}\cdots J_1$이고, 위의 두 함수 합성 결과를 $f_L$과 $g$에 쓰면 $J_f = J_L J_g = J_L J_{L-1} \cdots J_1$이다. $\square$

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 레이어 하나를 손으로 끝까지 따라간다

레이어는 $f_\theta(x) = \theta x + 1$ 하나다. 지금 $\theta = 2$이고 입력은 $x = 3$, 맞혀야 할 값은 $10$이다. 로스는 제곱오차, 러닝레이트는 $0.01$로 둔다.

$\partial \ell / \partial \theta$를 사슬 규칙으로 구하고, 한 스텝 간 뒤 로스가 얼마가 되는지까지 적어라.

</div>

??? success "풀이"
    **순전파.** 입력을 레이어에 넣고 로스까지 간다.

    $$
    h = 2 \cdot 3 + 1 = 7, \qquad \ell = (7 - 10)^2 = 9
    $$

    **역전파.** 끝에서 시작해 거꾸로 곱해 온다.

    | 어디까지 왔는가 | 무엇을 곱하는가 | 누적된 값 |
    |---|---|---|
    | $\partial \ell / \partial h$ | $2(h - 10)$ | $-6$ |
    | $\partial \ell / \partial \theta$ | $\times\, x = 3$ | $\mathbf{-18}$ |

    그래디언트가 **음수**다. $\theta$를 키우면 로스가 줄어든다는 뜻이다 — 출력 7이 맞혀야 할 값 10보다 작으니 $\theta$를 키워야 하는 것과 맞는다.

    **한 스텝.** 그래디언트의 반대쪽으로 $\lambda = 0.01$만큼 간다. 음수를 빼므로 $\theta$가 커진다.

    $$
    \theta \leftarrow 2 - 0.01 \times (-18) = 2.18
    $$

    **다시 통과시키면** $h = 2.18 \cdot 3 + 1 = 7.54$이고 $\ell = (7.54 - 10)^2 = 6.0516$이다. 9에서 6.05로 내려갔다.

    손으로 계산한 것을 파이토치로 확인한다. 이 쪽의 첫 코드이므로 줄마다 풀어 적는다.

    ```python
    """보기 1을 파이토치로 확인한다 — 레이어 하나, 파라메터 하나, 한 스텝.

    손으로 한 일을 같은 차례로 다시 한다.
      (1) 순전파   — 입력을 레이어에 넣어 로스까지 간다
      (2) 역전파   — 로스를 파라메터로 미분한 값(그래디언트)을 구한다
      (3) 확인     — 손으로 구한 값과 견준다
      (4) 한 스텝  — 그래디언트의 반대쪽으로 파라메터를 조금 옮긴다
      (5) 다시     — 옮긴 파라메터로 순전파를 다시 해 로스가 줄었는지 본다

    기호: 본문의 θ 를 t 로, ℓ 을 loss 로 적는다.
    실행: 이 쪽의 코드를 파일로 갈무리해 돌린다. 데이터도 GPU 도 필요 없다.
    """
    # 파이토치를 불러온다. 뒤에 나오는 torch.tensor, torch.no_grad 가 다 여기서 온다.
    import torch

    # === 0. 파라메터와 데이터를 만든다 =============================================
    # 레이어는 f_t(x) = t*x + 1 하나다.
    #
    # torch.tensor(2.0) 은 수 2.0 하나를 담은 「텐서」를 만든다. 텐서는 파이토치가
    # 다루는 수의 묶음이고, 여기서는 수 하나짜리(스칼라)다.
    #
    # requires_grad=True 는 「이 텐서로 미분할 것이니, 이 텐서가 들어간 계산을
    # 모두 기록해 두라」는 표시다. 이것을 경사 추적이라 한다.
    #
    # 기본값은 requires_grad=False 다. 적지 않으면 기록하지 않으므로, 그냥
    # torch.tensor(2.0) 으로 만든 텐서로는 미분할 수 없다. 그래서 트레인으로
    # 고칠 값, 곧 파라메터에만 True 로 건다.
    t = torch.tensor(2.0, requires_grad=True)     # θ = 2 — 레이어의 파라메터

    # 입력 x, 맞혀야 할 값 target, 러닝레이트 lr 는 고칠 값이 아니다. 미분할 일이
    # 없으므로 텐서로 만들지 않고 보통의 파이썬 실수로 둔다. 1e-2 는 0.01 이다.
    x, target, lr = 3.0, 10.0, 1e-2

    # === 1. 순전파 — 입력에서 로스까지 ============================================
    # 식을 그대로 적으면 된다. t 가 경사 추적 중인 텐서이므로 결과 h 도 텐서가
    # 되고, 파이토치는 값 7 을 계산하는 한편 「h 는 t 에 x 를 곱하고 1 을 더해
    # 만들었다」는 것을 함께 적어 둔다. 이 기록이 계산 그래프다.
    h = t * x + 1                 # 레이어의 출력    h = 2*3 + 1 = 7
    loss = (h - target) ** 2      # 제곱오차 로스    (7 - 10)^2 = 9   (** 는 거듭제곱)

    # === 2. 역전파 — 로스를 파라메터로 미분한다 ====================================
    # backward() 한 줄이 풀이의 표 전체를 한다. 위에서 적어 둔 계산 그래프를
    # loss 에서 시작해 거꾸로 따라가며 사슬 규칙으로 곱해 나가고, 그 결과를
    # 경사 추적을 건 텐서의 .grad 에 넣어 준다.
    #   t.grad = dl/dt (본문의 ∂ℓ/∂θ)
    # backward() 를 부르기 전에는 t.grad 가 None 이다.
    loss.backward()

    # === 3. 확인 — 손으로 구한 값과 견준다 ========================================
    # .item() 은 수 하나짜리 텐서에서 그 수를 파이썬 실수로 꺼낸다.
    # f"...{값:.3f}..." 은 문자열 안에 값을 끼워 넣는 파이썬 문법이고,
    # :.3f 는 소수점 아래 세 자리까지 찍으라는 뜻이다.
    print(f"  순전파   h = {h.item():.3f}   loss = {loss.item():.3f}")
    print(f"  autograd dl/dt = {t.grad.item():.3f}")
    # 풀이의 표를 수로 옮긴 것이다.   dl/dt = 2(h - 10) * x = 2*(-3)*3 = -18
    print(f"  손으로   dl/dt = {2*(7-10)*3:.3f}")

    # === 4. 한 스텝 — 경사 하강법 =================================================
    # 그래디언트는 로스가 가장 빨리 「커지는」 쪽을 가리킨다. 그래서 반대쪽으로,
    # 러닝레이트만큼만 옮긴다.   파라메터 ← 파라메터 - lr * 그래디언트
    #
    # with torch.no_grad(): 블록 안의 계산은 기록하지 않는다. 파라메터를 고치는
    # 일은 모델의 계산이 아니므로 계산 그래프에 들어가면 안 된다. 이 블록 없이
    # 경사 추적 중인 t 를 제자리에서 고치면 파이토치가 오류를 낸다.
    #
    # t -= ... 는 t 를 새로 만들지 않고 「제자리에서」 고친다. t = t - ... 로 적으면
    # t 가 새 텐서로 바뀌어 파라메터가 아니게 된다.
    with torch.no_grad():
        t -= lr * t.grad          # θ ← 2 - 0.01 * (-18) = 2.18
    print(f"  한 스텝  t = {t.item():.3f}")

    # === 5. 다시 — 옮긴 파라메터로 로스를 다시 잰다 ================================
    # 로스가 줄었는지 값만 보면 되고 미분은 하지 않는다. 그래서 여기서도
    # no_grad() 안에서 계산해 기록을 남기지 않는다.
    with torch.no_grad():
        h_new = t * x + 1                       # 새 θ 로 레이어를 다시 지난다
        new = (h_new - target) ** 2             # 로스를 다시 잰다
    print(f"  다시     h = {h_new.item():.3f}   loss = {new.item():.4f}")
    # loss 는 스텝 전에 잰 값(9)이고 new 는 스텝 뒤에 잰 값이다.
    print(f"  로스가 {loss.item():.3f} 에서 {new.item():.4f} 으로 내려갔다")
    ```

    **출력:**

    ```
      순전파   h = 7.000   loss = 9.000
      autograd dl/dt = -18.000
      손으로   dl/dt = -18.000
      한 스텝  t = 2.180
      다시     h = 7.540   loss = 6.0516
      로스가 9.000 에서 6.0516 으로 내려갔다
    ```

**손으로 구한 $-18$과 `backward()`가 채운 `.grad`가 같다.** 그리고 그래디언트의 반대쪽으로 한 스텝 가니 로스가 줄었다. 트레인은 이 다섯 단계를 되풀이하는 일이다.

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 두 레이어를 손으로 끝까지 따라간다

보기 1의 레이어 $f_\theta(x) = \theta x + 1$ 뒤에 $f_\varphi(h) = \varphi h + 1$을 하나 더 이어 붙인다. 지금 $\theta = 2$, $\varphi = 6$이고 입력은 $x = 3$, 맞혀야 할 값은 $40$이다. 로스는 제곱오차, 러닝레이트는 $0.001$로 둔다.

$\partial \ell / \partial \theta$와 $\partial \ell / \partial \varphi$를 사슬 규칙으로 구하고, 한 스텝 간 뒤 로스가 얼마가 되는지까지 적어라.

</div>

??? success "풀이"
    **순전파.** 두 레이어를 차례로 지난다.

    $$
    h = 2 \cdot 3 + 1 = 7, \qquad x_2 = 6 \cdot 7 + 1 = 43, \qquad
    \ell = (43 - 40)^2 = 9
    $$

    **역전파.** 끝에서 시작해 거꾸로 곱해 온다. 각 칸은 **그 자리의 값**으로 잰다.

    | 어디까지 왔는가 | 무엇을 곱하는가 | 누적된 값 |
    |---|---|---|
    | $\partial \ell / \partial x_2$ | $2(x_2 - 40)$ | $6$ |
    | $\partial \ell / \partial \varphi$ | $\times\, h = 7$ | $\mathbf{42}$ |
    | $\partial \ell / \partial h$ | $\times\, \varphi = 6$ | $36$ |
    | $\partial \ell / \partial \theta$ | $\times\, x = 3$ | $\mathbf{108}$ |

    둘째 줄과 넷째 줄이 갈라지는 자리를 보라. $\varphi$로 가는 길은 **거기서 끝나고**, $\theta$로 가는 길은 $h$를 한 번 더 지난다. 정리 1이 말하는 곱의 사슬이 이 표다.

    곱하는 값이 모두 **순전파 때 지나간 수**라는 점도 눈여겨볼 만하다. $h = 7$도 $\varphi = 6$도 $x = 3$도 앞으로 갈 때 이미 손에 있던 것이고, 그래서 역전파가 그 값들을 들고 있어야 한다. 「retain_graph」 쪽이 말하는 **역전파가 저장해 둔 값**이 이것이다.

    **한 스텝.** $\lambda = 0.001$로 빼면

    $$
    \theta \leftarrow 2 - 0.001 \times 108 = 1.892, \qquad
    \varphi \leftarrow 6 - 0.001 \times 42 = 5.958
    $$

    **다시 통과시키면** $h = 1.892 \cdot 3 + 1 = 6.676$이고 $x_2 = 5.958 \cdot 6.676 + 1 = 40.775608$이므로 $\ell = 0.601568$이다. 9에서 0.6으로 내려갔다.

    이것도 파이토치로 확인한다. 이번에는 함수값만 두 번 재는 **유한차분**으로도 한 번 더 받는다.

    ```python
    """보기 2를 파이토치로 확인한다 — 두 레이어, 파라메터 둘, 한 스텝.

    보기 1의 코드와 차례가 같다. 달라진 것은 셋이다.
      - 레이어가 둘이라 순전파가 두 줄이다
      - 파라메터가 둘이라 backward() 한 번이 그래디언트 둘을 한꺼번에 채운다
      - 유한차분으로 한 번 더 확인한다

    기호: 본문의 θ 를 t 로, φ 를 p 로, ℓ 을 loss 로 적는다.
    실행: 이 쪽의 코드를 파일로 갈무리해 돌린다. 데이터도 GPU 도 필요 없다.
    """
    import torch

    # === 0. 파라메터와 데이터를 만든다 =============================================
    # 손으로 계산한 보기를 그대로 — f_t(x) = t*x + 1, f_p(h) = p*h + 1
    # 고칠 값이 둘이므로 둘 다 requires_grad=True 로 만든다.
    t = torch.tensor(2.0, requires_grad=True)     # θ = 2 — 첫째 레이어의 파라메터
    p = torch.tensor(6.0, requires_grad=True)     # φ = 6 — 둘째 레이어의 파라메터
    x, target, lr = 3.0, 40.0, 1e-3               # 입력, 맞혀야 할 값, 러닝레이트 0.001

    # === 1. 순전파 — 입력에서 로스까지 ============================================
    # 첫째 레이어의 출력 h 가 둘째 레이어의 입력이 된다. 계산 그래프에는
    # 「x2 는 p 와 h 로 만들었고, 그 h 는 t 로 만들었다」가 이어져 적힌다.
    h = t * x + 1                 # 첫째 레이어의 출력  h  = 2*3 + 1 = 7
    x2 = p * h + 1                # 둘째 레이어의 출력  x2 = 6*7 + 1 = 43
    loss = (x2 - target) ** 2     # 제곱오차 로스       (43 - 40)^2 = 9

    # === 2. 역전파 — 로스를 두 파라메터로 미분한다 ==================================
    # backward() 한 번이 loss 에서 거꾸로 따라가며 두 길을 다 지난다. 풀이의 표에서
    # φ 로 가는 길은 둘째 줄에서 끝나고 θ 로 가는 길은 h 를 한 번 더 지난다.
    #   t.grad = dl/dt (본문의 ∂ℓ/∂θ),   p.grad = dl/dp (본문의 ∂ℓ/∂φ)
    loss.backward()

    # === 3. 확인 — 손으로 구한 값과 견준다 ========================================
    print(f"  순전파   h = {h.item():.3f}   x2 = {x2.item():.3f}   loss = {loss.item():.3f}")
    print(f"  autograd dl/dt = {t.grad.item():.3f}   dl/dp = {p.grad.item():.3f}")
    # 풀이의 표를 수로 옮긴 것이다.
    #   dl/dt = 2(x2 - 40) * φ * x = 2*3*6*3 = 108
    #   dl/dp = 2(x2 - 40) * h     = 2*3*7   = 42
    print(f"  손으로   dl/dt = {2*(43-40)*6*3:.3f}   dl/dp = {2*(43-40)*7:.3f}")

    # --- 유한차분으로 한 번 더 — 차분 간격을 바꿔 가며 ------------------------------
    # 미분의 정의로 돌아가 한 번 더 확인한다. t 를 e 만큼 늘린 로스와 e 만큼 줄인
    # 로스의 차이를 2e 로 나누면 기울기의 근삿값이 된다(중심차분). 사슬 규칙도
    # 파이토치도 쓰지 않고 함수값만 두 번 잰다.
    def loss_at(tv, pv):
        """파라메터가 (tv, pv) 일 때의 로스. 파이썬 실수만으로 순전파를 다시 한다."""
        return ((pv * (tv * x + 1) + 1) - target) ** 2

    print("  유한차분 [l(t+e) - l(t-e)] / 2e")
    for e in (1e-1, 1e-3, 1e-5):                  # 간격 0.1, 0.001, 0.00001 로 세 번
        # p 는 6.0 에 붙들어 두고 t 만 움직인다 — t 에 대한 미분이기 때문이다.
        g = (loss_at(2.0 + e, 6.0) - loss_at(2.0 - e, 6.0)) / (2 * e)
        # abs(g - 108) 은 손으로 구한 108 과 얼마나 다른가. 2.56e-13 은 2.56 × 10^-13.
        print(f"    e = {e:<7} dl/dt = {g:11.6f}   어긋남 {abs(g - 108):.2e}")

    # === 4. 한 스텝 — 경사 하강법 =================================================
    # 두 파라메터를 저마다 제 그래디언트의 반대쪽으로 옮긴다.
    with torch.no_grad():
        t -= lr * t.grad          # θ ← 2 - 0.001 * 108 = 1.892
        p -= lr * p.grad          # φ ← 6 - 0.001 * 42  = 5.958
    print(f"  한 스텝  t = {t.item():.3f}   p = {p.item():.3f}")

    # === 5. 다시 — 옮긴 파라메터로 로스를 다시 잰다 ================================
    with torch.no_grad():
        h2 = t * x + 1                          # 새 θ 로 첫째 레이어를 다시 지난다
        new = (p * h2 + 1 - target) ** 2        # 새 φ 로 둘째 레이어를 지나 로스까지
    print(f"  다시     h = {h2.item():.3f}   x2 = {(p*h2+1).item():.6f}   "
          f"loss = {new.item():.6f}")
    print(f"  로스가 {loss.item():.3f} 에서 {new.item():.6f} 으로 내려갔다")
    ```

    **출력:**

    ```
      순전파   h = 7.000   x2 = 43.000   loss = 9.000
      autograd dl/dt = 108.000   dl/dp = 42.000
      손으로   dl/dt = 108.000   dl/dp = 42.000
      유한차분 [l(t+e) - l(t-e)] / 2e
        e = 0.1     dl/dt =  108.000000   어긋남 2.56e-13
        e = 0.001   dl/dt =  108.000000   어긋남 4.09e-12
        e = 1e-05   dl/dt =  108.000000   어긋남 1.75e-10
      한 스텝  t = 1.892   p = 5.958
      다시     h = 6.676   x2 = 40.775608   loss = 0.601568
      로스가 9.000 에서 0.601568 으로 내려갔다
    ```

**세 길이 같은 수를 준다.** 사슬 규칙으로 손계산한 108과 42, `backward()`가 채운 `.grad`, 그리고 미분을 모르고 함수값만 두 번 재어 나눈 값이 모두 맞는다.

!!! warning "유한차분이 여기서 유난히 잘 맞는 까닭"
    차분 간격을 $0.1$로 크게 두었는데도 어긋남이 $10^{-13}$이다. 보통은 이렇지 않다 — 「[비선형 전체 야코비안](25_vector_backward_nonlinear_full_jacobian.md)」 쪽에서는 `float64`를 써야 $10^{-8}$까지 간다.

    까닭은 **이 로스가 파라메터 하나하나에 대해 이차식**이기 때문이다. $\ell$을 $\theta$의 함수로만 보면 (일차식)$^2$이고, 중심차분은 이차식에서 **정확하다** — 틀리게 만드는 삼차 이상 항이 아예 없다. 게다가 위의 유한차분은 파이썬 실수, 곧 배정밀도로 계산했다.

    곧 이 줄은 「유한차분은 정확하다」가 아니라 **「이 보기에서는 셋이 맞는다」**까지만 말한다.

**곱하는 차례가 자동 미분의 모드를 가른다.** 행렬 곱은 결합법칙이 성립하므로 $J_L(J_{L-1}(\cdots J_1))$로 묶든 $((J_L J_{L-1})\cdots)J_1$로 묶든 **결과는 같다.** 그러나 드는 비용이 다르다. 오른쪽부터 묶는 것이 순방향 모드, 왼쪽부터 묶는 것이 역방향 모드이며, 로스처럼 출력이 스칼라 하나인 경우($m = 1$) 역방향 모드가 훨씬 싸다. 딥러닝이 역전파를 쓰는 까닭이 이것이다.

---

## 3. PyTorch에서 그래디언트 계산하기

### `requires_grad` — 그래프를 세울지 정하는 플래그

아래 코드들이 `requires_grad=True`로 시작한다. 그 한 마디가 무엇을 하는지부터
적어 둔다. **자동 미분에서 이것 하나만 알아야 한다면 이것이다.**

```python
# torch 는 텐서와 자동 미분을, torch.nn 은 레이어와 모듈을 담고 있다.
# torch.nn 은 관례대로 nn 이라는 짧은 이름으로 불러온다.
import torch
import torch.nn as nn

# torch.randn 처럼 난수를 쓰는 함수는 부를 때마다 다른 수를 낸다. 시드를 0 으로
# 박아 두면 난수가 나오는 차례가 고정되어, 다시 돌려도 같은 수가 나온다.
torch.manual_seed(0)

# === 1. 기본은 꺼져 있고, 켜는 길이 셋이다 ==================================
# torch.randn(3) 은 표준정규분포에서 뽑은 난수 3개를 담은 텐서를 만든다.
# 형태는 (3,) — 수 셋이 한 줄로 선 벡터다. 아무것도 적지 않았으므로 꺼져 있다.
made = torch.randn(3)
# 길 1 — 만들 때 켠다.
at_birth = torch.randn(3, requires_grad=True)
# 길 2 — 만든 뒤에 켠다. 이름 끝의 밑줄(_)은 「새 텐서를 만들지 않고 이 텐서를
# 제자리에서 고친다」는 파이토치의 표기다. 세미콜론(;)은 두 문장을 한 줄에 적는다.
turned_on = torch.randn(3); turned_on.requires_grad_(True)
# 길 3 — nn.Parameter 로 감싼다. nn.Parameter 는 「트레인으로 고칠 값」이라는
# 표가 붙은 텐서이고, 감싸면 경사 추적이 저절로 켜진다.
as_param = nn.Parameter(torch.randn(3))

print("  1. 플래그의 기본값과 켜는 길")
# (이름표, 텐서) 짝 넷을 차례로 꺼내 플래그를 찍는다. .requires_grad 는 경사
# 추적이 켜져 있는지를 True/False 로 알려 주는 속성이다.
# {name:36s} 는 문자열을 36칸에 맞춰 찍으라는 뜻이다(줄을 맞추려는 것이다).
for name, obj in [("torch.randn(3)", made),
                  ("torch.randn(3, requires_grad=True)", at_birth),
                  (".requires_grad_(True) 로 켠 것", turned_on),
                  ("nn.Parameter(...)", as_param)]:
    print(f"     {name:36s} requires_grad={obj.requires_grad}")

print("     내장 레이어가 속에 두는 것도 그 셋째 길이다")
# nn.Linear(3, 2) 는 입력 3개를 받아 출력 2개를 내는 레이어 y = W x + b 다.
# 웨이트 W 는 형태 (2, 3), 바이어스 b 는 형태 (2,) 이고 둘 다 난수로 초기화된다.
lin = nn.Linear(3, 2)
# named_parameters() 는 모듈에 등록된 파라메터를 (이름, 텐서) 짝으로 하나씩
# 내준다. type(w).__name__ 은 그 객체의 클래스 이름이다 — "Parameter" 가 찍힌다.
for name, w in lin.named_parameters():
    print(f"       nn.Linear 의 {name:8s} {type(w).__name__:10s} "
          f"requires_grad={w.requires_grad}")

# --- 내가 만든 텐서를 넘겨 보면 ---------------------------------------
plain = torch.randn(3)                 # 맨 텐서 — 꺼져 있다
handed = nn.Parameter(plain)           # 그것을 nn.Parameter 에 넘겨 새 객체를 받는다
print("     맨 텐서를 넘기면")
print(f"       넘기기 전 plain  requires_grad={plain.requires_grad}")
# is_leaf 는 리프 텐서인지를 알려 준다. 리프 텐서는 연산의 결과가 아니라 직접
# 만든 텐서다.
print(f"       넘긴 뒤   handed requires_grad={handed.requires_grad}"
      f"  is_leaf={handed.is_leaf}")
# data_ptr() 는 텐서의 수가 실제로 놓인 메모리의 번지다. 두 번지가 같으면 두
# 텐서가 같은 저장소를 함께 쓰는 것이다.
print(f"       plain 은 그대로 꺼져 있다: {plain.requires_grad}"
      f"   그런데 저장소는 같다: {plain.data_ptr() == handed.data_ptr()}")
# 저장소가 하나이므로 plain 의 0번 칸을 99 로 고치면 handed 의 0번 칸도 99 다.
plain[0] = 99.0
print(f"       plain 을 고치면 handed 도 바뀐다: {handed[0].item()}"
      f"   <- 베낀 것이 아니다")


# nn.Module 은 모든 레이어와 모델의 밑바탕 클래스다. 이것을 물려받아 __init__ 에
# 파라메터를 두고 forward 에 순전파를 적으면 내 레이어가 된다.
class Scale(nn.Module):
    """같은 형태의 텐서 둘 — 하나만 nn.Parameter 로 감쌌다."""

    def __init__(self, d):
        # nn.Module 의 초기화를 먼저 부른다. 이 줄이 있어야 아래에서 속성으로
        # 넣는 nn.Parameter 가 모듈에 등록된다.
        super().__init__()
        # torch.ones(d) 는 1 이 d 개, torch.zeros(d) 는 0 이 d 개인 텐서다.
        self.gain = nn.Parameter(torch.ones(d))   # 감쌌다
        self.shift = torch.zeros(d)                     # 안 감쌌다

    def forward(self, x):
        # 원소별로 gain 을 곱하고 shift 를 더한다. 모듈을 scaler(x) 처럼 부르면
        # 이 forward 가 실행된다.
        return x * self.gain + self.shift


scaler = Scale(3)                      # d = 3 — 조금 아래 「내 모듈에 …」에서 쓴다
print("     저장소 공유는 nn.Parameter 만의 일이 아니다")
# 형태 (3, 4) 인 텐서 하나를 여섯 가지로 건드려 저장소가 그대로인지 본다.
#   view(4, 3)   같은 수 12개를 형태 (4, 3) 으로 다시 읽는다
#   base[0]      0번 행만 꺼낸다(슬라이스) — 형태 (4,)
#   t()          행과 열을 바꾼다(전치) — 형태 (4, 3)
#   detach()     계산 그래프와 이어지지 않은 텐서를 돌려준다
#   clone()      수를 새 메모리에 베낀다
# 반복 변수 made 는 맨 위의 made 를 덮어쓴다. 그 텐서는 더 쓰지 않으므로 괜찮다.
base = torch.randn(3, 4)
for tag, made in [("base.view(4, 3)", base.view(4, 3)),
                  ("base[0]", base[0]),
                  ("base.t()", base.t()),
                  ("base.detach()", base.detach()),
                  ("nn.Parameter(base)", nn.Parameter(base)),
                  ("base.clone()", base.clone())]:
    print(f"       {tag:20s} 저장소 공유 {base.data_ptr() == made.data_ptr()}")

print("     저장해 둔 값을 제자리에서 고치면")
# 두 함수로 같은 실험을 한다. lambda u: u ** 2 는 「u 를 받아 u ** 2 를 돌려주는」
# 이름 없는 함수이고, torch.exp 는 원소마다 e^x 를 계산하는 함수다.
for tag, fn in [("x ** 2  (입력을 저장)", lambda u: u ** 2),
                ("x.exp() (출력을 저장)", torch.exp)]:
    x = torch.tensor([2.0], requires_grad=True)   # 형태 (1,) 인 리프 텐서, 값 2
    y = fn(x)                                     # 2^2 = 4 또는 e^2 = 7.389...
    # y.detach() 는 y 와 저장소는 같고 계산 그래프와는 끊긴 텐서다. 그 0번 칸에
    # 99 를 써 넣으면 y 의 값도 99 로 바뀐다 — 계산 그래프를 거치지 않고.
    y.detach()[0] = 99.0          # 같은 저장소를 그래프 밖에서 덮어쓴다
    # try 안에서 오류가 나면 프로그램이 멈추지 않고 except 로 넘어간다.
    try:
        # .sum() 은 원소를 모두 더해 스칼라(0차원 텐서)로 만든다. 그 스칼라에서
        # 역전파를 시작한다.
        y.sum().backward()
        # x ** 2 의 역전파는 2x = 2*2 = 4 를 쓴다. 저장해 둔 것이 입력 x 이므로
        # y 를 덮어쓴 것이 걸리지 않고 4 가 그대로 나온다.
        print(f"       {tag:22s} x.grad = {x.grad.item():.1f}  <- 아무 말이 없다")
    except RuntimeError as err:
        # str(err) 는 오류 글, .splitlines()[0] 은 그 첫 줄, [:46] 은 앞 46자다.
        print(f"       {tag:22s} {str(err).splitlines()[0][:46]}")

print("     내 모듈에 둘을 나란히 두면")
# [n for n, _ in ...] 는 (이름, 텐서) 짝에서 이름만 모아 리스트로 만든다.
print("       parameters():", [n for n, _ in scaler.named_parameters()])
# state_dict() 는 모델을 저장할 때 파일에 들어가는 {이름: 텐서} 사전이다.
# list(...) 로 감싸면 그 이름만 남는다.
print("       state_dict ():", list(scaler.state_dict()))
# 형태 (2, 3) 인 입력을 넣고 출력을 모두 더해 스칼라로 만든 뒤 역전파한다.
# 합 = Σ (x * gain + shift) 이므로 gain 에 대한 그래디언트는 x 를 열마다 더한 값이다.
#   [1+4, 2+5, 3+6] = [5, 7, 9]
scaler(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).sum().backward()
# .tolist() 는 텐서를 파이썬 리스트로 바꾼다.
print("       gain.grad :", scaler.gain.grad.tolist())
# shift 는 경사 추적이 꺼진 맨 텐서라 .grad 가 None 으로 남는다.
print("       shift.grad:", scaler.shift.grad, "  <- 트레인에서 조용히 빠진다")

# === 2. 플래그는 연산을 타고 앞으로 번진다 ====================================
# 문자열 맨 앞의 \n 은 줄바꿈이다 — 빈 줄 하나를 띄우고 찍는다.
print("\n  2. 입력 가운데 하나라도 켜져 있으면 출력이 켜진다")
# 표의 머리 줄이다. {'왼쪽':>6} 은 6칸에 오른쪽으로 붙여 찍으라는 뜻이다.
print(f"     {'왼쪽':>6}{'오른쪽':>8}   {'출력':>6}   grad_fn")
# 왼쪽·오른쪽 플래그의 네 조합(꺼짐/켜짐 × 꺼짐/켜짐)을 모두 해 본다.
for left_on in (False, True):
    for right_on in (False, True):
        u = torch.randn(3, requires_grad=left_on)
        v = torch.randn(3, requires_grad=right_on)
        out = u * v                    # 원소별 곱 — 형태 (3,)
        # grad_fn 은 「이 텐서를 만든 연산」의 기록이고, 역전파가 이것을 따라
        # 거꾸로 간다. 곱셈으로 만든 텐서는 MulBackward0 을 받고, 기록된 것이
        # 없으면 None 이다. None 에는 클래스 이름이 없어 글자 "None" 을 따로 둔다.
        fn = type(out.grad_fn).__name__ if out.grad_fn is not None else "None"
        print(f"     {str(left_on):>6}{str(right_on):>8}   "
              f"{str(out.requires_grad):>6}   {fn}")

# === 3. 거는 것과 갖는 것과 받는 것은 서로 다르다 ==========================
print("\n  3. 리프만인 것과 아닌 것")
leaf = torch.randn(3, requires_grad=True)   # 직접 만들었다 — 리프 텐서
grown = leaf * 2                      # 계산으로 생긴 것
print(f"     leaf   is_leaf={leaf.is_leaf}   requires_grad={leaf.requires_grad}")
print(f"     grown  is_leaf={grown.is_leaf}  requires_grad={grown.requires_grad}"
      f"   <- 리프가 아닌데 켜져 있다")

print("     requires_grad_() 로 바꿔 보면")
# (이름표, 텐서를 만드는 함수, 바꿀 값) 셋을 차례로 꺼낸다. lambda: ... 는 인자
# 없는 함수이고 make() 로 부를 때마다 새 텐서를 만든다. 끝에 「* 2」 가 붙은
# 것은 계산의 결과이므로 리프가 아니다.
for tag, make, val in [("리프에 켜기  ", lambda: torch.randn(3), True),
                       ("리프에 끄기  ", lambda: torch.randn(3, requires_grad=True), False),
                       ("리프 아닌 것에 켜기", lambda: torch.randn(3, requires_grad=True) * 2, True),
                       ("리프 아닌 것에 끄기", lambda: torch.randn(3, requires_grad=True) * 2, False)]:
    try:
        # 플래그를 val 로 바꾸고, 바뀐 뒤의 플래그를 찍는다.
        print(f"       {tag:18s} 된다 -> {make().requires_grad_(val).requires_grad}")
    except RuntimeError as err:
        # 리프가 아닌 텐서를 끄려 할 때만 오류가 나 이리로 온다.
        print(f"       {tag:18s} {str(err).splitlines()[0][:44]}")
print("     requires_grad=False 인 텐서는 언제나 리프다 — 그래서 '리프가 아니다'")
print("     는 곧 '켜져 있다' 이고, 끄라는 요청만 거절당한다")

# === 4. 플래그가 하나도 없으면 그래프가 아예 안 세워진다 ======================
print("\n  4. 아무도 안 켜져 있으면")
cold_w = torch.randn(4, 3)             # 형태 (4, 3) 인 행렬 — 플래그가 꺼져 있다
cold_x = torch.randn(3)                # 형태 (3,) 인 벡터 — 이것도 꺼져 있다
# @ 는 행렬 곱이다. (4, 3) 행렬에 (3,) 벡터를 곱하면 형태 (4,) 인 벡터가 나온다.
# .sum() 으로 그 넷을 더해 스칼라로 만든다. 값은 평소와 똑같이 계산된다.
cold_out = (cold_w @ cold_x).sum()
print(f"     grad_fn          : {cold_out.grad_fn}")   # 기록된 연산이 없다 — None
try:
    cold_out.backward()                # 따라갈 계산 그래프가 없어 오류가 난다
except RuntimeError as err:
    print(f"     backward()       : {str(err).splitlines()[0][:58]}")

# === 5. 그래디언트는 어디에 어떤 형태로 쌓이는가 =================================
print("\n  5. .grad 가 사는 자리")
warm_w = torch.randn(4, 3, requires_grad=True)   # 이번에는 웨이트에 플래그를 켠다
warm_x = torch.randn(3)                          # 입력은 꺼 둔다
mid = warm_w @ warm_x                  # 형태 (4,) — 플래그가 번져 mid 도 켜진다
total = (mid ** 2).sum()               # 원소마다 제곱해 더한 스칼라
total.backward()                       # warm_w.grad 에 d(total)/d(warm_w) 가 들어간다
# .shape 는 텐서의 형태다. tuple(...) 로 감싸면 (4, 3) 처럼 간단히 찍힌다.
# 그래디언트의 (i, j) 칸은 total 을 warm_w 의 (i, j) 칸으로 미분한 값이므로
# 형태가 warm_w 와 같다.
print(f"     warm_w 형태      : {tuple(warm_w.shape)}")
print(f"     warm_w.grad 형태  : {tuple(warm_w.grad.shape)}   같은가 "
      f"{warm_w.shape == warm_w.grad.shape}")
# 첫 역전파가 넣은 값 하나를 따로 적어 둔다. [0, 0] 은 0번 행 0번 열의 칸이다.
first = warm_w.grad[0, 0].item()
# 같은 계산을 한 번 더 하고 한 번 더 역전파한다. 역전파가 끝나면 계산 그래프가
# 지워지므로 순전파부터 다시 한다. .grad 를 지우지 않았으므로 새 그래디언트가
# 앞의 값 위에 더해져 두 배가 된다.
mid = warm_w @ warm_x
(mid ** 2).sum().backward()
print(f"     한 번 더 역전파하면 {first:.4f} 가 {warm_w.grad[0, 0].item():.4f} 로"
      f" — 덮어쓰지 않고 더한다")

# === 6. 플래그를 입력에 걸면 다른 것을 얻는다 =================================
print("\n  6. 같은 로스, 플래그만 옮기면")
# 손으로 따라갈 수 있게 수를 직접 적는다. 로스는 L = Σ (W x - target)^2 이다.
fixed_w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])     # W — 형태 (2, 2)
fixed_x = torch.tensor([1.0, 1.0])                   # x — 형태 (2,)
target = torch.tensor([2.0, 2.0])                    # 맞혀야 할 값 — 형태 (2,)

# .clone() 으로 수를 베낀 새 텐서를 만들고 그 사본에만 플래그를 켠다. 원본
# fixed_w 는 꺼진 채로 남아 아래에서 다시 쓰인다.
w_on = fixed_w.clone().requires_grad_(True)
# W x = [1+2, 3+4] = [3, 7],   오차 r = [3-2, 7-2] = [1, 5],   L = 1 + 25 = 26
# dL/dW 의 (i, j) 칸 = 2 * r_i * x_j   →   [[2, 2], [10, 10]]
((w_on @ fixed_x - target) ** 2).sum().backward()
# .flatten() 은 형태 (2, 2) 를 한 줄 (4,) 로 편다.
print(f"     플래그가 W 에  dL/dW = {w_on.grad.flatten().tolist()}")

# 이번에는 입력의 사본에 플래그를 켠다. 순전파의 수는 위와 똑같다.
x_on = fixed_x.clone().requires_grad_(True)
# dL/dx 의 j 칸 = Σ_i 2 * r_i * W_ij
#   [2*1*1 + 2*5*3,  2*1*2 + 2*5*4] = [32, 44]
((fixed_w @ x_on - target) ** 2).sum().backward()
print(f"     플래그가 x 에  dL/dx = {x_on.grad.tolist()}")
print("     앞엣것으로 모델을 고치고, 뒤엣것으로 입력을 고친다")
```

**출력:**

```
1. 플래그의 기본값과 켜는 길
     torch.randn(3)                       requires_grad=False
     torch.randn(3, requires_grad=True)   requires_grad=True
     .requires_grad_(True) 로 켠 것          requires_grad=True
     nn.Parameter(...)                    requires_grad=True
     내장 레이어가 속에 두는 것도 그 셋째 길이다
       nn.Linear 의 weight   Parameter  requires_grad=True
       nn.Linear 의 bias     Parameter  requires_grad=True
     맨 텐서를 넘기면
       넘기기 전 plain  requires_grad=False
       넘긴 뒤   handed requires_grad=True  is_leaf=True
       plain 은 그대로 꺼져 있다: False   그런데 저장소는 같다: True
       plain 을 고치면 handed 도 바뀐다: 99.0   <- 베낀 것이 아니다
     저장소 공유는 nn.Parameter 만의 일이 아니다
       base.view(4, 3)      저장소 공유 True
       base[0]              저장소 공유 True
       base.t()             저장소 공유 True
       base.detach()        저장소 공유 True
       nn.Parameter(base)   저장소 공유 True
       base.clone()         저장소 공유 False
     저장해 둔 값을 제자리에서 고치면
       x ** 2  (입력을 저장)       x.grad = 4.0  <- 아무 말이 없다
       x.exp() (출력을 저장)       one of the variables needed for gradient compu
     내 모듈에 둘을 나란히 두면
       parameters(): ['gain']
       state_dict (): ['gain']
       gain.grad : [5.0, 7.0, 9.0]
       shift.grad: None   <- 트레인에서 조용히 빠진다

  2. 입력 가운데 하나라도 켜져 있으면 출력이 켜진다
         왼쪽     오른쪽       출력   grad_fn
      False   False    False   None
      False    True     True   MulBackward0
       True   False     True   MulBackward0
       True    True     True   MulBackward0

  3. 리프만인 것과 아닌 것
     leaf   is_leaf=True   requires_grad=True
     grown  is_leaf=False  requires_grad=True   <- 리프가 아닌데 켜져 있다
     requires_grad_() 로 바꿔 보면
       리프에 켜기              된다 -> True
       리프에 끄기              된다 -> False
       리프 아닌 것에 켜기         된다 -> True
       리프 아닌 것에 끄기         you can only change requires_grad flags of l
     requires_grad=False 인 텐서는 언제나 리프다 — 그래서 '리프가 아니다'
     는 곧 '켜져 있다' 이고, 끄라는 요청만 거절당한다

  4. 아무도 안 켜져 있으면
     grad_fn          : None
     backward()       : element 0 of tensors does not require grad and does not ha

  5. .grad 가 사는 자리
     warm_w 형태      : (4, 3)
     warm_w.grad 형태  : (4, 3)   같은가 True
     한 번 더 역전파하면 1.9227 가 3.8454 로 — 덮어쓰지 않고 더한다

  6. 같은 로스, 플래그만 옮기면
     플래그가 W 에  dL/dW = [2.0, 2.0, 10.0, 10.0]
     플래그가 x 에  dL/dx = [32.0, 44.0]
     앞엣것으로 모델을 고치고, 뒤엣것으로 입력을 고친다
```

**기본은 꺼져 있고, 켜는 길이 셋이다.** `torch.randn(3)`으로 만든 텐서는 플래그가
**내려가 있다.** 켜는 길이 셋인데, 셋째가
평소에 가장 많이 쓰인다.

| | 어떻게 |
|---|---|
| 만들 때 | `torch.randn(3, requires_grad=True)` |
| 만든 뒤에 | `t.requires_grad_(True)` |
| **`nn.Parameter`로 두면** | **`nn.Parameter(torch.randn(3))` — 저절로 켜진다** |

마지막 줄이 모델의 웨이트가 **전부 켜져 있는** 까닭이다. 레이어 하나를 열어 보면
그렇게 되어 있다.

```
내장 레이어가 속에 두는 것도 그 셋째 길이다
  nn.Linear 의 weight   Parameter  requires_grad=True
  nn.Linear 의 bias     Parameter  requires_grad=True
```

`nn.Linear`가 속에 두는 `weight`와 `bias`는 맨 텐서가 아니라 `nn.Parameter`이고,
`nn.Parameter`는 기본으로 플래그를 켠다. 그래서 **모델을 만드는 것만으로 트레인
대상이 정해진다** — 어디에도 `requires_grad=True`를 적은 적이 없는데 옵티마이저에
`model.parameters()`를 넘기면 그대로 트레인이 된다.

**내가 만든 텐서도 넘기면 된다.** 셋째 길은 내장 레이어만의 것이 아니다. 맨 텐서를
하나 만들어 `nn.Parameter`에 넘기면 그 자리에서 트레인 대상이 된다.

```
맨 텐서를 넘기면
  넘기기 전 plain  requires_grad=False
  넘긴 뒤   handed requires_grad=True  is_leaf=True
  plain 은 그대로 꺼져 있다: False   그런데 저장소는 같다: True
  plain 을 고치면 handed 도 바뀐다: 99.0   <- 베낀 것이 아니다
```

두 가지가 같이 나오고, 둘이 **서로 반대 방향**이다.

**플래그는 넘긴 쪽으로 옮아가지 않는다.** `nn.Parameter`는 받은 텐서의 플래그를
뒤집는 것이 아니라, **플래그가 켜진 새 텐서 객체**를 만들어 돌려준다. 넘긴
쪽(`plain`)은 그대로 `False`다.

**그런데 데이터는 베껴지지 않고 함께 쓰인다.** `plain`과 `handed`는 **같은
저장소(storage)** 를 가리킨다. 그래서 넘긴 뒤에 `plain`을 고치면 파라메터가
따라 바뀐다.

이 둘이 어긋나는 까닭은 텐서가 **두 계층**으로 되어 있기 때문이다.

| | 무엇이 들어 있나 | `nn.Parameter`를 거치면 |
|---|---|---|
| **텐서 객체** (메타데이터) | 형태(shape) · 스트라이드(stride) · 오프셋 · `requires_grad` · `grad_fn` · `.grad` | **새로 하나 생긴다** |
| **저장소** (storage) | 실제 수가 한 줄로 누워 있는 메모리 | **그대로 하나다** |

`plain`과 `handed`는 **객체는 둘, 저장소는 하나**다. 그래서 플래그는 따로 서고
숫자는 같이 움직인다. 이 모델을 세우는 쪽이
[메모리 배치와 스트라이드](../tensor_attrs/memory_layout_strides.md)다.

떼어 놓으려면 **저장소를 새로 잡아야** 한다 — `nn.Parameter(plain.clone())`.
`clone()`을 빼면, 같은 텐서 하나를 `nn.Parameter`에 두 번 넘겨 파라메터 둘을
만들었을 때 둘이 **같은 메모리를 고친다.**

**저장소 공유는 `nn.Parameter`만의 일이 아니다.** 텐서를 거의 어떻게 건드려도
새 메타데이터만 생기고 저장소는 그대로다.

```
base.view(4, 3)      저장소 공유 True
base[0]              저장소 공유 True
base.t()             저장소 공유 True
base.detach()        저장소 공유 True
nn.Parameter(base)   저장소 공유 True
base.clone()         저장소 공유 False
```

**복사하는 것은 `clone()` 하나뿐이다.** 뷰(view)·슬라이스·전치·`detach()`·
`nn.Parameter`는 전부 공짜다 — 그러니까 역전파가 레이어마다 생기는 중간 텐서를
감당할 수 있다. `detach()`가 그래프만 끊고 저장소는 안 끊는다는 것은
「[detach와 detach_](09_detach_vs_detach_.md)」 쪽이, NumPy까지 같은 메모리를
나눠 쓴다는 것은 「[NumPy로 변환하기](12_converting_to_numpy.md)」 쪽이 잰다.

!!! warning "공짜의 대가는 제자리 연산(in-place)이다"
    저장소가 하나이므로 `a[0] = 99`나 `a.add_(1)` 처럼 **제자리에서 고치면**
    그 저장소를 가리키는 모든 텐서에 보인다. 역전파가 저장해 둔 값이 거기
    있었다면 결과가 틀어진다.

    파이토치는 저장소마다 **버전 카운터**를 두어 이것을 잡는다. 다만 **늘
    잡지는 않는다.**

    ```
    x ** 2  (입력을 저장)       x.grad = 4.0  <- 아무 말이 없다
    x.exp() (출력을 저장)       one of the variables needed for gradient compu
    ```

    두 식 모두 출력 `y`를 제자리에서 덮어썼는데 한쪽만 막혔다. `x.exp()`는
    역전파에 **출력** $e^x$ 가 필요해서 그것을 저장해 두므로 버전이 어긋난
    것이 걸리고, `x ** 2`는 **입력** $x$ 만 저장하므로 `y`를 덮어쓴 것이
    아무 데도 안 걸린다. 무엇이 저장되는지는
    [retain_graph](03_retain_graph.md)가 재어 둔다.

    즉 **에러가 안 났다고 안전한 것이 아니다.** 역전파에 쓰일 텐서를
    제자리에서 고치지 않는 것이 규칙이고, 버전 카운터는 그물이지 보증이
    아니다.

**감싸는 것이 곧 등록이다.** 같은 형태의 텐서 둘을 내 모듈 안에 나란히 두되
하나만 감싸 보면, 둘이 어디서 갈리는지가 한눈에 보인다.

```python
# 위 프로그램의 Scale 클래스를 다시 적은 것이다.
class Scale(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))   # 감쌌다 — 모듈에 파라메터로 등록된다
        self.shift = torch.zeros(d)               # 안 감쌌다 — 보통의 속성일 뿐이다

    def forward(self, x):
        return x * self.gain + self.shift         # 순전파는 둘을 똑같이 쓴다
```

```
내 모듈에 둘을 나란히 두면
  parameters(): ['gain']
  state_dict (): ['gain']
  gain.grad : [5.0, 7.0, 9.0]
  shift.grad: None   <- 트레인에서 조용히 빠진다
```

`forward`는 둘을 똑같이 쓴다. 그런데 `shift`는 **세 군데에서 한꺼번에 빠진다** —
`parameters()`에 안 들고, `state_dict()`에 안 들고, `.grad`를 못 받는다.
옵티마이저에 `model.parameters()`를 넘기는 한 `shift`는 **영원히 `0`이다.**

!!! warning "이 실수는 에러를 내지 않는다"
    `self.shift = torch.zeros(d)`는 잘 돌아간다. 순전파도 되고 역전파도 되고
    로스도 내려간다. 다만 그 수 하나가 트레인되지 않을 뿐이고, **그걸 알려 주는
    것이 아무것도 없다.** 저장했다 불러와도 조용히 초기값으로 돌아온다.

    그래서 커스텀 모듈을 쓸 때는 `print(list(m.named_parameters()))` 한 줄로
    **내가 둔 것이 거기 있는지** 확인하고 넘어가는 편이 낫다.
    [레이어 출력하기](13_print_layers.md)가 세는 법을 모아 둔다.

    트레인은 안 되는데 **저장은 되어야 하는** 수라면 맨 텐서로 두지 말고
    `register_buffer`로 등록한다. 그것이 버퍼라는 갈래다.

거꾸로 **입력 데이터는 꺼져 있다.** 그냥 `torch.randn`이나 `ToTensor()`로 온
텐서이기 때문이고, 그것이 맞는 기본값이다. 왜 맞는지는 아래 「플래그를 어디에
거느냐」에서 본다.

**플래그는 연산을 타고 앞으로 번진다.**

```
   왼쪽     오른쪽       출력   grad_fn
  False   False    False   None
  False    True     True   MulBackward0
   True   False     True   MulBackward0
   True    True     True   MulBackward0
```

**입력 가운데 하나라도 켜져 있으면 출력이 켜지고 `grad_fn`을 받는다.** 켜진 것이
하나도 없으면 출력도 꺼지고 `grad_fn`이 `None`이다.

![두 줄 그림. 윗줄은 입력 둘 가운데 하나만 플래그가 켜져 있는데도 곱셈을 지난 출력에 플래그가 켜지고 grad_fn 이 붙는다. 아랫줄은 입력 둘 다 꺼져 있어 연산도 출력도 점선으로 흐리게 그려지고 grad_fn 이 비어 있다.](figures/flag_spreads.svg)

이 한 줄이 **그래프가 세워지는 메커니즘 전부**다. 파이토치는 「이제부터 기록한다」는
스위치를 따로 두지 않는다. 연산을 할 때마다 입력들의 플래그를 보고, 하나라도
켜져 있으면 그 연산을 적어 둔다. 적어 둔 것들이 이어 붙어 그래프가 된다.

그래서 **$x$ 하나만 켜도 그 뒤의 모든 것이 따라 켜진다.** 로스까지 가는 길에
놓인 연산이 전부 기록되고, 그 기록을 거꾸로 걸어 내려오는 것이 `backward()`다.

**거는 것과 갖는 것과 받는 것이 서로 다르다.** 셋을 한자리에 놓으면 이렇다. 여기서 **리프**는 리프 텐서(leaf tensor), 곧 연산의 결과가 아니라 직접 만든 텐서를 가리킨다.

```
leaf   is_leaf=True   requires_grad=True
grown  is_leaf=False  requires_grad=True   <- 리프가 아닌데 켜져 있다

  리프에 켜기              된다 -> True
  리프에 끄기              된다 -> False
  리프 아닌 것에 켜기         된다 -> True
  리프 아닌 것에 끄기         you can only change requires_grad flags of l
```

| | 리프만인가 |
|---|---|
| 만들 때 주는 **인자** `requires_grad=` | **그렇다** — 텐서를 만드는 함수에만 달려 있고, 만든 것은 다 리프다 |
| 나중에 부르는 **메서드** `requires_grad_()` | **끄는 것만** 그렇다 |
| `requires_grad`를 **갖는** 것 | **아니다** — 리프가 아니면 반드시 켜져 있다 |
| `.grad`를 **받는** 것 | **그렇다** |

둘째 줄과 셋째 줄은 같은 사실의 앞뒤다. **`requires_grad=False`인 텐서는 언제나
리프이므로**, 「리프가 아니다」는 곧 「켜져 있다」이다. 그래서 리프가 아닌 것에 켜라는
요청은 이미 켜져 있어 그냥 지나가고, **끄라는 요청만 거절당한다.**

그리고 기본값이 셋으로 갈린다.

| | `requires_grad` 기본값 |
|---|---|
| 그냥 만든 텐서, 입력 데이터 | **꺼짐** |
| `nn.Parameter` — 내장 레이어의 웨이트 | **켜짐** |
| 버퍼 — 배치 정규화의 `running_mean` 같은 것 | 꺼짐 |

가운뎃줄이 **모델을 만드는 것만으로 트레인 대상이 정해지는** 까닭이고, 윗줄이
**입력에 대한 그래디언트는 따로 청하지 않으면 안 나오는** 까닭이다. 아랫줄은
[레이어 출력하기](13_print_layers.md)가 따로 다룬다.

**아무도 안 켜져 있으면 적히지 않는다.**

```
grad_fn          : None
backward()       : element 0 of tensors does not require grad and does not ha
```

곱셈과 덧셈은 **그대로 일어난다.** 값은 똑같이 나온다. 다른 것은 **적히지
않는다**는 것뿐이고, 적힌 것이 없으니 되짚어 갈 길도 없다. 그래서
`backward()`가 거절한다.

이것이 추론이 트레인보다 싼 까닭이다 — 「경사 추적을 언제 멈출 것인가」 쪽이 재어 보인
대로, 되짚지 않을 것이라면 **중간값을 들고 있을 까닭이 없다.**

**그래디언트는 그 텐서 옆에, 같은 형태로, 더해져 쌓인다.**

```
warm_w 형태      : (4, 3)
warm_w.grad 형태  : (4, 3)   같은가 True
한 번 더 역전파하면 1.9227 가 3.8454 로 — 덮어쓰지 않고 더한다
```

그래디언트가 사는 자리는 **그 텐서의 `.grad` 속성**이고, 형태가 **텐서와 같다.**
$\partial L / \partial w_{ij}$가 $w_{ij}$가 있던 자리에 그대로 들어앉는다고
보면 된다. 그래서 갱신이 `w -= lr * w.grad` 한 줄로 적힌다.

그리고 **덮어쓰지 않고 더한다.** 두 번 역전파하면 두 배가 된다. 이 성질이
「[경사 누적](02_gradient_accumulation.md)」 쪽의 바탕이고, 트레인 루프마다 지우는 줄이
있어야 하는 까닭이다.

다만 `.grad`가 **아무 텐서에나 쌓이지는 않는다.** 플래그가 켜져 있어도 계산으로
생긴 중간 텐서는 받지 않고, 내가 만든 **리프**만 받는다. 그 구별은
「[스칼라에 대한 기본 역전파](01_basic_scalar_backward.md)」 쪽이 그림과 함께 다룬다.

**플래그를 어디에 거느냐가 무엇을 얻느냐를 정한다.**

```
플래그가 W 에  dL/dW = [2.0, 2.0, 10.0, 10.0]
플래그가 x 에  dL/dx = [32.0, 44.0]
```

**로스도 같고 수도 같은데 플래그만 옮겼더니 전혀 다른 것이 나온다.** 앞엣것은
웨이트를 어느 쪽으로 고쳐야 로스가 주는지이고, 뒤엣것은 **입력을** 어느 쪽으로
고쳐야 로스가 주는지다.

트레인에 필요한 것은 앞엣것이다. 그래서 입력 데이터의 플래그가 꺼져 있는 것이 맞고,
켜 두면 쓰지도 않을 그래디언트를 계산하느라 비용만 치른다.

뒤엣것이 쓸모없는 것은 아니다. **그 방향으로 입력을 밀면 로스가 커진다** — 모델이
틀리도록 그림을 고치는 것이고, 적대적 예제가 만들어지는 자리가 이 한 줄이다.
「[with torch.no_grad()](10_with_torch_no_grad.md)」 쪽이 그 그래디언트를 평가 루프 한가운데서
어떻게 얻는지, 그리고 예측이 실제로 뒤집히는지를 재어 둔다.

### 스칼라 로스: `.backward()` 메서드

스칼라 로스 $L: \mathbb{R}^n \rightarrow \mathbb{R}$에 대해 `L.backward()`를 호출하면 `requires_grad=True`인 모든 리프 텐서 $x$에 대한 그래디언트 $\nabla_x L$이 계산된다.

$$\nabla_x L = \left[\frac{\partial L}{\partial x_1}, \frac{\partial L}{\partial x_2}, \ldots, \frac{\partial L}{\partial x_n}\right]^T$$

```python
import torch

torch.manual_seed(0)                   # 시드를 박아 아래 난수를 고정한다

# 표준정규분포에서 뽑은 수 셋 — 형태 (3,). 경사 추적을 켠 리프 텐서다.
x = torch.randn(3, requires_grad=True)
# 텐서를 그대로 찍으면 값 옆에 requires_grad=True 가 함께 보인다.
print(f"x: {x}")

# 순전파: loss = sum(x²)
# x ** 2 는 원소마다 제곱한다 — 형태는 그대로 (3,). .sum() 이 그 셋을 더해
# 스칼라로 만든다.   1.5410² + (-0.2934)² + (-2.1788)² = 7.2079 (반올림)
loss = (x ** 2).sum()
# 찍히는 grad_fn=<SumBackward0> 은 loss 가 sum 연산으로 만들어졌다는 기록이다.
print("loss:", loss)   # f-string을 쓰면 0차원 텐서가 맨 실수로 찍혀 grad_fn이 사라진다

# 역전파: d(loss)/dx = 2x
# loss 를 x 의 원소 하나하나로 미분한 값 셋이 x.grad 에 들어간다. 형태가 x 와
# 같은 (3,) 이다.
loss.backward()

print(f"x.grad: {x.grad}")
# 손으로 구한 2x 와 견준다. x.detach() 는 값은 같고 계산 그래프와는 끊긴
# 텐서를 돌려준다. 견줄 값을 만드는 계산까지 기록할 까닭이 없어 떼어 놓고 쓴다.
print(f"Expected (2x): {2 * x.detach()}")
# torch.allclose 는 두 텐서의 모든 원소가 부동소수점 오차 안에서 같으면 True 다.
print(f"Match: {torch.allclose(x.grad, 2 * x.detach())}")
```

**출력:**
```
x: tensor([ 1.5410, -0.2934, -2.1788], requires_grad=True)
loss: tensor(7.2079, grad_fn=<SumBackward0>)
x.grad: tensor([ 3.0820, -0.5869, -4.3576])
Expected (2x): tensor([ 3.0820, -0.5869, -4.3576])
Match: True
```

### 그래디언트 저장: 리프 텐서와 리프가 아닌 텐서

기본적으로 그래디언트는 리프 텐서에 대해서**만** 저장된다.

```python
import torch

# 값을 직접 적어 만든 리프 텐서 — 형태 (3,). 1. 은 1.0 을 줄여 적은 것이다.
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = 2 * x         # 리프가 아니다 — 계산으로 생긴 중간 텐서.  y = [2, 4, 6]
z = (y ** 2).sum()   # 4 + 16 + 36 = 56 — 스칼라

z.backward()

# x 는 리프라 .grad 를 받는다.   dz/dx = dz/dy * dy/dx = 2y * 2 = 4y = [8, 16, 24]
print(f"x.grad: {x.grad}")  # tensor([ 8., 16., 24.]) — 저장된다
# y 는 중간 텐서다. 역전파가 y 를 지나가기는 하지만 그 값을 남겨 두지 않는다.
# (리프가 아닌 텐서의 .grad 를 읽었다고 파이토치가 경고를 한 줄 낼 수 있다.)
print(f"y.grad: {y.grad}")  # None — 기본으로는 저장되지 않는다
```

**출력:**

```
x.grad: tensor([ 8., 16., 24.])
y.grad: None
```

리프가 아닌 텐서의 그래디언트를 저장하려면 역전파 전에 `.retain_grad()`를 쓴다.

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = 2 * x             # y = [2, 4, 6] — 리프가 아닌 중간 텐서
# retain_grad() 는 「역전파 때 이 중간 텐서의 그래디언트도 .grad 에 남겨 달라」는
# 요청이다. backward() 보다 먼저 불러야 한다.
y.retain_grad()       # 그래디언트 저장을 요청한다
z = (y ** 2).sum()    # 4 + 16 + 36 = 56

z.backward()

print(f"x.grad: {x.grad}")  # tensor([ 8., 16., 24.])   dz/dx = 4y
print(f"y.grad: {y.grad}")  # tensor([ 4.,  8., 12.])   dz/dy = 2y — 이번에는 남았다
```

**출력:**

```
x.grad: tensor([ 8., 16., 24.])
y.grad: tensor([ 4.,  8., 12.])
```

**검증:** $y = 2x = [2, 4, 6]$이고 $z = \sum y_i^2$일 때 다음과 같다.

- $\frac{\partial z}{\partial y_i} = 2y_i$이므로 $\nabla_y z = [4, 8, 12]$이다
- $\frac{\partial z}{\partial x_i} = \frac{\partial z}{\partial y_i} \cdot \frac{\partial y_i}{\partial x_i} = 2y_i \cdot 2 = 4y_i$이므로 $\nabla_x z = [8, 16, 24]$이다

### `torch.autograd.grad` 인터페이스

좀 더 세밀하게 제어하려면 `torch.autograd.grad`를 쓴다. `.grad` 속성을 채우지 않고 그래디언트를 곧바로 반환한다.

```python
import torch

x = torch.tensor([2.0], requires_grad=True)   # 형태 (1,), 값 2
y = x ** 3                                    # 2³ = 8

# torch.autograd.grad(출력, 입력) 은 출력을 입력으로 미분한 값을 「돌려준다」.
# backward() 와 달리 x.grad 에 넣지 않는다 — 이 줄 뒤에도 x.grad 는 None 이다.
#
# 입력을 여럿 넘길 수 있으므로 결과는 늘 튜플이다. 입력이 하나여도 원소 하나짜리
# 튜플이 오므로 (grad_y,) = ... 로 받아 그 하나를 꺼낸다.
# 그래디언트들의 튜플을 반환한다
(grad_y,) = torch.autograd.grad(y, x)
print(f"dy/dx = 3x² = {grad_y}")  # tensor([12.])   3 * 2² = 12
```

**출력:**

```
dy/dx = 3x² = tensor([12.])
```

이는 고차 도함수(1.4절의 「야코비안과 고차 미분」 쪽 참고)를 다룰 때, 그리고 `retain_grad()`를 호출하지 않고 리프가 아닌 텐서의 그래디언트가 필요할 때 특히 유용하다.

---

## 4. 벡터-야코비 곱의 틀

### 왜 VJP인가?

PyTorch의 역전파는 전체 야코비 행렬을 계산하지 **않는다.** 고차원 파라메터 공간에서는 그 비용이 감당할 수 없이 크기 때문이다. 대신 그래디언트 기반 최적화에 필요한 전부인 **벡터-야코비 곱**(VJP)을 계산한다.

### 수학적 정의

야코비 행렬이 $J \in \mathbb{R}^{m \times n}$인 함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$과 상류 그래디언트(수반 변수, adjoint) $\bar{y} \in \mathbb{R}^m$이 주어졌을 때 다음과 같다.

$$\text{VJP:} \quad \bar{x} = J^T \bar{y} \in \mathbb{R}^n$$

동등하게 $\bar{x}^T = \bar{y}^T J$이다. 이것이 행렬 형태의 사슬 규칙이다. $\bar{y}$가 출력 쪽에서 온 그래디언트를 나르고, $J^T$를 곱하면 그것이 한 단계 더 입력 쪽으로 전파된다.

### 스칼라 출력에서의 암묵적 VJP

출력이 스칼라이면($m = 1$) 야코비 행렬이 행벡터(즉 그래디언트)로 줄어들고 상류 "벡터"는 스칼라 $\bar{y} = 1$이 된다. `loss.backward()`에 인수가 필요 없는 이유가 이것이다.

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
loss = (x ** 2).sum()                  # 1 + 4 + 9 = 14 — 스칼라

# backward() 에 넘기는 인수는 역전파의 출발값, 곧 본문의 상류 그래디언트 ȳ 다.
# 스칼라 로스에서 인수를 비우면 파이토치가 1 을 넣어 준다. 그래서 아래 두 호출은
# 같은 일을 한다.
# 동등한 호출:
# loss.backward()
loss.backward(torch.tensor(1.0))   # v = 1 을 드러내 놓고 넘긴다

print(f"x.grad: {x.grad}")  # tensor([2., 4., 6.])   d(loss)/dx = 2x
```

**출력:**

```
x.grad: tensor([2., 4., 6.])
```

### 스칼라가 아닌 출력에서의 명시적 VJP

스칼라가 아닌 출력에 대해서는 상류 그래디언트 벡터 $\bar{y}$를 반드시 넘겨야 한다.

```python
import torch

x = torch.tensor([0.5, 1.0, -0.5], requires_grad=True)   # 형태 (3,)
# torch.sin 은 원소마다 sin 을 취한다. 출력도 형태 (3,) 인 벡터 — 스칼라가 아니다.
y = torch.sin(x)   # y: R³ → R³, 원소별

# y와 같은 형태의 그래디언트 벡터 v를 반드시 넘겨야 한다
# v 는 「출력의 각 성분에 무게를 얼마씩 둘 것인가」다. backward(v) 는 야코비 행렬
# J 를 통째로 만들지 않고 벡터-야코비 곱(VJP) J^T v 만 계산해 x.grad 에 넣는다.
v = torch.tensor([0.1, 1.0, 0.01])
y.backward(v)

# 원소별 sin에 대해 J = diag(cos(x))
# y_i 가 x_i 하나에만 달려 있으므로 J 는 대각선에 cos(x_i) 만 선 대각 행렬이다.
# x.grad = J^T v = v ⊙ cos(x)        (⊙ 는 원소별 곱)
#   [0.1*cos(0.5), 1.0*cos(1.0), 0.01*cos(-0.5)] = [0.0878, 0.5403, 0.0088]
expected = v * torch.cos(x.detach())   # detach 로 계산 그래프에서 떼어 값만 쓴다
print(f"x.grad:   {x.grad}")
print(f"Expected: {expected}")
print(f"Match: {torch.allclose(x.grad, expected)}")
```

**출력:**

```
x.grad:   tensor([0.0878, 0.5403, 0.0088])
Expected: tensor([0.0878, 0.5403, 0.0088])
Match: True
```

`v` 없이 스칼라가 아닌 텐서에 `.backward()`를 호출하면 `RuntimeError`가 발생한다.

```python
# 난수의 시드를 고정한다. 아래에서 뽑는 텐서가 돌릴 때마다 같은 값이 된다.
torch.manual_seed(0)
# 위 블록에 이어서 돌린다 — torch 는 거기서 이미 불러왔다.
x = torch.randn(3, requires_grad=True)
y = torch.sin(x)                       # 형태 (3,) — 스칼라가 아니다

# 출발값 v 를 주지 않았다. 스칼라라면 파이토치가 1 을 넣어 주지만, 벡터에는 무엇을
# 넣어야 할지 정할 수 없어 RuntimeError 를 낸다. try/except 로 그 오류를 받아 찍는다.
try:
    y.backward()  # 실패한다 — 출력이 스칼라가 아니다
except RuntimeError as e:
    print(f"Error: {e}")               # e 를 찍으면 오류 글이 나온다
```

**출력:**

```
Error: grad can be implicitly created only for scalar outputs
```

### 선형 변환에서의 VJP

$A \in \mathbb{R}^{m \times n}$인 선형 사상 $y = Ax$에 대해 야코비 행렬은 단순히 $J = A$이다.

```python
import torch

A = torch.tensor([[2.0, 0.0, -1.0],
                  [0.5, 3.0,  1.0]])   # (2, 3) — 행 2, 열 3
x = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)   # 형태 (3,)
# @ 는 행렬 곱이다. (2, 3) 행렬 × (3,) 벡터 → (2,) 벡터.
#   y = [2*1 + 0*(-2) + (-1)*0.5,   0.5*1 + 3*(-2) + 1*0.5] = [1.5, -5.0]
y = A @ x   # (2,)

# 상류 그래디언트 v 는 출력 y 와 형태가 같아야 한다 — (2,).
v = torch.tensor([3.0, -1.0])
y.backward(v)

# x.grad = A^T @ v
# y = A x 의 야코비 행렬은 A 자신이므로 VJP 는 A^T v 다. A.T 는 A 의 전치이고
# 형태가 (3, 2) 다.
#   [2*3 + 0.5*(-1),   0*3 + 3*(-1),   (-1)*3 + 1*(-1)] = [5.5, -3.0, -4.0]
# 결과의 형태는 x 와 같은 (3,) 이다. v 의 형태 (2,) 가 아니다.
expected = A.T @ v
print(f"x.grad: {x.grad}")
print(f"A^T @ v: {expected}")
print(f"Match: {torch.allclose(x.grad, expected)}")
```

**출력:**

```
x.grad: tensor([ 5.5000, -3.0000, -4.0000])
A^T @ v: tensor([ 5.5000, -3.0000, -4.0000])
Match: True
```

---

## 5. 순방향 모드와 역방향 모드 자동 미분

야코비 행렬의 곱 $J_f = J_L \cdot J_{L-1} \cdot \ldots \cdot J_1$은 두 가지 순서로 계산할 수 있으며, 각각이 자동 미분의 한 모드를 정의한다.

![같은 3행 2열 야코비 행렬을 두 번 그린 그림. 왼쪽 것은 첫 열이 채워져 순방향 모드가 입력마다 한 번씩 지나며 열을 얻는다는 것을, 오른쪽 것은 첫 행이 채워져 역방향 모드가 출력마다 한 번씩 지나며 행을 얻는다는 것을 보인다.](figures/forward_vs_reverse.svg)

### 순방향 모드: 야코비-벡터 곱(JVP)

순방향 모드는 도함수를 **입력에서 출력으로** 전파하며, 입력에서의 접벡터(tangent vector, 입력을 흔드는 방향) $\dot{x}$에 대해 $\dot{y} = J \cdot \dot{x}$를 계산한다.

**계산 순서(오른쪽에서 왼쪽으로):**

$$J \cdot \dot{x} = J_L \cdot (J_{L-1} \cdot (\ldots \cdot (J_2 \cdot (J_1 \cdot \dot{x}))))$$

특정한 $\dot{x}$로 순전파를 한 번 하면 방향도함수 하나를 얻는다. 입력이 $n$개일 때 전체 야코비 행렬을 얻으려면 **순전파 $n$번** 이 필요하다.

```python
import torch
# torch.func 는 「함수」를 받아 그 미분을 계산해 주는 도구 모음이다.
# 그 가운데 jvp 는 야코비-벡터 곱(JVP) J v 를 순방향 모드로 계산한다.
# 이름이 같은 torch.autograd.functional.jvp 는 값은 같지만 안에서 역전파를
# 두 번 돌려 계산하므로, 순방향 모드를 보이는 이 자리에는 맞지 않는다.
from torch.func import jvp

# 입력 둘, 출력 둘인 함수다. x[0] 은 x 의 0번 원소(x₁), x[1] 은 1번 원소(x₂)다.
def f(x):
    """f(x) = [sin(x₁·x₂), x₁² + x₂]"""
    # torch.stack 은 텐서 여럿을 쌓아 텐서 하나로 만든다. 여기서는 스칼라 둘을
    # 쌓아 형태 (2,) 인 벡터가 된다.
    return torch.stack([
        torch.sin(x[0] * x[1]),        # f₁ = sin(x₁ x₂)
        x[0]**2 + x[1]                 # f₂ = x₁² + x₂
    ])

# 미분할 자리 (x₁, x₂) = (1, 2). 순방향 모드는 계산 그래프를 기록하지 않고
# 순전파를 하면서 미분을 함께 나르므로 requires_grad 를 적지 않는다.
x = torch.tensor([1.0, 2.0])

# 야코비 행렬을 손으로 적으면 (행이 출력, 열이 입력)
#   J = [[x₂ cos(x₁x₂),  x₁ cos(x₁x₂)],   →  (1, 2) 에서  [[-0.8323, -0.4161],
#        [2 x₁,          1           ]]                    [ 2,       1     ]]
# 접벡터 [1, 0]으로 JVP를 하면 야코비 행렬의 첫 열을 얻는다
# 접벡터는 「입력을 어느 방향으로 흔들 것인가」다. [1, 0] 은 x₁ 만 흔든다는 뜻이다.
tangent = torch.tensor([1.0, 0.0])
# jvp(함수, 입력들, 접벡터들) 은 (함수값, J @ 접벡터) 둘을 돌려준다. (x,) 는 원소
# 하나짜리 튜플이다. 순전파 한 번에 함수값과 J v 가 함께 나온다.
output, jvp_result = jvp(f, (x,), (tangent,))

print(f"f(x) = {output}")              # [sin 2, 1² + 2] = [0.9093, 3]
print(f"J @ [1,0] = {jvp_result}")  # J 의 첫 열 [2 cos 2, 2] = [-0.8323, 2]
```

**출력:**

```
f(x) = tensor([0.9093, 3.0000])
J @ [1,0] = tensor([-0.8323,  2.0000])
```

### 역방향 모드: 벡터-야코비 곱(VJP)

역방향 모드는 도함수를 **출력에서 입력으로** 전파하며 $\bar{x} = J^T \bar{y}$를 계산한다.

**계산 순서(왼쪽에서 오른쪽으로):**

$$\bar{x}^T = ((((\bar{y}^T \cdot J_L) \cdot J_{L-1}) \cdot \ldots) \cdot J_1)$$

일반적인 $\bar{y}$로는 $J$의 행들의 가중합을 얻는다. $\bar{y}=e_i$인 기저 벡터를 쓰면 $J^T$의 i번 열, 즉 $J$의 i번 행을 열벡터 형태로 얻는다. **스칼라** 출력($m = 1$)에서는 **역전파 한 번** 으로 전체 그래디언트를 얻는다.

```python
import torch
# vjp 는 벡터-야코비 곱(VJP) v^T J 를 계산한다. 위의 jvp 와 짝이다.
from torch.autograd.functional import vjp

# 위 블록과 같은 함수다.   f(x) = [sin(x₁x₂), x₁² + x₂]
def f(x):
    return torch.stack([
        torch.sin(x[0] * x[1]),
        x[0]**2 + x[1]
    ])

x = torch.tensor([1.0, 2.0], requires_grad=True)   # 미분할 자리 (1, 2)

# 수반 벡터 [1, 0]으로 VJP를 하면 J의 첫 행을 얻는다
# torch.autograd.functional.vjp는 수반 벡터를 세 번째 인자로 직접 받고
# (출력, VJP 결과)를 돌려준다. 함수를 돌려주는 torch.func.vjp와 혼동하기 쉽다.
#
# 수반 벡터는 「출력의 어느 성분을 볼 것인가」다. [1, 0] 은 첫 출력 f₁ 만 본다는
# 뜻이고, 그러면 f₁ 을 x₁ 과 x₂ 로 미분한 값, 곧 J 의 첫 행이 나온다.
adjoint = torch.tensor([1.0, 0.0])
output, vjp_result = vjp(f, x, adjoint)

print(f"f(x) = {output}")
# J 의 첫 행 [2 cos 2, cos 2] = [-0.8323, -0.4161]. 위의 JVP 가 준 첫 열
# [-0.8323, 2] 와 첫 칸이 같다 — 둘 다 J 의 맨 왼쪽 위 칸을 지나기 때문이다.
print(f"[1,0] @ J = {vjp_result}")
```

**출력:**

```
f(x) = tensor([0.9093, 3.0000])
[1,0] @ J = tensor([-0.8323, -0.4161])
```

### 뉴럴 네트워크에 역방향 모드를 쓰는 이유

딥러닝의 전형적인 상황은 수백만 개의 파라메터($n \gg 1$)가 하나의 스칼라 로스($m = 1$)로 대응되는 것이다.

| 모드 | 필요한 통과 횟수 | 복잡도 |
|------|-----------------|------------|
| 순방향 | $n$(파라메터마다 한 번) | $O(n \cdot T)$ |
| **역방향** | $m = 1$ | $O(T)$ |

여기서 $T$는 순전파 한 번의 비용이다. 역방향 모드가 **압도적으로 효율적** 이며, PyTorch가 역전파를 기본으로 삼는 이유가 이것이다.

```python
import torch
import torch.nn as nn

# 파라메터가 약 20만 개인 뉴럴 네트워크
# nn.Sequential 은 레이어를 적은 차례대로 이어 붙여 모델 하나로 만든다.
model = nn.Sequential(
    nn.Linear(784, 256),   # 784 → 256.  웨이트 (256, 784) + 바이어스 (256,) = 200,960개
    nn.ReLU(),             # 음수를 0 으로 바꾸는 엑티베이션 함수 — 파라메터가 없다
    nn.Linear(256, 10)     # 256 → 10.   웨이트 (10, 256) + 바이어스 (10,) = 2,570개
)                          # 합쳐서 203,530개

# 난수의 시드를 고정한다. 아래에서 뽑는 텐서가 돌릴 때마다 같은 값이 된다.
torch.manual_seed(0)
# 가짜 데이터로 배치 하나를 만든다. 이 블록은 그래디언트의 형태만 찍으므로 값은
# 무엇이든 된다(그래서 시드를 박지 않았다).
x = torch.randn(32, 784)               # 입력 32개, 하나가 수 784개 — 형태 (32, 784)
# torch.randint(0, 10, (32,)) 는 0 이상 10 미만의 정수 32개를 뽑는다. 입력마다
# 하나씩 붙는 정답 클래스 번호다 — 형태 (32,).
y = torch.randint(0, 10, (32,))

logits = model(x)                      # 순전파 — 형태 (32, 10). 입력마다 클래스 10개의 점수
# cross_entropy 는 분류에 쓰는 로스다. 점수(logits)와 정답 번호를 받아 배치
# 평균 로스를 스칼라 하나로 돌려준다.
loss = nn.functional.cross_entropy(logits, y)

# 한 번의 역전파가 모든 파라메터의 그래디언트를 계산한다
loss.backward()

# 파라메터마다 .grad 가 채워졌고 형태가 그 파라메터와 같다. 이름 앞의 0, 2 는
# Sequential 안에서의 자리 번호다(1번은 ReLU 라 파라메터가 없다).
for name, param in model.named_parameters():
    print(f"{name}: grad shape = {param.grad.shape}")
```

**출력:**

```
0.weight: grad shape = torch.Size([256, 784])
0.bias: grad shape = torch.Size([256])
2.weight: grad shape = torch.Size([10, 256])
2.bias: grad shape = torch.Size([10])
```

<div class="storybox" markdown>

**그림으로 보는 AI 1.** 역전파는 반올림 오차를 좇다가 나왔다

1970년, 헬싱키 대학교의 석사 논문 한 편이 나온다. 제목은 「알고리즘의 누적 반올림 오차를 국소 반올림 오차의 테일러 전개로 나타내기」(*Algoritmin kumulatiivinen pyöristysvirhe yksittäisten pyöristysvirheiden Taylor-kehitelmänä*)다. 쓴 사람은 세포 리나인마다.

그가 풀려던 것은 트레인 문제가 아니었다. **수치 계산의 신뢰도 문제** 였다. 연산을 수천 번 이어 붙이면 연산마다 작은 반올림 오차가 생기는데, 그것들이 최종 결과에 얼마나 섞여 드는가. 답하려면 **최종 값이 중간 값 하나하나에 얼마나 민감한지** 를 전부 알아야 한다.

![위쪽 줄은 u1에서 y까지 상자 넷이 화살표로 이어져 있고 상자마다 위에서 반올림 오차가 하나씩 더해진다. 아래쪽 줄은 같은 상자들을 y에서 u1 쪽으로 거꾸로 훑으며 상자 아래에 민감도 값을 하나씩 남긴다.](figures/linnainmaa_1970.svg)

민감도를 하나씩 따로 구하면 중간 값의 수만큼 계산을 되풀이해야 한다. 리나인마가 보인 것은 **거꾸로 한 번만 훑으면 전부 한꺼번에 나온다** 는 것이었다. 이 쪽의 역방향 모드가 바로 그것이고, 식도 같다.

**이 논문에는 뉴럴 네트워크가 한 번도 나오지 않는다.** 역전파라는 이름도, 트레인이라는 말도 없다. 쓰인 자리가 전혀 달랐을 뿐 기계는 같았고, 뒤에 그 기계가 네트워크 트레인에 쓰이게 된다.

**다만 이 논문이 말해 주지 않는 것도 분명하다.** 미분을 싸게 구하는 법은 최적화 방법이 아니다. 스텝 크기(러닝레이트)를 어떻게 고를지, 언제 멈출지, 어떤 최솟값에 닿을지는 여기에 없다 — 그 질문들이 이 장의 [경사 하강법 절](../gradient_descent/steepest_direction.md)에 있다. 거꾸로, 아무리 좋은 최적화 방법도 그래디언트를 싸게 못 구하면 쓸 수 없다. 두 갈래가 따로 자라다 만난 셈이다.

이름을 한 사람에게 돌리기도 어렵다. 1962년에 로젠블랫이 이미 "back-propagating error correction"이라는 말을 썼지만 **구현할 줄은 몰랐다** 고 적혀 있고, 1950년대 최적제어의 수반 상태법(adjoint state method)이 연속 시간 판의 역전파로 꼽힌다. 같은 생각이 여러 곳에서 따로 자랐다.

_확인한 곳 — 영문 위키백과 "Seppo Linnainmaa"(논문 제목·연도·대학, 그리고 뉴럴 네트워크에 대한 언급이 없다는 서술), "Backpropagation"(로젠블랫 1962와 수반 상태법에 대한 서술)._

</div>

### 메모리와 계산의 절충

역방향 모드는 역전파를 위해 **중간 엑티베이션 값(레이어마다 나온 출력)을 저장해야** 하므로 뉴럴 네트워크 깊이에 비례하는 메모리 부담이 생긴다.

| 측면 | 순방향 모드 | 역방향 모드 |
|--------|--------------|--------------|
| **전파 방향** | 입력 → 출력 | 출력 → 입력 |
| **계산 대상** | JVP: $J \dot{x}$ | VJP: $J^T \bar{y}$ |
| **효율적인 경우** | $n \ll m$ | $m \ll n$ |
| **전체 야코비 행렬** | $n$번 통과 | $m$번 통과 |
| **추가 메모리** | 한 방향의 접선과 현재 연산 상태 | 역전파용 저장 텐서(구조에 따라 다름) |

아주 깊은 뉴럴 네트워크에서는 **엑티베이션 값 체크포인팅** 이 엑티베이션 값을 저장하는 대신 역전파 중에 다시 계산함으로써 계산을 대가로 메모리를 아낀다.

```python
# checkpoint 는 「이 구간의 중간값을 저장하지 말고 역전파 때 다시 계산하라」고
# 시키는 함수다. 위 블록에 이어서 돌린다 — torch 는 이미 불러왔다.
from torch.utils.checkpoint import checkpoint

def expensive_layer(x):
    # x @ x.T 는 (1000, 1000) × (1000, 1000) 행렬 곱이고 결과도 (1000, 1000) 이다.
    # torch.relu 는 음수를 0 으로 바꾼다.
    return torch.relu(x @ x.T)

# 난수의 시드를 고정한다. 아래에서 뽑는 텐서가 돌릴 때마다 같은 값이 된다.
torch.manual_seed(0)
x = torch.randn(1000, 1000, requires_grad=True)   # 수 백만 개 — 형태 (1000, 1000)

# 체크포인팅: 역전파 중에 엑티베이션 값을 다시 계산한다
# y 의 값은 expensive_layer(x) 를 그냥 부른 것과 같다. 다른 것은 메모리다 —
# 순전파 때 구간 안의 중간값을 버리고 입력 x 만 들고 있다가, 역전파가 이 구간에
# 닿으면 순전파를 한 번 더 해서 중간값을 다시 만든다.
# use_reentrant=False 는 두 구현 가운데 파이토치 문서가 권하는 쪽을 고른다.
y = checkpoint(expensive_layer, x, use_reentrant=False)
```

---

## 6. 계산량을 재어 본다

위의 표는 역방향 모드가 $O(T)$, 순방향 모드가 $O(n \cdot T)$라고 적는다. 파라메터가
4,225개면 4,225배 차이가 나야 한다는 말이다. 재어 보면 **60배**다.

복잡도 계산이 틀린 것이 아니라 **상수가 숨어 있다.** 아래 코드가 그것을 재어 본다.

```python
"""순방향 모드와 역방향 모드 — 복잡도 계산으로 말고 재어 본다.

본문의 표는 역방향 모드가 $O(T)$, 순방향 모드가 $O(n \\cdot T)$ 라고
적는다. 파라메터가 4,225 개면 4,225 배 차이가 나야 한다는 말이다.

재어 보면 60 배다. 복잡도 계산이 틀린 것이 아니라 **상수가 숨어 있다.**
순방향 모드도 한 번에 여러 방향을 묶어 처리하므로 통과 횟수만큼
곱해지지 않는다.

그래도 자릿수가 같고, 방향을 뒤집으면 **정확히 거꾸로** 뒤집힌다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
# time 은 시각을 재는 파이썬 표준 모듈이다.
import time

import torch
import torch.nn as nn
# torch.nn.functional 은 로스와 엑티베이션 함수를 「함수」 형태로 모아 둔 곳이다.
# 관례대로 F 라는 이름으로 불러온다.
import torch.nn.functional as F
# torch.func 는 「함수를 받아 함수를 돌려주는」 자동 미분 도구 모음이다.
#   jacrev          역방향 모드(VJP)로 야코비 행렬 전체를 구하는 함수를 만든다
#   jacfwd          순방향 모드(JVP)로 같은 야코비 행렬을 구하는 함수를 만든다
#   functional_call 모델의 파라메터를 밖에서 넘긴 것으로 바꿔 끼워 순전파한다
from torch.func import functional_call, jacfwd, jacrev


def header(title):
    # 실험마다 제목 한 줄을 찍는다. \n 은 줄바꿈이라 앞에 빈 줄이 하나 생긴다.
    print(f"\n=== {title} ===")


def timeit(fn, rep=3):
    # fn 을 불러 걸린 시간을 밀리초로 돌려준다. rep 번 재어 가장 짧은 것을
    # 고른다 — 다른 프로그램이 끼어들어 느려진 측정을 버리려는 것이다.
    fn()                                   # 예열
    # 첫 호출에는 준비 비용이 섞여 느리므로 한 번 그냥 돌리고 재지 않는다.
    ts = []                                # 잰 시간(초)을 모을 리스트
    for _ in range(rep):                   # _ 는 쓰지 않는 반복 변수라는 표시다
        t0 = time.perf_counter()           # 시작 시각(초) — 짧은 시간을 재는 시계
        fn()
        ts.append(time.perf_counter() - t0)    # 끝 시각 - 시작 시각 = 걸린 초
    return min(ts) * 1e3                   # 가장 짧은 것을 밀리초로 (초 × 1000)


# ========================================================================
# 1. 딥러닝의 형태 — 입력 많고 출력 하나
# ========================================================================

def deep_learning_shape():
    header("1. 파라메터는 많고 로스는 하나")
    torch.manual_seed(0)                   # 웨이트 초기화와 데이터의 난수를 고정한다
    # 작은 네트워크 64 → 64 → 1 이다. nn.Sequential 은 레이어를 차례대로 이어
    # 붙이고, nn.Tanh 는 엑티베이션 함수 tanh 다.
    net = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    # p.numel() 은 텐서 p 에 든 수의 개수다. 모든 파라메터에 걸쳐 더한다.
    #   (64*64 + 64) + (64*1 + 1) = 4,160 + 65 = 4,225
    n_param = sum(p.numel() for p in net.parameters())
    # 입력 8개 — 형태 (8, 64), 맞혀야 할 값 8개 — 형태 (8, 1).
    x, y = torch.randn(8, 64), torch.randn(8, 1)
    # 파라메터를 {이름: 텐서} 사전으로 꺼낸다. {k: v for k, v in ...} 는 사전을
    # 만드는 파이썬 문법이다. .detach() 로 계산 그래프와 끊긴 텐서를 담는다 —
    # 아래에서 jacrev/jacfwd 가 미분할 「입력」으로 넘길 것이기 때문이다.
    params = {k: v.detach() for k, v in net.named_parameters()}

    # 「파라메터 사전 p → 로스」 인 함수다. jacrev/jacfwd 는 이런 함수를 받는다.
    def loss_fn(p):
        # functional_call(net, p, (x,)) 은 net 의 파라메터 자리에 p 를 끼우고
        # x 를 넣어 순전파한다 — 출력 형태 (8, 1). F.mse_loss 는 평균제곱오차,
        # 곧 출력과 y 의 차를 제곱해 평균한 스칼라다.
        return F.mse_loss(functional_call(net, p, (x,)), y)

    # jacrev(loss_fn) 은 「loss_fn 의 야코비 행렬을 구하는 함수」이고, 뒤의
    # (params) 가 그 함수를 부른다. 출력이 스칼라 하나이므로 결과는 로스를
    # 파라메터 4,225개로 미분한 그래디언트다(파라메터와 이름·형태가 같은 사전).
    # lambda: ... 로 감싸 「부르면 그 계산을 하는 함수」로 만들어 timeit 에 넘긴다.
    rev = timeit(lambda: jacrev(loss_fn)(params))    # 역방향 모드 — ms
    fwd = timeit(lambda: jacfwd(loss_fn)(params))    # 순방향 모드 — 같은 값, 다른 길
    # {n_param:,} 의 쉼표는 천 단위마다 쉼표를 찍으라는 뜻이다(4,225).
    print(f"  파라메터 {n_param:,}개, 로스는 스칼라 하나")
    # :8.2f 는 8칸에 맞춰 소수 둘째 자리까지 찍는다.
    print(f"  jacrev (역방향) {rev:8.2f} ms")
    print(f"  jacfwd (순방향) {fwd:8.2f} ms")
    # fwd / rev 가 「순방향이 몇 배 느린가」다. :.0f 는 소수점 아래 없이 찍는다.
    # 시간은 기계마다 다르므로 이 배수도 돌릴 때마다 조금씩 달라진다.
    print(f"  순방향이 {fwd / rev:.0f} 배 느리다")
    print(f"  표의 계산으로는 {n_param:,} 배여야 하는데 {fwd / rev:.0f} 배다")
    print("  순방향 모드도 여러 방향을 묶어 처리하므로 통과 횟수만큼 안 곱해진다")


# ========================================================================
# 2. 거꾸로 — 입력 하나에 출력 많기
# ========================================================================

def the_other_shape():
    header("2. 거꾸로 뒤집으면")
    torch.manual_seed(0)
    A = torch.randn(2048, 1)               # 형태 (2048, 1) — 열이 하나인 행렬

    # 입력이 수 하나, 출력이 2048개인 함수다.   h(t) = tanh(A t)
    def h(t):
        # t.view(1, 1)   형태 (1,) 인 t 를 (1, 1) 행렬로 다시 읽는다
        # A @ ...        (2048, 1) × (1, 1) → (2048, 1)
        # .squeeze(1)    길이가 1 인 1번 축을 없앤다 → (2048,)
        # .tanh()        원소마다 tanh 를 취한다
        return (A @ t.view(1, 1)).squeeze(1).tanh()

    xin = torch.randn(1)                   # 입력 — 수 하나, 형태 (1,)
    # 야코비 행렬의 형태는 (출력 수, 입력 수) = (2048, 1) 이다. 역방향 모드의 일은
    # 출력 수(2048)를 따라 늘고 순방향 모드의 일은 입력 수(1)를 따라 는다.
    rev = timeit(lambda: jacrev(h)(xin))
    fwd = timeit(lambda: jacfwd(h)(xin))
    print(f"  입력 1 개 -> 출력 2048 개")
    print(f"  jacrev (역방향) {rev:8.2f} ms")
    print(f"  jacfwd (순방향) {fwd:8.2f} ms")
    # 이번에는 나누는 차례가 거꾸로다 — rev / fwd 는 「순방향이 몇 배 빠른가」다.
    print(f"  이번에는 순방향이 {rev / fwd:.0f} 배 빠르다")
    print("  고르는 기준은 라이브러리가 아니라 함수의 모양이다")


# ========================================================================
# 3. 역방향 모드가 치르는 비용 — 엑티베이션 값을 들고 있어야 한다
# ========================================================================

def the_memory_cost():
    header("3. 역방향 모드가 치르는 비용")
    torch.manual_seed(0)
    # (Linear(256, 256) → ReLU) 묶음 여덟 개를 이어 붙인 네트워크다.
    # [... for _ in range(8)] 가 묶음 8개의 리스트를 만들고, 앞의 * 가 그 리스트를
    # 풀어 인수 8개로 넘긴다.
    net = nn.Sequential(*[nn.Sequential(nn.Linear(256, 256), nn.ReLU())
                          for _ in range(8)])
    x = torch.randn(64, 256)               # 입력 64개 — 형태 (64, 256)
    # no_grad() 안에서는 계산을 기록하지 않는다. 중간값을 들고 있지 않고 결과에
    # grad_fn 이 붙지 않는다.
    with torch.no_grad():
        no_graph = net(x)
    # 그냥 돌리면 net 의 파라메터가 경사 추적 중이므로 계산 그래프가 세워지고,
    # 역전파에 쓸 중간값이 메모리에 남는다.
    with_graph = net(x)
    print("  no_grad 로 돌린 결과 grad_fn :", no_graph.grad_fn)
    # 마지막 연산이 ReLU 였으므로 ReluBackward0 이 찍힌다.
    print("  그냥 돌린 결과 grad_fn       :",
          type(with_graph.grad_fn).__name__)
    # 저장되는 엑티베이션 값의 크기를 어림한다 — 메모리를 잰 것이 아니라 센 것이다.
    #   배치 64 × 폭 256 × 수 하나에 4바이트(float32) × 레이어 8
    #   × 레이어마다 텐서 둘(Linear 의 출력과 ReLU 의 출력)
    #   = 1,048,576 바이트.  1e6 으로 나눠 MB 로 적으면 1.05 MB
    act = 64 * 256 * 4 * 8 * 2 / 1e6
    # 파라메터의 크기는 개수 × 4바이트다.
    #   8 × (256*256 + 256) = 526,336개  →  2,105,344 바이트 = 2.11 MB
    par = sum(p.numel() * 4 for p in net.parameters()) / 1e6
    print(f"  파라메터 {par:.2f} MB   역전파를 위해 저장되는 엑티베이션 값 대략 {act:.2f} MB")
    print("  순방향 모드는 엑티베이션 값을 안 들고 있어도 된다 — 앞으로만 가니까")
    print("  역방향 모드의 속도는 공짜가 아니라 메모리와 바꾼 것이다")


# ========================================================================
# 4. 야코비 행렬을 모으는 두 방향
# ========================================================================

def two_directions():
    header("4. 야코비안을 행으로 모으기 vs 열로 모으기")

    # 입력 둘 (x, y) = (t[0], t[1]), 출력 셋 (x², xy, y²) 인 함수다.
    def f(t):
        # torch.stack 이 스칼라 셋을 쌓아 형태 (3,) 인 벡터로 만든다.
        return torch.stack([t[0] ** 2, t[0] * t[1], t[1] ** 2])

    x = torch.tensor([1.0, 2.0])           # 미분할 자리 (x, y) = (1, 2)
    print("  f: R^2 -> R^3,  야코비안은 3x2")
    print("  jacrev (역방향 — 출력 3 개의 행을 모은다):")
    # jacrev(f)(x) 는 형태 (3, 2) 인 야코비 행렬이다 — 행이 출력, 열이 입력.
    # .tolist() 로 행 셋의 리스트로 바꿔 한 행씩 찍는다. round(v, 4) 는 소수
    # 넷째 자리로 반올림한다.
    for r in jacrev(f)(x).tolist():
        print("    ", [round(v, 4) for v in r])
    print("  jacfwd (순방향 — 입력 2 개의 열을 모은다):")
    # 같은 행렬을 순방향 모드로 구한다.
    for r in jacfwd(f)(x).tolist():
        print("    ", [round(v, 4) for v in r])
    # torch.allclose 는 두 텐서의 모든 원소가 부동소수점 오차 안에서 같으면 True 다.
    print("  같은가:", torch.allclose(jacrev(f)(x), jacfwd(f)(x)))
    # 행마다 (x 로 미분, y 로 미분) 이다.   x² → (2x, 0),  xy → (y, x),  y² → (0, 2y)
    print("  손으로: [[2x, 0], [y, x], [0, 2y]] = [[2,0],[2,1],[0,4]]")
    print("  출력 3 > 입력 2 이므로 여기서는 순방향이 한 번 덜 돈다")


def main():
    # 네 실험을 차례로 돌린다.
    deep_learning_shape()
    the_other_shape()
    the_memory_cost()
    two_directions()


# 이 파일을 직접 돌렸을 때만 main() 을 부른다. 다른 파일이 import 로 불러올 때는
# __name__ 이 "__main__" 이 아니어서 돌지 않는다.
if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 파라메터는 많고 로스는 하나 ===
  파라메터 4,225개, 로스는 스칼라 하나
  jacrev (역방향)     0.24 ms
  jacfwd (순방향)    14.62 ms
  순방향이 60 배 느리다
  표의 계산으로는 4,225 배여야 하는데 60 배다
  순방향 모드도 여러 방향을 묶어 처리하므로 통과 횟수만큼 안 곱해진다

=== 2. 거꾸로 뒤집으면 ===
  입력 1 개 -> 출력 2048 개
  jacrev (역방향)     5.42 ms
  jacfwd (순방향)     0.11 ms
  이번에는 순방향이 48 배 빠르다
  고르는 기준은 라이브러리가 아니라 함수의 모양이다

=== 3. 역방향 모드가 치르는 비용 ===
  no_grad 로 돌린 결과 grad_fn : None
  그냥 돌린 결과 grad_fn       : ReluBackward0
  파라메터 2.11 MB   역전파를 위해 저장되는 엑티베이션 값 대략 1.05 MB
  순방향 모드는 엑티베이션 값을 안 들고 있어도 된다 — 앞으로만 가니까
  역방향 모드의 속도는 공짜가 아니라 메모리와 바꾼 것이다

=== 4. 야코비안을 행으로 모으기 vs 열로 모으기 ===
  f: R^2 -> R^3,  야코비안은 3x2
  jacrev (역방향 — 출력 3 개의 행을 모은다):
     [2.0, 0.0]
     [2.0, 1.0]
     [0.0, 4.0]
  jacfwd (순방향 — 입력 2 개의 열을 모은다):
     [2.0, 0.0]
     [2.0, 1.0]
     [0.0, 4.0]
  같은가: True
  손으로: [[2x, 0], [y, x], [0, 2y]] = [[2,0],[2,1],[0,4]]
  출력 3 > 입력 2 이므로 여기서는 순방향이 한 번 덜 돈다
```

### 읽기

```
파라메터 4,225개, 로스는 스칼라 하나
jacrev (역방향)     0.24 ms
jacfwd (순방향)    14.62 ms      순방향이 60 배 느리다

입력 1 개 -> 출력 2048 개
jacrev (역방향)     5.42 ms
jacfwd (순방향)     0.11 ms      순방향이 48 배 빠르다
```

![두 줄 그림. 윗줄은 입력이 많고 출력이 하나인 함수에서 역방향이 60배 빠르다는 것을, 아랫줄은 입력이 하나이고 출력이 많은 함수에서 순방향이 48배 빠르다는 것을 보인다.](figures/which_mode_wins.svg)

**방향이 정확히 뒤집힌다.** 같은 순방향 모드가 함수의 모양에 따라 60배 손해를 보기도
하고 48배 이득을 보기도 한다.

$n$배가 아니라 60배인 까닭은 순방향 모드가 **여러 방향을 묶어** 처리하기 때문이다.
통과 횟수가 $n$이어도 각 통과가 통째로 되풀이되는 것은 아니다. 그래도 자릿수의
이야기는 맞는다 — 파라메터가 수백만 개인 실제 모델에서는 순방향 모드로 트레인하는
것이 불가능하다.

### 역방향 모드가 치르는 비용

```
파라메터 2.11 MB   역전파를 위해 저장되는 엑티베이션 값 대략 1.05 MB
```

역방향 모드가 빠른 것은 공짜가 아니다. **역전파가 쓸 중간값을 들고 있어야** 한다.
순방향 모드는 앞으로만 가므로 그럴 필요가 없다.

「경사 추적을 언제 멈출 것인가」 쪽에서 `no_grad`가 아끼는 메모리가 바로 이것이고,
위 출력의 `grad_fn: None`이 그 표시다.

---

## 7. 전체 야코비 행렬 계산하기

### `torch.autograd.functional.jacobian` 사용하기

```python
import torch
# jacobian(함수, 입력) 은 야코비 행렬 전체를 한 번에 돌려준다.
from torch.autograd.functional import jacobian

def f(x):
    """f: R³ → R²"""
    # torch.stack 이 스칼라 둘을 쌓아 형태 (2,) 인 벡터로 만든다.
    return torch.stack([
        x[0] * x[1] + x[2],            # f₁ = x₁x₂ + x₃
        torch.sin(x[0]) + x[1]**2      # f₂ = sin(x₁) + x₂²
    ])

x = torch.tensor([1.0, 2.0, 3.0])      # 미분할 자리 (x₁, x₂, x₃) = (1, 2, 3)
# 형태는 (출력 수, 입력 수) = (2, 3) 이다. 행 하나가 출력 하나의 그래디언트다.
#   J = [[x₂, x₁, 1], [cos(x₁), 2x₂, 0]] = [[2, 1, 1], [0.5403, 4, 0]]
J = jacobian(f, x)

print(f"Input dim:  {x.shape[0]}")     # x.shape[0] 은 0번 축의 길이 — 3
print(f"Output dim: {f(x).shape[0]}")  # f(x) 의 길이 — 2
print(f"Jacobian shape: {J.shape}")
print(f"Jacobian:\n{J}")               # \n 은 줄바꿈이다
```

**출력:**

```
Input dim:  3
Output dim: 2
Jacobian shape: torch.Size([2, 3])
Jacobian:
tensor([[2.0000, 1.0000, 1.0000],
        [0.5403, 4.0000, 0.0000]])
```

### VJP로 야코비 행렬 직접 구하기(행 단위)

원핫 수반 벡터 $e_i$로 역전파를 하면 야코비 행렬의 $i$번 행을 뽑아낼 수 있다.

```python
import torch

def compute_jacobian_via_vjp(f, x):
    """역방향 모드 벡터-야코비 곱으로 야코비 행렬을 한 행씩 구한다."""
    # 먼저 f 를 한 번 불러 출력이 몇 개인지 알아낸다 — 이 y 는 개수를 세는 데만
    # 쓴다. x.detach().requires_grad_(True) 는 x 와 값이 같고 경사 추적이 켜진
    # 새 리프 텐서를 만든다(넘겨받은 x 의 플래그는 건드리지 않는다).
    y = f(x.detach().requires_grad_(True))
    m, n = y.numel(), x.numel()            # m = 출력 수, n = 입력 수 (아래에서는 3, 2)
    J = torch.zeros(m, n)                  # 0 으로 채운 (m, n) 행렬 — 한 행씩 채운다

    for i in range(m):                     # 출력마다 한 번, 모두 m 번 역전파한다
        # 행마다 새 리프 텐서로 순전파를 다시 한다. 까닭이 둘이다.
        #   - backward() 가 끝나면 계산 그래프가 지워져 같은 그래프로 또
        #     역전파할 수 없다
        #   - .grad 는 더해져 쌓이므로 같은 텐서를 쓰면 앞 행의 값이 섞인다
        x_copy = x.detach().requires_grad_(True)
        y_copy = f(x_copy)

        # 원핫 벡터 e_i 를 만든다 — i번 칸만 1 이고 나머지는 0 이다. 「i번 출력만
        # 본다」는 수반 벡터다.
        v = torch.zeros(m)
        v[i] = 1.0

        # VJP J^T e_i 는 J 의 i번 행이다. 그것이 x_copy.grad 에 들어온다 — 형태 (n,).
        y_copy.backward(v)
        J[i] = x_copy.grad                 # J 의 i번 행에 적어 넣는다

    return J

# 입력 둘, 출력 셋인 함수다.   f(x) = (x₁², x₁x₂, x₂²)
def f(x):
    return torch.stack([x[0]**2, x[0]*x[1], x[1]**2])

x = torch.tensor([2.0, 3.0])               # 미분할 자리 (x₁, x₂) = (2, 3)
J = compute_jacobian_via_vjp(f, x)         # 형태 (3, 2)
print(f"Jacobian:\n{J}")
# 손으로 구한 값이다. 행마다 (x₁ 로 미분, x₂ 로 미분) 이다.
# [[4, 0],    d(x₁²)/d(x₁, x₂)  = (2x₁, 0)
#  [3, 2],    d(x₁x₂)/d(x₁, x₂) = (x₂, x₁)
#  [0, 6]]    d(x₂²)/d(x₁, x₂)  = (0, 2x₂)
```

**출력:**

```
Jacobian:
tensor([[4., 0.],
        [3., 2.],
        [0., 6.]])
```

### JVP로 야코비 행렬 직접 구하기(열 단위)

원핫 접벡터 $e_j$로 순전파를 하면 $j$번 열을 뽑아낼 수 있다.

```python
import torch
from torch.func import jvp                 # 순방향 모드 JVP (5절과 같은 것)

# 5절의 JVP 보기와 같은 함수다.   f(x) = [sin(x₁x₂), x₁² + x₂]
def f(x):
    return torch.stack([
        torch.sin(x[0] * x[1]),
        x[0]**2 + x[1]
    ])

x = torch.tensor([1.0, 2.0])               # 미분할 자리 (1, 2)

# jvp 는 (함수값, J @ 접벡터) 를 돌려주므로 [1] 로 둘째 것만 꺼낸다.
# 접벡터 [1, 0] 은 J 의 0번 열 [-0.8323, 2] 를, [0, 1] 은 1번 열 [-0.4161, 1] 을 준다.
J_col0 = jvp(f, (x,), (torch.tensor([1.0, 0.0]),))[1]
J_col1 = jvp(f, (x,), (torch.tensor([0.0, 1.0]),))[1]
# 두 열을 나란히 세워 행렬로 만든다 — 형태 (2, 2). dim=1 은 「1번 축(열 방향)
# 으로 쌓으라」는 뜻이다. dim=0 으로 쌓으면 두 벡터가 행으로 들어가 전치된
# 행렬이 된다.
J_forward = torch.stack([J_col0, J_col1], dim=1)

print(f"Full Jacobian (forward mode):\n{J_forward}")
```

**출력:**

```
Full Jacobian (forward mode):
tensor([[-0.8323, -0.4161],
        [ 2.0000,  1.0000]])
```

---

## 8. 주요 속성과 메서드

| 속성 / 메서드 | 설명 |
|--------------------|-------------|
| `x.requires_grad` | `x`에 대한 연산이 미분을 위해 추적되는지 여부 |
| `x.grad` | 누적된 그래디언트(`.backward()` 후에 채워진다) |
| `x.grad_fn` | `x`를 만든 역전파 함수(리프 텐서에서는 `None`) |
| `x.is_leaf` | `x`가 리프 텐서이면 `True` |
| `x.backward(v)` | VJP를 계산하여 `.grad`에 누적한다 |
| `x.retain_grad()` | 리프가 아닌 `x`의 그래디언트를 저장하도록 요청한다 |
| `x.detach()` | 데이터는 공유하되 그래프에서 분리된 텐서를 반환한다 |
| `torch.autograd.grad(y, x)` | `.grad`를 채우지 않고 그래디언트를 계산한다 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$f(x_1, x_2) = (x_1^2, x_1 x_2, x_2^2)$의 $3 \times 2$ 야코비 행렬을 손으로 적고
`jacrev`와 `jacfwd`로 각각 확인하라. 둘이 같은가?

</div>

??? success "풀이"
    $$
    J = \begin{bmatrix}
    2x_1 & 0 \\ x_2 & x_1 \\ 0 & 2x_2
    \end{bmatrix}
    \;\xrightarrow{(1,2)}\;
    \begin{bmatrix} 2 & 0 \\ 2 & 1 \\ 0 & 4 \end{bmatrix}
    $$

    ```python
    # 앞의 코드에 이어서 돌린다 — torch 는 이미 불러왔다.
    from torch.func import jacrev, jacfwd

    # 입력 둘, 출력 셋인 함수다.   f(x₁, x₂) = (x₁², x₁x₂, x₂²)
    def f(t):
        return torch.stack([t[0] ** 2, t[0] * t[1], t[1] ** 2])

    x = torch.tensor([1.0, 2.0])               # 점 (1, 2)
    # jacrev(f) 와 jacfwd(f) 는 둘 다 「야코비 행렬을 구하는 함수」다. (x) 로 불러
    # 형태 (3, 2) 인 행렬을 받고 .tolist() 로 파이썬 리스트로 바꿔 찍는다.
    print("  jacrev:", jacrev(f)(x).tolist())  # 역방향 모드
    print("  jacfwd:", jacfwd(f)(x).tolist())  # 순방향 모드
    # 두 행렬의 모든 원소가 부동소수점 오차 안에서 같으면 True 다.
    print("  같은가:", torch.allclose(jacrev(f)(x), jacfwd(f)(x)))
    ```

    ```
      jacrev: [[2.0, 0.0], [2.0, 1.0], [0.0, 4.0]]
      jacfwd: [[2.0, 0.0], [2.0, 1.0], [0.0, 4.0]]
      같은가: True
    ```

    **값은 같고 구하는 길만 다르다.** `jacrev`는 출력 3개에 대해 VJP를 세 번,
    `jacfwd`는 입력 2개에 대해 JVP를 두 번 부른다.

    출력(3)이 입력(2)보다 많으니 **여기서는 순방향이 한 번 덜 돈다.** 6절에서
    본 규칙 그대로다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
점 $(1, 2)$에서 $f(x_1, x_2) = (x_1^2 + x_2,\ x_1 x_2^3)$의 야코비 행렬을 구하고,
$\bar{y} = (1, 1)$에 대한 VJP를 **손으로 계산한 뒤** 확인하라.

</div>

??? success "풀이"
    $$
    J = \begin{bmatrix} 2x_1 & 1 \\ x_2^3 & 3x_1 x_2^2 \end{bmatrix}
    \;\xrightarrow{(1,2)}\;
    \begin{bmatrix} 2 & 1 \\ 8 & 12 \end{bmatrix},
    \qquad
    J^\top \bar{y} = \begin{bmatrix} 2 & 8 \\ 1 & 12 \end{bmatrix}
    \begin{bmatrix} 1 \\ 1 \end{bmatrix}
    = \begin{bmatrix} 10 \\ 13 \end{bmatrix}
    $$

    ```python
    # 연습문제 1의 코드에 이어서 돌린다 — torch 와 jacrev 를 그대로 쓴다.
    # f(x₁, x₂) = (x₁² + x₂,  x₁ x₂³)
    def f(t):
        return torch.stack([t[0] ** 2 + t[1], t[0] * t[1] ** 3])

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = f(x)                                   # [1 + 2, 1 * 8] = [3, 8]
    # ȳ = (1, 1) 을 넘겨 VJP J^T ȳ 를 x.grad 에 받는다.   [2 + 8, 1 + 12] = [10, 13]
    y.backward(torch.tensor([1.0, 1.0]))
    # 야코비 행렬은 따로 jacrev 로 구한다. x.detach() 로 계산 그래프와 끊긴 텐서를
    # 넘긴다.
    print("  야코비:", jacrev(f)(x.detach()).tolist())
    print("  VJP   :", x.grad.tolist())
    ```

    ```
      야코비: [[2.0, 1.0], [8.0, 12.0]]
      VJP   : [10.0, 13.0]
    ```

    손으로 계산한 것과 맞는다.

    VJP가 $J^\top \bar{y}$이지 $J \bar{y}$가 아니라는 것을 형태로도 확인할 수
    있다. $J$가 $2 \times 2$라 여기서는 둘 다 되지만, $J$가 $3 \times 2$였다면
    $J\bar{y}$는 형태가 안 맞는다. 「선형 변환에서의 VJP」에서 본 그 점검이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
6절에서 순방향 모드가 60배 느렸다. **파라메터 수를 바꿔 가며** 그 배수가 어떻게
변하는지 재고, 이론의 $O(n)$과 견주어라.

</div>

??? success "풀이"
    ```python
    # 6절의 코드에 이어서 돌린다 — torch, nn, F 를 그대로 쓴다.
    import time
    from torch.func import functional_call, jacfwd, jacrev

    # 6절의 timeit 과 같다 — 예열로 한 번 돌린 뒤 rep 번 재어 가장 짧은 시간을
    # 밀리초로 돌려준다.
    def timeit(fn, rep=3):
        fn()
        ts = []
        for _ in range(rep):
            # 세미콜론으로 세 문장을 한 줄에 적었다 — 시작 시각, 실행, 걸린 초 기록.
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3

    # 표의 머리 줄이다. {'폭':>5} 는 5칸에 오른쪽으로 붙여 찍으라는 뜻이다.
    print(f"  {'폭':>5} {'파라메터':>10} {'역방향 ms':>9} {'순방향 ms':>9} {'배수':>7}")
    for width in (8, 16, 32, 64):              # 네트워크의 폭을 두 배씩 키운다
        torch.manual_seed(0)                   # 폭마다 같은 시드에서 시작한다
        # width → width → 1 인 네트워크다. 폭 64 가 6절의 네트워크와 같다.
        net = nn.Sequential(nn.Linear(width, width), nn.Tanh(),
                            nn.Linear(width, 1))
        # 파라메터 수 = (width² + width) + (width + 1).   폭 8 이면 72 + 9 = 81
        n = sum(p.numel() for p in net.parameters())
        # 입력 — 형태 (8, width), 맞혀야 할 값 — 형태 (8, 1).
        x, y = torch.randn(8, width), torch.randn(8, 1)
        # {이름: 텐서} 사전 — jacrev/jacfwd 가 미분할 입력이다.
        params = {k: v.detach() for k, v in net.named_parameters()}

        def loss_fn(p):
            # p 를 파라메터로 끼워 순전파하고 평균제곱오차를 돌려준다.
            return F.mse_loss(functional_call(net, p, (x,)), y)

        r = timeit(lambda: jacrev(loss_fn)(params))    # 역방향 모드 — ms
        f = timeit(lambda: jacfwd(loss_fn)(params))    # 순방향 모드 — ms
        # 마지막 칸 f / r 이 「순방향이 몇 배 느린가」다.
        print(f"  {width:>5} {n:>10,} {r:>9.2f} {f:>9.2f} {f / r:>7.1f}")
    ```

    ```
         폭     파라메터   역방향 ms   순방향 ms      배수
          8         81      0.23      0.33     1.4
         16        289      0.22      0.57     2.6
         32      1,089      0.22      1.78     8.2
         64      4,225      0.24     14.72    61.8
    ```

    **역방향 쪽은 거의 일정하고(0.22~0.24 ms) 순방향 쪽만 자란다.** 그것이 표가
    말하는 바다 — 역방향은 $O(T)$, 순방향은 $O(n \cdot T)$.

    배수를 보면 1.4 → 2.6 → 8.2 → 61.8이다. 파라메터가 네 배가 될 때마다
    대략 두 배, 세 배, 일곱 배씩 는다. **$n$에 비례하지는 않지만 $n$과 함께
    자라고, 커질수록 가팔라진다.**

    $n$에 못 미치는 까닭은 6절에서 적은 대로다. 그리고 작은 모델에서는 고정
    비용이 섞여 있어 배수가 더 눌린다 — 폭 8에서는 1.4배뿐이다.

    실제 모델로 외삽하면 뜻이 분명해진다. 파라메터가 백만 개면 이 추세로도
    수천 배이고, 그것이 **순방향 모드로는 트레인이 불가능하다**는 말의 내용이다.
    순방향 모드가 쓰이는 자리는 입력이 적은 민감도 분석이나, 역방향 모드와 섞어
    쓰는 고차 미분이다.

## 정리하며

| 개념 | 핵심 |
|---------|------------|
| **그래디언트** | 스칼라 $L$에 대해 역방향 모드 자동 미분으로 계산한 $\nabla_x L$ |
| **VJP** | $J^T \bar{y}$ — PyTorch가 역전파 중에 계산하는 것 |
| **JVP** | $J \dot{x}$ — 순방향 모드. 입력이 적을 때 효율적이다 |
| **역방향 모드** | 스칼라 로스에 대해 역전파 한 번. 비용 $O(T)$ |
| **순방향 모드** | 전체 야코비 행렬에 방향별 $n$번 통과. 현재 값과 접선의 메모리는 필요 |
| **전체 야코비 행렬** | 역전파 $m$번(역방향) 또는 순전파 $n$번 |
| **그래디언트 저장** | 기본적으로 리프 텐서만 `.grad`를 저장한다 |

**참고 문헌**

- Baydin, A.G., et al. (2018). Automatic Differentiation in Machine Learning: A Survey. *JMLR*.
- Griewank, A. & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*.
- PyTorch Autograd Documentation: [https://pytorch.org/docs/stable/autograd.html](https://pytorch.org/docs/stable/autograd.html)
