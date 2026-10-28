# 벡터 역전파의 함정 — 터지는 것과 조용한 것

「벡터 출력의 역전파」부터 여기까지 `backward(v)`를 여러 각도로 보았다. 이 쪽은
**틀렸을 때 어떻게 드러나는지**로 모아 정리한다.

| | 어떻게 드러나는가 |
|---|---|
| `v`를 아예 안 주면 | **터진다** |
| `v`의 모양이 다르면 | **터진다** |
| 여러 번 부르며 안 비우면 | **조용히 쌓인다** |
| `v`에 `requires_grad`를 켜면 | **조용히 무시된다** |
| `v`의 자료형이 다르면 | **조용히 맞춰 준다** |

위의 둘은 메시지만 읽으면 고쳐진다. 아래 셋이 이 쪽에서 볼 것이다.

## 1. 코드

```python
"""벡터 역전파의 함정 — 터지는 것과 조용한 것.

「벡터 출력의 역전파」부터 여기까지 `backward(v)` 를 여러 각도로
보았다. 이 쪽은 **틀렸을 때 어떻게 드러나는지**로 모아 정리한다.

터져 주는 쪽
- v 를 아예 안 주면
- v 의 모양이 출력과 다르면

조용한 쪽
- v 를 비우지 않고 여러 번 부르면 (경사가 쌓인다)
- v 에 requires_grad 를 켜면 (아무 일도 안 일어난다)
- v 의 자료형이 다르면 (말없이 맞춰 준다)

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 터져 주는 둘
# ========================================================================

def the_loud_ones():
    header("1. 터져 주는 둘")
    x = torch.randn(3, requires_grad=True)
    try:
        (x ** 2).backward()
    except RuntimeError as e:
        print("  v 를 안 주면 :", str(e).splitlines()[0])
    for shape in ((4,), (3, 1), ()):
        xx = torch.randn(3, requires_grad=True)
        try:
            (xx ** 2).backward(torch.ones(shape))
            print(f"  v 모양 {str(shape):6s} -> 통과")
        except RuntimeError as e:
            print(f"  v 모양 {str(shape):6s} -> {str(e).splitlines()[0][:50]}")
    print("  둘 다 고치기 쉽다. 메시지가 무엇이 어긋났는지 적어 준다")


# ========================================================================
# 2. 조용한 것 하나 — 쌓인다
# ========================================================================

def accumulates():
    header("2. 조용한 것 — 쌓인다")
    torch.manual_seed(0)
    A = torch.randn(3, 3)
    for clear in (True, False):
        x = torch.randn(3, requires_grad=True)
        y = A @ x
        rows = []
        for i in range(3):
            e = torch.zeros(3)
            e[i] = 1.0
            if clear:
                x.grad = None
            y.backward(e, retain_graph=True)
            rows.append(x.grad.clone())
        J = torch.stack(rows)
        print(f"  비움={clear!s:5s} J == A 인가: {torch.allclose(J, A)}")
        if not clear:
            print("    첫 줄은 맞다:", torch.allclose(J[0], A[0]))
            print("    둘째 줄 = A[0]+A[1] 인가:",
                  torch.allclose(J[1], A[0] + A[1]))
    print("  모양도 맞고 첫 줄도 맞아서 검산이 쉽게 속는다")


# ========================================================================
# 3. 조용한 것 둘 — v 의 requires_grad
# ========================================================================

def v_requires_grad():
    header("3. 조용한 것 — v 에 깃발을 켜면")
    x = torch.randn(3, requires_grad=True)
    v = torch.ones(3, requires_grad=True)
    (x ** 2).backward(v)
    print("  통과한다. x.grad :", [round(t, 4) for t in x.grad.tolist()])
    print("  그런데 v.grad    :", v.grad)
    print("  v 는 '가중치' 일 뿐 미분 대상이 아니다. 깃발은 무시된다")
    print("  v 에 대한 민감도가 필요하면 식을 다시 세워야 한다")
    x2 = torch.randn(3, requires_grad=True)
    v2 = torch.ones(3, requires_grad=True)
    (v2 * x2 ** 2).sum().backward()
    print("  (v*y).sum() 으로 적으면 v.grad 도 온다:",
          [round(t, 4) for t in v2.grad.tolist()])


# ========================================================================
# 4. 조용한 것 셋 — 자료형
# ========================================================================

def dtype_of_v():
    header("4. 조용한 것 — v 의 자료형")
    for dt in (torch.float32, torch.float64, torch.float16, torch.int64):
        x = torch.randn(3, requires_grad=True)
        (x ** 2).backward(torch.ones(3, dtype=dt))
        print(f"  v dtype {str(dt):16s} -> 통과, x.grad dtype {x.grad.dtype}")
    print("  말없이 맞춰 준다. 정수 v 도 받는다")
    print("  0.5 같은 가중치를 정수 텐서에 담으면 0 으로 잘린다")
    x = torch.randn(3, requires_grad=True)
    (x ** 2).backward(torch.tensor([1, 0, 1]))          # 정수
    print("  정수 [1,0,1] 로 준 x.grad:", [round(t, 4) for t in x.grad.tolist()])


# ========================================================================
# 5. 쌓임을 아예 피하는 길
# ========================================================================

def use_autograd_grad():
    header("5. autograd.grad 로 피하기")
    torch.manual_seed(0)
    A = torch.randn(3, 3)
    x = torch.randn(3, requires_grad=True)
    y = A @ x
    rows = []
    for i in range(3):
        e = torch.zeros(3)
        e[i] = 1.0
        g, = torch.autograd.grad(y, x, e, retain_graph=True)
        rows.append(g)
    J = torch.stack(rows)
    print("  autograd.grad 로 모은 J == A :", torch.allclose(J, A))
    print("  x.grad 는 그대로 :", x.grad)
    print("  .grad 를 건드리지 않으니 비울 일도, 쌓일 일도 없다")
    print("  야코비를 모을 때는 이쪽이 안전하다")


# ========================================================================
# 6. v 의 부호
# ========================================================================

def sign_of_v():
    header("6. v 의 부호")
    for sign, label in ((1.0, "v=+1 (경사 하강)"), (-1.0, "v=-1 (경사 상승)")):
        x = torch.tensor([3.0], requires_grad=True)
        opt = torch.optim.SGD([x], lr=0.1)
        for _ in range(5):
            opt.zero_grad()
            (x ** 2).backward(torch.tensor([sign]))
            opt.step()
        print(f"  {label}  3.0 -> {x.item():.4f}")
    print("  부호를 뒤집으면 최솟값에서 멀어진다")
    print("  GAN 처럼 올리고 싶은 로스가 있을 때 쓰는 길이기도 하다")


def main():
    the_loud_ones()
    accumulates()
    v_requires_grad()
    dtype_of_v()
    use_autograd_grad()
    sign_of_v()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 터져 주는 둘 ===
  v 를 안 주면 : grad can be implicitly created only for scalar outputs
  v 모양 (4,)   -> Mismatch in shape: grad_output[0] has a shape of t
  v 모양 (3, 1) -> Mismatch in shape: grad_output[0] has a shape of t
  v 모양 ()     -> Mismatch in shape: grad_output[0] has a shape of t
  둘 다 고치기 쉽다. 메시지가 무엇이 어긋났는지 적어 준다

=== 2. 조용한 것 — 쌓인다 ===
  비움=True  J == A 인가: True
  비움=False J == A 인가: False
    첫 줄은 맞다: True
    둘째 줄 = A[0]+A[1] 인가: True
  모양도 맞고 첫 줄도 맞아서 검산이 쉽게 속는다

=== 3. 조용한 것 — v 에 깃발을 켜면 ===
  통과한다. x.grad : [0.2454, -1.1326, 0.7462]
  그런데 v.grad    : None
  v 는 '가중치' 일 뿐 미분 대상이 아니다. 깃발은 무시된다
  v 에 대한 민감도가 필요하면 식을 다시 세워야 한다
  (v*y).sum() 으로 적으면 v.grad 도 온다: [0.7957, 2.2774, 0.1372]

=== 4. 조용한 것 — v 의 자료형 ===
  v dtype torch.float32    -> 통과, x.grad dtype torch.float32
  v dtype torch.float64    -> 통과, x.grad dtype torch.float32
  v dtype torch.float16    -> 통과, x.grad dtype torch.float32
  v dtype torch.int64      -> 통과, x.grad dtype torch.float32
  말없이 맞춰 준다. 정수 v 도 받는다
  0.5 같은 가중치를 정수 텐서에 담으면 0 으로 잘린다
  정수 [1,0,1] 로 준 x.grad: [1.0855, -0.0, 0.518]

=== 5. autograd.grad 로 피하기 ===
  autograd.grad 로 모은 J == A : True
  x.grad 는 그대로 : None
  .grad 를 건드리지 않으니 비울 일도, 쌓일 일도 없다
  야코비를 모을 때는 이쪽이 안전하다

=== 6. v 의 부호 ===
  v=+1 (경사 하강)  3.0 -> 0.9830
  v=-1 (경사 상승)  3.0 -> 7.4650
  부호를 뒤집으면 최솟값에서 멀어진다
  GAN 처럼 올리고 싶은 로스가 있을 때 쓰는 길이기도 하다
```

## 2. 터져 주는 둘

```
v 를 안 주면 : grad can be implicitly created only for scalar outputs
v 모양 (4,)   -> Mismatch in shape
v 모양 (3, 1) -> Mismatch in shape
v 모양 ()     -> Mismatch in shape
```

출력이 `(3,)`인데 `(4,)`, `(3,1)`, `()`가 전부 거절당한다. 「열벡터로 적으면」에서
본 대로 **브로드캐스팅되지 않는다.**

고마운 자리다. 고칠 것이 무엇인지 메시지가 적어 준다.

## 3. 조용한 것 하나 — 쌓인다

```
비움=True  J == A 인가: True
비움=False J == A 인가: False
  첫 줄은 맞다: True
  둘째 줄 = A[0]+A[1] 인가: True
```

야코비를 모을 때 `x.grad`를 안 비우면 아래 줄로 갈수록 누적된다.

**모양이 맞고 첫 줄도 맞는다**는 것이 고약하다. 눈으로 확인하려고 첫 줄을
찍어 보면 정상으로 보인다. 틀린 것은 둘째 줄부터다.

## 4. 조용한 것 둘 — `v`의 `requires_grad`

```
통과한다. x.grad : [0.2454, -1.1326, 0.7462]
그런데 v.grad    : None
```

`v`에 `requires_grad=True`를 켜도 **아무 일도 일어나지 않는다.** 오류도 없고
`v.grad`도 안 온다.

까닭은 `v`가 미분의 **대상**이 아니라 **가중치**이기 때문이다. `backward(v)`는
"이 가중치로 섞어서 뒤로 보내라"는 명령이지, $v$를 포함한 식을 미분하라는 것이
아니다.

$v$에 대한 민감도가 정말 필요하면 식을 다시 세워야 한다.

```python
(v * y).sum().backward()      # 이러면 v.grad 도 온다
```

```
(v*y).sum() 으로 적으면 v.grad 도 온다: [0.7957, 2.2774, 0.1372]
```

같은 값을 계산하는 두 길인데 **하나는 $v$를 그래프에 올리고 하나는 안 올린다.**

## 5. 조용한 것 셋 — 자료형

```
v dtype torch.float32  -> 통과, x.grad dtype torch.float32
v dtype torch.float64  -> 통과, x.grad dtype torch.float32
v dtype torch.float16  -> 통과, x.grad dtype torch.float32
v dtype torch.int64    -> 통과, x.grad dtype torch.float32
```

**넷 다 통과한다.** 「형상, 자료형, 장치」에서 본 자료형 승격과 달리, 여기서는
`v`를 출력의 자료형에 맞춰 조용히 바꾼다.

대개 편한데, **정수 `v`도 받는다**는 것이 걸린다. 0.5 같은 가중치를 실수로
정수 텐서에 담으면 0으로 잘리고, 아무 경고도 없다.

```python
torch.tensor([0.5, 1.0])      # float32 — 괜찮다
torch.tensor([0, 1])          # int64  — 0.5 를 넣었다면 이미 잘렸다
```

## 6. 쌓임을 아예 피하는 길

```
autograd.grad 로 모은 J == A : True
x.grad 는 그대로 : None
```

`torch.autograd.grad(y, x, v)`는 경사를 **돌려주고** `.grad`를 건드리지 않는다.

```python
g, = torch.autograd.grad(y, x, v, retain_graph=True)
```

비울 일이 없으니 쌓일 일도 없다. **야코비를 모을 때는 이쪽이 안전하다.** 3절의
함정이 구조적으로 사라진다.

| | `.grad` | 쓸 자리 |
|---|---|---|
| `y.backward(v)` | 채운다 (쌓인다) | 학습 루프 |
| `torch.autograd.grad(y, x, v)` | 안 건드린다 | 야코비, 고차 미분, 분석 |

## 7. `v`의 부호

```
v=+1 (경사 하강)  3.0 -> 0.9830
v=-1 (경사 상승)  3.0 -> 7.4650
```

부호를 뒤집으면 **최솟값에서 멀어진다.** $-\nabla$ 대신 $+\nabla$ 방향으로
걷는 것이니 당연하다.

실수로 뒤집으면 로스가 올라가므로 금방 알아챈다. 그런데 **일부러 뒤집는 자리**도
있다 — GAN의 생성자처럼 어떤 로스를 올리고 싶을 때다. 대개는 로스에 음수를
붙이는 쪽이 읽기 좋지만, 같은 일이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
출력 가운데 **하나만** 역전파하려면 어떻게 하는가? 두 가지 길을 써 보고 같은
결과인지 확인하라.

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    x1 = torch.randn(3, requires_grad=True)
    (x1 ** 2)[0].backward()                    # 길 1: 원소를 골라 역전파
    print("  y[0].backward()      :", [round(t, 4) for t in x1.grad.tolist()])

    x2 = x1.detach().clone().requires_grad_(True)
    (x2 ** 2).backward(torch.tensor([1.0, 0.0, 0.0]))   # 길 2: v 로 고른다
    print("  backward([1,0,0])    :", [round(t, 4) for t in x2.grad.tolist()])
    print("  같은가:", torch.allclose(x1.grad, x2.grad))
    ```

    ```
      y[0].backward()      : [3.082, -0.0, -0.0]
      backward([1,0,0])    : [3.082, -0.0, -0.0]
      같은가: True
    ```

    **같다.** `y[0]`을 떼어 내 역전파하는 것은 $v$를 첫 단위벡터로 주는 것과
    같은 일이다.

    `y[0]`이 0차원이므로 `backward()`에 인자가 필요 없다는 점이 편하다.
    고르는 출력이 하나면 이쪽이 짧다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
4절에서 `v.grad`가 안 왔다. 그렇다면 `backward(v)`에서 `v`를 바꾸면 결과가
**선형으로** 변하는가? 확인하라.

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    def vjp(v):
        torch.manual_seed(1)                 # 같은 x 에서 재려고 씨앗을 고정
        x = torch.randn(3, requires_grad=True)
        (A @ x).backward(v)
        return x.grad.clone()

    a, b = torch.randn(4), torch.randn(4)
    print("  vjp(a) + vjp(b) :", [round(t, 5) for t in (vjp(a) + vjp(b)).tolist()])
    print("  vjp(a + b)      :", [round(t, 5) for t in vjp(a + b).tolist()])
    print("  2*vjp(a)        :", [round(t, 5) for t in (2 * vjp(a)).tolist()])
    print("  vjp(2a)         :", [round(t, 5) for t in vjp(2 * a).tolist()])
    ```

    ```
      vjp(a) + vjp(b) : [-1.58776, -1.99875, 2.19892]
      vjp(a + b)      : [-1.58776, -1.99875, 2.19892]
      2*vjp(a)        : [-2.35213, -3.8263, 2.24002]
      vjp(2a)         : [-2.35213, -3.8263, 2.24002]
    ```

    **$v$에 대해 선형이다.** $v^\top J$가 $v$에 대한 선형 사상이니 당연하다.

    쓸모가 둘 있다.

    첫째, **검산에 쓴다.** VJP 구현이 맞는지 보려면 선형성을 깨뜨리는지 확인하면
    된다. 깨지면 어딘가 틀린 것이다.

    둘째, 야코비를 모을 때 **단위벡터 말고 다른 것을 써도 된다.** $v$들이
    기저를 이루기만 하면 되고, 모은 결과를 변환하면 야코비가 나온다. 보통은
    단위벡터가 가장 간단하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
6절의 `autograd.grad`가 `.grad`를 안 건드린다고 했다. 그렇다면 **학습 루프
한가운데서** 그것을 써도 되는가? 쌓여 있던 경사가 어떻게 되는지 확인하라.

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    m = torch.nn.Linear(3, 1)
    x, y = torch.randn(8, 3), torch.randn(8, 1)

    torch.nn.functional.mse_loss(m(x), y).backward()
    saved = m.weight.grad.clone()
    print("  backward 로 쌓인 경사:", [round(t, 5) for t in saved.flatten().tolist()])

    probe, = torch.autograd.grad(m(x).sum(), m.weight, retain_graph=True)
    print("  autograd.grad 로 딴 것:", [round(t, 5) for t in probe.flatten().tolist()])
    print("  쌓여 있던 경사가 그대로인가:",
          torch.allclose(m.weight.grad, saved))
    ```

    ```
      backward 로 쌓인 경사: [0.25892, 0.79127, -0.2553]
      autograd.grad 로 딴 것: [-0.31461, -1.96776, -0.77923]
      쌓여 있던 경사가 그대로인가: True
    ```

    **써도 된다.** 쌓여 있던 경사가 그대로다.

    그래서 학습 루프 한가운데서 **다른 양을 들여다보는 데** 쓸 수 있다.
    경사의 노름을 재거나, 특정 층의 기여를 보거나, 적대적 예제를 만들거나.
    「torch.no_grad 사용하기」 연습문제 2에서 입력에 대한 경사를 딴 것도
    같은 꼴이다.

    다만 `retain_graph=True`가 필요하다. 그 뒤에 `opt.step()`을 부를 것이라면
    이미 `backward()`가 끝났으므로 상관없지만, 순서가 반대라면 그래프를
    남겨야 한다.

    값이 공짜는 아니다. 그래프를 한 번 더 걷는 것이므로 역전파 한 번만큼 든다.
    걸음마다 하면 학습이 두 배 느려진다 — 몇 걸음마다 한 번 보는 쪽이 보통이다.
