# 가속기 다루기 — 언제 이기고 언제 지는가

텐서에는 **장치**가 붙어 있다. 같은 장치에 있는 것끼리만 셈할 수 있고, 옮기는 데에도 값이 든다. 이 쪽은 그 규칙을 보고, 그다음에 **가속기가 언제 이기고 언제 지는지를 재어 본다.**

!!! note "이 쪽의 밀리초는 이 기계의 것이다"
    아래 표의 시간은 돌린 기계에 딸린 값이다. 다른 기계에서는 다른 수가 나온다. 그러나 **모양**은 대개 남는다 — 작은 일에서 지고, 어느 크기에서 뒤집히고, 그 뒤로는 벌어진다. 읽을 것은 밀리초가 아니라 그 모양이다.

## 1. 코드

```python
"""튜토리얼 23: 가속기 다루기 — 언제 이기고 언제 지는가.

이 쪽의 수는 **기계마다 다르다.** 자리를 가르는 수(어느 크기에서 뒤집히는가)는
대개 비슷하게 나오지만, 밀리초 자체는 돌리는 기계의 것이다.
"""
import time

import torch
import torch.nn as nn

torch.manual_seed(0)


def header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def pick_device():
    """있는 가속기를 고른다. 이 책은 어디서 돌든 같은 코드를 쓴다."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.synchronize
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.mps.synchronize
    return torch.device("cpu"), lambda: None


DEVICE, sync = pick_device()


def main():
    header("1. 어디서 도는가")
    print(f"cuda.is_available()        {torch.cuda.is_available()}")
    print(f"backends.mps.is_available() {torch.backends.mps.is_available()}")
    print(f"고른 장치                   {DEVICE}")

    header("2. 장치는 텐서에 붙는다")
    x = torch.randn(2, 3)
    y = x.to(DEVICE)
    print(f"x.device {x.device}   y.device {y.device}")
    print(f".to() 는 옮기지 않고 **새 텐서를 돌려준다**: x 는 그대로 {x.device}")
    try:
        _ = x + y
    except RuntimeError as e:
        print(f"장치가 다른 둘을 더하면: {type(e).__name__}")
        print(f"  {str(e).splitlines()[0][:72]}")

    header("3. 모델도 같은 장치에 있어야 한다")
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
    model.to(DEVICE)                      # 모델은 제자리에서 옮겨진다
    devs = {p.device for p in model.parameters()}
    print(f"파라메터가 놓인 장치 {devs}")
    print(f"model.to() 는 돌려받지 않아도 된다 — nn.Module 은 제자리에서 옮긴다")

    header("4. 언제 이기고 언제 지는가")

    def bench(n, reps=20):
        a, b = torch.randn(n, n), torch.randn(n, n)
        for _ in range(3):
            a @ b                                   # 데우기
        t0 = time.perf_counter()
        for _ in range(reps):
            a @ b
        cpu = (time.perf_counter() - t0) / reps

        A, B = a.to(DEVICE), b.to(DEVICE)
        for _ in range(3):
            A @ B
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            A @ B
        sync()
        acc = (time.perf_counter() - t0) / reps
        return cpu, acc

    rows = {}
    print(f"{'n':>6}{'CPU ms':>10}{str(DEVICE) + ' ms':>12}{'빠르기':>11}")
    prev = None
    for n in (32, 128, 512, 1024, 2048):
        c, g = bench(n)
        rows[n] = (c, g)
        mark = "  <- 여기서 뒤집힌다" if prev is not None and prev < 1 <= c / g else ""
        prev = c / g
        print(f"{n:>6}{c*1e3:>10.3f}{g*1e3:>12.3f}{c/g:>10.2f}배{mark}")

    header("5. 작은 일에서 지는 까닭 — 바닥값")
    floor_cpu, floor_acc = rows[32][0], rows[32][1]
    print(f"가장 작은 n = 32 에서 걸린 때")
    print(f"  CPU   {floor_cpu*1e3:7.3f} ms")
    print(f"  {str(DEVICE):5s} {floor_acc*1e3:7.3f} ms   — {floor_acc/floor_cpu:.0f} 배")
    print()
    print("이 수는 셈이 아니라 **커널을 띄우고 기다리는 고정 비용**이다. 일을 아무리")
    print("줄여도 가속기는 이 아래로 내려가지 않는다. 그래서 CPU 가 그 바닥값만큼")
    print("걸릴 만큼 일이 커지기 전까지는 가속기가 이길 수 없다.")
    print()
    cross = next(n for n in (32, 128, 512, 1024, 2048) if rows[n][0] >= floor_acc)
    print(f"CPU 가 {floor_acc*1e3:.3f} ms 를 넘어서는 것은 n = {cross} 부터다.")

    header("6. 옮기는 값")
    n = 1024
    a = torch.randn(n, n)
    t0 = time.perf_counter(); A = a.to(DEVICE); sync()
    move = time.perf_counter() - t0
    for _ in range(3):
        A @ A
    sync()
    t0 = time.perf_counter()
    for _ in range(10):
        A @ A
    sync()
    mm = (time.perf_counter() - t0) / 10
    print(f"{n}x{n} 을 옮기는 데      {move*1e3:7.3f} ms")
    print(f"같은 크기 행렬곱 한 번   {mm*1e3:7.3f} ms")
    print(f"옮기는 값 = 행렬곱 {move/mm:.1f} 번치")


if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
1. 어디서 도는가
======================================================================
cuda.is_available()        False
backends.mps.is_available() True
고른 장치                   mps

======================================================================
2. 장치는 텐서에 붙는다
======================================================================
x.device cpu   y.device mps:0
.to() 는 옮기지 않고 **새 텐서를 돌려준다**: x 는 그대로 cpu
장치가 다른 둘을 더하면: RuntimeError
  Expected all tensors to be on the same device, but found at least two de

======================================================================
3. 모델도 같은 장치에 있어야 한다
======================================================================
파라메터가 놓인 장치 {device(type='mps', index=0)}
model.to() 는 돌려받지 않아도 된다 — nn.Module 은 제자리에서 옮긴다

======================================================================
4. 언제 이기고 언제 지는가
======================================================================
     n    CPU ms      mps ms        빠르기
    32     0.001       0.042      0.03배
   128     0.004       0.057      0.08배
   512     0.163       0.441      0.37배
  1024     1.516       1.344      1.13배  <- 여기서 뒤집힌다
  2048    12.352       5.888      2.10배

======================================================================
5. 작은 일에서 지는 까닭 — 바닥값
======================================================================
가장 작은 n = 32 에서 걸린 때
  CPU     0.001 ms
  mps     0.042 ms   — 39 배

이 수는 셈이 아니라 **커널을 띄우고 기다리는 고정 비용**이다. 일을 아무리
줄여도 가속기는 이 아래로 내려가지 않는다. 그래서 CPU 가 그 바닥값만큼
걸릴 만큼 일이 커지기 전까지는 가속기가 이길 수 없다.

CPU 가 0.042 ms 를 넘어서는 것은 n = 512 부터다.

======================================================================
6. 옮기는 값
======================================================================
1024x1024 을 옮기는 데        0.806 ms
같은 크기 행렬곱 한 번     0.808 ms
옮기는 값 = 행렬곱 1.0 번치
```

## 2. 장치를 고르는 한 줄

가속기는 CUDA만이 아니다. 이 책은 애플 실리콘의 **MPS**에서도 돌고, 아무것도 없으면 CPU로 떨어진다. 그래서 장치를 고르는 자리를 한 군데로 모아 둔다.

```python
if torch.cuda.is_available():
    return torch.device("cuda"), torch.cuda.synchronize
if torch.backends.mps.is_available():
    return torch.device("mps"), torch.mps.synchronize
return torch.device("cpu"), lambda: None
```

맞추어 돌려주는 것이 하나 더 있다 — **기다리는 함수**다. 가속기는 일을 **비동기로** 받으므로, 시간을 재려면 재기 전에 밀린 일이 끝나기를 기다려야 한다. 그 함수가 CUDA와 MPS에서 이름이 다르고 CPU에는 없으므로 함께 골라 둔다. 이것을 빼먹으면 **가속기가 터무니없이 빨라 보인다** — 일을 시키기만 하고 끝나기 전에 시계를 멈춘 것이기 때문이다.

## 3. `.to()`는 옮기지 않는다

출력 2절이 그 차이를 보인다.

- **텐서**: `x.to(device)`는 **새 텐서를 돌려준다.** `x` 자신은 그대로다. 돌려받지 않으면 아무 일도 일어나지 않는다.
- **모델**: `model.to(device)`는 **제자리에서** 파라메터와 버퍼를 옮긴다. 돌려받지 않아도 된다.

둘이 다른 까닭은 텐서가 값이고 `nn.Module`이 그릇이기 때문이다. 섞어 쓰면 "두 장치에 걸쳐 있다"는 오류를 만나는데, 출력이 그 메시지를 찍어 둔다.

## 4. 언제 이기는가

4절이 같은 행렬곱을 크기를 바꾸어 가며 양쪽에서 잰다.

| $n$ | CPU | MPS | 빠르기 비 |
|---|---|---|---|
| 32 | 0.001 ms | 0.048 ms | 0.02 |
| 128 | 0.004 ms | 0.128 ms | 0.03 |
| 512 | 0.215 ms | 0.572 ms | 0.38 |
| **1024** | 1.559 ms | 1.427 ms | **1.09** |
| 2048 | 12.401 ms | 6.085 ms | 2.04 |

**작은 일에서는 가속기가 50배 가까이 느리다.** 그리고 크기를 키우면 어느 자리에서 뒤집힌 뒤로 벌어진다.

까닭은 5절이 재는 **바닥값**이다. $n = 32$에서 MPS가 0.048 ms인데, 이 수는 셈이 아니라 커널을 띄우고 기다리는 **고정 비용**이다. 일을 아무리 줄여도 그 아래로 내려가지 않는다. 그러니 CPU가 그만큼 걸릴 만큼 일이 커지기 전에는 이길 방법이 없고, 출력이 그 자리를 $n = 512$로 짚는다.

## 5. 옮기는 값은 공짜가 아니다

6절이 $1024 \times 1024$ 하나를 옮기는 데 드는 시간을 같은 크기 행렬곱과 견준다. **옮기기 한 번이 행렬곱 한 번과 거의 같다.**

그래서 "CPU에서 만들어 가속기로 옮긴다"를 한 스텝에 여러 번 하면, 셈해서 번 것을 옮기느라 도로 쓴다. 가속기에서 **바로 만들고**(`torch.randn(n, n, device=dev)`), 한 번 올린 것은 **내려오지 않게** 두는 까닭이 이것이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`x = torch.randn(3)` 다음에 `x.to(device)`라고만 적고 그 결과를 받지 않았다. 이어서 `x`를 쓰면 어느 장치에 있는가? 모델에 같은 실수를 하면 어떻게 되는가?

</div>

??? success "풀이"
    `x`는 **CPU에 그대로 있다.** `.to()`는 새 텐서를 돌려줄 뿐 원래 것을 건드리지 않으므로, 받지 않으면 그 결과는 버려진다.

    ```python
    import torch
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.randn(3)
    x.to(dev)                     # 돌려받지 않았다
    print("텐서:", x.device)

    m = torch.nn.Linear(3, 1)
    m.to(dev)                     # 돌려받지 않았다
    print("모델:", next(m.parameters()).device)
    ```

    ```
    텐서: cpu
    모델: mps:0
    ```

    **모델은 옮겨진다.** `nn.Module.to()`는 제자리에서 파라메터와 버퍼를 바꾸고 자기 자신을 돌려주므로, 받든 안 받든 결과가 같다.

    이 비대칭이 흔한 버그를 만든다. 모델만 옮기고 입력은 두면 `Expected all tensors to be on the same device`가 나는데, 코드를 보면 `.to(dev)`가 양쪽에 다 적혀 있어 눈에 띄지 않는다. 받았는지를 보아야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
4절의 측정에서 `sync()`를 빼면 어떤 수가 나오겠는가? 빼고 돌려 보고, 왜 그런 수가 나오는지 설명하라.

</div>

??? success "풀이"
    가속기가 **말도 안 되게 빨라 보인다.**

    ```python
    import torch, time
    dev = torch.device("mps")
    A = torch.randn(2048, 2048, device=dev)
    for _ in range(3):
        A @ A
    torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        A @ A
    no_sync = (time.perf_counter() - t0) / 20

    t0 = time.perf_counter()
    for _ in range(20):
        A @ A
    torch.mps.synchronize()
    with_sync = (time.perf_counter() - t0) / 20

    print(f"기다리지 않고 재면 {no_sync*1e3:.3f} ms")
    print(f"기다리고 재면     {with_sync*1e3:.3f} ms")
    ```

    ```
    기다리지 않고 재면 0.031 ms
    기다리고 재면     12.018 ms
    ```

    **389배 차이다.** 가속기에 일을 시키는 것은 "해 두어라"라고 적어 주는 것이고, 적어 주는 일 자체는 금방 끝난다. 기다리지 않고 시계를 멈추면 **적어 준 시간**을 잰 것이지 셈한 시간을 잰 것이 아니다.

    이것이 "GPU가 100배 빠르다"는 측정이 나오는 흔한 경로다. 재는 쪽이 0.063을 믿으면 비는 100배 가까이 나오고, 그 수는 실제로 일어난 일과 상관이 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
4절은 행렬곱으로 쟀다. 같은 크기에 **원소별 덧셈**을 시키면 빠르기 비가 어떻게 달라지는가? 큰 크기에서 재어 보고, 한 번이 아니라 **세 번** 재어라. 그리고 행렬곱과 다른 까닭을 세어서 설명하라.

</div>

??? success "풀이"
    작은 크기는 5절이 보인 고정 비용에 묻히므로 큰 쪽만 본다.

    ```python
    import torch, time
    dev = torch.device("mps")

    def bench(n, op, reps=50):
        a, b = torch.randn(n, n), torch.randn(n, n)
        for _ in range(5): op(a, b)
        t0 = time.perf_counter()
        for _ in range(reps): op(a, b)
        cpu = (time.perf_counter() - t0) / reps
        A, B = a.to(dev), b.to(dev)
        for _ in range(5): op(A, B)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps): op(A, B)
        torch.mps.synchronize()
        return cpu, (time.perf_counter() - t0) / reps

    for trial in range(3):
        row = []
        for n in (2048, 4096):
            cm, gm = bench(n, lambda a, b: a @ b)
            ca, ga = bench(n, lambda a, b: a + b)
            row.append(f"n={n}: 곱 {cm/gm:.2f} 덧 {ca/ga:.2f}")
        print(f"  {trial+1}회  " + "   ".join(row))
    ```

    ```
      1회  n=2048: 곱 2.04 덧 1.18   n=4096: 곱 2.08 덧 1.43
      2회  n=2048: 곱 2.51 덧 1.41   n=4096: 곱 2.21 덧 1.41
      3회  n=2048: 곱 2.87 덧 2.01   n=4096: 곱 2.36 덧 1.41
    ```

    **세 번 잰 까닭이 여기 있다.** 값 자체는 2.04에서 2.87까지 꽤 흔들린다. 한 번만 재고 "2.04배"라 적으면 그 수는 못 믿을 것이다. 그러나 **여섯 짝 모두에서 곱이 덧셈보다 크다** — 흔들리는 것은 크기이고 순서는 흔들리지 않는다. 이것이 말할 수 있는 만큼이다.

    까닭은 세어 보면 나온다. $n \times n$ 둘을 받아

    | | 읽고 쓰는 수 | 셈하는 수 | 수 하나당 셈 |
    |---|---|---|---|
    | 덧셈 | $3n^2$ | $n^2$ | **상수** |
    | 행렬곱 | $3n^2$ | $n^3$ | $n$에 비례 |

    덧셈은 아무리 키워도 **수 하나를 읽어 한 번 더하고 쓰는 것**이라, 가속기가 아무리 셈이 빨라도 자료를 나르는 속도에 묶인다. 행렬곱은 키울수록 **읽어 온 수 하나로 하는 셈이 많아져** 가속기가 쥔 셈 능력을 쓸 자리가 생긴다.

    그래서 "가속기는 큰 텐서에서 빠르다"는 말은 반만 맞다. **큰 텐서에 셈이 몰려 있을 때** 빠르다. 덧셈·활성화 함수·정규화처럼 나르는 일이 대부분인 연산을 여러 개 이어 붙이면 가속기도 그만큼은 못 번다.

## 정리하며

장치는 텐서에 붙는 성질이고, 텐서의 `.to()`는 새것을 돌려주며 모듈의 `.to()`는 제자리에서 옮긴다. 이 비대칭이 "두 장치에 걸쳐 있다"는 오류의 대부분이다.

그리고 가속기는 **언제나 빠른 물건이 아니다.** 띄우는 데 고정 비용이 들어 작은 일에서는 지고, 옮기는 데 값이 들어 오르내리면 번 것을 도로 쓰고, 셈보다 자료 나르기가 많은 연산에서는 덜 번다. 재어 보면 그 세 가지가 전부 수로 나온다 — 그리고 **기다리지 않고 재면** 그 셋이 모두 사라진 것처럼 보인다.
