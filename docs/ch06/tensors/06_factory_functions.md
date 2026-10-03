# 팩토리 함수

텐서를 처음부터 만들어 주는 함수가 여러 벌 있다. 다 채워 주는 것(`zeros`, `ones`, `full`), 수열을 만드는 것(`arange`, `linspace`), 무작위로 뽑는 것(`rand`, `randn`), 남의 모양을 따르는 것(`*_like`), 그리고 **아무것도 채우지 않는 것**(`empty`)이다. 어느 것을 고르느냐에 따라 자료형이 달라지고, 하나는 값이 아예 정해져 있지 않다.

## 1. 코드

```python
"""생성기 함수."""
import torch

# ========================================================================
# 메인
# ========================================================================


def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    # -------------------------------------------------------------------------
    # 재현성을 위한 시드
    # -------------------------------------------------------------------------
    # 이 프로세스의 CPU/CUDA 난수 생성기를 제어한다(rand/randn/normal 등).
    # 참고: CUDA는 자체 난수 스트림을 가지지만 여기서 함께 시드가 설정된다.
    # PyTorch/BLAS 버전이나 장치가 다르면 결정성이 보장되지 않는다.
    torch.manual_seed(123)  # controls rand/randn/normal etc.

    # -------------------------------------------------------------------------
    # 장치 선택(기본은 CPU. 가능하면 CUDA를 쓴다)
    # -------------------------------------------------------------------------
    # 애플 실리콘에서는 'mps'(Metal)를 따로 확인할 수도 있다.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # 아래에서 재사용할 흔한 dtype(단정밀도가 좋은 기본값이다)
    fp = torch.float32

    # -------------------------------------------------------------------------
    # 기본 "채우기" 팩토리 함수
    # -------------------------------------------------------------------------
    header("1) Basic fills: zeros / ones / full / empty / eye")

    t_zeros = torch.zeros((2, 3), dtype=fp, device=device, requires_grad=False)
    t_ones  = torch.ones((2, 3), dtype=fp, device=device)
    t_full  = torch.full((2, 3), fill_value=7.7, dtype=fp, device=device)
    t_empty = torch.empty((2, 3), dtype=fp, device=device)  # ⚠️ uninitialized memory (values are garbage)
    t_eye   = torch.eye(4, dtype=fp, device=device)         # 4x4 identity (float because dtype=fp)

    print("zeros:\n", t_zeros)
    print("ones:\n",  t_ones)
    print("full(7.7):\n", t_full)
    # empty의 값은 찍지 않는다. 정해져 있지 않으므로 돌릴 때마다 달라질 수 있고,
    # 하필 0이 나오면 "empty는 빠른 zeros"라는 잘못된 인상을 준다. 아래 연습문제
    # 3이 실제로 쓰레기 값이 나오는 것을 보인다.
    print(f"empty: shape={tuple(t_empty.shape)} dtype={t_empty.dtype} — 값은 정해져 있지 않다")
    print("eye(4):\n", t_eye)

    # -------------------------------------------------------------------------
    # 범위와 등간격 값
    # -------------------------------------------------------------------------
    header("2) Ranges: arange / linspace / logspace / randint / randperm")

    # arange: 반개구간 [start, end). 인수 중 하나라도 실수면 → 실수 출력.
    t_arange_i = torch.arange(0, 10, 2, device=device)          # ints by default when step is int
    t_arange_f = torch.arange(0.0, 1.0, 0.2, device=device)     # floats when any arg is float

    # linspace: 끝값을 포함하는 균등 간격의 점 N개(닫힌 구간)
    t_lin = torch.linspace(0, 1, steps=5, device=device)        # [0., .25, .5, .75, 1.]

    # logspace: base**start와 base**end 사이(양끝 포함)의 등비 간격 점들
    t_log = torch.logspace(start=0, end=3, steps=4, base=10.0, device=device)  # [1, 10, 100, 1000]

    # randint: [low, high) 구간의 정수
    t_randi = torch.randint(low=0, high=10, size=(3, 4), device=device)

    # randperm: 0..n-1의 무작위 순열(중복 없음)
    t_perm = torch.randperm(10, device=device)

    print("arange int:", t_arange_i)
    print("arange float:", t_arange_f)
    print("linspace(0,1,5):", t_lin)
    print("logspace(0,3,4):", t_log)
    print("randint[0,10):\n", t_randi)
    print("randperm(10):", t_perm)

    # -------------------------------------------------------------------------
    # 무작위 연속 분포
    # -------------------------------------------------------------------------
    header("3) Random: rand / randn / normal")

    # rand: 지정한 device/dtype에서 독립 동일 분포 U(0,1)
    t_rand  = torch.rand((2, 3), dtype=fp, device=device)

    # randn: 표준 정규분포 N(0,1)
    t_randn = torch.randn((2, 3), dtype=fp, device=device)

    # normal: N(mean, std). mean/std가 텐서면 브로드캐스팅을 지원한다
    t_norm  = torch.normal(mean=5.0, std=2.0, size=(2, 3), dtype=fp, device=device)

    print("rand U(0,1):\n", t_rand)
    print("randn N(0,1):\n", t_randn)
    print("normal N(5,2):\n", t_norm)

    # -------------------------------------------------------------------------
    # *_like: 다른 텐서의 모양/dtype/device에 맞는 텐서를 만든다
    # -------------------------------------------------------------------------
    header("4) *_like variants: zeros_like / ones_like / full_like")

    base = torch.randn((3, 2), dtype=torch.float64, device=device)
    # *_like는 기본적으로 모양/dtype/device를 복사한다. 키워드 인자로 덮어쓸 수 있다.
    z_like = torch.zeros_like(base)                 # dtype=float64 because base is float64
    o_like = torch.ones_like(base)
    f_like = torch.full_like(base, fill_value=3.14)

    print("base (float64):\n", base)
    print("zeros_like(base):\n", z_like)
    print("ones_like(base):\n",  o_like)
    print("full_like(base, 3.14):\n", f_like)

    # -------------------------------------------------------------------------
    # 삼각/대각 관련 도우미 함수
    # -------------------------------------------------------------------------
    header("5) Triangular / diagonal: triu / tril / diag / diagonal")

    M = torch.arange(1, 10, device=device, dtype=fp).reshape(3, 3)
    print("M:\n", M)

    M_triu = torch.triu(M)         # upper triangular (copies lower part to zero)
    M_tril = torch.tril(M)         # lower triangular
    d_main = torch.diagonal(M)     # view of the main diagonal (shares storage)
    D = torch.diag(torch.tensor([9., 8., 7.], device=device))  # 1-D → diag matrix (new tensor)

    print("triu(M):\n", M_triu)
    print("tril(M):\n", M_tril)
    print("diagonal(M):", d_main)
    print("diag([9,8,7]):\n", D)

    # -------------------------------------------------------------------------
    # requires_grad: autograd를 위해 계산을 추적한다
    # -------------------------------------------------------------------------
    header("6) requires_grad example")

    # 실수 텐서에 requires_grad=True이면 PyTorch가 그래프를 만들고 경사를 누적한다.
    w = torch.ones((2, 2), dtype=fp, device=device, requires_grad=True)
    b = torch.zeros((2, 2), dtype=fp, device=device, requires_grad=True)
    x = torch.rand((2, 2), dtype=fp, device=device)  # input (no grad)

    # y = sum(w * x + b) → dy/dw = x, dy/db = 1 (b와 같은 모양)
    y = (w * x + b).sum()
    y.backward()  # populates w.grad and b.grad

    print("w:\n", w)
    print("x:\n", x)
    print("b:\n", b)
    print("y (sum):", y.item())
    print("w.grad:\n", w.grad)  # ≈ x
    print("b.grad:\n", b.grad)  # all ones

    # -------------------------------------------------------------------------
    # 요령: 이식성 있는 장치 생성
    # -------------------------------------------------------------------------
    header("7) Portable device tip")

    # 권장 패턴: 생성 시 `device=device`를 넘긴다 → 불필요한 복사/이동을 피한다.
    t_portable = torch.ones((2, 2), device=device)
    print("Portable tensor on chosen device:\n", t_portable)

    # 이미 CPU에 만들었다면 .to(device)로 옮긴다(목표 장치에 새 텐서를 만든다).
    t_moved = torch.ones((2, 2)).to(device)
    print("Moved tensor to device:\n", t_moved)

    # -------------------------------------------------------------------------
    # 요약
    # -------------------------------------------------------------------------
    header("8) Summary")
    print(
        "• Use zeros/ones/full/empty/eye for basic shapes\n"
        "• Use arange/linspace/logspace/randint/randperm for sequences\n"
        "• Use rand/randn/normal for random continuous values\n"
        "• Use *_like to mirror another tensor's shape/dtype/device\n"
        "• Use triu/tril/diag/diagonal for structured matrices\n"
        "• Always set dtype/device/requires_grad explicitly when it matters\n"
    )


if __name__ == "__main__":
    main()
```

??? note "전체 출력 (128줄)"

    ```
    Using device: cpu

    ================================================================================
    1) Basic fills: zeros / ones / full / empty / eye
    ================================================================================
    zeros:
     tensor([[0., 0., 0.],
            [0., 0., 0.]])
    ones:
     tensor([[1., 1., 1.],
            [1., 1., 1.]])
    full(7.7):
     tensor([[7.7000, 7.7000, 7.7000],
            [7.7000, 7.7000, 7.7000]])
    empty: shape=(2, 3) dtype=torch.float32 — 값은 정해져 있지 않다
    eye(4):
     tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])

    ================================================================================
    2) Ranges: arange / linspace / logspace / randint / randperm
    ================================================================================
    arange int: tensor([0, 2, 4, 6, 8])
    arange float: tensor([0.0000, 0.2000, 0.4000, 0.6000, 0.8000])
    linspace(0,1,5): tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
    logspace(0,3,4): tensor([   1.,   10.,  100., 1000.])
    randint[0,10):
     tensor([[2, 9, 2, 0],
            [0, 2, 6, 7],
            [9, 4, 1, 1]])
    randperm(10): tensor([6, 8, 2, 9, 4, 0, 7, 1, 5, 3])

    ================================================================================
    3) Random: rand / randn / normal
    ================================================================================
    rand U(0,1):
     tensor([[0.5932, 0.6367, 0.9826],
            [0.2745, 0.6584, 0.2775]])
    randn N(0,1):
     tensor([[ 0.5455, -0.6713,  1.2182],
            [-0.7725, -1.9249,  0.3442]])
    normal N(5,2):
     tensor([[4.6445, 8.9105, 5.2568],
            [3.9897, 3.5005, 3.4265]])

    ================================================================================
    4) *_like variants: zeros_like / ones_like / full_like
    ================================================================================
    base (float64):
     tensor([[ 0.8551, -0.9193],
            [-0.2089,  1.0131],
            [ 0.0520, -0.8516]], dtype=torch.float64)
    zeros_like(base):
     tensor([[0., 0.],
            [0., 0.],
            [0., 0.]], dtype=torch.float64)
    ones_like(base):
     tensor([[1., 1.],
            [1., 1.],
            [1., 1.]], dtype=torch.float64)
    full_like(base, 3.14):
     tensor([[3.1400, 3.1400],
            [3.1400, 3.1400],
            [3.1400, 3.1400]], dtype=torch.float64)

    ================================================================================
    5) Triangular / diagonal: triu / tril / diag / diagonal
    ================================================================================
    M:
     tensor([[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])
    triu(M):
     tensor([[1., 2., 3.],
            [0., 5., 6.],
            [0., 0., 9.]])
    tril(M):
     tensor([[1., 0., 0.],
            [4., 5., 0.],
            [7., 8., 9.]])
    diagonal(M): tensor([1., 5., 9.])
    diag([9,8,7]):
     tensor([[9., 0., 0.],
            [0., 8., 0.],
            [0., 0., 7.]])

    ================================================================================
    6) requires_grad example
    ================================================================================
    w:
     tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    x:
     tensor([[0.1634, 0.3009],
            [0.5201, 0.3834]])
    b:
     tensor([[0., 0.],
            [0., 0.]], requires_grad=True)
    y (sum): 1.367701530456543
    w.grad:
     tensor([[0.1634, 0.3009],
            [0.5201, 0.3834]])
    b.grad:
     tensor([[1., 1.],
            [1., 1.]])

    ================================================================================
    7) Portable device tip
    ================================================================================
    Portable tensor on chosen device:
     tensor([[1., 1.],
            [1., 1.]])
    Moved tensor to device:
     tensor([[1., 1.],
            [1., 1.]])

    ================================================================================
    8) Summary
    ================================================================================
    • Use zeros/ones/full/empty/eye for basic shapes
    • Use arange/linspace/logspace/randint/randperm for sequences
    • Use rand/randn/normal for random continuous values
    • Use *_like to mirror another tensor's shape/dtype/device
    • Use triu/tril/diag/diagonal for structured matrices
    • Always set dtype/device/requires_grad explicitly when it matters

    ```
## 2. 논의

**기본 자료형이 함수마다 다르다.** 실수를 주는 것과 정수를 주는 것이 갈리는데, 이름만 보고는 알 수 없다.

| `float32`를 준다 | `int64`를 준다 |
|---|---|
| `zeros`, `ones`, `eye`, `rand`, `randn`, `empty`, `linspace` | `arange`, `randint`, `randperm` |

`full`은 넣는 값을 보고 정한다 — `full((3,), 7)`은 `int64`, `full((3,), 7.0)`은 `float32`다. 색인으로 쓸 텐서를 `zeros`로 만들면 실수가 되어 색인에 쓸 수 없고, 반대로 `arange`로 만든 것을 실수 셈에 섞으면 자료형이 어긋난다.

**`*_like`는 모양만 베끼는 것이 아니다.** 자료형과 장치와 메모리 배치까지 함께 따른다. 그래서 `torch.zeros(x.shape)`와 `torch.zeros_like(x)`는 같지 않다.

```python
x = torch.ones(2, 3, dtype=torch.float64)
torch.zeros(2, 3).dtype        # torch.float32  — 기본값으로 떨어진다
torch.zeros_like(x).dtype      # torch.float64  — x를 따른다
```

`x`가 GPU에 있으면 차이가 더 커진다. `zeros(x.shape)`는 CPU에 만들어지므로 둘을 더하는 자리에서 장치가 다르다는 오류가 난다. **남과 짝을 맞출 텐서라면 `*_like`를 쓴다.**

**`torch.diag`는 넣는 계수에 따라 하는 일이 반대다.**

```python
torch.diag(torch.tensor([1., 2., 3.]))        # (3, 3) 행렬을 만든다
torch.diag(torch.arange(9.).reshape(3, 3))    # tensor([0., 4., 8.]) — 대각선을 꺼낸다
```

만드는 함수와 꺼내는 함수가 이름을 함께 쓰는 셈이다. 모양이 바뀌었는데 오류가 나지 않으니, 계수를 잘못 짚으면 한참 뒤에서 드러난다.

!!! warning "`torch.empty`의 값은 정해져 있지 않다"
    `empty`는 메모리만 받아 오고 **채우지 않는다.** 그래서 그 안에 무엇이 있을지 정해져 있지 않다 — 앞서 다른 텐서가 쓰고 버린 값이 그대로 보일 수 있다.

    위험한 점은 흔히 **0처럼 보인다**는 것이다. 갓 받은 메모리는 0으로 차 있기 쉬워서, 작은 예제에서는 `zeros`와 구별되지 않는다. 그래서 "`empty`는 빠른 `zeros`"라고 잘못 배우고, 메모리를 많이 쓰는 코드에 가서야 쓰레기 값이 나온다.

    `empty`를 쓰는 자리는 **바로 다음 줄에서 전부 덮어쓸 때**뿐이다. 그렇지 않으면 `zeros`를 쓴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
아래 여섯 가지의 `dtype`을 찍어 보기 전에 적어 보고 확인하라. 실수가 나오는 것과 정수가 나오는 것을 가르는 기준이 무엇인지 한 문장으로 적어라.

```python
torch.zeros(3)   torch.eye(3)   torch.arange(3)
torch.randint(0, 5, (3,))       torch.full((3,), 7)      torch.full((3,), 7.0)
```

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    for e in ["torch.zeros(3)", "torch.eye(3)", "torch.arange(3)",
              "torch.randint(0, 5, (3,))", "torch.full((3,), 7)", "torch.full((3,), 7.0)"]:
        print(f"{e:26} -> {eval(e).dtype}")
    ```

    ```
    torch.zeros(3)             -> torch.float32
    torch.eye(3)               -> torch.float32
    torch.arange(3)            -> torch.int64
    torch.randint(0, 5, (3,))  -> torch.int64
    torch.full((3,), 7)        -> torch.int64
    torch.full((3,), 7.0)      -> torch.float32
    ```

    기준은 **그 함수가 세는 것을 주는가, 재는 것을 주는가**다. `arange`와 `randint`는
    개수나 색인처럼 세는 값을 주므로 `int64`이고, `zeros`·`eye`·`rand` 들은 재는 값을
    주므로 `float32`다. `full`만 내가 넣은 값의 꼴을 보고 따라간다.

    이름에서 짐작할 수 없다는 것이 요점이다. 색인에 쓸 텐서를 `zeros`로 만들면 실수가
    되어 색인에 못 쓴다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`x = torch.ones(2, 3, dtype=torch.float64)`에 대해 `torch.zeros(x.shape)`와 `torch.zeros_like(x)`의 `dtype`을 비교하라. 둘이 다른 까닭을 밝히고, `x`가 GPU에 있을 때 어느 쪽이 오류를 내는지 답하라.

</div>

??? success "연습문제 2 풀이"
    ```python
    import torch

    x = torch.ones(2, 3, dtype=torch.float64)
    print("x           :", x.dtype)
    print("zeros(x.shape):", torch.zeros(x.shape).dtype)
    print("zeros_like(x) :", torch.zeros_like(x).dtype)
    ```

    ```
    x           : torch.float64
    zeros(x.shape): torch.float32
    zeros_like(x) : torch.float64
    ```

    `x.shape`는 **모양만** 담고 있다. 자료형도 장치도 거기에 들어 있지 않으므로
    `torch.zeros(x.shape)`는 기본값인 `float32`, CPU로 떨어진다. `zeros_like(x)`는
    `x`를 통째로 보고 자료형·장치·메모리 배치까지 따른다.

    `x`가 GPU에 있으면 `torch.zeros(x.shape)`가 **CPU에** 만들어지므로, 둘을 더하는
    자리에서 장치가 다르다는 오류가 난다. `zeros_like(x)`는 같은 GPU에 만들어져
    문제가 없다.

    그래서 남과 짝을 맞출 텐서는 `*_like`로 만든다. 모양만 맞추면 되는 줄 알았던 것이
    실은 셋을 맞추는 일이다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`torch.empty(5)`를 몇 번 찍어 보면 대개 0이 나온다. 그런데 `empty`가 돌려주는 값은 정해져 있지 않다. 아래처럼 **메모리를 한 번 쓰고 버린 뒤에** 받아 보고, 무슨 일이 일어났는지 설명하라.

```python
a = torch.full((1000,), 3.14)
del a
b = torch.empty(1000)
```

그리고 `empty`를 써도 되는 경우가 언제인지 답하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch

    print("작게 받으면:", torch.empty(5))

    a = torch.full((1000,), 3.14)   # 메모리를 3.14로 채운다
    del a                            # 돌려준다
    b = torch.empty(1000)            # 같은 자리를 다시 받는다
    print("앞머리 8개:", b[:8])
    print("0이 아닌 원소 수:", int((b != 0).sum()), "/ 1000")
    ```

    ```
    작게 받으면: tensor([0., 0., 0., 0., 0.])
    앞머리 8개: tensor([3.1400, 3.1400, 3.1400, 3.1400, 3.1400, 3.1400, 3.1400, 3.1400])
    0이 아닌 원소 수: 1000 / 1000
    ```

    !!! note "이 출력은 돌릴 때마다, 기계마다 다를 수 있다"
        값이 정해져 있지 않다는 것이 바로 이 문제의 요점이므로, 위의 3.1400이 그대로
        나오지 않아도 맞다. 똑같이 나오는 쪽이 오히려 우연이다.

    `empty`는 메모리를 받아 오기만 하고 **채우지 않는다.** 그래서 앞서 그 자리를 쓰던
    텐서의 값이 그대로 남아 보인다. 위에서는 버린 3.14가 1000개 모두 되돌아왔다.

    위험한 점은 첫 줄이다. 작게 받으면 **0처럼 보인다.** 갓 받은 메모리는 0으로 차
    있기 쉬우므로 작은 예제에서는 `zeros`와 구별되지 않고, 그래서 "`empty`는 빠른
    `zeros`"라고 잘못 익히게 된다. 쓰레기 값은 메모리를 많이 돌려 쓰는 코드에 가서야
    나타나는데, 그때는 원인이 멀리 있다.

    써도 되는 경우는 하나다. **바로 다음에 전부 덮어쓸 때**다.

    ```python
    out = torch.empty(1000)
    torch.add(x, y, out=out)     # 모든 자리를 덮어쓴다 — 받아 둔 값은 쓰이지 않는다
    ```

    조금이라도 덮어쓰지 않는 자리가 남으면 `zeros`를 쓴다. `empty`가 버는 것은 한 번
    채우는 값뿐이고, 그것은 대개 아낄 만한 값이 아니다.

## 정리하며

텐서를 처음부터 만드는 함수를 고를 때 걸리는 것 셋이다.

- **기본 자료형이 이름에서 드러나지 않는다.** `arange`·`randint`·`randperm`은 `int64`를, 나머지는 `float32`를 준다. `full`만 넣는 값을 따른다. 세는 값과 재는 값의 차이다.
- **`*_like`는 모양만 베끼지 않는다.** 자료형·장치·메모리 배치까지 따른다. 그래서 `zeros(x.shape)`와 `zeros_like(x)`는 다르고, `x`가 GPU에 있으면 앞의 것은 장치가 어긋난다. 남과 짝을 맞출 텐서는 `*_like`로 만든다.
- **`empty`의 값은 정해져 있지 않다.** 앞서 그 메모리를 쓰던 값이 그대로 보일 수 있다. 하필 0으로 보이기 쉬운 것이 함정이다. 바로 다음에 전부 덮어쓸 때만 쓴다.

`torch.diag`는 계수에 따라 하는 일이 반대라는 것도 기억할 만하다. 1차원을 넣으면 행렬을 만들고, 2차원을 넣으면 대각선을 꺼낸다.

세 가지가 모두 같은 자리를 가리킨다. **텐서를 만들 때 내가 적지 않은 것은 PyTorch가 정하고, 그 기본값이 함수마다 다르다.** 모양만 맞추었다고 끝난 것이 아니다.
