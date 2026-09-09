# 팩토리 함수

이 스크립트는 텐서 팩토리 함수을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

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
    print("empty (uninitialized):\n", t_empty)
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

## 2. 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

브로드캐스팅은 작은 텐서를 가상으로 확장하여 모양이 다른 텐서 사이의 원소별 연산을 가능하게 한다. PyTorch는 차원을 오른쪽부터 맞추며, 각 차원 쌍이 서로 같거나, 둘 중 하나가 1이거나, 아예 없을 것을 요구한다. 이로써 데이터를 명시적으로 복제하지 않아도 되어 메모리 효율이 좋고 빠르다.

## 연습문제

**연습문제 1.**
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


**연습문제 2.**
`retain_graph=True` 없이 같은 계산 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    기본적으로 PyTorch는 메모리를 아끼기 위해 `.backward()` 후에 계산 그래프를 해제한다. `.backward()`를 두 번째로 호출하면 더 이상 존재하지 않는 그래프를 훑으려 하므로 `RuntimeError`가 발생한다. `retain_graph=True`로 두면 그래프가 메모리에 남아 재사용할 수 있지만, 모든 중간 텐서가 할당된 채로 남으므로 메모리 소비가 늘어난다.

---


**연습문제 3.**
잎 텐서 `w`를 만들고 손실을 계산한 뒤, 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하는 코드를 작성하라. 관찰된 값을 설명하라.

??? success "연습문제 3 풀이"
    ```python
    import torch

    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        print(f'After backward {i+1}: w.grad = {w.grad}')
    # 출력: 4.0, 8.0, 12.0
    # 경사가 누적된다. 매 backward가 기존 경사에 2*w = 4.0을 더한다.
    ```

## 정리하며

**다룬 것** — 팩토리 함수

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다.

앞의 연습문제 3개로 직접 확인할 수 있다.
