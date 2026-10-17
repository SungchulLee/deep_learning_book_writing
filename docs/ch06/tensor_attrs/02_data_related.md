# 데이터 관련 속성

텐서의 데이터와 자동 미분 연결은 따로 관리된다. `clone`·`detach`·NumPy 변환이 저장소와 그래프에 각각 어떤 영향을 주는지 살펴본다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
텐서의 데이터 쪽 잔손질 도구.

Covers:
- item(), tolist()
- detach()과 detach().clone()
- .cpu().numpy()(그리고 detach/cpu이 왜 종요로운가)
- .data의 조심할 점(자동 미분을 건너뛴다)
- CUDA에서 바꿀 때의 기본 함정
"""

import torch

# ========================================================================
# 메인
# ========================================================================

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(0)

    # -------------------------------------------------------------------------
    header("Scalars and Python numbers: .item()")
    s = torch.tensor(3.14159)  # 0-dim tensor
    print("s:", s, "| s.item():", s.item())

    # -------------------------------------------------------------------------
    header("Lists: .tolist()")
    v = torch.tensor([[1., 2.], [3., 4.]])
    print("v:\n", v, "\nv.tolist():", v.tolist())

    # -------------------------------------------------------------------------
    header("detach() vs detach().clone() (storage alias vs snapshot)")
    a = torch.randn(4, requires_grad=True)
    b = a.detach()             # shares storage, no grad
    c = a.detach().clone()     # copy to new storage, no grad
    # no_grad로 감싼다. a는 requires_grad=True인 잎이라 제자리 변경이
    # 그대로는 허용되지 않는다
    with torch.no_grad():
        a.add_(10)
    print("a:", a)
    print("b (shares storage, updated with a):", b)
    print("c (snapshot, unchanged):", c)

    # -------------------------------------------------------------------------
    header("NumPy interop: .cpu().numpy()")
    x = torch.randn(5, requires_grad=True)
    x_np = x.detach().cpu().numpy()       # detach if requires_grad=True
    print("x_np (shape, dtype):", x_np.shape, x_np.dtype)

    # GPU 예제(조건부 실행)
    if torch.cuda.is_available():
        g = torch.randn(3, device="cuda")
        try:
            _ = g.numpy()
        except Exception as e:
            print("CUDA tensor .numpy() fails (expected):", e)
        print("g.cpu().numpy() works shape:", g.cpu().numpy().shape)
    else:
        print("CUDA not available; skipping GPU numpy demo.")

    # -------------------------------------------------------------------------
    header(".data caveat: bypasses autograd (use sparingly)")
    q = torch.tensor([1., 2., 3.], requires_grad=True)
    print("Before .data in-place, q:", q)
    q.data.mul_(1000.)  # no autograd tracking
    print("After  .data in-place,  q:", q)

if __name__ == "__main__":
    main()
```

**출력:**

```
================================================================================
Scalars and Python numbers: .item()
================================================================================
s: tensor(3.1416) | s.item(): 3.141590118408203

================================================================================
Lists: .tolist()
================================================================================
v:
 tensor([[1., 2.],
        [3., 4.]]) 
v.tolist(): [[1.0, 2.0], [3.0, 4.0]]

================================================================================
detach() vs detach().clone() (storage alias vs snapshot)
================================================================================
a: tensor([11.5410,  9.7066,  7.8212, 10.5684], requires_grad=True)
b (shares storage, updated with a): tensor([11.5410,  9.7066,  7.8212, 10.5684])
c (snapshot, unchanged): tensor([ 1.5410, -0.2934, -2.1788,  0.5684])

================================================================================
NumPy interop: .cpu().numpy()
================================================================================
x_np (shape, dtype): (5,) float32
CUDA not available; skipping GPU numpy demo.

================================================================================
.data caveat: bypasses autograd (use sparingly)
================================================================================
Before .data in-place, q: tensor([1., 2., 3.], requires_grad=True)
After  .data in-place,  q: tensor([1000., 2000., 3000.], requires_grad=True)
```

## 2. 논의

`clone()`은 새 저장소를 만들지만 추적 중인 입력과의 미분 연결은 유지한다. `detach()`는 연결을 끊되 저장소를 공유한다. 독립적인 기록이 필요하면 `detach().clone()`을 사용한다.

`.data`를 통한 수정은 autograd의 변경 검사를 우회할 수 있어 올바른 갱신 방법으로 삼지 않는다. 파라메터 수정은 옵티마이저나 `no_grad` 안에서 수행한다. NumPy와 공유한 데이터를 바꿀 때도 원래 텐서의 역전파에 필요한 값이 바뀔 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
텐서를 하나 만들고 `clone()` 복사본과 `detach()` 복사본을 각각 만들어라. 원본을 제자리에서 수정한 뒤 어느 복사본이 영향을 받는지 보여라.

</div>

??? success "풀이"
    ```python
    original = torch.tensor([1., 2., 3.], requires_grad=True)
    cloned = original.clone()
    detached = original.detach()
    original.add_(10)  # 제자리(no_grad 안에서)
    print(cloned)    # 변하지 않음(독립적인 저장소)
    print(detached)  # 변함(저장소를 공유한다)
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`clone()`만 쓰거나 `detach()`만 쓰는 경우와 비교하여 `detach().clone()`을 언제 쓰는지 설명하라.

</div>

??? success "풀이"
    `clone()`만 쓰면 데이터는 복사하지만 autograd 그래프 연결은 유지된다. `detach()`만 쓰면 저장소는 공유하되 그래프에서 떨어져 나온다. `detach().clone()`은 그래프 연결이 없는 독립적인 복사본을 준다. 학습에 영향을 주지 않으면서 기록, 비교, 직렬화를 위한 스냅숏을 만들 때 적합하다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
슬라이스를 수정한 뒤 원본을 확인하여, 텐서를 슬라이싱하면 뷰(저장소 공유)가 만들어짐을 보여라.

</div>

??? success "풀이"
    ```python
    x = torch.tensor([1., 2., 3., 4., 5.])
    s = x[1:4]  # 뷰
    s[0] = 99.
    print(x)  # tensor([ 1., 99.,  3.,  4.,  5.])
    # 슬라이스가 저장소를 공유하므로 원본이 바뀌었다.
    ```

## 정리하며

텐서의 데이터와 자동 미분 연결은 따로 관리된다. `clone`·`detach`·NumPy 변환이 저장소와 그래프에 각각 어떤 영향을 주는지 살펴본다.
