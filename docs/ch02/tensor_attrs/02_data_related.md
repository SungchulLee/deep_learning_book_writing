# 데이터 관련 속성

이 스크립트는 데이터와 관련된 텐서 속성을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
#!/usr/bin/env python3
"""
텐서의 자료 쪽 잔손질 도구.

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
    main()```

## 논의

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

CPU 텐서에서 PyTorch와 NumPy의 상호 운용은 매끄럽다. `torch.from_numpy()`는 배열과 메모리를 공유하는 텐서를 만들고, `torch.tensor()`는 항상 복사한다. 어떤 연산이 저장소를 공유하고 어떤 연산이 독립적인 복사본을 만드는지 이해하는 것이 미묘한 버그를 피하는 데 결정적이다.

GPU 가속은 텐서 연산, 특히 신경망 계산을 지배하는 행렬 곱에 대해 몇 자릿수의 속도 향상을 제공한다. `.to(device)`로 텐서와 모델을 GPU로 옮기는 것은 간단하지만, 성능을 유지하려면 CPU-GPU 사이의 데이터 전송을 최소화하는 것이 결정적이다.

## 연습문제

**연습문제 1.**
텐서를 하나 만들고 `clone()` 복사본과 `detach()` 복사본을 각각 만들어라. 원본을 제자리에서 수정한 뒤 어느 복사본이 영향을 받는지 보여라.

??? success "연습문제 1 풀이"
    ```python
    original = torch.tensor([1., 2., 3.], requires_grad=True)
    cloned = original.clone()
    detached = original.detach()
    original.add_(10)  # 제자리(no_grad 안에서)
    print(cloned)    # 변하지 않음(독립적인 저장소)
    print(detached)  # 변함(저장소를 공유한다)
    ```

---


**연습문제 2.**
`clone()`만 쓰거나 `detach()`만 쓰는 경우와 비교하여 `detach().clone()`을 언제 쓰는지 설명하라.

??? success "연습문제 2 풀이"
    `clone()`만 쓰면 데이터는 복사하지만 autograd 그래프 연결은 유지된다. `detach()`만 쓰면 저장소는 공유하되 그래프에서 떨어져 나온다. `detach().clone()`은 그래프 연결이 없는 독립적인 복사본을 준다. 학습에 영향을 주지 않으면서 기록, 비교, 직렬화를 위한 스냅숏을 만들 때 가장 안전한 선택이다.

---


**연습문제 3.**
슬라이스를 수정한 뒤 원본을 확인하여, 텐서를 슬라이싱하면 뷰(저장소 공유)가 만들어짐을 보여라.

??? success "연습문제 3 풀이"
    ```python
    x = torch.tensor([1., 2., 3., 4., 5.])
    s = x[1:4]  # 뷰
    s[0] = 99.
    print(x)  # tensor([ 1., 99.,  3.,  4.,  5.])
    # 슬라이스가 저장소를 공유하므로 원본이 바뀌었다.
    ```
