# NumPy로 변환하기

autograd를 사용하는 텐서를 NumPy로 전달할 때 그래프 연결, 장치, 저장 공간을 각각 확인한다. 예제는 원본과 독립적인 복사본도 만든다.

## 1. 코드

```python
"""넘파이로 바꾸기."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():
    t = torch.randn(4, requires_grad=True)

    # ----------------------------------------------------------
    # 경우 1: requires_grad=True
    # ----------------------------------------------------------
    # 경사가 필요한 텐서에는 .numpy()가 허용되지 않는다.
    # 이유: NumPy에는 autograd가 없다. 뷰를 노출하면 autograd 몰래 값을
    # 바꿀 수 있게 된다. PyTorch는 버그를 막기 위해 이를 차단한다.
    #
    # 참고: .numpy()는 텐서가 CPU에 있을 것도 요구한다. CUDA/MPS를 쓴다면
    # 먼저 .cpu()를 호출한다(아래 참고).
    try:
        _ = t.numpy()
    except Exception as e:
        print("t.numpy() fails (requires_grad=True):", e)

    # 같은 저장소의 NumPy 뷰를 얻는 올바른 절차:
    #   1) .detach()  → autograd 이력을 버린다 (requires_grad=False, grad_fn=None)
    #   2) (텐서가 CPU에 있으면 생략 가능) .cpu()  → CPU로 옮긴다. NumPy 배열은 CPU 전용이다
    #   3) .numpy()   → 무복사 뷰(텐서와 메모리를 공유한다)
    t_cpu_np = t.detach().clone().cpu().numpy()
    print("Detached .cpu().numpy() shape:", t_cpu_np.shape, "| dtype:", t_cpu_np.dtype)

    # ----------------------------------------------------------
    # 경우 2: requires_grad=False
    # ----------------------------------------------------------
    # 텐서가 경사를 필요로 하지 않고 이미 CPU에 있다면,
    # .numpy()가 바로 동작하며 뷰를 반환한다(메모리 공유).
    t2 = torch.randn(4, requires_grad=False)  # CPU by default

    # 같은 저장소의 NumPy 뷰를 얻는 올바른 절차:
    #   1) (requires_grad=False이면 생략 가능) .detach()  → autograd 이력을 버린다 (requires_grad=False, grad_fn=None)
    #   2) (텐서가 CPU에 있으면 생략 가능) .cpu()  → CPU로 옮긴다. NumPy 배열은 CPU 전용이다
    #   3) .numpy()   → 무복사 뷰(텐서와 메모리를 공유한다)
    t2_np = t2.numpy()
    print("t2.requires_grad:", t2.requires_grad,
          "| numpy() works directly, shape:", t2_np.shape, "| dtype:", t2_np.dtype)

    # ----------------------------------------------------------
    # 참고(중요!)
    # ----------------------------------------------------------
    # clone() → 데이터와 메타데이터를 복사한다
    #   데이터: 새 저장소에 데이터를 복제한다(공유 없음).
    #   경사: requires_grad와 grad_fn을 복제한다. 원본이 requires_grad=True이면 복제본도 경사 추적을 유지하며 그래프에 연결된다(grad_fn=CloneBackward를 가진다).
    #   쓰는 때: 독립적인 텐서가 필요하지만 경사는 여전히 원본으로 되돌아 흐르기를 원할 때.
    # detach() → 데이터는 공유하되 메타데이터를 초기화한다
    #   데이터: 원본과 같은 저장소를 공유한다(새 할당 없음).
    #   경사: requires_grad=False, grad_fn=None으로 설정하여 autograd를 멈춘다. 반환된 텐서는 requires_grad=False, grad_fn=None이다.
    #   쓰는 때: 원본의 그래프를 깨뜨리지 않고 경사와 무관한 작업(NumPy, 기록 등)을 위해 데이터의 뷰가 필요할 때.
    # detach_() → 메타데이터를 제자리에서 초기화한다
    #   requires_grad=False, grad_fn=None으로 바꾸어 더 이상 경사를 추적하지 않게 한다.
    #   많은 텐서(잎이든 아니든)에 허용되지만 주의해야 한다. 그래프에서 아직 쓰이는 텐서에 이렇게 하면 경사 흐름이 조용히 끊길 수 있다.

if __name__ == "__main__":
    main()
```

**출력:**

```
t.numpy() fails (requires_grad=True): Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
Detached .cpu().numpy() shape: (4,) | dtype: float32
t2.requires_grad: False | numpy() works directly, shape: (4,) | dtype: float32
```

## 2. 논의

### 변환 전에 확인할 조건

일반적인 `.numpy()` 호출은 CPU에 있고 경사를 추적하지 않는 등 변환 조건을 만족하는 텐서에 사용한다. 예제의 `t`는 경사를 추적하므로 직접 변환하면 오류가 난다. `detach()`로 연결을 끊고 `cpu()`로 CPU에 옮기면 NumPy에서 사용할 수 있다.

### 복사가 필요한 경우

예제는 `t.detach().clone().cpu().numpy()`를 사용한다. `clone()`이 원본과 독립적인 저장 공간을 만들므로, 반환한 배열을 수정해도 원래 `t`의 값은 바뀌지 않는다. 반면 CPU 텐서에 `detach().numpy()`만 사용하면 저장 공간이 공유될 수 있다. `cpu()` 역시 이미 CPU에 있는 텐서에 항상 복사를 수행하는 것은 아니다.

NumPy에서 수행한 연산은 PyTorch의 계산 그래프에 기록되지 않는다. 변환 후 계산한 값을 다시 텐서로 만든다고 원래 미분 경로가 복구되지는 않는다. 학습 로스까지 미분해야 하는 계산은 PyTorch 연산 안에 유지한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
텐서를 하나 만들고 `clone()` 복사본과 `detach()` 복사본을 각각 만들어라. 원본을 제자리에서 수정한 뒤 어느 복사본이 영향을 받는지 보여라.

</div>

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


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`clone()`만 쓰거나 `detach()`만 쓰는 경우와 비교하여 `detach().clone()`을 언제 쓰는지 설명하라.

</div>

??? success "연습문제 2 풀이"
    `clone()`만 쓰면 데이터는 복사하지만 autograd 그래프 연결은 유지된다. `detach()`만 쓰면 저장소는 공유하되 그래프에서 떨어져 나온다. `detach().clone()`은 그래프 연결이 없는 독립적인 복사본을 준다. 학습에 영향을 주지 않으면서 기록, 비교, 직렬화를 위한 스냅숏을 만들 때 적합하다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
슬라이스를 수정한 뒤 원본을 확인하여, 텐서를 슬라이싱하면 뷰(저장소 공유)가 만들어짐을 보여라.

</div>

??? success "연습문제 3 풀이"
    ```python
    x = torch.tensor([1., 2., 3., 4., 5.])
    s = x[1:4]  # 뷰
    s[0] = 99.
    print(x)  # tensor([ 1., 99.,  3.,  4.,  5.])
    # 슬라이스가 저장소를 공유하므로 원본이 바뀌었다.
    ```

## 정리하며

NumPy 변환에서는 그래프 연결을 끊을지, CPU 이동이 필요한지, 독립 복사본이 필요한지를 따로 판단한다. NumPy 계산은 원래 autograd 경로를 이어주지 않는다.
