# NumPy로 변환하기

이 스크립트는 텐서를 NumPy로 변환하는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

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

## 2. 논의

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

CPU 텐서에서 PyTorch와 NumPy의 상호 운용은 매끄럽다. `torch.from_numpy()`는 배열과 메모리를 공유하는 텐서를 만들고, `torch.tensor()`는 항상 복사한다. 어떤 연산이 저장소를 공유하고 어떤 연산이 독립적인 복사본을 만드는지 이해하는 것이 미묘한 버그를 피하는 데 결정적이다.

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다. `torch.zeros`, `torch.randn`, `torch.arange` 같은 팩토리 함수는 `dtype`, `device`, `requires_grad` 매개변수를 받으므로 불필요한 복사 없이 목표 장치에 곧바로 할당할 수 있다.

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

## 정리하며

**다룬 것** — NumPy로 변환하기

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
