# 리스트를 텐서로

이 스크립트는 리스트를 텐서로 바꾸는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""목록에서 텐서로."""
import torch

# ========================================================================
# 메인
# ========================================================================

def print_info(t):
    # 텐서의 흔한 속성들을 살펴보는 도우미 함수.
    # - t            : 값을 출력한다(PyTorch는 dtype에 따라 보기 좋게 출력한다)
    # - t.shape      : 계수/길이를 보여준다. []는 0차원 스칼라, [N]은 1차원, [R,C]는 2차원 등이다.
    # - t.dtype      : 명시하지 않으면 추론된다(기본적으로 정수→int64, 실수→float32)
    # - requires_grad: autograd 추적 플래그(기본값 False. 실수/복소수에만 의미가 있다)
    print(f"{t = }", f"{t.shape = }", f"{t.dtype = }", f"{t.requires_grad = }", sep="\n", end="\n\n")

def main():
    # --------------------------------------------
    # 1) 1차원 파이썬 리스트  →  1차원 텐서 (COPY)
    # --------------------------------------------
    # torch.tensor(...)는 파이썬 수열의 데이터를 완전히 새로운 텐서로 **복사한다**.
    # 결과 dtype이 추론된다. 실수 리스트는 기본적으로 float32가 된다.
    list1 = [1.0, 2.0, 3.0]
    t1 = torch.tensor(list1)   # copy data from list → independent storage
    print_info(t1)
    # 기댓값: tensor([1., 2., 3.])   torch.Size([3])   torch.float32

    # --------------------------------------------
    # 2) 중첩된 (직사각형) 리스트  →  다차원 텐서
    # --------------------------------------------
    # 모든 내부 리스트의 **길이가 같아야** 한다(직사각형). 아니면 들쭉날쭉하다.
    # 정수 값은 기본적으로 dtype=int64로 추론된다.
    list2 = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.tensor(list2)
    print_info(t2)
    # 기댓값: tensor([[1, 2, 3],
    #                 [4, 5, 6]])   torch.Size([2, 3])   torch.int64

    # --------------------------------------------
    # 3) 명시적 dtype 지정
    # --------------------------------------------
    # 추론을 덮어쓸 수 있다. 여기서는 float64(배정밀도)를 강제한다.
    t3 = torch.tensor(list1, dtype=torch.float64)
    print_info(t3)

    # --------------------------------------------
    # 4) 리스트에 여러 데이터형이 섞이면  →  dtype 승격
    # --------------------------------------------
    # PyTorch는 모든 값을 표현할 수 있는 공통 dtype으로 승격한다.
    # 정수 + 실수 → 실수(따로 강제하지 않으면 기본은 float32).
    list4 = [1, 2.5, 3]  # int + float
    t4 = torch.tensor(list4)  # auto-promotes to float
    print_info(t4)
    # 기댓값 dtype: torch.float32

    # --------------------------------------------
    # 5) 빈 리스트  →  빈 1차원 텐서(길이 0)
    # --------------------------------------------
    # 기본 실수 dtype(float32). 모양은 [0]이다.
    empty_list = []
    t5 = torch.tensor(empty_list)
    print_info(t5)
    # 기댓값: tensor([])   torch.Size([0])   torch.float32

    # --------------------------------------------
    # 6) 불리언 리스트  →  torch.bool 텐서
    # --------------------------------------------
    # 마스크와 인덱싱에 유용하다.
    bool_list = [True, False, True]
    t6 = torch.tensor(bool_list)
    print_info(t6)
    # 기댓값: tensor([ True, False,  True])   dtype=torch.bool

    # --------------------------------------------
    # 7) 들쭉날쭉한(직사각형이 아닌) 중첩 리스트는 오류를 낸다
    # --------------------------------------------
    # 내부 리스트의 길이가 다르면 → PyTorch가 제대로 된 텐서 모양을 만들 수 없다.
    try:
        ragged = [[1, 2], [3, 4, 5]]
        torch.tensor(ragged)  # inconsistent inner lengths → ValueError
    except Exception as e:
        print("Ragged list error:", e)

    # ---------------------- 추가 참고 사항 ----------------------
    # • NumPy에서의 COPY와 SHARE:
    #     - torch.tensor(np_array)      → **항상 복사한다**(새로운 독립 저장소).
    #     - torch.as_tensor(np_array)   → **복사를 피하려 한다**(흔히 from_numpy처럼 공유하며,
    #                                    dtype/스트라이드/쓰기 가능 여부가 허용하면 공유, 아니면 복사).
    #     - torch.from_numpy(np_array)  → **항상 공유한다**(복사 없음. 변경이 양쪽에 반영된다).
    # • **파이썬 리스트/튜플**을 쓸 때(NumPy가 아닐 때):
    #     - torch.tensor(list_like)     → 복사한다(위에서 쓴 대로).
    #     - torch.as_tensor(list_like)  → 여전히 복사한다(공유할 것이 없다).
    # • 위의 모든 생성에서 requires_grad의 기본값은 False이다. 역전파를 원한다면 실수
    #   역전파를 위해 autograd가 연산을 추적하기를 원한다면 텐서에 설정한다.
    # • 장치 배치를 위해서는 텐서 생성 시 device=...를 넘긴다(예: device='cuda').

if __name__ == "__main__":
    main()
```

**출력:**

```
t = tensor([1., 2., 3.])
t.shape = torch.Size([3])
t.dtype = torch.float32
t.requires_grad = False

t = tensor([[1, 2, 3],
        [4, 5, 6]])
t.shape = torch.Size([2, 3])
t.dtype = torch.int64
t.requires_grad = False

t = tensor([1., 2., 3.], dtype=torch.float64)
t.shape = torch.Size([3])
t.dtype = torch.float64
t.requires_grad = False

t = tensor([1.0000, 2.5000, 3.0000])
t.shape = torch.Size([3])
t.dtype = torch.float32
t.requires_grad = False

t = tensor([])
t.shape = torch.Size([0])
t.dtype = torch.float32
t.requires_grad = False

t = tensor([ True, False,  True])
t.shape = torch.Size([3])
t.dtype = torch.bool
t.requires_grad = False

Ragged list error: expected sequence of length 2 at dim 1 (got 3)
```

## 2. 논의

CPU 텐서에서 PyTorch와 NumPy의 상호 운용은 매끄럽다. `torch.from_numpy()`는 배열과 메모리를 공유하는 텐서를 만들고, `torch.tensor()`는 항상 복사한다. 어떤 연산이 저장소를 공유하고 어떤 연산이 독립적인 복사본을 만드는지 이해하는 것이 미묘한 버그를 피하는 데 결정적이다.

GPU 가속은 텐서 연산, 특히 신경망 계산을 지배하는 행렬 곱에 대해 몇 자릿수의 속도 향상을 제공한다. `.to(device)`로 텐서와 모델을 GPU로 옮기는 것은 간단하지만, 성능을 유지하려면 CPU-GPU 사이의 데이터 전송을 최소화하는 것이 결정적이다.

PyTorch는 (저장소를 공유하는 뷰를 반환하는) 기본 슬라이싱과 (복사본을 반환하는) 불리언 마스크나 정수 배열을 이용한 고급 인덱싱을 모두 지원한다. 이 구분을 이해하는 것은 메모리 효율을 위해서도, 인덱싱한 결과를 수정할 때 의도치 않은 부작용을 피하기 위해서도 중요하다.

## 연습문제

**연습문제 1.**
$5 \times 5$ 행렬을 만들고 불리언 마스킹으로 0.5보다 큰 모든 원소를 뽑아내라. 결과는 뷰인가 복사본인가?

??? success "연습문제 1 풀이"
    ```python
    m = torch.rand(5, 5)
    selected = m[m > 0.5]
    print(selected)
    # 불리언 인덱싱은 뷰가 아니라 항상 복사본(COPY)을 반환한다.
    # `selected`를 수정해도 `m`에는 영향이 없다.
    ```

---


**연습문제 2.**
기본 슬라이싱(예: `a[1:3]`)과 정수 배열 인덱싱(예: `a[torch.tensor([1,2])]`)의 차이를 뷰와 복사본의 관점에서 설명하라.

??? success "연습문제 2 풀이"
    기본 슬라이싱은 원래 텐서와 저장소를 공유하는 뷰(VIEW)를 반환한다. 슬라이스를 바꾸면 원본에 영향을 준다. 정수 배열 인덱싱은 독립적인 저장소를 가진 복사본(COPY)을 반환한다. 인덱싱한 결과를 바꾸어도 원본에는 영향이 없다. 이 구분은 정확성과 메모리 효율 모두에 중요하다.

---


**연습문제 3.**
`torch.where`를 사용하여 조각별 함수를 구현하라. $x > 0$일 때 $f(x) = x^2$이고 그 밖에는 $f(x) = 0$이다.

??? success "연습문제 3 풀이"
    ```python
    x = torch.tensor([-3., -1., 0., 1., 3.])
    f_x = torch.where(x > 0, x**2, torch.zeros_like(x))
    print(f_x)  # tensor([0., 0., 0., 1., 9.])
    ```

## 정리하며

**다룬 것** — 리스트를 텐서로

CPU 텐서에서 PyTorch와 NumPy의 상호 운용은 매끄럽다.

앞의 연습문제 3개로 직접 확인할 수 있다.
