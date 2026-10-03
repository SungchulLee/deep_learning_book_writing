# 리스트를 텐서로

리스트를 텐서로 감쌀 때 PyTorch는 두 가지를 리스트의 **내용을 보고** 정한다. 모양과 자료형이다. 그래서 리스트 안에 무엇이 섞여 있느냐가 결과를 바꾼다. 이 쪽은 섞인 자료형이 어떻게 하나로 승격되는지, 그리고 길이가 들쭉날쭉한 리스트가 왜 텐서가 될 수 없는지 본다.

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
    # • 위의 모든 생성에서 requires_grad의 기본값은 False이다. 역전파가 필요하면
    #   실수 텐서에 requires_grad=True를 직접 준다.
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

**텐서는 한 가지 자료형만 담는다.** 파이썬 리스트는 정수와 실수와 참거짓을 한자리에 섞어 담을 수 있지만 텐서는 못 한다. 그래서 섞인 리스트를 주면 PyTorch가 **모두 담을 수 있는 가장 좁은 자료형**을 골라 전부 그리로 올린다. 이것을 승격(promotion)이라 한다.

| 리스트 | 자료형 | 왜 |
|---|---|---|
| `[1, 2, 3]` | `int64` | 모두 정수다 |
| `[1.0, 2.0]` | `float32` | 모두 실수다 |
| `[1, 2.5, 3]` | `float32` | 실수 하나가 정수들을 끌어올린다 |
| `[True, False]` | `bool` | 모두 참거짓이다 |
| `[True, 1]` | `int64` | 참거짓이 정수로 올라간다 |
| `[True, 1.5]` | `float32` | 둘 다 실수로 올라간다 |

올라가는 길은 한 방향이다. $\text{bool} \to \text{int64} \to \text{float32}$. 내려가는 일은 없다 — 실수 하나가 섞이면 리스트 전체가 실수가 된다. 정수 색인으로 쓸 자리에 실수가 하나 끼어 있으면 그 자리에서 터지지 않고 **텐서를 만들 때 조용히** 전부 실수가 되어 버린다.

**빈 리스트는 미룰 거리가 없다.** `torch.tensor([])`는 `float32`다. 리스트에 아무것도 없으니 보고 정할 것이 없어서 기본 실수형으로 떨어진다. 곧 `int64`를 기대하고 빈 리스트로 시작했다면 자료형이 어긋난 채로 출발한다. 모양도 눈여겨볼 만하다 — `[]`는 `torch.Size([0])`이고 `[[], []]`는 `torch.Size([2, 0])`이다. 원소가 없어도 계수는 남는다.

**텐서는 직사각형이어야 한다.** 안쪽 리스트의 길이가 다르면 모양을 정할 수 없다. `[[1, 2], [3, 4, 5]]`에서 둘째 차원의 길이는 2인가 3인가 — 답이 없으므로 `ValueError`다. 모양이란 "각 차원의 길이 하나"이고, 들쭉날쭉한 자료에는 그 하나가 존재하지 않는다. 길이가 다른 자료를 담아야 한다면 짧은 쪽을 채워 길이를 맞추거나, 텐서 하나로 만들지 않고 텐서의 리스트로 둔다.

!!! note "리스트에서는 `tensor`와 `as_tensor`가 같다"
    `torch.tensor`는 늘 복사하고 `torch.as_tensor`는 되도록 복사를 피한다. 그런데 **파이썬 리스트에서는 둘이 똑같이 복사한다.** 리스트는 텐서가 쓰는 꼴로 메모리에 놓여 있지 않아서 공유할 것이 없기 때문이다. 이 구별이 뜻을 갖는 것은 NumPy 배열에서다 — `from_numpy`는 메모리를 공유하므로 한쪽을 고치면 다른 쪽도 바뀐다. 「NumPy 배열을 텐서로」에서 다룬다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
아래 다섯 리스트를 텐서로 만들었을 때의 `dtype`을 찍어 보기 전에 적어 보고, 확인하라.

```python
[1, 2, 3]      [1.0, 2.0]      [1, 2.5, 3]      [True, False]      [True, 1]
```

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    for L in [[1, 2, 3], [1.0, 2.0], [1, 2.5, 3], [True, False], [True, 1]]:
        print(f"{str(L):16} -> {torch.tensor(L).dtype}")
    ```

    ```
    [1, 2, 3]        -> torch.int64
    [1.0, 2.0]       -> torch.float32
    [1, 2.5, 3]      -> torch.float32
    [True, False]    -> torch.bool
    [True, 1]        -> torch.int64
    ```

    텐서는 한 가지 자료형만 담으므로, 섞인 리스트는 **모두 담을 수 있는 자료형**으로
    올라간다. 승격은 $\text{bool} \to \text{int64} \to \text{float32}$ 한 방향이다.
    `[1, 2.5, 3]`에서 실수 하나가 정수 둘을 끌어올리고, `[True, 1]`에서 `True`가
    정수로 올라간다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`torch.tensor([])`의 자료형은 무엇인가? 리스트가 비어 있어 미룰 거리가 없는데 어떻게 자료형이 정해지는지 설명하라. 또 `torch.tensor([])`와 `torch.tensor([[], []])`의 모양을 비교하고, 원소가 하나도 없는데도 둘이 다른 까닭을 밝혀라.

</div>

??? success "연습문제 2 풀이"
    ```python
    import torch

    for e in ["torch.tensor([])", "torch.tensor([[], []])"]:
        t = eval(e)
        print(f"{e:24} shape={tuple(t.shape)!s:8} dtype={t.dtype} numel={t.numel()}")
    ```

    ```
    torch.tensor([])         shape=(0,)    dtype=torch.float32 numel=0
    torch.tensor([[], []])   shape=(2, 0)  dtype=torch.float32 numel=0
    ```

    **자료형**은 `float32`다. 미룰 값이 없으면 PyTorch는 기본 실수형으로 떨어진다.
    따라서 정수 텐서를 쌓아 갈 생각으로 빈 리스트에서 출발하면 자료형이 처음부터
    어긋나 있다. 그럴 때는 `torch.tensor([], dtype=torch.int64)`처럼 직접 밝혀야 한다.

    **모양**은 다르다. 원소 개수는 둘 다 0이지만, `[]`는 "길이 0인 1차원"이고
    `[[], []]`는 "길이 0인 벡터가 둘 있는 2차원"이다. 대괄호가 알려 주는 계수는
    원소가 없어도 그대로 남는다. 모양의 어느 자리에 0이 있느냐가 뒤에 무엇을 이어
    붙일 수 있는지를 정하므로 이 차이가 중요하다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`torch.tensor([[1, 2], [3, 4, 5]])`는 오류를 낸다. 오류의 갈래와 메시지를 적고, **모양이란 무엇인가**에 비추어 왜 이것이 고칠 수 없는 요구인지 설명하라. 그리고 이 자료를 다루는 두 가지 길을 코드로 보여라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch

    try:
        torch.tensor([[1, 2], [3, 4, 5]])
    except ValueError as e:
        print(type(e).__name__, "-", e)
    ```

    ```
    ValueError - expected sequence of length 2 at dim 1 (got 3)
    ```

    텐서의 모양은 **차원마다 길이 하나**다. 첫 줄은 둘째 차원의 길이가 2라고 말하고
    둘째 줄은 3이라고 말한다. 둘 다 맞는 모양은 없다. 메시지도 그대로 그 말이다 —
    1번 차원에서 길이 2를 기대했는데 3이 왔다. 그러므로 이것은 PyTorch의 모자람이
    아니라 요구 자체가 성립하지 않는 경우다. 텐서는 직사각형이어야 한다.

    두 가지 길이 있다. 첫째, **짧은 쪽을 채워** 직사각형으로 만든다.

    ```python
    from torch.nn.utils.rnn import pad_sequence

    rag = [[1, 2], [3, 4, 5]]
    seqs = [torch.tensor(r) for r in rag]
    print(pad_sequence(seqs, batch_first=True))
    ```

    ```
    tensor([[1, 2, 0],
            [3, 4, 5]])
    ```

    둘째, **하나로 합치지 않고** 텐서의 리스트로 둔다.

    ```python
    print([torch.tensor(r) for r in rag])
    ```

    ```
    [tensor([1, 2]), tensor([3, 4, 5])]
    ```

    고르는 기준은 채워 넣은 0이 셈에 섞여도 괜찮은지다. 길이가 들쭉날쭉한 문장을 다룰 때 이 선택이 다시 나온다.

## 정리하며

리스트를 텐서로 감쌀 때 PyTorch는 리스트의 **내용을 보고** 모양과 자료형을 정한다. 거기서 제약 둘이 나온다.

- **자료형은 하나뿐이다.** 섞인 리스트는 모두 담을 수 있는 자료형으로 올라간다. $\text{bool} \to \text{int64} \to \text{float32}$ 한 방향이고, 실수 하나가 리스트 전체를 실수로 만든다.
- **모양은 직사각형이어야 한다.** 차원마다 길이가 하나여야 하므로 안쪽 길이가 다르면 `ValueError`다. 채워서 맞추거나, 텐서 하나로 만들지 않는다.

빈 리스트는 둘 다 걸린다. 미룰 값이 없어 `float32`로 떨어지고, 계수는 대괄호가 알려 주는 대로 남아 `[]`는 `(0,)`, `[[], []]`는 `(2, 0)`이 된다.

리스트에서는 `torch.tensor`와 `torch.as_tensor`가 똑같이 복사한다. 공유할 메모리가 없기 때문이다. 그 구별은 NumPy 배열에서 뜻을 갖는다.
