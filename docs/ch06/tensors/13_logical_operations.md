# 논리 연산 - 불리언 연산과 마스킹

조건을 적는 일은 파이썬과 가장 많이 어긋나는 자리다. `and`와 `or`와 `not`이 **텐서에서는 쓸 수 없고**, 그 자리에 쓰는 `&`와 `|`와 `~`는 우선순위가 달라서 괄호를 빠뜨리면 엉뚱한 곳에서 오류가 난다. 실수를 `==`로 견주는 것도 믿을 수 없다. 이 쪽은 그 세 가지를 본다.

## 1. 코드

```python
"""튜토리얼 13: 논리 셈 - 참거짓 셈과 가리기"""
import torch

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Comparison Operations")
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([5, 4, 3, 2, 1])
    print(f"a = {a}\nb = {b}\n")
    print(f"a > b: {a > b}")
    print(f"a == b: {a == b}")
    print(f"torch.eq(a, b): {torch.eq(a, b)}")
    print(f"torch.gt(a, b): {torch.gt(a, b)}")
    
    header("2. Logical Operations - AND, OR, NOT")
    x = torch.tensor([True, True, False, False])
    y = torch.tensor([True, False, True, False])
    print(f"x = {x}\ny = {y}\n")
    print(f"x & y (AND): {x & y}")
    print(f"x | y (OR): {x | y}")
    print(f"~x (NOT): {~x}")
    print(f"x ^ y (XOR): {x ^ y}")
    print(f"torch.logical_and(x, y): {torch.logical_and(x, y)}")
    
    header("3. Boolean Masking")
    data = torch.tensor([10, 20, 5, 30, 15])
    print(f"Data: {data}")
    mask = data > 15
    print(f"Mask (data > 15): {mask}")
    filtered = data[mask]
    print(f"Filtered data: {filtered}")
    complex_mask = (data > 10) & (data < 25)
    print(f"Complex mask: {complex_mask}")
    print(f"Filtered: {data[complex_mask]}")
    
    header("4. Conditional Selection")
    x = torch.tensor([-2, -1, 0, 1, 2])
    print(f"x = {x}")
    result = torch.where(x > 0, x, torch.zeros_like(x))  # ReLU
    print(f"ReLU (where x>0, x, 0): {result}")
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([10, 20, 30])
    condition = torch.tensor([True, False, True])
    selected = torch.where(condition, a, b)
    print(f"\nSelect from a or b: {selected}")
    
    header("5. Element-wise Comparison")
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.tensor([[2, 2], [2, 4]])
    print(f"x:\n{x}\ny:\n{y}\n")
    print(f"torch.eq(x, y):\n{torch.eq(x, y)}")
    print(f"torch.allclose(x, y): {torch.allclose(x.float(), y.float())}")
    z = torch.tensor([[1.0001, 2.0], [3.0, 4.0]])
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"\nClose values? {torch.allclose(z, w, atol=1e-3)}")
    
    header("6. Practical Example: Data Cleaning")
    data = torch.tensor([1.0, 2.0, float('nan'), 4.0, float('inf')])
    print(f"Raw data: {data}")
    is_finite = torch.isfinite(data)
    print(f"is_finite: {is_finite}")
    clean_data = data[is_finite]
    print(f"Clean data: {clean_data}")
    data_with_outliers = torch.tensor([1, 2, 100, 3, 4, 200])
    mask = (data_with_outliers > 0) & (data_with_outliers < 50)
    cleaned = data_with_outliers[mask]
    print(f"\nOutlier removal: {cleaned}")

if __name__ == "__main__":
    main()
```

**출력:**

```

======================================================================
1. Comparison Operations
======================================================================
a = tensor([1, 2, 3, 4, 5])
b = tensor([5, 4, 3, 2, 1])

a > b: tensor([False, False, False,  True,  True])
a == b: tensor([False, False,  True, False, False])
torch.eq(a, b): tensor([False, False,  True, False, False])
torch.gt(a, b): tensor([False, False, False,  True,  True])

======================================================================
2. Logical Operations - AND, OR, NOT
======================================================================
x = tensor([ True,  True, False, False])
y = tensor([ True, False,  True, False])

x & y (AND): tensor([ True, False, False, False])
x | y (OR): tensor([ True,  True,  True, False])
~x (NOT): tensor([False, False,  True,  True])
x ^ y (XOR): tensor([False,  True,  True, False])
torch.logical_and(x, y): tensor([ True, False, False, False])

======================================================================
3. Boolean Masking
======================================================================
Data: tensor([10, 20,  5, 30, 15])
Mask (data > 15): tensor([False,  True, False,  True, False])
Filtered data: tensor([20, 30])
Complex mask: tensor([False,  True, False, False,  True])
Filtered: tensor([20, 15])

======================================================================
4. Conditional Selection
======================================================================
x = tensor([-2, -1,  0,  1,  2])
ReLU (where x>0, x, 0): tensor([0, 0, 0, 1, 2])

Select from a or b: tensor([ 1, 20,  3])

======================================================================
5. Element-wise Comparison
======================================================================
x:
tensor([[1, 2],
        [3, 4]])
y:
tensor([[2, 2],
        [2, 4]])

torch.eq(x, y):
tensor([[False,  True],
        [False,  True]])
torch.allclose(x, y): False

Close values? True

======================================================================
6. Practical Example: Data Cleaning
======================================================================
Raw data: tensor([1., 2., nan, 4., inf])
is_finite: tensor([ True,  True, False,  True, False])
Clean data: tensor([1., 2., 4.])

Outlier removal: tensor([1, 2, 3, 4])
```

## 2. 논의

**파이썬의 `and`·`or`·`not`은 텐서에서 쓸 수 없다.** 원소별로 셈하는 것이 아니라 텐서 하나를 참거짓 하나로 바꾸려 하기 때문이다. 원소가 둘 이상이면 어느 것을 따라야 할지 알 수 없으니 오류다.

```python
a & b     # tensor([ True, False, False])  ← 원소별
a and b   # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
```

| 뜻 | 텐서 | 파이썬 |
|---|---|---|
| 그리고 | `&` | `and` |
| 또는 | `\|` | `or` |
| 아니다 | `~` | `not` |

같은 오류가 `if`에 텐서를 넣을 때도 난다. `if t:`는 `t`를 참거짓 하나로 바꾸려 하므로, 원소가 여럿이면 `.any()`나 `.all()`로 **무엇을 묻는지 밝혀야** 한다.

!!! warning "`&`는 비교보다 먼저 묶인다"
    `&`·`|`·`~`는 본래 비트 연산자여서 **`>`나 `<`보다 우선순위가 높다.** 그래서 괄호가 없으면 뜻이 뒤집힌다.

    ```python
    x > 1 & x < 4        # (1 & x) 가 먼저 묶인다 → RuntimeError
    (x > 1) & (x < 4)    # 이것이 뜻한 것
    ```

    고약한 점은 오류 메시지가 **우선순위를 말해 주지 않는다**는 것이다. `Boolean value of Tensor ... is ambiguous`라고만 나오므로, `and`를 잘못 쓴 줄로 알고 엉뚱한 곳을 고치게 된다. 조건을 둘 이상 묶을 때는 늘 괄호를 친다.

**`~`는 자료형을 보고 하는 일이 달라진다.** 참거짓 텐서에서는 뒤집지만, 정수 텐서에서는 **비트를 반전**한다.

```python
i = torch.tensor([0, 1, 2])
~i          # tensor([-1, -2, -3])   ← 비트 반전
~(i > 0)    # tensor([ True, False, False])  ← 뜻한 것
```

`~i`는 오류가 아니라 그럴듯한 정수를 돌려주므로 알아채기 어렵다. 뒤집으려면 **먼저 비교해서 참거짓으로 만든 다음** 뒤집는다.

**실수를 `==`로 견주지 않는다.** 같은 수를 다른 길로 셈하면 마지막 자리가 어긋난다.

```python
torch.tensor([2.0]).sqrt() ** 2     # 1.9999998807907104
... == 2.0                          # False
torch.isclose(..., 2.0)             # True
```

더 심한 경우도 있다. `float32`는 유효숫자가 일곱 자리 남짓이라 큰 수에 작은 수를 더하면 **작은 쪽이 아예 사라진다.**

```python
(torch.tensor([1e8]) + 1) - 1e8     # 0.0   ← 1을 더한 적이 없는 셈이 된다
```

`float64`로 같은 셈을 하면 1.0이 나온다. 그러므로 `==`가 `False`라고 해서 값이 다른 것이 아니고, `0.0`이 나왔다고 해서 코드가 틀린 것도 아니다. 실수 비교는 `torch.isclose`나 `torch.allclose`로 **얼마나 가까우면 같다고 볼지** 밝혀서 한다.

**마스크로 고르는 두 길.** `a[mask]`는 **고른 것만** 모아 더 짧은 텐서를 주고, `torch.where(mask, a, b)`는 **모양을 그대로 두고** 자리마다 둘 중 하나를 고른다. 모양을 지켜야 하는 자리에서는 `where`를 쓴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`a = torch.tensor([True, False, True])`와 `b = torch.tensor([True, True, False])`에 대해 `a & b`를 찍어라. 그 다음 `a and b`를 해 보고 오류 메시지를 적어라. 파이썬 키워드가 왜 쓸 수 없는지 설명하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    a = torch.tensor([True, False, True])
    b = torch.tensor([True, True, False])
    print("a & b  :", a & b)
    try:
        a and b
    except RuntimeError as e:
        print("a and b:", e)
    ```

    ```
    a & b  : tensor([ True, False, False])
    a and b: Boolean value of Tensor with more than one value is ambiguous
    ```

    `&`는 원소별로 셈해서 길이 3인 참거짓 텐서를 준다. `and`는 그런 일을 하지 못한다 —
    파이썬의 `and`는 **왼쪽 값이 참인지 거짓인지 먼저 판단한 뒤** 둘 중 하나를
    돌려주는 문법이기 때문이다. 그러려면 텐서 하나가 참거짓 하나로 바뀌어야 하는데,
    원소가 셋이면 어느 것을 따를지 정할 수 없다. 그래서 "모호하다"고 한다.

    같은 이유로 `if t:`도 쓸 수 없다. 무엇을 묻는지 밝혀야 한다.

    ```python
    print("하나라도 참:", a.any().item(), "  모두 참:", a.all().item())
    ```

    ```
    하나라도 참: True   모두 참: False
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`x = torch.tensor([0, 2, 4])`에서 1보다 크고 4보다 작은 원소를 고르려 한다. `x > 1 & x < 4`를 해 보고 무엇이 일어나는지 적어라. 오류 메시지가 **원인을 가리키지 않는** 까닭을 설명하고 바르게 고쳐라.

</div>

??? success "연습문제 2 풀이"
    ```python
    import torch

    x = torch.tensor([0, 2, 4])
    try:
        x > 1 & x < 4
    except RuntimeError as e:
        print("괄호 없이:", e)
    print("괄호 치면:", (x > 1) & (x < 4))
    print("고른 값  :", x[(x > 1) & (x < 4)])
    ```

    ```
    괄호 없이: Boolean value of Tensor with more than one value is ambiguous
    괄호 치면: tensor([False,  True, False])
    고른 값  : tensor([2])
    ```

    `&`는 본래 비트 연산자여서 `>`와 `<`보다 **먼저 묶인다.** 그래서 파이썬은 이 줄을
    이렇게 읽는다.

    ```text
    x > (1 & x) < 4
    ```

    `1 & x`가 먼저 셈해지고, 그 다음 비교가 사슬로 이어져 텐서를 참거짓 하나로 바꾸려
    하므로 "모호하다"는 오류가 난다.

    오류 메시지가 원인을 가리키지 않는 까닭이 여기에 있다. 메시지는 **마지막에 터진
    자리**를 말하는데, 그 자리는 비교 사슬이고 진짜 원인은 우선순위다. 그래서 `and`를
    잘못 쓴 줄로 알고 엉뚱한 곳을 고치게 된다.

    조건을 둘 이상 묶을 때는 **늘 괄호를 친다.** 괄호는 값이 들지 않는 보험이다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
두 가지를 확인하라.

1. `i = torch.tensor([0, 1, 2])`에 `~i`를 해 보라. 참거짓이 뒤집히기를 바랐는데 무엇이 나오는가? 왜 오류가 나지 않는지 설명하고 바르게 고쳐라.
2. `torch.tensor([2.0]).sqrt() ** 2`가 `2.0`과 같은지 `==`로 견주어라. 이어서 `(torch.tensor([1e8]) + 1) - 1e8`을 찍어라. 둘을 묶어, 실수를 `==`로 견주면 안 되는 까닭을 밝혀라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch

    i = torch.tensor([0, 1, 2])
    print("~i      :", ~i)
    print("~(i > 0):", ~(i > 0))
    ```

    ```
    ~i      : tensor([-1, -2, -3])
    ~(i > 0): tensor([ True, False, False])
    ```

    **1.** `~`는 자료형을 보고 하는 일이 달라진다. 참거짓 텐서에서는 뒤집지만 **정수
    텐서에서는 비트를 반전한다.** 두 보수 표현에서 비트를 모두 뒤집으면 $-(n+1)$이
    되므로 $0, 1, 2$가 $-1, -2, -3$이 된다.

    오류가 나지 않는 까닭은 정수에 대한 비트 반전이 **올바른 연산**이기 때문이다.
    PyTorch가 보기에 내가 요청한 일을 해 준 것이다. 그래서 그럴듯한 정수가 돌아오고,
    뒤집혔다고 믿은 채 넘어간다. 뒤집으려면 **먼저 비교해서 참거짓으로 만든 뒤**
    뒤집는다.

    ```python
    import torch

    s = torch.tensor([2.0]).sqrt() ** 2
    print("sqrt(2)**2     :", repr(s.item()))
    print("== 2.0         :", (s == 2.0).item())
    print("isclose(2.0)   :", torch.isclose(s, torch.tensor([2.0])).item())
    print()
    a = (torch.tensor([1e8]) + 1) - 1e8
    print("(1e8+1)-1e8    :", a.item())
    b = (torch.tensor([1e8], dtype=torch.float64) + 1) - 1e8
    print("float64 로 하면:", b.item())
    ```

    ```
    sqrt(2)**2     : 1.9999998807907104
    == 2.0         : False
    isclose(2.0)   : True

    (1e8+1)-1e8    : 0.0
    float64 로 하면: 1.0
    ```

    **2.** 두 경우가 다른 것을 보여 준다.

    앞은 **마지막 자리의 어긋남**이다. $\sqrt{2}$를 실수로 적을 때 이미 잘린 값이고,
    그것을 제곱하면 2에 아주 가깝지만 같지는 않다. 값은 사실상 맞으므로 `isclose`가
    `True`를 준다.

    뒤는 **작은 쪽이 사라진 것**이다. `float32`는 유효숫자가 일곱 자리 남짓이라
    $10^8$ 옆의 1을 담을 자리가 없다. 그래서 더해도 값이 그대로이고, 다시 빼면 0이
    된다. 1을 더한 적이 없는 셈이 되는데 `float64`로 하면 1.0이 나온다.

    묶어서 말하면 **실수에서 `==`는 "값이 같은가"를 묻는 것이 아니라 "비트가 같은가"를
    묻는다.** 비트는 셈하는 길에 따라 달라지므로, 값을 견주려면 얼마나 가까우면 같다고
    볼지 내가 밝혀야 한다.

    ```python
    torch.isclose(a, b)      # 원소별
    torch.allclose(a, b)     # 전부 가까운가 — 참거짓 하나
    ```

## 정리하며

조건을 적는 일은 파이썬과 가장 많이 어긋나는 자리다.

- **`and`·`or`·`not`은 쓸 수 없다.** 텐서를 참거짓 하나로 바꾸려 하므로 원소가 여럿이면 "모호하다"는 오류다. `&`·`|`·`~`를 쓰고, `if`에 넣을 때는 `.any()`나 `.all()`로 무엇을 묻는지 밝힌다.
- **`&`는 비교보다 먼저 묶인다.** 괄호가 없으면 `x > (1 & x) < 4`로 읽힌다. 오류 메시지는 터진 자리만 말하고 우선순위를 알려 주지 않으므로, 조건을 묶을 때는 늘 괄호를 친다.
- **`~`는 자료형을 보고 하는 일이 달라진다.** 정수에서는 비트 반전이라 `-1, -2, -3` 같은 그럴듯한 값이 나오고 오류가 없다. 먼저 비교해서 참거짓으로 만든 뒤 뒤집는다.
- **실수에 `==`를 쓰지 않는다.** 그것은 값이 같은지가 아니라 비트가 같은지를 묻는다. `isclose`·`allclose`로 얼마나 가까우면 같다고 볼지 밝힌다.

마스크로 고를 때는 모양을 보고 고른다. `a[mask]`는 고른 것만 모아 짧아지고, `torch.where(mask, a, b)`는 모양을 지킨다.
