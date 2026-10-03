# 텐서 비교 - 원소별 비교와 텐서 비교

"두 텐서가 같은가"라는 물음에 답이 여럿이다. 원소마다 답하는 `==`, 텐서 하나에 참거짓 하나로 답하는 `torch.equal`, 얼마나 가까우면 같다고 볼지 정해 주는 `torch.allclose`가 저마다 다른 것을 묻는다. 게다가 `nan`은 **자기 자신과도 같지 않다.** 이 쪽은 어느 물음에 어느 함수를 쓸지 가린다.

## 1. 코드

```python
"""튜토리얼 17: 텐서 견주기 - 원소별 견줌과 텐서 견줌"""
import torch
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Element-wise Comparison")
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([5, 4, 3, 2, 1])
    print(f"a = {a}\nb = {b}\n")
    print(f"a > b: {a > b}")
    print(f"a >= b: {a >= b}")
    print(f"a < b: {a < b}")
    print(f"a == b: {a == b}")
    print(f"a != b: {a != b}")
    
    header("2. Tensor Equality - torch.equal()")
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([1, 2, 3])
    z = torch.tensor([1, 2, 4])
    print(f"x = {x}\ny = {y}\nz = {z}\n")
    print(f"torch.equal(x, y): {torch.equal(x, y)}")  # True
    print(f"torch.equal(x, z): {torch.equal(x, z)}")  # False
    print("\nNote: equal() requires EXACT match")
    
    header("3. Approximate Equality - torch.allclose()")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0001, 2.0001, 3.0001])
    print(f"a = {a}\nb = {b}\n")
    print(f"equal(): {torch.equal(a, b)}")  # False
    print(f"allclose() default: {torch.allclose(a, b)}")  # True
    print(f"allclose(atol=1e-5): {torch.allclose(a, b, atol=1e-5)}")  # False
    print(f"allclose(atol=1e-3): {torch.allclose(a, b, atol=1e-3)}")  # True
    
    header("4. Finding Matches - torch.eq()")
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b = torch.tensor([[1, 0, 3], [4, 0, 6]])
    print(f"a =\n{a}\nb =\n{b}\n")
    matches = torch.eq(a, b)
    print(f"Element-wise equality:\n{matches}")
    num_matches = matches.sum().item()
    print(f"Number of matching elements: {num_matches}")
    
    header("5. Top-k and Sorting")
    scores = torch.tensor([3.2, 1.5, 4.7, 2.1, 5.3])
    print(f"Scores: {scores}")
    top_k_values, top_k_indices = torch.topk(scores, k=3)
    print(f"Top 3 values: {top_k_values}")
    print(f"Top 3 indices: {top_k_indices}")
    sorted_values, sorted_indices = torch.sort(scores, descending=True)
    print(f"\nSorted (descending): {sorted_values}")
    print(f"Sorted indices: {sorted_indices}")
    
    header("6. Element-wise Max/Min")
    a = torch.tensor([1, 5, 3])
    b = torch.tensor([2, 4, 6])
    print(f"a = {a}\nb = {b}\n")
    max_elem = torch.max(a, b)
    min_elem = torch.min(a, b)
    print(f"Element-wise max: {max_elem}")
    print(f"Element-wise min: {min_elem}")
    
    header("7. Practical: Finding Best Predictions")
    logits = torch.randn(5, 10)  # 5 samples, 10 classes
    print(f"Logits shape: {logits.shape}")
    predictions = torch.argmax(logits, dim=1)
    print(f"Predicted classes: {predictions}")
    max_scores, _ = torch.max(logits, dim=1)
    print(f"Max scores: {max_scores}")

if __name__ == "__main__":
    main()
```

**출력:**

```

======================================================================
1. Element-wise Comparison
======================================================================
a = tensor([1, 2, 3, 4, 5])
b = tensor([5, 4, 3, 2, 1])

a > b: tensor([False, False, False,  True,  True])
a >= b: tensor([False, False,  True,  True,  True])
a < b: tensor([ True,  True, False, False, False])
a == b: tensor([False, False,  True, False, False])
a != b: tensor([ True,  True, False,  True,  True])

======================================================================
2. Tensor Equality - torch.equal()
======================================================================
x = tensor([1, 2, 3])
y = tensor([1, 2, 3])
z = tensor([1, 2, 4])

torch.equal(x, y): True
torch.equal(x, z): False

Note: equal() requires EXACT match

======================================================================
3. Approximate Equality - torch.allclose()
======================================================================
a = tensor([1., 2., 3.])
b = tensor([1.0001, 2.0001, 3.0001])

equal(): False
allclose() default: False
allclose(atol=1e-5): False
allclose(atol=1e-3): True

======================================================================
4. Finding Matches - torch.eq()
======================================================================
a =
tensor([[1, 2, 3],
        [4, 5, 6]])
b =
tensor([[1, 0, 3],
        [4, 0, 6]])

Element-wise equality:
tensor([[ True, False,  True],
        [ True, False,  True]])
Number of matching elements: 4

======================================================================
5. Top-k and Sorting
======================================================================
Scores: tensor([3.2000, 1.5000, 4.7000, 2.1000, 5.3000])
Top 3 values: tensor([5.3000, 4.7000, 3.2000])
Top 3 indices: tensor([4, 2, 0])

Sorted (descending): tensor([5.3000, 4.7000, 3.2000, 2.1000, 1.5000])
Sorted indices: tensor([4, 2, 0, 3, 1])

======================================================================
6. Element-wise Max/Min
======================================================================
a = tensor([1, 5, 3])
b = tensor([2, 4, 6])

Element-wise max: tensor([2, 5, 6])
Element-wise min: tensor([1, 4, 3])

======================================================================
7. Practical: Finding Best Predictions
======================================================================
Logits shape: torch.Size([5, 10])
Predicted classes: tensor([4, 3, 3, 4, 3])
Max scores: tensor([0.8487, 1.2377, 1.8530, 1.0554, 3.4105])
```

## 2. 논의

**세 함수가 서로 다른 것을 묻는다.**

| | 돌려주는 것 | 모양이 다르면 | 값이 조금 다르면 |
|---|---|---|---|
| `a == b` | 참거짓 **텐서** | 펴 맞추어 셈한다 | `False` |
| `torch.equal(a, b)` | 참거짓 **하나** | `False` | `False` |
| `torch.allclose(a, b)` | 참거짓 **하나** | 펴 맞추어 셈한다 | 허용오차 안이면 `True` |

모양을 다루는 데서 `==`와 `equal`이 갈린다. `==`는 펴 맞추기를 하므로 `(1, 2)`와 `(2,)`를 견주면 `(1, 2)` 모양의 텐서가 나오지만, `equal`은 **모양까지 같아야** 하므로 `False`다.

```python
x = torch.tensor([[1, 2]]); y = torch.tensor([1, 2])
(x == y).shape            # torch.Size([1, 2])  — 값은 모두 True
torch.equal(x, y)         # False               — 모양이 다르다
```

그래서 "완전히 같은 텐서인가"를 묻고 싶으면 `equal`을, "어느 자리가 같은가"를 묻고 싶으면 `==`를 쓴다. `==`의 결과를 `.all()`로 줄이는 것은 `equal`과 다르다 — 모양이 다른 경우를 놓친다.

**실수에는 `allclose`를 쓴다.** 허용오차는 $|a - b| \le \text{atol} + \text{rtol} \cdot |b|$로 재며, 기본값은 `rtol=1e-5`, `atol=1e-8`이다. 생각보다 빡빡하다.

```python
torch.allclose(torch.tensor([1.0]), torch.tensor([1.000001]))   # True
torch.allclose(torch.tensor([1.0]), torch.tensor([1.00001]))    # False
```

여섯째 자리에서 갈린다. 학습한 모델의 출력을 견주는 자리에서는 기본값이 너무 좁은 일이 많으므로 `rtol`을 직접 준다.

!!! warning "`nan`은 자기 자신과도 같지 않다"
    IEEE 754는 `nan`을 어떤 값과도 — 자기 자신과도 — 같지 않게 정했다. 그래서 세 함수가 모두 `False`를 돌려준다.

    ```python
    n = torch.tensor([float('nan')])
    n == n                        # tensor([False])
    torch.equal(n, n)             # False
    torch.allclose(n, n)          # False
    torch.allclose(n, n, equal_nan=True)   # True
    ```

    같은 객체를 두 번 넣었는데 "다르다"고 하는 셈이다. 그러므로 **텐서가 같은지 보는 검사는 `nan`이 끼면 늘 실패한다.** 저장했다 불러온 가중치를 견주는 검사가 까닭 없이 깨지면 `nan`을 의심한다. `equal_nan=True`를 주면 `nan` 자리끼리는 같다고 본다.

    `nan`을 찾으려면 `==`가 아니라 `torch.isnan`을 쓴다. `t == float('nan')`은 늘 전부 `False`다.

**`torch.max`는 세 가지 일을 한다.** 인수에 따라 하는 일이 달라지는데, 이름은 하나다.

```python
torch.max(p)             # 5.0                  — 전체에서 가장 큰 값 하나 (줄인다)
torch.max(p, dim=0)      # (values, indices)    — 그 차원에서 줄이고 자리까지 준다
torch.max(p, q)          # tensor([4., 5., 3.]) — 두 텐서를 원소별로 견준다
```

둘째와 셋째가 헷갈리기 쉽다. `torch.max(p, 0)`은 "0번 차원으로 줄여라"이고 `torch.max(p, q)`는 "p와 q를 견주어라"인데, 둘째 자리에 무엇이 오는지로만 갈린다. 원소별로 견줄 뜻이라면 **`torch.maximum`을 쓰는 것이 분명하다** — 그 이름은 한 가지 일만 한다.

차원을 준 `max`가 값과 자리를 **묶어서** 돌려주는 것도 기억할 만하다. 분류에서 예측 라벨을 얻을 때 쓰는 것이 그 자리 쪽이다.

```python
values, preds = logits.max(dim=1)      # preds 가 예측 라벨이다
```

`argmax`는 같은 것의 자리만 돌려주는 줄임말이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`x = torch.tensor([[1, 2]])`와 `y = torch.tensor([1, 2])`에 대해 `(x == y)`와 `torch.equal(x, y)`를 찍어라. 값이 모두 같은데 `equal`이 `False`인 까닭을 설명하라. 이어서 `nan` 하나만 담은 텐서를 자기 자신과 견주어 보고 무엇이 나오는지 적어라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor([[1, 2]])
    y = torch.tensor([1, 2])
    print("x == y      :", x == y)
    print("torch.equal :", torch.equal(x, y))

    n = torch.tensor([float('nan')])
    print()
    print("n == n                       :", (n == n).item())
    print("torch.equal(n, n)            :", torch.equal(n, n))
    print("torch.allclose(n, n)         :", torch.allclose(n, n))
    print("allclose(n, n, equal_nan=True):", torch.allclose(n, n, equal_nan=True))
    ```

    ```
    x == y      : tensor([[True, True]])
    torch.equal : False

    n == n                       : False
    torch.equal(n, n)            : False
    torch.allclose(n, n)         : False
    allclose(n, n, equal_nan=True): True
    ```

    `==`는 **펴 맞추기**를 한다. `(1, 2)`와 `(2,)`를 오른쪽에 맞추면 둘 다 `(1, 2)`가
    되므로 값끼리 견주어 모두 `True`가 나온다. `torch.equal`은 "같은 텐서인가"를 묻는
    함수여서 **모양까지 같아야** 한다. 계수가 2와 1로 다르므로 `False`다.

    그래서 `(x == y).all()`과 `torch.equal(x, y)`는 다른 검사다. 앞은 모양이 달라도
    `True`가 될 수 있다.

    `nan` 쪽은 IEEE 754가 그렇게 정했기 때문이다. `nan`은 **어떤 값과도, 자기 자신과도**
    같지 않다. 같은 객체를 두 번 넣었는데 "다르다"는 답이 나오는 까닭이고, 그래서
    가중치를 견주는 검사가 `nan` 하나 때문에 까닭 없이 깨진다. `equal_nan=True`를 주면
    `nan` 자리끼리는 같다고 본다.

    덧붙여 `nan`을 **찾을** 때 `==`는 쓸 수 없다. `t == float('nan')`은 늘 전부
    `False`이므로 `torch.isnan(t)`을 쓴다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`p = torch.tensor([1., 5., 3.])`, `q = torch.tensor([4., 2., 3.])`에 대해 `torch.max(p)`, `torch.max(p, 0)`, `torch.max(p, q)` 셋을 찍어라. 같은 이름이 서로 다른 일을 하는 것을 정리하고, 원소별로 견줄 때 `torch.maximum`을 쓰는 편이 나은 까닭을 밝혀라.

</div>

??? success "연습문제 2 풀이"
    ```python
    import torch

    p = torch.tensor([1., 5., 3.])
    q = torch.tensor([4., 2., 3.])
    print("max(p)      :", torch.max(p))
    print("max(p, 0)   :", torch.max(p, 0))
    print("max(p, q)   :", torch.max(p, q))
    print("maximum(p,q):", torch.maximum(p, q))
    ```

    ```
    max(p)      : tensor(5.)
    max(p, 0)   : torch.return_types.max(
    values=tensor(5.),
    indices=tensor(1))
    max(p, q)   : tensor([4., 5., 3.])
    maximum(p,q): tensor([4., 5., 3.])
    ```

    한 이름이 세 가지 일을 한다.

    | 적는 법 | 하는 일 | 돌려주는 것 |
    |---|---|---|
    | `max(p)` | 전체에서 가장 큰 값 | 홑값 하나 |
    | `max(p, dim)` | 그 차원으로 줄인다 | **값과 자리**를 묶어서 |
    | `max(p, q)` | 두 텐서를 원소별로 견준다 | 같은 모양의 텐서 |

    가르는 것은 **둘째 자리에 무엇이 오는가**뿐이다. 정수가 오면 차원으로 읽고 텐서가
    오면 견줄 상대로 읽는다. 그래서 차원 번호를 담은 변수를 넘기려다 텐서를 넘기면
    뜻이 통째로 바뀌는데, 오류가 아니라 **다른 모양의 결과**가 나온다.

    `torch.maximum`은 원소별 견주기 **한 가지만** 한다. 그래서 읽는 사람이 둘째 인수를
    보고 뜻을 되짚을 필요가 없다. 원소별로 견줄 뜻이라면 이 이름을 쓴다.

    차원을 준 `max`가 값과 자리를 함께 주는 것은 분류에서 요긴하다.

    ```python
    logits = torch.tensor([[1., 4., 2.], [5., 0., 3.]])
    values, preds = logits.max(dim=1)
    print("가장 큰 값:", values, " 예측 라벨:", preds)
    ```

    ```
    가장 큰 값: tensor([4., 5.])  예측 라벨: tensor([1, 0])
    ```

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
상위 $k$ 정확도를 구하라. 모양 $(N, C)$인 로짓과 모양 $(N,)$인 정답이 주어졌을 때, 정답이 **상위 3개 예측 안에 들어 있는** 비율을 셈한다. 씨앗을 고정해 결과가 다시 나오게 하고, 왜 `unsqueeze(1)`이 필요한지 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch

    torch.manual_seed(0)
    logits = torch.randn(6, 5)
    labels = torch.tensor([0, 1, 2, 3, 4, 0])

    top3 = logits.topk(3, dim=1).indices        # (6, 3)
    hit = (top3 == labels.unsqueeze(1)).any(dim=1)
    print("상위 3개 예측:", top3.tolist())
    print("정답이 들었나:", hit.tolist())
    print("top-3 정확도 :", hit.float().mean().item())
    ```

    ```
    상위 3개 예측: [[4, 2, 3], [0, 3, 1], [3, 0, 1], [4, 2, 0], [0, 1, 4], [4, 0, 3]]
    정답이 들었나: [False, True, False, False, True, True]
    top-3 정확도 : 0.5
    ```

    `unsqueeze(1)`이 필요한 까닭은 모양 때문이다. `top3`는 `(6, 3)`이고 `labels`는
    `(6,)`다. 그대로 `==`로 견주면 오른쪽에 맞추어 `(6, 3)`과 `(1, 6)`이 되는데, 3과 6이
    어긋나므로 오류가 난다. `labels.unsqueeze(1)`로 `(6, 1)`을 만들면 마지막 자리가
    1이므로 3으로 늘어나, **줄마다 정답 하나를 세 예측과 견주는** 셈이 된다.

    ```python
    try:
        top3 == labels
    except RuntimeError as e:
        print("unsqueeze 없이:", str(e)[:62])
    ```

    ```
    unsqueeze 없이: The size of tensor a (3) must match the size of tensor b (6)
    ```

    여기서는 오류가 나 주어 다행이다. $N$과 $k$가 우연히 같으면 — 이를테면 자료가 세
    개이고 $k = 3$이면 — 오류 없이 엉뚱한 값이 나온다.

    `.any(dim=1)`은 "세 예측 가운데 하나라도 맞았는가"를 줄마다 묻는다. `.all()`이 아니라
    `.any()`인 까닭이 거기 있다. 마지막으로 참거짓을 실수로 바꾸어 평균을 내면 비율이 된다.

## 정리하며

"같은가"라는 물음에 답이 셋이고, 서로 다른 것을 묻는다.

- **`==`** — 원소마다 답한다. 모양이 다르면 펴 맞춘다. 어느 자리가 같은지 알고 싶을 때.
- **`torch.equal`** — 텐서 하나에 참거짓 하나. **모양까지** 같아야 한다. 그래서 `(x == y).all()`과 다르다.
- **`torch.allclose`** — 얼마나 가까우면 같다고 볼지 정한다. 기본값 `rtol=1e-5`는 생각보다 빡빡해서 여섯째 자리에서 갈린다.

실수를 견줄 때는 `allclose`를 쓴다. 그리고 **`nan`은 자기 자신과도 같지 않다** — 세 함수가 모두 `False`를 준다. 가중치를 견주는 검사가 까닭 없이 깨지면 `nan`을 의심하고, 찾을 때는 `==`가 아니라 `torch.isnan`을 쓴다.

`torch.max`는 둘째 인수에 따라 세 가지 일을 한다. 정수면 차원으로 읽어 **값과 자리를 묶어** 돌려주고, 텐서면 원소별로 견준다. 원소별로 견줄 뜻이라면 한 가지 일만 하는 `torch.maximum`이 분명하다.
