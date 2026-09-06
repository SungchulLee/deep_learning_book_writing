# 꼬리 호출 최적화

꼬리 호출 최적화(TCO)는 꼬리 위치의 함수 호출에 대해 새 프레임을 쌓는 대신 현재 스택 프레임을 재사용하는 컴파일러 기법이다. 꼬리 재귀 함수에 적용하면 TCO는 $O(n)$의 스택 사용을 $O(1)$로 바꾸며, 사실상 기계 수준에서 재귀를 반복으로 바꾼다.

## TCO의 동작 방식

TCO가 없으면 호출마다 새 프레임이 쌓인다.

```
factorial_tail(5, 1)  → frame 1
  factorial_tail(4, 5)  → frame 2
    factorial_tail(3, 20) → frame 3
      ...                   → frame n
```

TCO가 있으면 컴파일러는 프레임 1이 프레임 2를 호출한 뒤에 더 할 일이 없음을 알아본다. 그래서 프레임 1을 프레임 2로 대체한다.

```
factorial_tail(5, 1)    → frame (reused)
factorial_tail(4, 5)    → frame (reused)
factorial_tail(3, 20)   → frame (reused)
...                      → frame (reused)
```

## 예제: 파이썬에서 TCO 흉내 내기

파이썬은 TCO를 자체적으로 지원하지 않는다. 그러나 트램펄린을 써서 흉내 낼 수 있다. 트램펄린은 최종 값이 나올 때까지 반환된 함수를 거듭 호출하는 반복문이다.

```python
"""트램펄린으로 꼬리 호출 최적화 흉내 내기."""


# === 트램펄린 틀 ===

class TailCall:
    """트램펄린으로 처리할 꼬리 호출을 나타내는 감싸개."""
    def __init__(self, func, *args):
        self.func = func
        self.args = args


def trampoline(result):
    """최종 값에 도달할 때까지 꼬리 호출을 반복적으로 수행한다."""
    while isinstance(result, TailCall):
        result = result.func(*result.args)
    return result


# === 트램펄린을 쓰는 계승 ===

def factorial_tco(n, acc=1):
    """트램펄린을 위해 TailCall을 반환하는 꼬리 재귀 계승."""
    if n <= 1:
        return acc
    return TailCall(factorial_tco, n - 1, acc * n)


# === 메인 ===

if __name__ == "__main__":
    for n in [5, 10, 100]:
        result = trampoline(factorial_tco(n))
        print(f"{n}! = {result}")
```

**출력:**
```
5! = 120
10! = 3628800
100! = 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000
```

## 언어별 지원

| 언어 | TCO 지원 |
|---|---|
| Scheme | 명세로 보장됨 |
| Haskell | 지원(지연 평가) |
| Scala | `@tailrec` 어노테이션 |
| 파이썬 | 미지원 |
| 자바 | 미지원 |
| C/C++ | 컴파일러에 따라 다름 |

## 참고 문헌

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
꼬리 호출 최적화(TCO)를 어셈블리 수준에서 설명하라. 꼬리 호출이 최적화될 때 스택 프레임에 어떤 일이 생기는가?

??? success "연습문제 1 풀이"
    TCO가 없으면 재귀 호출마다 반환 주소, 지역 변수, 인수를 담은 새 프레임이 호출 스택에 쌓인다. TCO가 있으면 컴파일러가 꼬리 호출을 알아보고 현재 프레임의 인수를 새 호출의 인수로 바꾼 뒤 (호출하는 대신) 함수의 진입점으로 점프한다. 반환 주소는 원래 호출자의 것으로 남는다. 이렇게 모든 재귀 호출이 프레임 하나를 재사용한다.

---

**연습문제 2.**
어떤 언어가 TCO를 지원하는가? 어떤 언어(Scheme)에서는 보장되고 다른 언어(파이썬, 자바)에서는 그렇지 않은 이유는 무엇인가?

??? success "연습문제 2 풀이"
    보장되는 언어: Scheme(R5RS에서 강제), Haskell, Erlang, Scala(`@tailrec` 사용). 보장되지 않는 언어: 파이썬(디버깅 편의를 위한 설계 선택), 자바(JVM이 자체적으로 지원하지 않는다), 자바스크립트(ES6가 명시하지만 사파리만 구현했다). 함수형 언어가 TCO를 강제하는 이유는 재귀가 주된 반복 구조이기 때문이다. TCO가 없으면 단순한 반복도 스택을 넘치게 한다.

---

**연습문제 3.**
재귀적 트리 순회 `inorder(node) = inorder(left) + [node.val] + inorder(right)`를 연속 전달 방식(CPS)을 사용해 꼬리 재귀 형태로 바꾸어라.

??? success "연습문제 3 풀이"
    CPS 형태: `inorder(node, cont) = inorder(node.left, lambda left_result: inorder(node.right, lambda right_result: cont(left_result + [node.val] + right_result)))`. TCO가 있으면 스택 프레임은 $O(1)$이지만 연속을 위해 힙을 $O(n)$ 쓴다. 실무에서는 명시적 스택(반복 순회)이 더 간단하고 효율적이다.

---

**연습문제 4.**
트램펄린 패턴은 TCO가 없는 언어에서 TCO를 흉내 낸다. 파이썬으로 계승에 대한 트램펄린을 구현하라.

??? success "연습문제 4 풀이"
    ```python
    def factorial(n, acc=1):
        if n <= 1: return acc
        return lambda: factorial(n-1, n*acc)  # 호출하지 말고 thunk를 반환한다
    
    def trampoline(f):
        while callable(f): f = f()
        return f
    
    result = trampoline(factorial(1000))
    ```
    재귀 "호출"마다 thunk(인수가 없는 람다)를 반환한다. 트램펄린 반복문이 thunk들을 반복적으로 호출하므로 스택 공간을 $O(1)$만 쓴다.