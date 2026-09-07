# 함수 호출 흉내 내기

프로그램이 함수를 부를 때마다 실행 환경은 돌아갈 주소, 지역 변수, 매개변수를 담은 **스택 프레임**을 쌓는다. 함수가 반환하면 그 프레임을 빼고 저장해 둔 주소에서 실행을 이어 간다. 이 장치, 곧 **호출 스택**이 있기에 재귀 알고리즘이 작동한다. 재귀 호출마다 자기만의 지역 상태를 담은 프레임을 갖기 때문이다. 이 과정을 이해하는 것은 재귀 알고리즘을 반복으로 바꾸고 스택 넘침 오류를 진단하는 데 꼭 필요하다. 이 쪽은 호출 스택의 작동 방식을 자세히 설명하고, 눈에 보이는 스택 자료 구조로 그것을 흉내 내는 법을 보인다.

## 스택 프레임의 구조

스택 프레임에는 세 가지 정보가 담긴다.

1. **돌아갈 주소** — 함수가 끝난 뒤 실행을 이어 갈 곳
2. **매개변수** — 함수에 넘긴 인수
3. **지역 변수** — 함수 본문 안에서 선언한 변수

함수 `A`이 함수 `B`을 부르면 실행 환경은 다음과 같이 한다.

1. `B`의 매개변수와 지역 변수를 담은 새 프레임을 호출 스택에 쌓는다
2. 제어를 `B`의 첫 명령으로 넘긴다
3. `B`이 반환하면 `B`의 프레임을 뺀다
4. 호출 다음 명령에서 `A`을 이어 간다

스택의 후입선출 순서 덕분에 가장 최근에 부른 함수가 먼저 끝나며, 이는 함수 호출의 중첩 구조와 들어맞는다.

## 호출 스택의 깊이와 공간

함수 호출이 $n$겹으로 중첩되면 호출 스택의 깊이가 $n$이 되어 $O(n)$의 메모리를 쓴다. 대부분의 언어 실행 환경은 최대 스택 깊이를 정해 둔다(예: 파이썬의 기본값은 1000). 재귀 깊이가 $d$인 재귀 함수는 $O(d)$의 스택 공간을 쓴다. $d$이 한계를 넘으면 **스택 넘침**이 일어난다.

$$
\text{Space}(n) = n \times (\text{frame size}) = O(n)
$$

깊이 재귀하는 알고리즘을 때때로 눈에 보이는 스택을 쓰는 반복 판본으로 바꾸어야 하는 까닭이 여기에 있다.

## 눈에 보이는 스택으로 재귀 흉내 내기

암묵적인 호출 스택을 사용자가 관리하는 스택으로 바꾸면 어떤 재귀 알고리즘도 반복으로 바꿀 수 있다. 눈에 보이는 스택의 각 "프레임"은 재귀 호출이 필요로 했을 상태를 담는다. 다음 예는 계승 계산으로 이를 보인다.

```python
"""
함수 호출 흉내 내기 — 눈에 보이는 스택으로 호출 스택 본뜨기.

재귀 호출이 스택의 넣기/빼기 연산에 어떻게 대응하는지, 그리고 이를 써서
재귀를 반복으로 바꾸는 법을 보인다.
"""


# === 스택 프레임의 표현 ===============================================

class Frame:
    """흉내 낸 호출 스택의 프레임 하나를 나타낸다."""

    def __init__(self, func_name, params, return_addr=None):
        self.func_name = func_name
        self.params = params
        self.return_addr = return_addr
        self.local_vars = {}

    def __repr__(self):
        return f"Frame({self.func_name}, params={self.params}, locals={self.local_vars})"


# === 재귀적 계승 (암묵적인 호출 스택을 쓴다) ===========================

def factorial_recursive(n):
    """n!을 재귀로 계산한다. 호출마다 암묵적인 스택 프레임이 생긴다."""
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)


# === 눈에 보이는 스택을 쓰는 반복적 계승 ===================================

def factorial_stack_simulation(n):
    """호출 스택을 명시적으로 흉내 내어 n!을 계산한다.

    '호출'마다 프레임을 쌓고, '반환'마다 프레임을 빼며 결과를
    부른 쪽에 넘긴다.
    """
    # 1단계: 재귀적으로 내려가는 과정 흉내 내기 (프레임 쌓기)
    call_stack = []
    print("  === Pushing frames (recursive descent) ===")
    for i in range(n, 0, -1):
        frame = Frame("factorial", {"n": i})
        call_stack.append(frame)
        print(f"    PUSH {frame}")

    # 2단계: 반환 과정 흉내 내기 (프레임을 빼며 결과 누적)
    print("  === Popping frames (returning results) ===")
    result = 1  # 기저 사례: factorial(1) = 1
    while call_stack:
        frame = call_stack.pop()
        current_n = frame.params["n"]
        result *= current_n
        print(f"    POP  {frame.func_name}(n={current_n}) → result so far = {result}")

    return result


# === 상호 재귀 흉내 내기 ==============================================

def simulate_even_odd(n):
    """명시적 스택으로 서로 재귀하는 is_even/is_odd를 흉내 낸다.

    is_even(n) = is_odd(n-1), is_odd(n) = is_even(n-1)
    is_even(0) = True, is_odd(0) = False
    """
    call_stack = [Frame("is_even", {"n": n})]
    print(f"  Checking if {n} is even via mutual recursion simulation:")

    while True:
        frame = call_stack[-1]
        fn = frame.func_name
        current_n = frame.params["n"]

        if current_n == 0:
            # 기저 사례
            result = (fn == "is_even")
            call_stack.pop()
            print(f"    BASE  {fn}(0) = {result}")
            break

        # 상호 호출 흉내 내기
        next_fn = "is_odd" if fn == "is_even" else "is_even"
        new_frame = Frame(next_fn, {"n": current_n - 1})
        call_stack.append(new_frame)
        print(f"    CALL  {fn}({current_n}) → {next_fn}({current_n - 1})")

    # 남은 프레임 풀기
    while call_stack:
        frame = call_stack.pop()
        print(f"    RETURN {frame.func_name}({frame.params['n']}) = {result}")

    return result


# === 시연 ============================================================

if __name__ == "__main__":
    # 재귀적 계승
    n = 5
    print(f"Recursive factorial({n}) = {factorial_recursive(n)}")
    print()

    # 스택으로 흉내 낸 계승
    print(f"Stack-simulated factorial({n}):")
    result = factorial_stack_simulation(n)
    print(f"  Result: {result}")
    print()

    # 상호 재귀 흉내 내기
    for test_n in [4, 7]:
        answer = simulate_even_odd(test_n)
        print(f"  is_even({test_n}) = {answer}")
        print()
```

**출력:**
```
Recursive factorial(5) = 120

Stack-simulated factorial(5):
  === Pushing frames (recursive descent) ===
    PUSH Frame(factorial, params={'n': 5}, locals={})
    PUSH Frame(factorial, params={'n': 4}, locals={})
    PUSH Frame(factorial, params={'n': 3}, locals={})
    PUSH Frame(factorial, params={'n': 2}, locals={})
    PUSH Frame(factorial, params={'n': 1}, locals={})
  === Popping frames (returning results) ===
    POP  factorial(n=1) → result so far = 1
    POP  factorial(n=2) → result so far = 2
    POP  factorial(n=3) → result so far = 6
    POP  factorial(n=4) → result so far = 24
    POP  factorial(n=5) → result so far = 120
  Result: 120

  Checking if 4 is even via mutual recursion simulation:
    CALL  is_even(4) → is_odd(3)
    CALL  is_odd(3) → is_even(2)
    CALL  is_even(2) → is_odd(1)
    CALL  is_odd(1) → is_even(0)
    BASE  is_even(0) = True
    RETURN is_odd(1) = True
    RETURN is_even(2) = True
    RETURN is_odd(3) = True
    RETURN is_even(4) = True
  is_even(4) = True

  Checking if 7 is even via mutual recursion simulation:
    CALL  is_even(7) → is_odd(6)
    CALL  is_odd(6) → is_even(5)
    CALL  is_even(5) → is_odd(4)
    CALL  is_odd(4) → is_even(3)
    CALL  is_even(3) → is_odd(2)
    CALL  is_odd(2) → is_even(1)
    CALL  is_even(1) → is_odd(0)
    BASE  is_odd(0) = False
    RETURN is_even(1) = False
    RETURN is_odd(2) = False
    RETURN is_even(3) = False
    RETURN is_odd(4) = False
    RETURN is_even(5) = False
    RETURN is_odd(6) = False
    RETURN is_even(7) = False
  is_even(7) = False

```

이 흉내 내기는 평소에는 보이지 않는 일을 눈앞에 드러낸다. 재귀 호출 하나가 `PUSH` 연산 하나에, 반환 하나가 `POP` 하나에 대응한다. 어느 순간의 스택 깊이는 그때의 재귀 깊이와 같다.

## 눈에 보이는 스택을 쓸 때

눈에 보이는 스택으로 재귀를 반복으로 바꾸는 것은 세 가지 상황에서 쓸모 있다.

1. **스택 넘침 피하기** — 호출 스택의 크기가 제한된 언어(파이썬, 자바)는 깊은 재귀에서 넘칠 수 있다. 눈에 보이는 스택은 대개 훨씬 큰 힙 메모리를 쓴다.
2. **성능** — 함수 호출의 부담(매개변수 전달, 프레임 할당)을 없애면 상수 배만큼 빨라질 수 있다.
3. **상태 들여다보기** — 눈에 보이는 스택은 실행 중에 살펴보거나 직렬화하거나 고칠 수 있는데, 암묵적인 호출 스택으로는 불가능하다.

!!! tip "일반적인 변환 방법"
    재귀 함수를 반복으로 바꾸려면 (1) 재귀 호출마다 어떤 상태가 필요한지 밝히고, (2) 그 상태를 담을 프레임 구조를 정의하고, (3) 각 재귀 호출을 넣기로 바꾸고, (4) 각 반환을 빼기로 바꾼다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.


## 연습문제

**연습문제 1.**
함수 호출 흉내 내기의 추상 자료형이 지원하는 연산을 시간 복잡도와 함께 모두 열거하라. 어느 연산이 병목인가?

??? success "연습문제 1 풀이"
    추상 자료형은 구현과 무관하게 지원하는 연산을 정한다. 무엇이 병목인지는 쓰임새에 달렸다. 실시간 시스템에서는 최악의 복잡도가 중요하고, 일괄 처리에서는 상각 복잡도로 충분하다.

---

**연습문제 2.**
함수 호출 흉내 내기을(를) 서로 다른 두 자료 구조로 구현하라. 각각의 절충을 비교하라.

??? success "연습문제 2 풀이"
    구현 1: 배열 기반 — 접근은 상수 시간이지만 크기를 다시 잡아야 할 수 있다. 구현 2: 연결 리스트 기반 — 삽입과 삭제는 상수 시간이지만 접근은 $O(n)$이다. 어느 쪽을 고를지는 응용에서 예상되는 연산의 구성에 달렸다.

---

**연습문제 3.**
함수 호출 흉내 내기을(를) 쓰는 딥러닝 응용을 하나 설명하라(예: 그래프 신경망의 너비 우선 탐색, 기호 미분에서의 식 계산, 데이터 적재의 스케줄링).

??? success "연습문제 3 풀이"
    구체적인 응용은 그 추상 자료형의 순서 성질에 달렸다. 선입선출(큐)은 GNN의 너비 우선 그래프 순회에 쓰이고, 후입선출(스택)은 자동 미분 테이프 처리에 쓰이며, 우선순위 순서는 빔 탐색과 예정 표집에 쓰인다.

---

**연습문제 4.**
함수 호출 흉내 내기을(를) 원형 배열로 구현하면 모든 연산이 상각 $O(1)$ 시간임을 증명하라.

??? success "연습문제 4 풀이"
    원형 배열은 머리와 꼬리 인덱스를 용량으로 나눈 나머지로 관리한다. 넣기와 빼기는 인덱스를 $O(1)$에 조정한다. 배열이 가득 차면 용량을 두 배로 늘리는 데 $O(n)$이 들지만, 이는 값싼 연산 $O(n)$번 뒤에 한 번 일어나므로 동적 배열과 같은 논법으로 상각 $O(1)$이 된다. $\square$