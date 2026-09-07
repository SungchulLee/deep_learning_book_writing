# 잠금 없는 스택

한꺼번에 쓰는 스택은 실 여럿이 벌이는 `push`와 `pop`을 망가짐 없이 받쳐 주어야 한다. 가장 쉬운 옳은 길은 연산을 뮤텍스로 감싸는 것이지만, 이러면 모든 닿기가 한 줄로 늘어선다. **트라이버 스택**(1986)은 잠금 없는 스택의 본보기다. 원자적인 `top` 손가락질을 둔 한 겹 이음 목록을 쓰고 모든 고침이 견주어 바꾸기(CAS)를 거친다. 그 열매는 적어도 한 실이 늘 앞으로 나아가는, 막히지 않는 스택이다.

## 트라이버 스택 알고리즘

### 얼개

스택은 마디마다 그 아래 마디를 가리키는 한 겹 이음 목록이다. 함께 쓰는 손가락질 `top` 하나가 지금 스택 꼭대기를 가리킨다.

### 넣기

1. 넣을 값을 담은 새 마디를 만든다.
2. `new_node.next = top`으로 둔다.
3. `CAS(top, old_top, new_node)`을 꾀한다.
    - 이루면 넣기가 끝난다.
    - 어그러지면(다른 실이 `top`을 고쳤으면) 2번으로 가서 다시 꾀한다.

### 빼기

1. `old_top = top`을 읽는다.
2. `old_top`이 널이면 스택이 비었다.
3. `new_top = old_top.next`을 읽는다.
4. `CAS(top, old_top, new_top)`을 꾀한다.
    - 이루면 `old_top.value`을 돌려준다.
    - 어그러지면 1번으로 가서 다시 꾀한다.

다투지 않는 경우 두 연산 모두 $O(1)$이다. 다툼이 있을 때에도 CAS를 다시 꾀하는 어림 횟수는 매여 있다.

## 구현

```python
"""
잠금 없는 스택(트라이버 스택) 흉내 내기.

꼭대기 손가락질에 흉내 낸 CAS를 쓰는 한 겹 이음 목록을 쓴다.
실제 서비스에서 CAS는 기계의 원자적인 명령이다.
"""

import threading

# ===================================================================
# 트라이버 스택 (잠금 없음을 흉내 냄)
# ===================================================================

class StackNode:
    """한 겹 이음 목록의 스택 마디."""

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


class TreiberStack:
    """흉내 낸 CAS를 쓰는 잠금 없는 스택.

    알고리즘은 트라이버 스택 설계를 따른다.
    파이썬의 GIL과 드러낸 잠금이 원자적인 CAS를 흉내 낸다.
    """

    def __init__(self):
        self.top = None
        self._lock = threading.Lock()  # CAS를 흉내 낸다
        self._size = 0

    def push(self, value):
        """값을 스택에 넣는다.

        인수:
            value: 넣을 항목
        """
        new_node = StackNode(value)
        with self._lock:
            new_node.next = self.top
            self.top = new_node
            self._size += 1

    def pop(self):
        """꼭대기 값을 빼내어 돌려준다.

        돌려주는 값:
            꼭대기 값, 비었으면 None
        """
        with self._lock:
            if self.top is None:
                return None
            value = self.top.value
            self.top = self.top.next
            self._size -= 1
            return value

    def peek(self):
        """빼내지 않고 꼭대기 값을 돌려준다."""
        if self.top is None:
            return None
        return self.top.value

    def is_empty(self):
        """스택이 비었는지 살핀다."""
        return self.top is None

    def size(self):
        """지금 스택 크기를 돌려준다."""
        return self._size

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    stack = TreiberStack()

    # 실 하나로 시험
    for x in [10, 20, 30, 40]:
        stack.push(x)

    print("Single-threaded (push 10,20,30,40 then pop all):")
    while not stack.is_empty():
        print(f"  pop: {stack.pop()}")

    # 실 여럿으로 넣고 빼기
    stack = TreiberStack()
    pushed = []
    popped = []
    barrier = threading.Barrier(4)

    def pusher(items):
        barrier.wait()
        for item in items:
            stack.push(item)
            pushed.append(item)

    def popper(count):
        barrier.wait()
        local = []
        attempts = 0
        while len(local) < count and attempts < count * 20:
            val = stack.pop()
            if val is not None:
                local.append(val)
            attempts += 1
        popped.extend(local)

    t1 = threading.Thread(target=pusher, args=([1, 2, 3, 4, 5],))
    t2 = threading.Thread(target=pusher, args=([6, 7, 8, 9, 10],))
    t3 = threading.Thread(target=popper, args=(5,))
    t4 = threading.Thread(target=popper, args=(5,))

    for t in [t1, t2, t3, t4]:
        t.start()
    for t in [t1, t2, t3, t4]:
        t.join()

    print(f"\nMulti-threaded test:")
    print(f"  Pushed: {sorted(pushed)}")
    print(f"  Popped: {sorted(popped)}")
    remaining = []
    while not stack.is_empty():
        remaining.append(stack.pop())
    print(f"  Remaining in stack: {sorted(remaining)}")
    all_items = sorted(popped + remaining)
    print(f"  All accounted for: {all_items == list(range(1, 11))}")
```

**출력:**
```
Single-threaded (push 10,20,30,40 then pop all):
  pop: 40
  pop: 30
  pop: 20
  pop: 10

Multi-threaded test:
  Pushed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  Popped: [5, 6, 7, 8, 9, 10]
  Remaining in stack: [1, 2, 3, 4]
  All accounted for: True
```

## 나아감 보장

트라이버 스택은 **잠금 없는** 것이다. 어떤 실이 연산 도중에 멈추어도 다른 실은 저마다의 넣기와 빼기를 마칠 수 있다. 이는 잠금을 쥔 실이 다른 모든 실을 끝없이 막을 수 있는 뮤텍스 스택보다 힘센 성질이다.

그러나 **기다림 없는** 것은 아니다. 다른 실이 계속 먼저 이루면 어떤 실 하나는 CAS를 아무리 여러 번이라도 다시 꾀할 수 있다. 실제로 다툼이 어중간하면 다시 꾀하는 횟수는 아주 작다.

## ABA 문제

트라이버 스택은 ABA 문제에 걸리기 쉽다.

1. 실 A가 `top = X`을 읽고 `top`을 `X`에서 `X.next`으로 CAS 할 채비를 한다.
2. 실 B가 `X`을 빼내고 `Y`을 빼낸 뒤 `X`을 다시 넣는다(같은 마디이지만 스택 상태는 다르다).
3. 실 A의 CAS가 (`X`을 보고) 이루어지지만 `X.next`은 이제 엉뚱한 마디를 가리킨다.

!!! warning "ABA를 푸는 길"

    - **표를 단 손가락질**: 손가락질마다 판 세개를 짝지운다. CAS가 손가락질과 세개를 함께 살핀다.
    - **위험 손가락질**: 어떤 실이든 가리키고 있는 동안에는 기억을 거둬들이지 못하게 막는다.
    - **시대에 바탕을 둔 거둬들이기**: 모든 실이 잠잠한 상태를 지날 때까지 마디 놓아주기를 미룬다.

## 지워 없애는 스택

다툼이 심할 때에는 **물러서기**나 **지워 없애기** 다듬기가 도움이 된다. CAS가 어그러진 실들이 값을 바로 주고받는다(넣는 실 하나와 빼는 실 하나가 함께 쓰는 스택을 건드리지 않고 서로를 지운다). 이로써 다툼이 처리량으로 바뀐다.

## 복잡도

| 연산 | 어림 때 |
|---|---|
| `push` | $O(1)$ 나눠 갚음 |
| `pop` | $O(1)$ 나눠 갚음 |
| 자리 | $O(n)$ |

## 참고 문헌

- Treiber, R. K. (1986). "Systems programming: Coping with parallelism." *IBM Research Report RJ 5118*.
- Herlihy, M. and Shavit, N. *The Art of Multiprocessor Programming*, 11장.

## 연습문제

**연습문제 1.**
CAS를 써서 트라이버 스택의 `push`와 `pop`을 밝혀라. 이 스택이 잠금 없는 것이면서 기다림 없는 것은 아닌 까닭은 무엇인가?

??? success "연습문제 1 풀이"
    **Push**: 새 마디를 마련하고 `node.next = top`으로 둔다. `top`을 지금 값에서 새 마디로 CAS 한다. CAS가 어그러지면(다른 실이 `top`을 고쳤으면) `top`을 다시 읽고 `node.next`을 고친 뒤 다시 꾀한다. **Pop**: `top`을 읽는다. 널이면 스택이 비었다. 아니면 `top.next`을 읽는다. `top`을 `top`에서 `top.next`으로 CAS 한다. CAS가 어그러지면 다시 꾀한다. 이루면 옛 `top`의 값을 돌려준다. 이 스택이 잠금 없는 까닭은 다툼이 벌어지는 판마다 적어도 한 실의 CAS가 이루어지기 때문이다. CAS가 어그러진 실은 다른 실이 이루었음을 알므로 온 시스템의 나아감이 보장된다. 기다림 없는 것이 아닌 까닭은 다른 실이 계속 이루면 낱낱의 실의 CAS가 매임 없이 여러 번 어그러질 수 있기 때문이다. 이치상 한 실이 굶주리는 동안 다른 실들이 끊임없이 넣고 뺄 수 있다. 기다림 없으려면 모든 실이 매인 걸음 수 안에 마쳐야 한다. $\square$

---

**연습문제 2.**
트라이버 스택에서 ABA 문제를 풀고, 그것이 그른 거동을 낳는 또렷한 자리를 보여라.

??? success "연습문제 2 풀이"
    자리: 스택이 $[A \to B \to C]$을 담는다(top = A). 실 1이 `pop`을 시작해 `top = A`, `next = B`을 읽는다. 실 1이 밀려난다. 실 2가 A을 빼내고(top이 B가 된다) B을 빼낸 뒤(top이 C가 된다) A을 다시 넣는다(top = A이고 `A.next = C`이다). 실 1이 다시 돌아 `top`을 A에서 B로 CAS 하고 이룬다(top이 A이므로). 이제 `top = B`이지만 B는 이미 놓아주어졌거나 빼내진 허공 손가락질이다. 스택이 망가졌다. 손가락질 값이 같았기에(A) CAS가 이루어졌으나 그 밑의 스택 얼개는 바뀌었다. 푸는 길은 이렇다. (1) 판 세개를 곁들인 곱절 너비 CAS를 쓴다(CAS마다 세개를 올리므로 세개가 1인 A와 3인 A가 다르다). (2) 위험 손가락질로 A의 기억이 다시 쓰이지 못하게 막는다. (3) 시대에 바탕을 둔 거둬들이기를 쓴다. $\square$

---

**연습문제 3.**
지워 없애며 물러서는 스택은 트라이버 스택과 지워 없애기 배열을 엮는다. 다툼이 있을 때 어떻게 더 높은 처리량을 이루는지 밝혀라.

??? success "연습문제 3 풀이"
    다툼이 심하면 모든 실이 캐시 줄 하나를 두고 겨루므로 트라이버 스택의 `top` 손가락질에 대한 CAS 다시 꾀하기가 목을 죈다. 지워 없애며 물러서는 스택은 실들이 짝을 지을 수 있는 곁 배열을 더한다. 배열에서 부딪친 `push`와 `pop`은 함께 쓰는 스택을 건드리지 않고 값을 곧바로 주고받을 수 있다. CAS가 어그러지면 실은 지워 없애기 배열에서 아무 자리를 골라 제 연산을(값을 지닌 넣기이거나 값을 바라는 빼기이거나) 내건다. 짝이 되는 실이 정한 때 안에 같은 자리에 오면 둘이 주고받고 함께 마친다. 짝을 찾지 못하면 그 실은 으뜸 스택에서 다시 꾀한다. 짝지은 연산이 다툼 자리를 아예 비껴가므로 처리량이 늘어난다. 다툼이 적을 때에는 지워 없애기 배열이 거의 쓰이지 않고 트라이버 스택이 연산을 곧바로 다룬다. $\square$

---

**연습문제 4.**
트라이버 스택이 줄 세울 수 있음을 증명하여라. 곧 한꺼번에 도는 모든 실행이, 넣기와 빼기가 저마다 CAS를 이룬 자리에서 효험을 내는 어떤 차례 실행과 같음을 보여라.

??? success "연습문제 4 풀이"
    연산마다 줄 세우는 자리를 매긴다. 곧 이루어진 CAS 명령이다. `push`에서 줄 세우는 자리는 `top`을 옛것에서 새 마디로 돌리는 CAS다. `pop`에서는 `top`을 지금 마디에서 `top.next`으로 돌리는 CAS다. 빈 스택에서 (널을 돌려주는) `pop`에서는 `top == null`을 읽는 자리다. CAS가 원자적인 명령이므로 줄 세우는 자리들은 기계가 실행한 때로 온전히 차례가 매겨진다. 이 차례에서 넣기의 CAS 뒤에는 넣은 원소가 스택 꼭대기에 있고, 빼기의 CAS 뒤에는 꼭대기 원소가 없어진다. 한꺼번에 도는 어떤 실행이든 연산을 그 줄 세우는 자리에 따라 늘어놓아 되돌려 볼 수 있고, 이는 옳은 차례 스택 실행을 낳는다. CAS가 스택 상태를 원자적으로 살피고 고쳐 중간 상태가 드러나지 않게 하므로 이것이 성립한다. $\square$

---

**연습문제 5.**
트라이버 스택과 잠금을 쓰는 스택을 (가) 옳음 보장, (나) 다툼이 없을 때의 성능, (다) 다툼이 심할 때의 성능, (라) 만들기 품에서 견주어라.

??? success "연습문제 5 풀이"
    (가) **옳음**: 둘 다 줄 세울 수 있다. 잠금을 쓰는 스택은 (잠금이 하나이면) 맞물려 멈추지 않고 (공평한 잠금이면) 굶주리지 않는다. 트라이버 스택은 잠금 없지만(온 시스템의 나아감이 보장된다) 낱낱의 실이 굶주리지 않음은 보장되지 않는다. (나) **다툼 없음**: 잠금을 쓰는 스택은 잠금을 얻고 놓는 덧듦이 있다(다투지 않는 뮤텍스에서 약 20 ns). 트라이버 스택은 CAS 한 번을 벌인다(약 10 ns). 트라이버 스택이 조금 빠르다. (다) **다툼 심함**: 잠금을 쓰는 스택은 모든 연산을 잠금에서 한 줄로 늘어세우며 처리량이 약 1 / (잠금 비용 + 연산 비용)으로 매인다. 트라이버 스택도 CAS 자리에서 한 줄로 늘어서므로 처리량이 비슷하지만, 다시 꾀하는 CAS가 CPU 걸음을 더 버린다. 둘 다 잘 늘어나지 않으므로 지워 없애며 물러서는 스택이 있어야 한다. (라) **품**: 잠금을 쓰는 스택은 아주 쉽다(연산을 `lock`/`unlock`으로 감싼다). 트라이버 스택은 ABA와 놓아준 뒤 쓰기를 막으려 꼼꼼한 기억 거둬들이기(위험 손가락질이나 시대에 바탕을 둔 것)가 있어야 하므로 품이 크게 는다. $\square$
