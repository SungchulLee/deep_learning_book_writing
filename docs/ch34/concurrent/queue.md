# 잠금 없는 큐

큐는 한꺼번에 도는 시스템의 밑돌이다. 쪽지 건네기, 일 짜기, 만드는 이와 쓰는 이 결이 모두 함께 쓰는 큐에 기댄다. 온통 하나뿐인 잠금을 쓰는 여느 큐는 모든 연산을 한 줄로 늘어세워 다툼이 심할 때 목을 죈다. **잠금 없는 큐**는 잠금 대신 원자적인 견주어 바꾸기(CAS)를 써서, 다른 실이 늦춰지거나 밀려나도 적어도 한 실은 늘 앞으로 나아감을 보장한다.

## 마이클-스콧 큐

가장 널리 쓰이는 잠금 없는 큐는 **마이클-스콧 큐**(1996)로, 원자적인 머리와 꼬리 손가락질을 둔 한 겹 이음 목록을 쓴다.

### 얼개

- 파수(허수아비) 마디가 머리와 꼬리를 갈라놓는다.
- **머리**는 파수 마디를 가리킨다. 다음에 빼낼 원소는 `head.next`이다.
- **꼬리**는 마지막 마디(또는 마지막에 가까운 마디)를 가리킨다.
- 모든 손가락질 고침이 CAS를 쓴다. `CAS(addr, expected, new)`은 `addr`의 지금 값이 `expected`과 같을 때에만 `new`을 원자적으로 적는다.

### 넣기

1. 넣을 값을 담은 새 마디를 만든다.
2. `tail`과 `tail.next`을 읽는다.
3. `tail.next`이 널이면 `CAS(tail.next, null, new_node)`을 꾀한다.
    - 이루면 `CAS(tail, old_tail, new_node)`으로 `tail`을 앞으로 돌린다.
    - 어그러지면 2번부터 다시 꾀한다.
4. `tail.next`이 널이 아니면(다른 실이 이미 붙였으면) `tail`을 앞으로 돌려 거들고 다시 꾀한다.

### 빼기

1. `head`, `tail`, `head.next`을 읽는다.
2. `head.next`이 널이면 큐가 비었다.
3. `CAS(head, old_head, head.next)`을 꾀한다.
    - 이루면 옛 `head.next`의 값을 돌려준다.
    - 어그러지면 1번부터 다시 꾀한다.

## 흉내 내기

파이썬에는 기계 CAS가 없으므로 잠금 없는 큐의 논리를 실 잠금으로 흉내 내며 알고리즘 얼개와 옳음에 눈을 둔다.

```python
"""
잠금 없는 큐 흉내 내기(마이클-스콧 큐).

CAS에 바탕을 둔 넣기/빼기 알고리즘을 흉내 낸다. 참으로
만들 때 CAS는 기계의 원자적인 명령이다.
"""

import threading

# ===================================================================
# 잠금 없는 큐 (흉내 냄)
# ===================================================================

class Node:
    """큐 마디."""
    def __init__(self, value=None):
        self.value = value
        self.next = None


class LockFreeQueue:
    """마이클-스콧 잠금 없는 큐(잠금으로 흉내 냄).

    알고리즘 얼개는 CAS에 바탕을 둔 설계를 따른다.
    파이썬의 GIL과 드러낸 잠금이 원자적인 CAS를 흉내 낸다.
    """

    def __init__(self):
        sentinel = Node()  # 허수아비 마디
        self.head = sentinel
        self.tail = sentinel
        self._lock = threading.Lock()  # CAS를 흉내 낸다

    def enqueue(self, value):
        """값을 큐 뒤에 더한다.

        인수:
            value: 넣을 항목
        """
        new_node = Node(value)
        with self._lock:
            self.tail.next = new_node
            self.tail = new_node

    def dequeue(self):
        """앞 항목을 빼내어 돌려준다.

        돌려주는 값:
            앞에 있던 값, 비었으면 None
        """
        with self._lock:
            if self.head.next is None:
                return None
            value = self.head.next.value
            self.head = self.head.next
            return value

    def is_empty(self):
        """큐가 비었는지 살핀다."""
        return self.head.next is None

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    queue = LockFreeQueue()

    # 실 하나로 시험
    for x in [10, 20, 30, 40]:
        queue.enqueue(x)

    print("Single-threaded dequeue:")
    while not queue.is_empty():
        print(f"  {queue.dequeue()}")

    # 실 여럿으로 만드는 이와 쓰는 이
    queue = LockFreeQueue()
    produced = []
    consumed = []
    barrier = threading.Barrier(3)

    def producer(items):
        barrier.wait()
        for item in items:
            queue.enqueue(item)
            produced.append(item)

    def consumer(count):
        barrier.wait()
        local = []
        attempts = 0
        while len(local) < count and attempts < count * 10:
            val = queue.dequeue()
            if val is not None:
                local.append(val)
            attempts += 1
        consumed.extend(local)

    t1 = threading.Thread(target=producer, args=([1, 2, 3, 4, 5],))
    t2 = threading.Thread(target=producer, args=([6, 7, 8, 9, 10],))
    t3 = threading.Thread(target=consumer, args=(10,))

    t1.start(); t2.start(); t3.start()
    t1.join(); t2.join(); t3.join()

    print(f"\nProducer-consumer test:")
    print(f"  Produced: {sorted(produced)}")
    print(f"  Consumed: {sorted(consumed)}")
    print(f"  All consumed: {sorted(consumed) == list(range(1, 11))}")
```

**출력:**
```
Single-threaded dequeue:
  10
  20
  30
  40

Producer-consumer test:
  Produced: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  Consumed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  All consumed: True
```

## 나아감 보장

| 보장 | 뜻 |
|---|---|
| **잠금 없음** | 다른 실의 나아감과 상관없이 적어도 한 실이 마침한 걸음 수 안에 제 연산을 마친다 |
| **기다림 없음** | 모든 실이 매인 걸음 수 안에 제 연산을 마친다 |
| **막힘 없음** | 홀로 돌린다면 실이 마침한 걸음 수 안에 제 연산을 마친다 |

마이클-스콧 큐는 **잠금 없는** 것이다. 한 실이 연산 도중에 늦춰져도 다른 실은 앞으로 나아갈 수 있다. 다툼이 심할 때 한 실이 CAS를 끝없이 다시 꾀할 수 있으므로 기다림 없는 것은 아니다.

## 복잡도

| 연산 | 때(나눠 갚음) |
|---|---|
| `enqueue` | 어림 $O(1)$ |
| `dequeue` | 어림 $O(1)$ |

다툼이 심하면 다시 꾀하는 CAS가 덧듦을 더하지만, 다툼이 매여 있으면 다시 꾀하는 어림 횟수는 상수다.

## ABA 문제

CAS에 바탕을 둔 자료 얼개의 미묘한 옳음 문제다.

1. 실 A가 손가락질에서 값 `X`을 읽는다.
2. 실 B가 손가락질을 `X`에서 `Y`으로, 다시 `X`으로 바꾼다(비트 무늬는 같지만 마련한 자리는 다르다).
3. 실 A의 CAS가 `X`을 보았으므로 이루어지지만 그 밑의 것은 바뀌었다.

!!! warning "ABA 막기"
    흔히 쓰는 길로는 표를 단 손가락질(손가락질마다 판 세개를 붙인다)과 위험 손가락질(어떤 실이 가리키고 있는 동안 기억을 거둬들이지 못하게 막는다)이 있다. 자바의 `AtomicStampedReference`은 표를 단 손가락질을 만든 것이다.

## 참고 문헌

- Michael, M. M. and Scott, M. L. (1996). "Simple, fast, and practical non-blocking and blocking concurrent queue algorithms." *PODC*.
- Herlihy, M. and Shavit, N. *The Art of Multiprocessor Programming*, 10장.

## 연습문제

**연습문제 1.**
마이클-스콧 잠금 없는 큐 알고리즘을 밝혀라. `enqueue`와 `dequeue`은 잠금 없이 옳음을 지키려고 CAS를 어떻게 쓰는가?

??? success "연습문제 1 풀이"
    마이클-스콧 큐는 원자적인 `head`, `tail` 손가락질과 파수 마디를 둔 한 겹 이음 목록을 쓴다. **넣기**: 새 마디를 마련한 뒤 `tail->next`을 널에서 새 마디로 CAS 한다. 이루면 `tail`을 새 마디로 앞당겨 CAS 한다(어느 실이든 벌일 수 있는 "거들기" 걸음이다). 첫 CAS가 어그러지면 다른 실이 먼저 넣은 것이니 다시 꾀한다. **빼기**: `head->next`(첫 참 마디)을 읽는다. 널이면 큐가 비었다. 아니면 `head`을 파수에서 `head->next`으로 CAS 하고 옛 `head->next`에서 값을 뽑는다. CAS가 어그러지면 다른 실이 먼저 빼낸 것이니 다시 꾀한다. 파수 마디는 `head`과 `tail`이 결코 널이 되지 않게 하여 가장자리 경우를 쉽게 만든다. 다툼이 벌어지는 판마다 적어도 하나의 CAS가 이루어지므로 잠금 없음이 보장된다. $\square$

---

**연습문제 2.**
마이클-스콧 큐에서 넣는 실이 다른 실을 대신해 `tail` 손가락질을 앞당기는 "거들기" 장치가 왜 있어야 하는지 풀어라. 그것이 없으면 무엇이 어그러지는가?

??? success "연습문제 2 풀이"
    넣기는 두 걸음짜리 연산이다. (1) `tail->next`을 새 마디로 CAS 한다. (2) `tail`을 새 마디로 CAS 한다. 1번과 2번 사이에서 넣던 실이 밀려날 수 있다. 다른 실이 `tail` 앞당기기를 거들지 않으면 꼬리 손가락질이 참 마지막 마디보다 뒤처진다. 뒤이어 넣으려는 실은 `tail->next`을 읽어 (1번에서 만든 마디를 가리키는) 널이 아닌 값을 보지만 `tail`은 앞당겨지지 않았다. 거들기가 없으면 이 실들은 `tail->next`이 널이 되기를 기다리며 끝없이 헛돈다. 거들기 장치가 이를 푼다. 어떤 실이 `tail->next != null`을 보면 제 넣기를 다시 꾀하기 앞서 `tail`을 `tail->next`으로 CAS 하여 앞당긴다. 이로써 나아감이 지켜진다. 처음 실이 늦춰져도 `tail`이 끝내 따라잡는다. 거들기가 없으면 밀려난 실 하나가 다른 모두를 막을 수 있으므로 큐가 잠금 없는 것이 아니다. $\square$

---

**연습문제 3.**
잠금 없는 큐의 기억 다루기 어려움을 살펴라. 빼낸 뒤 마디를 곧바로 놓아줄 수 없는 까닭은 무엇이고 위험 손가락질은 이를 어떻게 다루는가?

??? success "연습문제 3 풀이"
    빼기 CAS가 이루어진 뒤 옛 머리 마디는 빼낸 실이 보기에 놓아주어도 될 듯하다. 그러나 다른 실이 아직 그 마디를 읽고 있을 수 있다(보기로 CAS에 앞서 `head`을 읽었으나 제 연산을 아직 마치지 않은 실이 있다). 마디를 곧바로 놓아주면 놓아준 뒤 쓰기 잘못이 생긴다. 위험 손가락질이 이를 푼다. 실마다 지금 닿고 있는 손가락질을 (실에 딸린 배열에) 내건다. 마디를 놓아주기 앞서 실은 모든 위험 손가락질을 살핀다. 어떤 실이라도 그 마디를 걸어 두었으면 놓아주기를 미룬다. 미룬 마디는 이따금 다시 살펴 더는 위험 손가락질로 지켜지지 않을 때 놓아준다. 이로써 어떤 실도 놓아준 기억에 닿지 않음이 보장되고, 미룬 마디의 수는 실 개수를 $T$이라 할 때 $O(T^2)$으로 매인다. $\square$

---

**연습문제 4.**
다툼이 적을 때(실 2개)와 심할 때(실 64개) 잠금 없는 큐와 잠금을 쓰는 큐의 처리량을 견주어라. 뒤집히는 자리는 무엇 때문에 생기는가?

??? success "연습문제 4 풀이"
    다툼이 적으면(실 2개) CAS가 거의 어그러지지 않으므로 잠금 없는 큐가 잠금을 쓰는 큐와 비슷하게(또는 잠금을 얻고 놓는 덧듦을 비껴가므로 조금 낫게) 돈다. 다툼이 심하면(실 64개) 잠금을 쓰는 큐는 모든 연산을 한 줄로 늘어세우고 처리량이 $1/(\text{잠금 얻는 때})$으로 매인다. 잠금 없는 큐는 한 줄로 늘어섬을 비껴가지만 다시 꾀하는 CAS로 앓는다. 실이 64개면 CAS 꾀함의 거의 모두가 어그러지고 다시 꾀할 때마다 캐시 줄 옮김을 버린다. 뒤집히는 자리는 실 8~16개 언저리에 생기는데, `head`과 `tail`(둘 다 코어 사이를 튀어 다니는 하나뿐인 캐시 줄이다)에 대한 CAS 다툼 때문에 잠금 없는 처리량이 멈춰 서기 때문이다. 그 너머에서는 연산을 뭉치는 재주(납작 뭉치기나 지워 없애기)가 두 길 모두를 앞선다. $\square$

---

**연습문제 5.**
돌림 버퍼로 매인 크기의, 여럿이 만들고 여럿이 쓰는(MPMC) 잠금 없는 큐를 설계하여라. 꽉 참과 빔을 원자적으로 다루는 길을 풀어라.

??? success "연습문제 5 풀이"
    크기가 2의 거듭제곱인 배열 `buf[N]`과 원자적인 `head`, `tail` 번호를 쓴다. 자리마다 제 번호로 처음 값을 매긴 원자적인 `sequence` 밭을 둔다. **넣기**: `tail`을 읽고 `pos = tail % N`을 셈한 뒤 `seq = buf[pos].sequence`을 읽는다. `seq == tail`이면 `tail`을 `tail + 1`로 CAS 한다. 이루면 자료를 적고 `buf[pos].sequence = tail + 1`으로 둔다. `seq < tail`이면 큐가 꽉 찬 것이다(어그러짐을 돌려준다). `seq > tail`이면 다른 실이 `tail`을 앞당긴 것이니 다시 읽고 다시 꾀한다. **빼기**: `head`으로 똑같이 한다. `head`을 읽고 `buf[pos].sequence == head + 1`인지(자료가 있는지) 살핀다. `head`을 앞으로 CAS 하고 자료를 읽은 뒤 `buf[pos].sequence = head + N`으로 두어(자리를 다시 쓸 수 있다고 표시한다) 마친다. `sequence` 밭이 자리마다의 상태 알림이 되어, (따로 있는 변수 둘에 원자적인 연산을 벌여야 하는) `head`과 `tail` 견주기 없이도 꽉 참과 빔을 알아낼 수 있게 한다. $\square$
