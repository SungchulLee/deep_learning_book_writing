# 요세푸스 문제

요세푸스 문제는 원형 연결 리스트에서 자연스럽게 나오는 고전적인 세어 나가기 퍼즐이다. $n$명이 원을 이루고 서 있다고 하자. 정해진 첫 사람에서 시작하여 한 사람만 남을 때까지 $k$번째 사람을 차례로 없앤다. 물음은 이것이다. **어느 자리가 살아남는가?** 이 문제는 라운드 로빈 스케줄링, 게임 이론, 암호학에 등장한다. 또한 원형 연결 리스트가 왜 존재하는지를 설득력 있게 보여준다. 원형 구조가 문제의 기하를 그대로 비추기 때문이다.

---

## 1. 문제 서술

$0, 1, \ldots, n-1$으로 번호가 매겨진 $n$명이 원을 이루고 있고 걸음 크기가 $k \geq 1$일 때 다음과 같이 한다.

1. 0번 사람에서 시작한다.
2. (출발한 사람을 포함하여) 시계 방향으로 $k$명을 센다.
3. $k$번째 사람을 없앤다.
4. 한 사람이 남을 때까지 다음 사람에서부터 되풀이한다.

**요세푸스 수** $J(n, k)$은 마지막 생존자의 자리이다.

---

## 2. 원형 연결 리스트로 모의실험하기

가장 직관적인 방법은 원형 단일 연결 리스트로 그 과정을 그대로 흉내 내는 것이다. 각 노드가 한 사람을 나타낸다. 어떤 사람이 없어지면 그에 해당하는 노드를 지우고 고리를 따라 계속 세어 나간다.

```python
"""
원형 연결 리스트에서 제거를 모의실험하여 푸는 요세푸스 문제.

원형 리스트의 순회, 노드 삭제, 그리고 검증을 위한 요세푸스
점화식을 보인다.
"""

# === 노드와 원형 리스트 ===

class Node:
    """원형 단일 연결 리스트의 노드."""

    def __init__(self, data):
        self.data = data
        self.next = None

def build_circle(n):
    """노드 0, 1, ..., n-1로 이루어진 원형 단일 연결 리스트를 만든다.

    0번 사람의 노드를 돌려준다.
    """
    head = Node(0)
    current = head
    for i in range(1, n):
        current.next = Node(i)
        current = current.next
    current.next = head       # 고리 닫기
    return head

# === 모의실험 방식 ===

def josephus_simulation(n, k):
    """원형 리스트 모의실험으로 생존자의 자리를 돌려준다.

    시간: O(n * k) -- n-1번의 제거마다 k 걸음이 필요하다.
    공간: 원형 연결 리스트를 위해 O(n).
    """
    head = build_circle(n)
    current = head

    # 시작 위치 바로 앞의 노드 찾기
    prev = head
    while prev.next is not head:
        prev = prev.next

    for _ in range(n - 1):
        # 앞으로 k-1 걸음 세기 (현재가 1걸음째)
        for _ in range(k - 1):
            prev = current
            current = current.next
        # 현재 노드 제거
        prev.next = current.next
        current = current.next

    return current.data

# === 점화식 방식 ===

def josephus_recurrence(n, k):
    """수학적 점화식으로 생존자의 자리를 돌려준다.

    0부터 세는 위치에 대한 요세푸스 점화식:
        J(1, k) = 0
        J(n, k) = (J(n-1, k) + k) mod n

    시간: O(n)
    공간: O(1)
    """
    survivor = 0
    for i in range(2, n + 1):
        survivor = (survivor + k) % i
    return survivor

# === 메인 ===

if __name__ == "__main__":
    # 예: 7명, 세 번째마다 제거
    n, k = 7, 3
    print(f"Josephus({n}, {k})")
    print(f"  Simulation:  {josephus_simulation(n, k)}")
    print(f"  Recurrence:  {josephus_recurrence(n, k)}")

    # 여러 입력에 대해 두 방법이 일치하는지 확인
    print("\nVerification:")
    for n in range(1, 11):
        sim = josephus_simulation(n, 3)
        rec = josephus_recurrence(n, 3)
        status = "OK" if sim == rec else "MISMATCH"
        print(f"  n={n:2d}, k=3: survivor={sim}  [{status}]")
```

**출력:**

```
Josephus(7, 3)
  Simulation:  3
  Recurrence:  3

Verification:
  n= 1, k=3: survivor=0  [OK]
  n= 2, k=3: survivor=1  [OK]
  n= 3, k=3: survivor=1  [OK]
  n= 4, k=3: survivor=0  [OK]
  n= 5, k=3: survivor=3  [OK]
  n= 6, k=3: survivor=0  [OK]
  n= 7, k=3: survivor=3  [OK]
  n= 8, k=3: survivor=6  [OK]
  n= 9, k=3: survivor=0  [OK]
  n=10, k=3: survivor=3  [OK]
```

---

## 3. 요세푸스 점화식

없애는 과정을 실제로 흉내 내는 대신 점화식으로도 이 문제를 풀 수 있다. (위치 $k - 1$의) 첫 사람을 없애고 나면 남은 $n - 1$명이 새 원을 이루는데, 번호가 $k$자리만큼 밀린다. 여기서 0에서 시작하는 인덱스의 점화식이 나온다.

$$
J(1, k) = 0
$$

$$
J(n, k) = \bigl(J(n-1, k) + k\bigr) \bmod n \quad \text{for } n \geq 2
$$

이 점화식은 (위 코드에 있는) 간단한 반복문으로 $O(n)$ 시간과 $O(1)$ 공간에 답을 구한다. 모의실험의 $O(nk)$과 견주어 보라.

??? note "점화식의 유도"
    $n$명의 원에서 (0에서 시작하는 인덱스로) 위치 $k - 1$의 사람을 없앤 뒤, 남은 $n - 1$명에게 위치 $k$에서부터 번호를 다시 매긴다. 옛 위치 $j$의 사람은 새 위치 $(j - k) \bmod (n-1)$으로 간다. 번호를 다시 매긴 원에서의 생존자가 $J(n-1, k)$이면, 원래 원에서의 생존자는 위치 $(J(n-1, k) + k) \bmod n$에 있다.

---

## 4. 특수한 경우: k = 2

$k = 2$일 때 요세푸스 문제에는 닫힌 형태의 해가 있다. $0 \leq \ell < 2^m$인 $n = 2^m + \ell$으로 쓰면 다음이 성립한다.

$$
J(n, 2) = 2\ell + 1
$$

이는 비트 연산으로 $O(\log n)$ 시간에 계산할 수 있다. $n$에서 켜진 최상위 비트를 찾고 이진 표현을 왼쪽으로 한 자리 돌리면 된다.

---

## 5. 복잡도 요약

| 방법 | 시간 | 공간 |
|---|---|---|
| 원형 리스트 모의실험 | $O(nk)$ | $O(n)$ |
| 반복적 점화식 | $O(n)$ | $O(1)$ |
| 닫힌 형태 ($k = 2$일 때만) | $O(\log n)$ | $O(1)$ |

모의실험 방식이 가장 직관적이고 원형 연결 리스트의 힘을 잘 보여주지만, 입력이 클 때는 점화식이 압도적으로 효율적이다.

---

## 연습문제

**연습문제 1.**
요세푸스 문제에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 요세푸스 문제을(를) 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
요세푸스 문제이(가) PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 요세푸스 문제의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$

## 정리하며

이 마당은 문제 서술、원형 연결 리스트로 모의실험하기、요세푸스 점화식、특수한 경우: k = 2을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.), Problem 14-2. MIT Press.
- Graham, R. L., Knuth, D. E., & Patashnik, O. *Concrete Mathematics*
  (2nd ed.), Section 1.3. Addison-Wesley.
