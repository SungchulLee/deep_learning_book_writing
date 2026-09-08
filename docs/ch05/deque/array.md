# 배열 기반 덱

[덱 추상 자료형](adt.md)은 양쪽 끝에서 $O(1)$ 넣기와 빼기를 요구한다. 소박한 배열은 뒤에서만 효율적이다. 앞에 넣으려면 모든 원소를 오른쪽으로 옮겨야 하므로 $O(n)$이 든다. **원형 배열**(링 버퍼)은 바탕 배열의 양 끝이 이어져 있는 것처럼 다루어 이 문제를 없앤다. 그러면 간단한 인덱스 산술만으로 앞과 뒤가 모두 늘고 줄 수 있다. 이 쪽은 원형 배열 덱을 설명하고, 상각 복잡도를 유도하며, 완전한 파이썬 구현을 제시한다.

---

## 1. 원형 버퍼의 착상

용량이 $C$인 원형 배열은 원소를 위치 $0, 1, \dots, C-1$에 저장한다. 두 인덱스가 덱의 경계를 나타낸다.

- **front**: 첫 원소의 인덱스.
- **back**: 마지막 원소의 *바로 다음* 인덱스(뒤쪽의 다음 빈 칸).

덱의 크기는 다음과 같다.

$$
n = (\texttt{back} - \texttt{front}) \bmod C
$$

모든 인덱스 산술은 나머지 연산을 쓰므로 $C-1$을 넘어 늘면 $0$으로 감기고, $0$ 아래로 줄면 $C-1$으로 감긴다.

---

## 2. 연산

### 뒤에 넣기

원소 $x$을 뒤에 넣으려면 다음과 같이 한다.

1. 인덱스 `back`에 $x$을 저장한다.
2. `back = (back + 1) mod C`으로 갱신한다.

### 앞에 넣기

원소 $x$을 앞에 넣으려면 다음과 같이 한다.

1. `front = (front - 1) mod C`으로 갱신한다.
2. 새 `front` 인덱스에 $x$을 저장한다.

### 뒤에서 빼기

뒤의 원소를 빼려면 다음과 같이 한다.

1. `back = (back - 1) mod C`으로 갱신한다.
2. 새 `back` 인덱스의 원소를 돌려준다.

### 앞에서 빼기

앞의 원소를 빼려면 다음과 같이 한다.

1. 인덱스 `front`의 원소를 따로 둔다.
2. `front = (front + 1) mod C`으로 갱신한다.
3. 따로 둔 원소를 돌려준다.

이 네 연산은 각각 상수 번의 단계를 밟으므로, 배열의 크기를 다시 잡을 필요가 없을 때 최악의 경우에도 $O(1)$ 시간이다.

---

## 3. 동적 크기 조정

용량이 고정된 원형 배열은 $n = C - 1$일 때 가득 찬다(가득 참과 비어 있음을 가르려고 한 칸을 비워 둔다). 원소를 얼마든지 담으려면 가득 찼을 때 배열을 **두 배로** 늘리고, 크기가 $C / 4$ 아래로 떨어지면 **절반으로** 줄인다.

**두 배로 늘리기** (배열이 가득 찼을 때):

1. 크기가 $2C$인 새 배열을 할당한다.
2. 원소 $n$개를 모두 새 배열의 위치 $0, 1, \dots, n-1$으로 복사한다.
3. `front = 0`, `back = n`으로 둔다.

**절반으로 줄이기** ($n < C / 4$일 때):

1. 크기가 $C / 2$인 새 배열을 할당한다.
2. 원소 $n$개를 모두 위치 $0, 1, \dots, n-1$으로 복사한다.
3. `front = 0`, `back = n`으로 둔다.

### 상각 분석

크기를 다시 잡지 않는다면 넣기나 빼기 한 번에 $O(1)$이 든다. 크기를 다시 잡을 때에는 원소 $n$개를 복사하므로 $O(n)$이 든다. 표준적인 두 배 늘리기 논법을 쓰면(참고: [상각 분석 쪽](../../ch02/amortized/aggregate.md)), 빈 덱에서 시작하여 넣기와 빼기를 $m$번 하는 어떤 열이든 전체 $O(m)$ 시간이 든다.

$$
\text{Amortized cost per operation} = O(1)
$$

!!! tip "회계법으로 보는 직관"
    넣기 한 번에 $3$단위를 매긴다. $1$은 실제 삽입에 쓰고 $2$은 적립해 둔다. 두 배로 늘릴 때 배열의 원소 $n$개가 모아 둔 적립금이 적어도 $n$이므로 $O(n)$의 복사 비용을 치를 수 있다.

---

## 4. 파이썬 구현

```python
"""크기를 동적으로 조정하는 원형 버퍼로 만든 배열 기반 덱."""

# === 원형 배열 덱 ===

class ArrayDeque:
    """순환 배열로 뒷받침한 양방향 큐.

    네 가지 넣기/빼기 연산이 모두 상각 O(1) 시간에 돌아간다.
    내부 배열은 가득 차면 두 배로 늘고 4분의 1만 차면 절반으로 준다.
    """

    _MIN_CAPACITY = 8

    def __init__(self):
        self._capacity = self._MIN_CAPACITY
        self._data = [None] * self._capacity
        self._front = 0
        self._size = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    # --- 넣기 연산 ---

    def push_back(self, x):
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        back = (self._front + self._size) % self._capacity
        self._data[back] = x
        self._size += 1

    def push_front(self, x):
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        self._front = (self._front - 1) % self._capacity
        self._data[self._front] = x
        self._size += 1

    # --- 빼기 연산 ---

    def pop_front(self):
        if self.is_empty():
            raise IndexError("pop from empty deque")
        value = self._data[self._front]
        self._data[self._front] = None
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        if self._size > 0 and self._size <= self._capacity // 4:
            self._resize(max(self._MIN_CAPACITY, self._capacity // 2))
        return value

    def pop_back(self):
        if self.is_empty():
            raise IndexError("pop from empty deque")
        back = (self._front + self._size - 1) % self._capacity
        value = self._data[back]
        self._data[back] = None
        self._size -= 1
        if self._size > 0 and self._size <= self._capacity // 4:
            self._resize(max(self._MIN_CAPACITY, self._capacity // 2))
        return value

    # --- 들여다보기 연산 ---

    def front(self):
        if self.is_empty():
            raise IndexError("front from empty deque")
        return self._data[self._front]

    def back(self):
        if self.is_empty():
            raise IndexError("back from empty deque")
        return self._data[(self._front + self._size - 1) % self._capacity]

    # --- 내부 ---

    def _resize(self, new_capacity):
        new_data = [None] * new_capacity
        for i in range(self._size):
            new_data[i] = self._data[(self._front + i) % self._capacity]
        self._data = new_data
        self._front = 0
        self._capacity = new_capacity

# === 시연 ===

if __name__ == "__main__":
    dq = ArrayDeque()
    for v in [10, 20, 30]:
        dq.push_back(v)
    dq.push_front(5)
    print(f"Front: {dq.front()}")   # 5
    print(f"Back:  {dq.back()}")    # 30
    print(f"Size:  {len(dq)}")      # 4
    print(f"Pop front: {dq.pop_front()}")  # 5
    print(f"Pop back:  {dq.pop_back()}")   # 30
    print(f"Size:  {len(dq)}")      # 2
```

---

## 5. 복잡도 요약

| 연산 | 최악의 경우 | 상각 |
|---|---|---|
| `push_front(x)` | $O(n)$ (크기 재조정) | $O(1)$ |
| `push_back(x)` | $O(n)$ (크기 재조정) | $O(1)$ |
| `pop_front()` | $O(n)$ (크기 재조정) | $O(1)$ |
| `pop_back()` | $O(n)$ (크기 재조정) | $O(1)$ |
| `front()` / `back()` | $O(1)$ | $O(1)$ |
| `is_empty()` / `size()` | $O(1)$ | $O(1)$ |
| 공간 | — | $O(n)$ |

---

## 연습문제

**연습문제 1.**
배열 기반 덱의 추상 자료형이 지원하는 연산을 시간 복잡도와 함께 모두 열거하라. 어느 연산이 병목인가?

??? success "연습문제 1 풀이"
    추상 자료형은 구현과 무관하게 지원하는 연산을 정한다. 무엇이 병목인지는 쓰임새에 달렸다. 실시간 시스템에서는 최악의 복잡도가 중요하고, 일괄 처리에서는 상각 복잡도로 충분하다.

---

**연습문제 2.**
배열 기반 덱을(를) 서로 다른 두 자료 구조로 구현하라. 각각의 절충을 비교하라.

??? success "연습문제 2 풀이"
    구현 1: 배열 기반 — 접근은 상수 시간이지만 크기를 다시 잡아야 할 수 있다. 구현 2: 연결 리스트 기반 — 삽입과 삭제는 상수 시간이지만 접근은 $O(n)$이다. 어느 쪽을 고를지는 응용에서 예상되는 연산의 구성에 달렸다.

---

**연습문제 3.**
배열 기반 덱을(를) 쓰는 딥러닝 응용을 하나 설명하라(예: 그래프 신경망의 너비 우선 탐색, 기호 미분에서의 식 계산, 데이터 적재의 스케줄링).

??? success "연습문제 3 풀이"
    구체적인 응용은 그 추상 자료형의 순서 성질에 달렸다. 선입선출(큐)은 GNN의 너비 우선 그래프 순회에 쓰이고, 후입선출(스택)은 자동 미분 테이프 처리에 쓰이며, 우선순위 순서는 빔 탐색과 예정 표집에 쓰인다.

---

**연습문제 4.**
배열 기반 덱을(를) 원형 배열로 구현하면 모든 연산이 상각 $O(1)$ 시간임을 증명하라.

??? success "연습문제 4 풀이"
    원형 배열은 머리와 꼬리 인덱스를 용량으로 나눈 나머지로 관리한다. 넣기와 빼기는 인덱스를 $O(1)$에 조정한다. 배열이 가득 차면 용량을 두 배로 늘리는 데 $O(n)$이 들지만, 이는 값싼 연산 $O(n)$번 뒤에 한 번 일어나므로 동적 배열과 같은 논법으로 상각 $O(1)$이 된다. $\square$

## 정리하며

이 마당은 원형 버퍼의 착상、연산、동적 크기 조정、파이썬 구현을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
