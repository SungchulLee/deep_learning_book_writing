# 스택의 응용

스택의 후입선출 성질은 가장 최근의 정보가 가장 중요한 문제라면 어디에나 딱 들어맞는다. 컴파일러는 괄호를 맞추고 식을 계산하는 데 스택을 쓴다. 운영체제는 함수 호출을 관리하는 데 쓴다. 그래프 알고리즘은 깊이 우선 순회에 쓴다. 이 쪽은 스택의 중요한 알고리즘적 응용을 훑어보며 각각에 대해 구체적인 예와 복잡도 분석을 제시한다.

---

## 1. 문자열 뒤집기

스택은 넣은 순서의 역순으로 원소를 내놓으므로, 문자열의 모든 문자를 스택에 넣었다가 모두 빼면 뒤집힌 문자열을 얻는다. 문자열의 길이를 $n$이라 할 때 시간은 $O(n)$, 공간은 $O(n)$이다.

```python
"""
스택의 응용 — 스택 자료 구조의 흔한 알고리즘적 쓰임.

후입선출 성질에 기댄 문자열 뒤집기, 되돌리기 장치, 깊이 우선 탐색,
다음 큰 원소 문제를 보인다.
"""

# === 스택 구현 =====================================================

class Stack:
    """시연을 위한 최소한의 스택."""

    def __init__(self):
        self._items = []

    def push(self, x):
        self._items.append(x)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)

# === 응용 1: 문자열 뒤집기 ==========================================

def reverse_string(text):
    """스택으로 문자열을 뒤집는다.

    문자를 모두 넣었다가 모두 빼면 후입선출 순서가 뒤집힌 문자열을 만든다.
    시간: O(n), 공간: O(n).
    """
    stack = Stack()
    for ch in text:
        stack.push(ch)

    result = []
    while not stack.is_empty():
        result.append(stack.pop())
    return "".join(result)

# === 응용 2: 되돌리기 장치 ===========================================

def simulate_undo(actions):
    """스택으로 되돌리기 기능을 흉내 낸다.

    동작을 하나씩 넣는다. 되돌리기는 가장 최근 동작을 뺀다.
    되돌린 동작을 시간 역순의 목록으로 돌려준다.
    """
    stack = Stack()
    for action in actions:
        stack.push(action)
        print(f"  Performed: {action}")

    undone = []
    while not stack.is_empty():
        action = stack.pop()
        undone.append(action)
        print(f"  Undo:      {action}")
    return undone

# === 응용 3: 깊이 우선 탐색 =======================================

def dfs_iterative(graph, start):
    """명시적 스택을 쓰는 반복 방식 깊이 우선 탐색.

    재귀적 깊이 우선 탐색의 암묵적인 호출 스택을 사용자가 관리하는 스택으로
    바꾸어 후입선출 순회 순서를 눈에 보이게 한다.
    시간: O(V + E), 공간: O(V).
    """
    visited = set()
    stack = Stack()
    stack.push(start)
    order = []

    while not stack.is_empty():
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            # 왼쪽에서 오른쪽으로 한결같이 순회하려고 이웃을 역순으로 넣는다
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.push(neighbor)
    return order

# === 응용 4: 다음 큰 원소 =====================================

def next_greater_element(arr):
    """단조 스택으로 각 위치의 다음 큰 원소를 찾는다.

    원소마다 다음 큰 원소는 그 오른쪽에서 처음으로 엄격히 더 큰 원소이다.
    스택을 써서 소박한 O(n^2) 대신 O(n) 시간에 해낸다.
    
    """
    n = len(arr)
    result = [-1] * n
    stack = Stack()  # 인덱스를 담는다

    for i in range(n):
        while not stack.is_empty() and arr[stack.peek()] < arr[i]:
            idx = stack.pop()
            result[idx] = arr[i]
        stack.push(i)
    return result

# === 시연 ============================================================

if __name__ == "__main__":
    # 문자열 뒤집기
    original = "stack"
    print(f"Original: '{original}' → Reversed: '{reverse_string(original)}'")
    print()

    # 되돌리기 장치
    print("Undo mechanism:")
    actions = ["type 'H'", "type 'i'", "bold text", "insert image"]
    simulate_undo(actions)
    print()

    # 반복적 깊이 우선 탐색
    graph = {
        "A": ["B", "C"],
        "B": ["D", "E"],
        "C": ["F"],
        "D": [],
        "E": [],
        "F": [],
    }
    print(f"DFS from 'A': {dfs_iterative(graph, 'A')}")
    print()

    # 다음 큰 원소
    arr = [4, 5, 2, 10, 8]
    print(f"Array:                {arr}")
    print(f"Next greater element: {next_greater_element(arr)}")
```

**출력:**
```
Original: 'stack' → Reversed: 'kcats'

Undo mechanism:
  Performed: type 'H'
  Performed: type 'i'
  Performed: bold text
  Performed: insert image
  Undo:      insert image
  Undo:      bold text
  Undo:      type 'i'
  Undo:      type 'H'

DFS from 'A': ['A', 'B', 'D', 'E', 'C', 'F']

Array:                [4, 5, 2, 10, 8]
Next greater element: [5, 10, 10, -1, -1]
```

---

## 2. 되돌리기와 이력

편집기, 그림판, 버전 관리 시스템은 모두 동작의 이력을 남긴다. 동작을 스택에 하나씩 넣고 "되돌리기" 때 빼면 자연스럽게 올바른 순서로 되돌려진다. 되돌리기나 다시 하기 한 번에 $O(1)$이 든다. 많은 시스템은 다시 하기를 위해 두 번째 스택을 더 둔다. 사용자가 동작을 되돌리면 그 동작을 되돌리기 스택에서 빼서 다시 하기 스택에 넣는다.

---

## 3. 깊이 우선 탐색

재귀적인 깊이 우선 탐색은 호출 스택을 암묵적으로 쓴다. 반복적인 구현은 이를 눈에 보이는 스택으로 바꾸어 메모리 사용을 제어할 수 있게 하고 깊은 그래프에서 스택 넘침을 피한다. 후입선출 성질 덕분에 가장 최근에 찾은 마디를 먼저 살피게 되는데, 이것이 바로 깊이 우선 전략이다. 시간은 $O(V + E)$, 공간은 $O(V)$이다.

---

## 4. 단조 스택 문제

**단조 스택**은 원소를 (증가 또는 감소하는) 정렬된 차례로 유지한다. 이 방식은 그러지 않으면 $O(n^2)$이 걸릴 여러 문제를 $O(n)$ 시간에 푼다.

- **다음 큰 원소**: 원소마다 오른쪽에서 처음으로 더 큰 원소를 찾는다
- **히스토그램에서 가장 큰 직사각형**: 이어진 막대들이 이루는 넓이가 최대인 직사각형을 찾는다
- **주가 스팬 문제**: 날마다 그 앞에서 값이 같거나 낮았던 연속된 날의 수를 센다

핵심은 각 원소가 많아야 한 번 들어가고 한 번 나온다는 점이며, 그래서 입력과 무관하게 전체 연산이 $O(n)$이 된다.

---

## 5. 응용 요약

| 응용 | 스택의 구실 | 시간 | 공간 |
|---|---|---|---|
| 문자열 뒤집기 | 문자의 순서를 뒤집는다 | $O(n)$ | $O(n)$ |
| 되돌리기 장치 | 동작의 이력을 남긴다 | 연산당 $O(1)$ | $O(n)$ |
| 깊이 우선 탐색 | 아직 살피지 않은 마디를 기록한다 | $O(V + E)$ | $O(V)$ |
| 다음 큰 원소 | 단조로운 차례를 유지한다 | $O(n)$ | $O(n)$ |
| 균형 잡힌 괄호 | 여는 괄호와 닫는 괄호를 맞춘다 | $O(n)$ | $O(n)$ |
| 식의 계산 | 피연산자와 연산자를 다룬다 | $O(n)$ | $O(n)$ |

식의 계산, 중위에서 후위로의 변환, 괄호 균형 확인, 함수 호출 흉내 내기는 각각의 이웃 쪽에서 자세히 다룬다.

---

## 연습문제

**연습문제 1.**
스택의 응용의 추상 자료형이 지원하는 연산을 시간 복잡도와 함께 모두 열거하라. 어느 연산이 병목인가?

??? success "연습문제 1 풀이"
    추상 자료형은 구현과 무관하게 지원하는 연산을 정한다. 무엇이 병목인지는 쓰임새에 달렸다. 실시간 시스템에서는 최악의 복잡도가 중요하고, 일괄 처리에서는 상각 복잡도로 충분하다.

---

**연습문제 2.**
스택의 응용을(를) 서로 다른 두 자료 구조로 구현하라. 각각의 절충을 비교하라.

??? success "연습문제 2 풀이"
    구현 1: 배열 기반 — 접근은 상수 시간이지만 크기를 다시 잡아야 할 수 있다. 구현 2: 연결 리스트 기반 — 삽입과 삭제는 상수 시간이지만 접근은 $O(n)$이다. 어느 쪽을 고를지는 응용에서 예상되는 연산의 구성에 달렸다.

---

**연습문제 3.**
스택의 응용을(를) 쓰는 딥러닝 응용을 하나 설명하라(예: 그래프 신경망의 너비 우선 탐색, 기호 미분에서의 식 계산, 데이터 적재의 스케줄링).

??? success "연습문제 3 풀이"
    구체적인 응용은 그 추상 자료형의 순서 성질에 달렸다. 선입선출(큐)은 GNN의 너비 우선 그래프 순회에 쓰이고, 후입선출(스택)은 자동 미분 테이프 처리에 쓰이며, 우선순위 순서는 빔 탐색과 예정 표집에 쓰인다.

---

**연습문제 4.**
스택의 응용을(를) 원형 배열로 구현하면 모든 연산이 상각 $O(1)$ 시간임을 증명하라.

??? success "연습문제 4 풀이"
    원형 배열은 머리와 꼬리 인덱스를 용량으로 나눈 나머지로 관리한다. 넣기와 빼기는 인덱스를 $O(1)$에 조정한다. 배열이 가득 차면 용량을 두 배로 늘리는 데 $O(n)$이 들지만, 이는 값싼 연산 $O(n)$번 뒤에 한 번 일어나므로 동적 배열과 같은 논법으로 상각 $O(1)$이 된다. $\square$

## 정리하며

이 마당은 문자열 뒤집기、되돌리기와 이력、깊이 우선 탐색、단조 스택 문제을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
