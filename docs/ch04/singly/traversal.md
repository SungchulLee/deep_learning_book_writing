# 순회

순회는 연결 리스트의 가장 기본적인 연산으로, 머리에서 꼬리까지 모든 노드를 차례로 방문하는 일이다. 탐색, 세기, 출력, 합 구하기, 최댓값 찾기 등 연결 리스트의 다른 거의 모든 연산이 순회 위에 세워진다. 단일 연결 리스트의 노드는 메모리에 흩어져 있고 `next` 포인터로만 이어져 있으므로, 모든 원소를 방문하는 유일한 길은 사슬을 한 노드씩 따라가는 것이다. 반복 형태에서는 $O(n)$ 시간과 $O(1)$ 공간이 들며, 이것이 다른 모든 연결 리스트 연산을 재는 기준선이 된다.

## 반복적 순회

표준적인 순회 방식은 머리에서 시작해 노드를 하나씩 나아가는 포인터 하나를 쓴다.

**알고리즘:**

1. `current = head`로 둔다.
2. `current`가 `None`이 아닌 동안 다음을 반복한다.
    - `current.data`를 처리한다(출력, 누적, 변환 등).
    - 나아간다: `current = current.next`.

**시간 복잡도:** $O(n)$이다. $n$개의 노드를 정확히 한 번씩 방문한다.

**공간 복잡도:** $O(1)$이다. 포인터 변수 하나뿐이다.

이 방식은 for 반복문으로 배열을 훑는 것에 해당하는 연결 리스트판이다. 핵심 차이는 각 "걸음"이 인덱스 증가가 아니라 포인터 역참조라는 점이며, 그래서 메모리 접근이 순차적이지 않고 캐시에 덜 친화적이다.

## 재귀적 순회

순회는 재귀로도 표현할 수 있다. 재귀 형태는 현재 노드를 처리한 뒤 리스트의 나머지에 대해 자기 자신을 호출한다.

**알고리즘:**

```
traverse(node):
    if node is None:
        return
    process(node.data)
    traverse(node.next)
```

**시간 복잡도:** $O(n)$.

**공간 복잡도:** 재귀 스택 프레임이 $n$개 쌓이므로 $O(n)$이다.

!!! warning "재귀의 깊이"

    재귀적 순회는 우아하지만 $O(n)$의 스택 공간을 쓰므로 긴 리스트에서는 스택 넘침이 난다. 파이썬의 기본 재귀 한도는 1000 프레임이다. 실제로 쓰는 코드에서는 언제나 반복 방식이 낫다.

## 흔한 순회 패턴

몇 가지 실용적인 연산은 처리 단계만 달리한 순회의 직접적인 응용이다.

### 노드 세기

순회 중에 계수기를 늘려 노드의 개수를 센다.

**시간:** $O(n)$ | **공간:** $O(1)$

### 값 더하기

한 번의 순회로 모든 노드 값의 합을 누적한다.

**시간:** $O(n)$ | **공간:** $O(1)$

### 최댓값 찾기

노드를 방문할 때마다 지금까지 본 최댓값을 갱신한다.

**시간:** $O(n)$ | **공간:** $O(1)$

### 리스트로 모으기

모든 노드 값을 담은 파이썬 리스트를 만든다. 출력하거나 기대 결과와 비교할 때 유용하다.

**시간:** $O(n)$ | **공간:** 결과 리스트에 $O(n)$

### 리스트 출력하기

순회 중에 값을 모으고 화살표로 이어 `a -> b -> c -> None` 형식으로 리스트를 보여준다.

## 구현

```python
"""단일 연결 리스트의 순회 연산."""


# === 노드 클래스 ===
class Node:
    """단일 연결 리스트의 노드 하나."""

    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node

    def __repr__(self):
        return f"Node({self.data})"


# === 만들기 도우미 ===
def build_list(values):
    """반복 가능한 객체에서 연결 리스트를 만들고 머리 노드를 돌려준다."""
    head = None
    for val in reversed(values):
        head = Node(val, head)
    return head


# === 반복적 순회 ===
def print_list(head):
    """연결 리스트를 화살표 표기로 출력한다."""
    parts = []
    current = head
    while current is not None:
        parts.append(str(current.data))
        current = current.next
    print(" -> ".join(parts) + " -> None")


# === 노드 세기 ===
def count_nodes(head):
    """리스트의 노드 수를 돌려준다. 시간: O(n), 공간: O(1)."""
    count = 0
    current = head
    while current is not None:
        count += 1
        current = current.next
    return count


# === 값 더하기 ===
def sum_values(head):
    """모든 노드 값의 합을 돌려준다. 시간: O(n), 공간: O(1)."""
    total = 0
    current = head
    while current is not None:
        total += current.data
        current = current.next
    return total


# === 최댓값 찾기 ===
def find_max(head):
    """리스트의 최댓값을 돌려준다. 시간: O(n), 공간: O(1)."""
    if head is None:
        raise ValueError("Cannot find max of empty list")
    max_val = head.data
    current = head.next
    while current is not None:
        if current.data > max_val:
            max_val = current.data
        current = current.next
    return max_val


# === 파이썬 리스트로 모으기 ===
def to_list(head):
    """모든 노드 값을 파이썬 리스트로 모은다. 시간: O(n), 공간: O(n)."""
    result = []
    current = head
    while current is not None:
        result.append(current.data)
        current = current.next
    return result


# === 재귀적 순회 ===
def print_recursive(node):
    """값을 재귀적으로 출력한다. 시간: O(n), 공간: O(n)."""
    if node is None:
        print("None")
        return
    print(f"{node.data} -> ", end="")
    print_recursive(node.next)


# === 각 노드에 함수 적용 ===
def for_each(head, func):
    """각 노드의 data에 func을 적용한다. 시간: O(n), 공간: O(1)."""
    current = head
    while current is not None:
        func(current.data)
        current = current.next


# === 시연 ===
if __name__ == "__main__":
    head = build_list([10, 20, 30, 40, 50])

    # 반복적 순회로 출력
    print("Iterative print:")
    print_list(head)

    # 재귀적 순회로 출력
    print("\nRecursive print:")
    print_recursive(head)

    # 개수, 합, 최댓값
    print(f"\nCount: {count_nodes(head)}")
    print(f"Sum:   {sum_values(head)}")
    print(f"Max:   {find_max(head)}")

    # 파이썬 리스트로 모으기
    print(f"As list: {to_list(head)}")

    # 함수 적용
    print("\nDoubled values:")
    for_each(head, lambda x: print(f"  {x * 2}"))
```

**출력:**
```
Iterative print:
10 -> 20 -> 30 -> 40 -> 50 -> None

Recursive print:
10 -> 20 -> 30 -> 40 -> 50 -> None

Count: 5
Sum:   150
Max:   50
As list: [10, 20, 30, 40, 50]

Doubled values:
  20
  40
  60
  80
  100
```

## 복잡도 요약

| 연산         | 시간   | 공간  |
|-------------------|--------|--------|
| 반복적 순회 | $O(n)$ | $O(1)$ |
| 재귀적 순회 | $O(n)$ | $O(n)$ |
| 노드 세기       | $O(n)$ | $O(1)$ |
| 값 더하기        | $O(n)$ | $O(1)$ |
| 최댓값 찾기      | $O(n)$ | $O(1)$ |
| 리스트로 모으기   | $O(n)$ | $O(n)$ |

## 참고 문헌

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
순회에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 순회을(를) 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
순회이(가) PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 순회의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$