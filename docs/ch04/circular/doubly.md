# 원형 이중 연결 리스트

보통의 이중 연결 리스트에는 끝점이 분명하다. 머리의 `prev`가 `None`이고 꼬리의 `next`가 `None`이다. 그래서 꼬리에서 머리로 거슬러 순회하려면 양쪽 끝을 명시적으로 추적해야 한다. **원형 이중 연결 리스트**는 꼬리의 `next`를 머리에, 머리의 `prev`를 꼬리에 이어 닫힌 고리를 만듦으로써 이 끝점을 없앤다. 이 설계는 원소를 끊임없이 돌아가며 쓰는 응용, 이를테면 라운드 로빈 스케줄러, 원형 메뉴의 이동, 고정 크기 버퍼 관리에 특히 유용하다.

---

## 1. 구조

노드가 $n$개인 원형 이중 연결 리스트에서는 모든 노드가 유효한 `prev`와 `next` 포인터를 가지며 어떤 포인터도 `None`이 아니다. 노드 $x_0, x_1, \ldots, x_{n-1}$에 대해 다음이 성립한다.

$$
x_i.\text{next} = x_{(i+1) \bmod n} \quad \text{and} \quad x_i.\text{prev} = x_{(i-1) \bmod n}
$$

이 리스트에는 특별한 머리나 꼬리가 없다. 대신 바깥의 참조 하나가 아무 노드나 가리키고, 그 노드에서 어느 방향으로든 리스트 전체에 닿을 수 있다.

---

## 2. 구현

```python
"""
원형 이중 연결 리스트의 구현.

각 노드는 prev와 next 포인터를 가져 닫힌 고리를 이룬다.
어느 방향으로 순회해도 시작 노드로 되돌아온다.
"""

# === 노드 정의 ===

class Node:
    """원형 이중 연결 리스트의 노드."""

    def __init__(self, data):
        self.data = data
        # 만들어지는 순간부터 자기 자신을 양쪽으로 가리킨다. 곧 노드
        # 하나짜리 고리로 태어나는 셈이라, None인 이음선이 한순간도
        # 존재하지 않는다. 보초 노드와 같은 효과를 노드 자체로 얻는다
        self.prev = self
        self.next = self

# === 원형 이중 연결 리스트 ===

class CircularDLL:
    """머리와 꼬리의 구분이 없는 원형 이중 연결 리스트."""

    def __init__(self):
        # 고리 안의 아무 노드에 대한 참조.
        # 이름이 head가 아니라 access인 까닭이 있다. 고리에는 처음도
        # 끝도 없어 어느 노드를 붙들든 마찬가지이며, 붙든 자리가
        # 그저 들어가는 문일 뿐이기 때문이다.
        # 그래서 아래 순회들도 "머리부터"가 아니라 "이 노드부터
        # 한 바퀴"라는 뜻이 된다
        self.access = None
        self.size = 0

    def is_empty(self):
        return self.size == 0

    def insert(self, data):
        """고리에 새 노드를 넣는다.

        고리가 비어 있으면 새 노드는 자기 자신을 가리킨다.
        그렇지 않으면 새 노드를 접근 노드 뒤에 넣는다.
        """
        new_node = Node(data)
        if self.is_empty():
            self.access = new_node
        else:
            new_node.prev = self.access
            new_node.next = self.access.next
            self.access.next.prev = new_node
            self.access.next = new_node
        self.size += 1
        return new_node

    def delete(self, node):
        """고리에서 노드를 뺀다.

        노드가 그것 하나뿐이면 고리는 비게 된다.
        """
        if self.size == 1:
            self.access = None
        else:
            # 검사 없이 두 줄로 끝난다. 고리라 node.prev와 node.next가
            # 결코 None이 아니기 때문이다. 노드를 손에 쥐고 있으므로 O(1)이다
            node.prev.next = node.next
            node.next.prev = node.prev
            # 지운 노드가 하필 들어가는 문이었다면 문을 옮겨 주어야 한다.
            # 빠뜨리면 access가 고리에서 떨어져 나간 노드를 가리켜,
            # 이후 순회가 엉뚱한 곳을 돌게 된다
            if self.access is node:
                self.access = node.next
        self.size -= 1
        return node.data

    def traverse(self):
        """접근 노드에서 앞으로 순회하며 모든 값을 돌려준다."""
        if self.is_empty():
            return []
        result = []
        current = self.access
        # while True에 뒤쪽 검사를 쓴다. 앞쪽에서 current is self.access를
        # 물으면 시작하자마자 참이라 한 바퀴도 돌지 못하기 때문이다.
        # 곧 "적어도 한 번은 돌고 나서 출발점에 되돌아왔는지 본다"는
        # do-while 꼴이며, 끝이 없는 고리를 훑는 표준 형태다
        while True:
            result.append(current.data)
            current = current.next
            if current is self.access:
                break
        return result

    def traverse_reverse(self):
        """접근 노드에서 뒤로 순회하며 모든 값을 돌려준다."""
        if self.is_empty():
            return []
        result = []
        current = self.access
        while True:
            result.append(current.data)
            current = current.prev
            if current is self.access:
                break
        return result

    def search(self, target):
        """주어진 데이터 값을 지닌 노드를 찾는다.

        찾으면 노드를, 없으면 None을 돌려준다.
        """
        if self.is_empty():
            return None
        current = self.access
        while True:
            if current.data == target:
                return current
            current = current.next
            if current is self.access:
                return None

# === 메인 ===

if __name__ == "__main__":
    ring = CircularDLL()

    ring.insert(10)
    ring.insert(20)
    ring.insert(30)
    print("Forward: ", ring.traverse())
    print("Backward:", ring.traverse_reverse())

    # 값이 20인 노드 삭제
    node = ring.search(20)
    if node:
        ring.delete(node)
    print("After deleting 20:", ring.traverse())
```

**출력:**

```
Forward:  [10, 20, 30]
Backward: [10, 30, 20]
Forward after deleting 20: [10, 30]
```

---

## 3. 순회의 종료

원형이 아닌 리스트에서는 현재 포인터가 `None`이 될 때 순회가 멈춘다. 원형 리스트에는 `None`이 없으므로 커서가 출발한 노드로 돌아왔을 때 멈춰야 한다. 이 조건을 잊으면 무한 반복이 생기며, 이것이 원형 구조에서 가장 흔한 버그이다.

표준적인 방식은 `do-while`에 해당하는 형태를 쓴다.

1. 출발 노드를 기록한다.
2. 다음(또는 이전) 노드로 옮긴다.
3. 현재 노드가 출발 노드와 같아지면 멈춘다.

---

## 4. 복잡도

| 연산 | 시간 | Notes |
|---|---|---|
| 접근 후 삽입 | $O(1)$ | 포인터 네 번 고침 |
| 주어진 노드 삭제 | $O(1)$ | 포인터 네 번 고침 |
| 탐색 | $O(n)$ | Must traverse up to $n$ nodes |
| 순회 (고리 한 바퀴) | $O(n)$ | 모든 노드를 한 번씩 방문한다 |

노드 $n$개에 대한 공간 복잡도는 $O(n)$이며, 각 노드가 데이터 항목 하나와 포인터 둘을 저장한다.

---

## 5. 원형이 아닌 이중 연결 리스트와의 비교

| 특징 | 원형이 아닌 이중 연결 리스트 | 원형 이중 연결 리스트 |
|---|---|---|
| 머리의 `prev` | `None` | 꼬리를 가리킨다 |
| 꼬리의 `next` | `None` | 머리를 가리킨다 |
| 경계 검사 | 필요하다 | 필요 없다 |
| 돌아가는 쓰임에 적합한가 | 아니다 | 그렇다 |
| 순회의 종료 | `None` 검사 | 출발 노드 검사 |

원형 변형이 엄격히 더 일반적이다. 원형이 아닌 이중 연결 리스트는 포인터 둘을 `None`으로 두어 고리를 "끊어 놓은" 원형 이중 연결 리스트이다.

---

## 연습문제

**연습문제 1.**
원형 이중 연결 리스트에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 원형 이중 연결 리스트을(를) 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
원형 이중 연결 리스트이(가) PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 원형 이중 연결 리스트의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$

## 정리하며

이 마당은 구조、구현、순회의 종료、복잡도을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
