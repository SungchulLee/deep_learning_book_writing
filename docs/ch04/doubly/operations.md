# 삽입과 삭제
<div align="center"><img src="https://cdn-images-1.medium.com/max/1600/1*ETR5djgjMV_M2Oeitf4XZQ.png" width="100%" height="30%"></div>

```python
class Node:
    def __init__(self, prev_data=None, data=None, next_data=None):
        self.prev = prev_data
        self.data = data
        self.next = next_data
```

```python
class DoublyLinkedList:
    def __init__(self, data_list=None):
        self.head_prev = None
        self.head = None
        if data_list is not None:
            for data in data_list:
                self.insert_at_end(data)
        
    def get_length(self):
        count = 0
        itr = self.head
        while itr:
            count += 1
            itr = itr.next
        return count
    
    def insert_at_start(self, data):
        node = Node(None, data, self.head)
        self.head = node

    def insert_at_end(self, data):
        if self.head is None:
            self.insert_at_start(data)
            return
        itr = self.head
        while itr.next:
            itr = itr.next
        node = Node(itr, data, None)
        itr.next = node
        
    def insert(self, index, data):
        if (index < 0) or (index > self.get_length()):
            raise Exception("Invalid Index")
        if index == 0:
            self.insert_at_start(data)
            return
        itr = self.head
        for _ in range(1,index):
            itr = itr.next
        node = Node(itr, data, itr.next)
        itr.next = node
        
    def remove(self, index):
        if (index < 0) or (index >= self.get_length()):
            raise Exception("Invalid Index")
        if index==0:
            self.head = self.head.next
            return
        itr = self.head
        for _ in range(1,index):
            itr = itr.next
        itr.next = itr.next.next
        itr.next.prev = itr
        
    def print_linked_list(self):
        if self.head is None:
            print("Linked list is empty")
            return
        msg = ''
        itr = self.head
        while itr:
            msg += str(itr.data) + ' <--> ' if itr.next else str(itr.data)
            itr = itr.next
        print(msg)
```

```python
l = DoublyLinkedList()
l.insert_at_start(2)
l.insert_at_start(3)
l.insert_at_start(5)
l.print_linked_list()
```

**출력:**
```
5 <--> 3 <--> 2
```

```python
l = DoublyLinkedList()
l.insert_at_end(2)
l.insert_at_end(3)
l.insert_at_end(5)
l.print_linked_list()
```

**출력:**
```
2 <--> 3 <--> 5
```

```python
l = DoublyLinkedList()
l.insert_at_end(2)
l.insert_at_end(3)
l.insert_at_end(5)
l.remove(1)
l.print_linked_list()
```

**출력:**
```
2 <--> 5
```

```python
l = DoublyLinkedList(["banana","mango","grapes","orange"])
l.print_linked_list()
```

**출력:**
```
banana <--> mango <--> grapes <--> orange
```

```python
l = DoublyLinkedList(["banana","mango","grapes","orange"])
l.insert(1,"blueberry")
l.print_linked_list()
```

**출력:**
```
banana <--> blueberry <--> mango <--> grapes <--> orange
```

# 참고 문헌

[Linked List - Data Structures & Algorithms Tutorials in Python #4](https://www.youtube.com/watch?v=qp8u-frRAnU&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=4)

## 연습문제

**연습문제 1.**
이중 연결 리스트가 단일 연결 리스트보다 나은 점은 무엇인가?

??? success "연습문제 1 풀이"
    이중 연결 리스트는 노드 참조만 있으면 (앞 노드를 찾을 필요 없이) $O(1)$ 삭제, $O(1)$ 역방향 순회, 주어진 노드 앞으로의 $O(1)$ 삽입을 지원한다. 그 대가는 `prev` 포인터를 위한 추가 메모리이다.

---

**연습문제 2.**
노드 참조가 주어졌을 때 이중 연결 리스트에서 그 노드를 삭제하는 코드를 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def delete_node(node):
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
    ```
    이는 단일 연결 리스트의 $O(n)$에 비해 $O(1)$이다.

---

**연습문제 3.**
이중 연결 리스트와 해시 맵으로 LRU 캐시를 구현하라.

??? success "연습문제 3 풀이"
    해시 맵이 $O(1)$ 조회를 제공하고, 이중 연결 리스트가 $O(1)$의 맨 앞으로 옮기기와 제거로 접근 순서를 유지한다. 접근할 때는 노드를 맨 앞으로 옮긴다. 캐시가 가득 찬 상태에서 삽입할 때는 꼬리 노드를 제거한다.

---

**연습문제 4.**
단일 연결 리스트와 이중 연결 리스트의 메모리 부담을 비교하라.

??? success "연습문제 4 풀이"
    단일 연결은 노드당 포인터 하나(64비트에서 8바이트), 이중 연결은 노드당 포인터 둘(16바이트)이다. 데이터 크기가 $d$인 노드 $n$개에 대해 단일 연결은 $n(d + 8)$바이트를, 이중 연결은 $n(d + 16)$바이트를 쓴다. 이 부담은 데이터 크기가 포인터 크기에 비해 작을 때에만 유의미하다.
