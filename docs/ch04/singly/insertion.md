# 삽입
<div align="center"><img src="https://cdn-images-1.medium.com/max/1600/1*rEC8Te27eo5TSYCHMA7Ttw.png" width="100%" height="30%"></div>

```python
class Node:
    def __init__(self, data=None, next_data=None):
        self.data = data
        self.next = next_data
```

```python
class LinkedList:
    def __init__(self, data_list=None):
        self.head = None
        if data_list is not None:
            for data in data_list:
                self.insert_at_end(data)
        
    def get_length(self):
        # 길이를 따로 들고 있지 않아 셀 때마다 처음부터 훑는다. O(n)이다.
        # 배열이 len을 O(1)에 내놓는 것과 다른 점이며, 아래 insert와
        # remove가 이 함수를 부르는 탓에 그쪽도 함께 O(n)이 된다.
        # self.length를 필드로 두고 갱신하면 O(1)로 줄일 수 있다
        count = 0
        itr = self.head
        while itr:
            count += 1
            itr = itr.next
        return count
    
    def insert_at_start(self, data):
        # 새 노드가 옛 머리를 가리키게 만든 뒤 머리를 옮긴다. 훑을 것이
        # 없으므로 O(1)이며, 연결 리스트가 배열보다 나은 대표적인 자리다.
        # 배열의 맨 앞에 넣으려면 뒤의 원소를 모두 한 칸씩 밀어야 한다
        node = Node(data, self.head)
        self.head = node

    def insert_at_end(self, data):
        # 빈 리스트에는 꼬리가 없으므로 맨 앞 넣기로 넘긴다. 이 검사가
        # 없으면 아래 itr.next에서 None을 건드려 오류가 난다
        if self.head is None:
            self.insert_at_start(data)
            return
        # 꼬리를 따로 들고 있지 않아 끝까지 걸어가야 한다. O(n)이다.
        # 조건이 itr.next인 점에 주의하라. itr이었다면 마지막 노드를
        # 지나쳐 None에 닿아, 이을 자리를 잃는다
        itr = self.head
        while itr.next:
            itr = itr.next
        itr.next = Node(data, None)
        
    def insert(self, index, data):
        # 넣기에서는 index == length가 허용된다. 맨 뒤에 덧붙이는 뜻이라
        # 부등호가 >이다. 아래 remove는 지울 노드가 있어야 하므로
        # >=로 한 칸 더 좁다
        if (index < 0) or (index > self.get_length()):
            raise Exception("Invalid Index")
        if index == 0:
            self.insert_at_start(data)
            return
        # 넣을 자리의 "앞" 노드까지만 간다. 단일 연결 리스트는 뒤로만
        # 갈 수 있어, 어떤 노드 앞에 끼우려면 그 앞 노드를 붙들고
        # 있어야 하기 때문이다. 그래서 range가 1부터 시작한다
        itr = self.head
        for _ in range(1,index):
            itr = itr.next
        # 순서가 중요하다. 새 노드가 뒤를 먼저 가리키게 한 다음
        # 앞 노드를 새 노드로 이어야 한다. 거꾸로 하면 itr.next를
        # 덮어쓴 뒤라 뒤쪽 사슬을 통째로 잃는다
        node = Node(data, itr.next)
        itr.next = node
        
    def remove(self, index):
        if (index < 0) or (index >= self.get_length()):
            raise Exception("Invalid Index")
        if index==0:
            # 머리를 다음 노드로 옮기면 옛 머리는 아무도 가리키지 않게
            # 되어 파이썬이 알아서 거두어 간다. C였다면 free가 필요하다
            self.head = self.head.next
            return
        # 넣기와 마찬가지로 지울 노드의 "앞"까지만 간다
        itr = self.head
        for _ in range(1,index):
            itr = itr.next
        # 앞 노드가 건너뛰어 그다음을 가리키게 하면 사이에 낀 노드가
        # 사슬에서 떨어져 나간다. 노드를 지우는 것이 아니라
        # 이음선을 고쳐 끼우는 것이 연결 리스트의 삭제다
        itr.next = itr.next.next
        
    def print_linked_list(self):
        if self.head is None:
            print("Linked list is empty")
            return
        msg = ''
        itr = self.head
        while itr:
            msg += str(itr.data) + ' --> ' if itr.next else str(itr.data)
            itr = itr.next
        print(msg)
```

```python
l = LinkedList()
l.insert_at_start(2)
l.insert_at_start(3)
l.insert_at_start(5)
l.print_linked_list()
```

```python
l = LinkedList()
l.insert_at_end(2)
l.insert_at_end(3)
l.insert_at_end(5)
l.print_linked_list()
```

```python
l = LinkedList()
l.insert_at_end(2)
l.insert_at_end(3)
l.insert_at_end(5)
l.remove(1)
l.print_linked_list()
```

```python
l = LinkedList(["banana","mango","grapes","orange"])
l.print_linked_list()
```

```python
l = LinkedList(["banana","mango","grapes","orange"])
l.insert(1,"blueberry")
l.print_linked_list()
```

**출력:**

```
banana --> blueberry --> mango --> grapes --> orange
```

# 참고 문헌

[Linked List - Data Structures & Algorithms Tutorials in Python #4](https://www.youtube.com/watch?v=qp8u-frRAnU&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=4)

[109. Convert Sorted List to Binary Search Tree](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/)

[141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

[160. Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)

[234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)

[237. Delete Node in a Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/)

## 연습문제

**연습문제 1.**
단일 연결 리스트의 머리, 꼬리, 중간에 삽입할 때의 시간 복잡도는 각각 얼마인가?

??? success "연습문제 1 풀이"
    머리: $O(1)$(머리 포인터를 갱신). 꼬리: 꼬리 포인터가 없으면 $O(n)$, 있으면 $O(1)$. 중간(주어진 노드 뒤): 앞 노드의 참조를 갖고 있으면 $O(1)$, 위치로 찾아야 하면 $O(n)$.

---

**연습문제 2.**
단일 연결 리스트의 위치 $k$에 삽입하는 코드를 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def insert_at(head, k, val):
        if k == 0:
            node = ListNode(val)
            node.next = head
            return node
        curr = head
        for _ in range(k - 1):
            curr = curr.next
        node = ListNode(val)
        node.next = curr.next
        curr.next = node
        return head
    ```

---

**연습문제 3.**
단일 연결 리스트의 삽입과 파이썬 리스트(동적 배열)의 삽입을 비교하라.

??? success "연습문제 3 풀이"
    연결 리스트는 머리에서 $O(1)$이고 위치를 찾는 데 $O(n)$이다. 배열은 끝에서 상각 $O(1)$이고 임의의 위치에서는 (원소를 밀어야 하므로) $O(n)$이다. 위치를 이미 아는 곳에서 삽입과 삭제가 잦으면 연결 리스트가, 임의 접근이 많으면 배열이 유리하다.

---

**연습문제 4.**
연결 리스트에 삽입할 때 포인터를 올바른 순서로 갱신해야 하는 이유는 무엇인가?

??? success "연습문제 4 풀이"
    `new_node.next = prev.next`보다 먼저 `prev.next = new_node`를 설정하면 리스트의 나머지 부분에 대한 참조를 잃는다. 언제나 새 노드의 next 포인터를 먼저 설정한 뒤에 앞 노드의 포인터를 갱신하라.
