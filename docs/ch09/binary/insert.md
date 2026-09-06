# 삽입 (위로 올리기)
```python
# 최대 힙
# 공개 함수: push, peek, pop
# 비공개 함수: __swap, __floatUp, __bubbleDown

class MaxHeap:
    def __init__(self, items=None):
        self.heap = [None]
        for item in items:
            self.heap.append(item)
            self.__floatUp(len(self.heap)-1)
    def push(self, data):
        self.heap.append(data)
        self.__floatUp(len(self.heap)-1)
    def peek(self):
        if self.heap[1]:
            return self.heap[1]
        else:
            return False
    def pop(self):
        if len(self.heap) > 2:
            self.__swap(1, len(self.heap)-1)
            popped = self.heap.pop()
            self.__bubbleDown(1)
        elif len(self.heap) == 2:
            popped = self.heap.pop()
        else:
            popped = False
        return popped
    def __swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    def __floatUp(self, index):
        parent = index//2
        if index <= 1:
            return
        elif self.heap[index] > self.heap[parent]:
            self.__swap(index, parent)
            self.__floatUp(parent)
    def __bubbleDown(self, index):
        left = index * 2
        right = index * 2 + 1 
        largest = index
        if len(self.heap) > left and self.heap[largest] < self.heap[left]:
            largest = left
        if len(self.heap) > right and self.heap[largest] < self.heap[right]:
            largest = right
        if largest != index:
            self.__swap(index, largest)
            self.__bubbleDown(largest)
            
            
def main():
    m = MaxHeap([95, 3, 21])
    print(m.heap)
    
    m.push(10)
    print(m.heap)
    
    m.pop()
    print(m.heap)

    
if __name__ == "__main__":
    main()
```

**출력:**
```
[None, 95, 3, 21]
[None, 95, 10, 21, 3]
[None, 21, 10, 3]
```

# 참고 문헌

[파이썬: MaxHeap 힙 정렬](https://www.youtube.com/watch?v=GnKHVXv_rlQ&list=PLj8W7XIvO93qsmdxbaDpIvM1KCyNO1K_c&index=5)

## 연습문제

**연습문제 1.**
힙 삽입의 위로 올리기(거품 올리기) 연산을 설명하라.

??? success "연습문제 1 풀이"
    새 원소를 다음 빈 자리(배열의 끝)에 넣는다. 부모와 견주어 새 원소가 힙 성질을 어기면 부모와 맞바꾼다. 그 원소가 제자리에 닿거나 뿌리에 이를 때까지 되풀이한다. 시간은 (힙의 높이인) $O(\log n)$이다.

---

**연습문제 2.**
파이썬에서 위로 올리기를 쓰는 힙 삽입을 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def heap_insert(heap, val):
        heap.append(val)
        i = len(heap) - 1
        while i > 0:
            parent = (i - 1) // 2
            if heap[i] < heap[parent]:  # 최소 힙
                heap[i], heap[parent] = heap[parent], heap[i]
                i = parent
            else:
                break
    ```

---

**연습문제 3.**
위로 올리기 연산에 $O(\log n)$ 시간이 듦을 증명하라.

??? success "연습문제 3 풀이"
    힙은 높이가 $h = \lfloor\log_2 n\rfloor$인 완전 이진 트리이다. 위로 올리기는 층마다 비교와 자리바꿈을 많아야 한 번씩 하며 잎에서 뿌리까지 훑으므로 연산이 많아야 $h = O(\log n)$번이다. $\square$

---

**연습문제 4.**
처음에 빈 이진 힙에 잇달아 $n$번 넣을 때의 분할 상환 비용은 얼마인가?

??? success "연습문제 4 풀이"
    삽입마다 최악의 경우 $O(\log n)$이므로 $n$번 넣으면 $O(n \log n)$이다. 그러나 원소 대부분이 아래쪽에 들어가 자리바꿈이 적다. 평균 비용은 삽입당 $O(1)$이다(위로 올리기 대부분이 $O(1)$개 층 뒤에 끝난다).
