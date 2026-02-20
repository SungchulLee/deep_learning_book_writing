# Insert (Sift Up)


```python
# MaxHeap
# public functions : push, peek, pop
# private functions : __swap, __floatUp, __bubbleDown

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

**Output:**
```
[None, 95, 3, 21]
[None, 95, 10, 21, 3]
[None, 21, 10, 3]
```


# Reference

[Python: MaxHeap heapsort](https://www.youtube.com/watch?v=GnKHVXv_rlQ&list=PLj8W7XIvO93qsmdxbaDpIvM1KCyNO1K_c&index=5)
