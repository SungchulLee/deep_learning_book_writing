# 해시 테이블 클래스 구현
```python
class HashMap:
    def __init__(self):
        self.size = 6
        self.map = [None] * self.size
    def __get_hash(self, key):
        hash_val = 0
        for char in str(key):
            hash_val += ord(char)
        return hash_val % self.size
    def add(self, key, value):
        key_hash = self.__get_hash(key)
        key_value = [key, value]
        if self.map[key_hash] is None:
            self.map[key_hash] = list([key_value])
            return True
        else:
            for pair in self.map[key_hash]:
                if pair[0] == key:
                    pair[1] = value
                    return True
            self.map[key_hash].append(key_value)
            return True
    def get(self, key):
        key_hash = self.__get_hash(key)
        if self.map[key_hash] is not None:
            for pair in self.map[key_hash]:
                if pair[0] == key:
                    return pair[1]
        return None
    def delete(self, key):
        key_hash = self.__get_hash(key)
        if self.map[key_hash] is None:
            return False
        for i, lst in enumerate(self.map[key_hash]):
            if lst[0] == key:
                self.map[key_hash].pop(i)
                return True
        return False
    def keys(self):
        arr = []
        for i in range(0, len(self.map)):
            if self.map[i]:
                arr.append(self.map[i][0])
        return arr
    def print(self):
        print('---PHONEBOOK----')
        for item in self.map:
            if item is not None:
                print(str(item))

                                
def main():
    h = HashMap()
    h.add('Bob', '567-8888')
    h.add('Ming', '293-6753')
    h.add('Ming', '333-8233')
    h.add('Ankit', '293-8625')
    h.add('Aditya', '852-6551')
    h.add('Alicia', '632-4123')
    h.add('Mike', '567-2188')
    h.add('Aditya', '777-8888')
    h.print()
    print()
    
    h.delete('Bob')
    h.print()
    print()
    
    print('Ming: ' + h.get('Ming'))
    print(h.keys())

    
if __name__ == "__main__":
    main()
```

**출력:**
```
---PHONEBOOK----
[['Mike', '567-2188']]
[['Alicia', '632-4123']]
[['Aditya', '777-8888']]
[['Bob', '567-8888'], ['Ming', '333-8233'], ['Ankit', '293-8625']]

---PHONEBOOK----
[['Mike', '567-2188']]
[['Alicia', '632-4123']]
[['Aditya', '777-8888']]
[['Ming', '333-8233'], ['Ankit', '293-8625']]

Ming: 333-8233
[['Mike', '567-2188'], ['Alicia', '632-4123'], ['Aditya', '777-8888'], ['Ming', '333-8233']]
```

# 참고 문헌

Python: Creating a HASHMAP using Lists [youtube](https://www.youtube.com/watch?v=9HFbhPscPU0&list=PLj8W7XIvO93qsmdxbaDpIvM1KCyNO1K_c&index=4) [code](https://github.com/joeyajames/Python/blob/master/HashMap.py)

## 연습문제

**연습문제 1.**
충돌 해결에 체이닝을 쓰는 기본적인 해시 테이블을 구현하라.

??? success "연습문제 1 풀이"
    ```python
    class HashTable:
        def __init__(self, size=10):
            self.size = size
            self.table = [[] for _ in range(size)]
        def put(self, key, value):
            idx = hash(key) % self.size
            for i, (k, v) in enumerate(self.table[idx]):
                if k == key:
                    self.table[idx][i] = (key, value)
                    return
            self.table[idx].append((key, value))
        def get(self, key):
            idx = hash(key) % self.size
            for k, v in self.table[idx]:
                if k == key:
                    return v
            raise KeyError(key)
    ```

---

**연습문제 2.**
해시 테이블 연산의 평균 및 최악의 경우 시간 복잡도를 분석하라.

??? success "연습문제 2 풀이"
    평균의 경우(좋은 해시 함수와 적재율 < 1)에는 get, put, delete가 모두 $O(1)$이다. 최악의 경우(모든 키가 같은 버킷으로 해시될 때)는 $O(n)$이다. 체이닝을 쓰고 적재율이 $\alpha = n/m$이면 기대 사슬 길이가 $\alpha$이므로 평균의 경우는 $O(1 + \alpha)$이다.

---

**연습문제 3.**
적재율이 문턱값을 넘으면 크기를 자동으로 조정하도록 구현하라.

??? success "연습문제 3 풀이"
    적재율이 0.75를 넘으면 크기가 두 배인 새 테이블을 만들고 기존 항목을 모두 재해싱한다. 이렇게 하면 사슬을 짧게 유지하면서 상각 $O(1)$ 성능을 지킬 수 있다.

---

**연습문제 4.**
사전 자료 구조로서 해시 테이블과 이진 탐색 나무를 비교하라.

??? success "연습문제 4 풀이"
    해시 테이블은 조회와 삽입이 평균 $O(1)$, 최악의 경우 $O(n)$이며 순서가 없다. 이진 탐색 나무는 (균형 잡혀 있다면) $O(\log n)$이 보장되고 순서가 있어 구간 질의와 최솟값·최댓값을 지원한다. 순수한 조회에는 해시 테이블이 빠르고, 순서나 구간 연산이 필요하면 이진 탐색 나무가 낫다.
