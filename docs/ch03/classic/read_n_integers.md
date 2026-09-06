# 정수 N개 읽기
<img src="img/Screen Shot 2022-05-01 at 8.58.12 PM.png" width=50%>

```python
class Read_N_Integers_Using_Recursion:    
    def __init__(self, n, scanner_in_generator):
        self.n = n
        self.generator = scanner_in_generator
        self.data = [None] * (n + 1)
    def read_n_integers_using_recursion(self, k):
        if k == 0:
            return 
        self.read_n_integers_using_recursion(k-1)
        self.data[k] = next(self.generator)    
    def run(self):
        self.read_n_integers_using_recursion(self.n)
        
        
def range_generator(n):
    for i in range(n):
        yield i

def main():
    n = 10
    scanner_in_generator = range_generator(n)
    
    obj = Read_N_Integers_Using_Recursion(n, scanner_in_generator)
    obj.run()
    print(obj.data)
    
        
if __name__ == "__main__":
    main()
```

**출력:**
```
[None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

# 참고 자료

[[알고리즘] 제1-2강 Recursion의 개념과 기본 예제들 (2/3)](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)

## 연습문제

**연습문제 1.**
$n$을 미리 모르는 상태에서 보초 값(예: -1)이 입력될 때까지 정수를 읽도록 함수를 수정하라.

??? success "연습문제 1 풀이"
    ```python
    def read_until_sentinel(sentinel=-1):
        val = int(input())
        if val == sentinel:
            return []
        return [val] + read_until_sentinel(sentinel)
    ```

---

**연습문제 2.**
정수 $n$개를 읽어 역순으로 반환하는 재귀 함수를 작성하라.

??? success "연습문제 2 풀이"
    ```python
    def read_reversed(n):
        if n == 0:
            return []
        val = int(input())
        rest = read_reversed(n - 1)
        return rest + [val]
    ```
    핵심은 재귀 호출이 반환된 뒤에 `val`을 덧붙여 모으는 순서를 뒤집는 것이다.

---

**연습문제 3.**
값 $n$개를 재귀적으로 모을 때와 반복적으로 모을 때의 공간 복잡도는 각각 얼마인가?

??? success "연습문제 3 풀이"
    둘 다 결과 리스트에 값 $n$개를 저장하므로 $O(n)$이다. 다만 재귀 버전은 호출 프레임을 위해 $O(n)$의 스택 공간을 추가로 쓰므로 보조 공간이 총 $O(n)$이다. 반복 버전은 결과 리스트 외에 $O(1)$의 보조 공간만 쓴다.

---

**연습문제 4.**
정수 $n$개를 읽어 짝수만 반환하는 재귀 함수를 작성하라.

??? success "연습문제 4 풀이"
    ```python
    def read_evens(n):
        if n == 0:
            return []
        val = int(input())
        rest = read_evens(n - 1)
        if val % 2 == 0:
            return [val] + rest
        return rest
    ```
