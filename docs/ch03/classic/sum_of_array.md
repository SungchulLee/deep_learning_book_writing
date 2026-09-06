# 배열의 합
<img src="img/Screen Shot 2022-05-01 at 8.42.09 PM.png" width=70%>

```python
def compute_sum_of_array_using_recursion(lst):
    if len(lst) == 0:
        return 0    
    return compute_sum_of_array_using_recursion(lst[:-1]) + lst[-1]

def compute_sum_of_array_using_python(lst):
    return sum(lst)

def main():
    lsts = (
    [],
    [0],
    [2,0,],
    [2,1,-1],
    [2,5,-2,4],
    [1,2,3,4,5,6,7,8,9,10]
    )
    for lst in lsts:
        result_0 = compute_sum_of_array_using_recursion(lst)
        result_1 = compute_sum_of_array_using_python(lst)
        print(f"Sum of list {lst} using recursion : {result_0}")
        print(f"Sum of list {lst} using python    : {result_1}")
        print()

        
if __name__ == "__main__":
    main()
```

**출력:**
```
Sum of list [] using recursion : 0
Sum of list [] using python    : 0

Sum of list [0] using recursion : 0
Sum of list [0] using python    : 0

Sum of list [2, 0] using recursion : 2
Sum of list [2, 0] using python    : 2

Sum of list [2, 1, -1] using recursion : 2
Sum of list [2, 1, -1] using python    : 2

Sum of list [2, 5, -2, 4] using recursion : 9
Sum of list [2, 5, -2, 4] using python    : 9

Sum of list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] using recursion : 55
Sum of list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] using python    : 55
```

# 참고 자료

[[알고리즘] 제1-2강 Recursion의 개념과 기본 예제들 (2/3)](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)

## 연습문제

**연습문제 1.**
배열에서 최댓값을 찾는 재귀 함수를 작성하라.

??? success "연습문제 1 풀이"
    ```python
    def recursive_max(arr, n=None):
        if n is None:
            n = len(arr)
        if n == 1:
            return arr[0]
        return max(arr[n-1], recursive_max(arr, n-1))
    ```

---

**연습문제 2.**
배열을 재귀적으로 반으로 나누는 분할 정복 합을 구현하라. 선형 재귀에 비해 어떤 이점이 있는가?

??? success "연습문제 2 풀이"
    ```python
    def dc_sum(arr, lo=0, hi=None):
        if hi is None:
            hi = len(arr) - 1
        if lo == hi:
            return arr[lo]
        if lo > hi:
            return 0
        mid = (lo + hi) // 2
        return dc_sum(arr, lo, mid) + dc_sum(arr, mid+1, hi)
    ```
    최대 재귀 깊이가 선형 재귀의 $O(n)$ 대신 $O(\log n)$이므로 큰 배열에서 스택 넘침을 피할 수 있다.

---

**연습문제 3.**
마스터 정리를 사용해 분할 정복 합의 시간 복잡도를 분석하라.

??? success "연습문제 3 풀이"
    $T(n) = 2T(n/2) + O(1)$이다. $a=2, b=2, f(n)=O(1)$인 마스터 정리에 의해 $n^{\log_b a} = n^1$이 $f(n) = O(1)$을 지배하므로 $T(n) = \Theta(n)$이다. 시간 복잡도는 선형 재귀와 같지만 스택 깊이가 더 낫다.

---

**연습문제 4.**
배열의 모든 원소의 곱을 계산하는 재귀 함수를 작성하라.

??? success "연습문제 4 풀이"
    ```python
    def recursive_product(arr, n=None):
        if n is None:
            n = len(arr)
        if n == 0:
            return 1  # empty product
        if n == 1:
            return arr[0]
        return arr[n-1] * recursive_product(arr, n-1)
    ```
