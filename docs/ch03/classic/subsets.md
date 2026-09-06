# 부분집합 생성
<img src="img/Screen Shot 2022-05-02 at 6.07.29 PM.png" width=50%>

```python
def print_power_set(initial_set):
    pre = set() 
    post = initial_set 
    print_pre_post(pre, post)
    
    
def print_pre_post(pre, post):
    if len(post) == 0:
        print(pre)
        return
    
    t = post.pop()
    
    pre_add_t = pre.copy()
    pre_add_t.add(t)
    post_copy = post.copy()
    
    print_pre_post(pre, post)
    print_pre_post(pre_add_t, post_copy)
        

def main():
    initial_set = {0,1,2}
    print_power_set(initial_set)
    
        
if __name__ == "__main__":
    main()
```

**출력:**
```
set()
{2}
{1}
{1, 2}
{0}
{0, 2}
{0, 1}
{0, 1, 2}
```

```python
def main():
    initial_set = {0,1,2}
    for i in range(2**len(initial_set)):
        print(f"{format(i, f'0{len(initial_set)}b')}")
    
        
if __name__ == "__main__":
    main()
```

**출력:**
```
000
001
010
011
100
101
110
111
```

```python
def main():
    initial_set = {0,1,2}
    initial_list = list(initial_set)
    for i in range(2**len(initial_set)):
        binary_representation = f"{format(i, '03b')}"
        tmp = set()
        for i, rep in enumerate(binary_representation):
            if rep == "1":
                tmp.add(initial_list[i])
        print(tmp)
    
        
if __name__ == "__main__":
    main()
```

**출력:**
```
set()
{2}
{1}
{1, 2}
{0}
{0, 2}
{0, 1}
{0, 1, 2}
```

<img src="img/Screen Shot 2022-05-02 at 6.50.43 PM.png" width=50%>

# 참고 자료

[[알고리즘] 제2-4강 멱집합 (powerset)](https://www.youtube.com/watch?v=nkeMRRIVW9s&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=7)

## 연습문제

**연습문제 1.**
원소가 $n$개인 집합의 부분집합은 몇 개인가? 재귀 구조를 사용해 증명하라.

??? success "연습문제 1 풀이"
    $2^n$개이다. 각 원소가 포함되거나 제외되므로 원소마다 선택지가 2개이다. 곱의 법칙에 의해 전체 부분집합은 $2^n$개이다.

    재귀적으로는 $S(0) = 1$(공집합 하나)인 $S(n) = 2 \cdot S(n-1)$이다. 풀면 $S(n) = 2^n$이다.

---

**연습문제 2.**
비트마스크 방식으로 부분집합 생성을 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def subsets_bitmask(arr):
        n = len(arr)
        for mask in range(1 << n):
            subset = [arr[i] for i in range(n) if mask & (1 << i)]
            print(subset)
    ```

---

**연습문제 3.**
주어진 크기 $k$의 부분집합만 만들어 내도록 재귀적 부분집합 생성기를 수정하라.

??? success "연습문제 3 풀이"
    ```python
    def subsets_size_k(arr, k, start=0, current=[]):
        if len(current) == k:
            print(current)
            return
        if start >= len(arr):
            return
        # arr[start]를 포함한다
        subsets_size_k(arr, k, start + 1, current + [arr[start]])
        # arr[start]를 제외한다
        subsets_size_k(arr, k, start + 1, current)
    ```

---

**연습문제 4.**
$\{1, \ldots, n\}$의 모든 부분집합에 걸친 원소의 총 개수는 얼마인가? 각 원소가 정확히 $2^{n-1}$개의 부분집합에 나타나므로 총합은 $n \cdot 2^{n-1}$이다. $n = 3$에 대해 확인하라.

??? success "연습문제 4 풀이"
    $n = 3$일 때 부분집합은 $\emptyset, \{1\}, \{2\}, \{3\}, \{1,2\}, \{1,3\}, \{2,3\}, \{1,2,3\}$이다.

    전체 원소 수: $0 + 1 + 1 + 1 + 2 + 2 + 2 + 3 = 12 = 3 \cdot 2^2 = 3 \cdot 4$. 확인되었다.

    증명: $n$개의 원소 각각은 $2^n$개의 부분집합 중 정확히 절반에(즉 그 원소가 "포함"되는 경우에) 나타나므로 $2^{n-1}$개이다. 총합은 $n \cdot 2^{n-1}$이다. $\square$
