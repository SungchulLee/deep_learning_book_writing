# 순열과 조합
# 버그가 있는 코드

```python
def print_permutation(initial_list, k):
    if len(initial_list) < k:
        return
    
    pre = [] 
    post = initial_list 
    print_permutation_pre_post(pre, post, k)
    
    
def print_permutation_pre_post(pre, post, k):
    if len(pre) == k:
        print(pre)
        return
    
    for i, item in enumerate(post):
        del post[i]      # there are bugs in these two lines
        pre.append(item) # there are bugs in these two lines
        
        print_permutation_pre_post(pre, post, k)
        

def main():
    initial_list = [0,1,2]
    print_permutation(initial_list, 2)
    
        
if __name__ == "__main__":
    main()
```

**출력:**
```
[0, 1]
```

# 버그는 고쳤지만 메모리를 너무 많이 쓴다

```python
def print_permutation(initial_list, k):
    if len(initial_list) < k:
        return
    
    pre = [] 
    post = initial_list 
    print_permutation_pre_post(pre, post, k)
    
    
def print_permutation_pre_post(pre, post, k):
    if len(pre) == k:
        print(pre)
        return
    
    for i, item in enumerate(post):
        post_copy = post.copy()
        del post_copy[i]
        pre_copy = pre.copy()
        pre_copy.append(item)

        print_permutation_pre_post(pre_copy, post_copy, k)
        

def main():
    initial_list = [0,1,2]
    print_permutation(initial_list, 2)
    
        
if __name__ == "__main__":
    main()
```

**출력:**
```
[0, 1]
[0, 2]
[1, 0]
[1, 2]
[2, 0]
[2, 1]
```

# 메모리 사용을 줄이기 위해 전역 변수를 쓴다(작업 중)

```python
def print_permutation(i, k):
    """
    i : index you can change lst memebrs
    k : lenth of final printed list
    """
    if i > k:
        print(lst[:k])
        return
    
    for j in range(i,len(lst)):
        swap(i, j) # after all these recursive ops, we would like to have
        print_permutation_after_member_change(i, k) # lst[i:]
        swap(i, j)                                  # unchanged
    
    
def print_permutation_after_member_change(i, k):
    """
    i : index you can change lst memebrs, lst members are already changed
    k : lenth of final printed list
    """
    print_permutation(i+1, k)
    
    
def swap(i, j):
    lst[i], lst[j] = lst[j], lst[i]
        

def main():
    global lst
    lst = [0,1,2]
    print_permutation(0, 2)
    
        
if __name__ == "__main__":
    main()
```

**출력:**
```
[0, 1]
[0, 2]
[1, 0]
[1, 2]
[2, 1]
[2, 0]
```

# 참고 자료

[[알고리즘] 제2-5강 순열(permutation)](https://www.youtube.com/watch?v=MjW10t9ppok&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=8)

## 연습문제

**연습문제 1.**
서로 다른 원소 $n$개인 집합의 순열은 몇 개인가? 증명하라.

??? success "연습문제 1 풀이"
    $n!$개이다. 귀납법으로 증명한다. 기저 사례 $n=1$은 순열이 $1 = 1!$개이다. 귀납 단계에서는 $n$개의 원소 각각을 맨 앞에 놓을 수 있고, 남은 $n-1$개 원소는 가정에 의해 $(n-1)!$개의 순열을 가지므로 $n \cdot (n-1)! = n!$이 된다. $\square$

---

**연습문제 2.**
원소 $n$개에서 길이 $k$인 순열($k$-순열)만 만들어 내도록 순열 생성기를 수정하라.

??? success "연습문제 2 풀이"
    ```python
    def k_permutations(arr, k, current=[]):
        if len(current) == k:
            print(current)
            return
        for i, x in enumerate(arr):
            k_permutations(arr[:i] + arr[i+1:], k, current + [x])
    ```

---

**연습문제 3.**
$\{1, \ldots, n\}$에서 크기 $k$인 부분집합(조합)을 모두 재귀적으로 생성하라.

??? success "연습문제 3 풀이"
    ```python
    def combinations(arr, k, start=0, current=[]):
        if len(current) == k:
            print(current)
            return
        for i in range(start, len(arr)):
            combinations(arr, k, i + 1, current + [arr[i]])
    ```
    순열과 달리 현재 인덱스 이후의 원소만 고려하도록 `start`를 사용하여 중복을 피한다.

---

**연습문제 4.**
원소 $n$개의 모든 순열을 생성하는 시간 복잡도와 공간 복잡도는 얼마인가?

??? success "연습문제 4 풀이"
    시간: $O(n \cdot n!)$이다. 순열이 $n!$개이고 각각을 출력하거나 복사하는 데 $O(n)$이 든다.

    공간: 재귀 스택(깊이 $n$)을 위한 $O(n)$에 현재 순열을 위한 $O(n)$이 더해진다. 모든 순열을 저장한다면 $O(n \cdot n!)$이다.
