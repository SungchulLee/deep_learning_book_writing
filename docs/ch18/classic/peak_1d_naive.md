# 1차원 봉우리 — 막무가내
```python
def compare(left, center, right):
    if left > center: # 왼쪽이 중앙보다 크면 왼쪽 영역 (left 포함) 에서 찿는다.
        return 'left'
    elif right > center: # 오른쪽이 중앙보다 크면 오른쪽 영역 (right 포함) 에서 찿는다.
        return 'right'
    else: # 왼쪽과 오른쪽 모두 중앙보다 작거나 같으면 center가 찿고자하는 1D Peak 이다.
        return 'center'
```

```python
def compare_left_edge(center, right):
    if right > center: # 오른쪽이 중앙보다 크면 오른쪽 영역 (right 포함) 에서 찿는다.
        return 'right'
    else: # 왼쪽과 오른쪽 모두 중앙보다 작거나 같으면 center가 찿고자하는 1D Peak 이다.
        return 'center'
```

```python
def compare_right_edge(left, center):
    if left > center: # 왼쪽이 중앙보다 크면 왼쪽 영역 (left 포함) 에서 찿는다.
        return 'left'
    else: # 왼쪽과 오른쪽 모두 중앙보다 작거나 같으면 center가 찿고자하는 1D Peak 이다.
        return 'center'
```

```python
def peak(lst):
    
    # 바닥 경우
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return lst[0] 
    
    # 막무가내 한 줄 찾기
    for i in range(len(lst)):
        if i == 0:
            center = lst[i]
            right = lst[i+1]
            result = compare_left_edge(center, right) 
        elif 0 < i < (len(lst)-1):
            left = lst[i-1]
            center = lst[i]
            right = lst[i+1]
            result = compare(left,center, right)
        else:
            left = lst[i-1]
            center = lst[i]
            result = compare_right_edge(left, center)
            
        if result == 'center':
            return center
```

```python
#lst = [1,2,3,4,5]
#lst = [5,4,3,2,1]
lst = [1,2,3,4,5,4,3,2,1]
peak(lst)
```

# 참고 문헌

[1. Algorithmic Thinking, Peak Finding](https://www.youtube.com/watch?v=HtSuA80QTyo&list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb&index=1)

## 연습문제

**연습문제 1.**
막무가내 글자열 짝짓기 알고리즘, KMP, 보이어-무어의 가장 나쁜 경우 시간 복잡도를 견주어라.

??? success "연습문제 1 풀이"
    | 알고리즘 | 가장 나쁜 경우 | 가장 좋은 경우 | 공간 |
    |-----------|-----------|-----------|-------|
    | 막무가내 | $O(nm)$ | $O(n)$ | $O(1)$ |
    | KMP | $O(n + m)$ | $O(n)$ | 어그러짐 함수에 $O(m)$ |
    | Boyer-Moore | $O(nm)$ (pathological) | $O(n/m)$ (sublinear!) | $O(m + |\Sigma|)$ |

    KMP는 한 줄 시간을 보장한다. 보이어-무어는 (글자를 건너뛰므로) 실전에서 대개 더 빠르지만 갈릴 다듬기를 쓰지 않으면 가장 나쁜 경우 $O(nm)$이다.

---

**연습문제 2.**
글 $T$ = "ABABCABABD"과 무늬 $P$ = "ABABD"에 대해 알고리즘이 도는 과정을 견줌마다 보이며 좇아라.

??? success "연습문제 2 풀이"
    자리 0에서 시작: P[0]='A'와 T[0]='A' 견줌(맞음), P[1]='B'와 T[1]='B'(맞음), P[2]='A'와 T[2]='A'(맞음), P[3]='B'와 T[3]='B'(맞음), P[4]='D'와 T[4]='C'(어긋남). 어그러짐 함수 또는 밀기 규칙으로 무늬를 민다. 자리 2에서 시작(KMP는 어그러짐 함수로 다시 견주지 않는다). 끝내 자리 5에서 맞는 곳을 찾는다. 이 알고리즘은 모두 많아야 $2n$번 견준다.

---

**연습문제 3.**
KMP의 어그러짐 함수란 무엇인가? 무늬 "ABABCAB"에 대해 셈하여라.

??? success "연습문제 3 풀이"
    The failure function $\pi[i]$ gives the length of the longest proper prefix of $P[0..i]$ that is also a suffix. For "ABABCAB":

    | $i$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
    |-----|---|---|---|---|---|---|---|
    | $P[i]$ | A | B | A | B | C | A | B |
    | $\pi[i]$ | 0 | 0 | 1 | 2 | 0 | 1 | 2 |

    For example, $\pi[3] = 2$ because "AB" is both a prefix and suffix of "ABAB".

---

**연습문제 4.**
라빈-카프에 쓰이는 굴리는 해시 재주를 설명하여라. 헛맞음이 일어날 확률은 얼마인가?

??? success "연습문제 4 풀이"
    Rabin-Karp computes a hash of the pattern and slides a hash window over the text. The **rolling hash** updates in $O(1)$: $h(T[i+1..i+m]) = (h(T[i..i+m-1]) - T[i] \cdot d^{m-1}) \cdot d + T[i+m] \pmod{q}$ where $d$ is the base and $q$ is a prime. A spurious hit occurs when hashes match but strings differ. For a random prime $q$, the probability of a single spurious hit is $O(1/q)$, and for $n-m+1$ positions, the expected number of spurious hits is $O(n/q)$. Choosing $q \approx n^2$ gives expected $O(1)$ spurious hits.
