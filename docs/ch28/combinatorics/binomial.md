# 이항 계수

이항 계수 $\binom{n}{k}$은 서로 다른 $n$개의 모임에서 차례를 따지지 않고 $k$개를 고르는 가짓수를 센다. 이 양은 알고리즘 살피기(나누어 다스리기 되돌이 식, 확률 따짐)와 얽음학(부분 모임 세기, 격자 길)에 두루 나타난다.

---

## 1. 직관

$n$명 가운데 $k$명의 위원회를 고른다고 그려 보라. 고르기마다 크기 $k$의 차례 없는 부분 모임이다. 그런 위원회의 온 개수가 $\binom{n}{k}$이며 "$n$에서 $k$ 고르기"라 읽는다.

---

## 2. 정의

$0 \le k \le n$인 음이 아닌 정수 $n$과 $k$에 대해:

$$
\binom{n}{k} = \frac{n!}{k!\,(n-k)!}
$$

약속에 따라 $k < 0$이거나 $k > n$이면 $\binom{n}{k} = 0$이다.

---

## 3. 핵심 성질

**맞섬.** 어느 $k$개를 넣을지 고르는 것은 어느 $n - k$개를 뺄지 고르는 것과 같다:

$$
\binom{n}{k} = \binom{n}{n-k}
$$

**빨아들이기(뽑아내기).** 분자에서 인자 하나를 뽑아내면:

$$
\binom{n}{k} = \frac{n}{k}\,\binom{n-1}{k-1} \quad (k \ge 1)
$$

**파스칼 항등식.** $k$번째 원소는 고른 부분 모임에 들거나 들지 않는다:

$$
\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k} \quad (1 \le k \le n-1)
$$

??? note "파스칼 항등식의 밝힘"
    계승 뜻매김에서:

    $$
    \binom{n-1}{k-1} + \binom{n-1}{k} = \frac{(n-1)!}{(k-1)!(n-k)!} + \frac{(n-1)!}{k!(n-k-1)!}
    $$

    $(n-1)!$을 묶어내고 공통 분모 $k!(n-k)!$을 찾으면:

    $$
    = \frac{(n-1)!\,k + (n-1)!\,(n-k)}{k!(n-k)!} = \frac{(n-1)!\,n}{k!(n-k)!} = \frac{n!}{k!(n-k)!} = \binom{n}{k}
    $$

**반데르몽드 항등식.** 음이 아닌 정수 $m$, $n$, $r$에 대해:

$$
\binom{m+n}{r} = \sum_{k=0}^{r} \binom{m}{k}\binom{n}{r-k}
$$

**이항 정리.** 어떤 실수 $x$, $y$과 음이 아닌 정수 $n$에 대해:

$$
(x + y)^n = \sum_{k=0}^{n} \binom{n}{k}\, x^k\, y^{n-k}
$$

$x = y = 1$으로 두면 $\sum_{k=0}^{n} \binom{n}{k} = 2^n$을 얻어 원소 $n$개의 모임에 부분 모임이 $2^n$개임을 확인한다.

---

## 4. 이항 계수 셈하기

### 곱셈 공식

계승 뜻매김은 $n$이 어지간해도 넘침을 일으킨다. 더 나은 방식은 조금씩 곱하고 나눈다:

$$
\binom{n}{k} = \frac{n \cdot (n-1) \cdots (n-k+1)}{k!} = \prod_{i=1}^{k} \frac{n - k + i}{i}
$$

부분 곱 $\prod_{i=1}^{j} \frac{n-k+i}{i}$마다 정수이므로 왼쪽에서 오른쪽으로 하면 나눗셈이 늘 딱 떨어진다.

```python
def binom(n: int, k: int) -> int:
    """곱셈 공식으로 C(n, k)을 셈한다.

    O(min(k, n-k)) 시간과 O(1) 공간에 돈다.
    """
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)  # exploit symmetry
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

if __name__ == "__main__":
    # === 쓰기 보기 ===
    print(f"C(10, 3) = {binom(10, 3)}")   # 120
    print(f"C(20, 10) = {binom(20, 10)}") # 184756
```

### 파스칼 삼각형(짜 넣기)

파스칼 항등식으로 $O(nk)$ 시간과 $O(nk)$ 공간에 표를 세운다:

```python
def pascal_table(n: int) -> list[list[int]]:
    """n번째 줄까지 파스칼 삼각형을 짓는다.

    C[i][j] = C(i, j)인 2차원 목록을 돌려준다.
    """
    C = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        C[i][0] = 1
        for j in range(1, i + 1):
            C[i][j] = C[i - 1][j - 1] + C[i - 1][j]
    return C

if __name__ == "__main__":
    # === 처음 6줄 찍기 ===
    table = pascal_table(5)
    for i in range(6):
        print([table[i][j] for j in range(i + 1)])
```

### 공간을 아낀 파스칼 줄

$n$번째 줄 하나만 필요하면 오른쪽에서 왼쪽으로 고치는 1차원 배열을 쓴다:

```python
def pascal_row(n: int) -> list[int]:
    """자리 O(n)으로 파스칼 삼각형의 n번째 줄을 셈한다."""
    row = [0] * (n + 1)
    row[0] = 1
    for i in range(1, n + 1):
        for j in range(i, 0, -1):
            row[j] += row[j - 1]
    return row
```

---

## 5. 한계와 점근

알고리즘 살피기에 쓸모 있는 한계:

$$
\left(\frac{n}{k}\right)^k \le \binom{n}{k} \le \left(\frac{en}{k}\right)^k
$$

붙박이 $k$에 대해 $\binom{n}{k} = \Theta(n^k)$이다. 가운데 계수는:

$$
\binom{2n}{n} \sim \frac{4^n}{\sqrt{\pi n}} \quad \text{(Stirling's approximation)}
$$

---

## 6. 알고리즘에서의 쓰임새

| 쓰임새 | $\binom{n}{k}$이 나타나는 곳 |
|---|---|
| 부분 모임 세기 | 원소 $n$개 모임의 $k$원소 부분 모임 개수 |
| 병합 줄 세우기 살피기 | 자리바꿈의 뒤바뀜 개수 |
| 확률로 살피기 | 표시 아무 변수로 얻은 기댓값 |
| 해싱 살피기 | 생일 문제 꼴의 부딪힘 한계 |
| 나누어 다스리기 | $\binom{n}{k}$ 항이 든 되돌이 식의 풀이 |

---

## 연습문제

**연습문제 1.**
반데르몽드 항등식 $\binom{m+n}{r} = \sum_{k=0}^{r} \binom{m}{k}\binom{n}{r-k}$을 밝혀라.

??? success "연습문제 1 풀이"
    얽음으로 밝히기: A 무리에 $m$개, B 무리에 $n$개가 있는 $m + n$개에서 $r$개를 고르는 경우를 보자. 올바른 $k$마다 A에서 $k$개, B에서 $r - k$개를 고를 수 있다. 그 가짓수는 $\binom{m}{k}\binom{n}{r-k}$이다. 올바른 $k$ 모두에 대해 더하면 온 개수 $\binom{m+n}{r}$을 얻는다.

---

**연습문제 2.**
파스칼 항등식을 되돌이로 써서 $\binom{10}{3}$을 셈하라. 계승 공식으로 확인하라.

??? success "연습문제 2 풀이"
    파스칼: $\binom{10}{3} = \binom{9}{2} + \binom{9}{3}$이다. $\binom{9}{2} = \binom{8}{1} + \binom{8}{2} = 8 + 28 = 36$이고 $\binom{9}{3} = \binom{8}{2} + \binom{8}{3} = 28 + 56 = 84$이다. 따라서 $\binom{10}{3} = 36 + 84 = 120$이다. 계승: $10!/(3! \cdot 7!) = 720/6 = 120$. 확인되었다.

---

**연습문제 3.**
하키 스틱 항등식 $\sum_{i=0}^{r} \binom{n+i}{i} = \binom{n+r+1}{r}$을 밝혀라.

??? success "연습문제 3 풀이"
    $r$에 대한 귀납으로 밝힌다. 바탕 경우 $r = 0$: $\binom{n}{0} = 1 = \binom{n+1}{0}$이다. 귀납 걸음: $r-1$에 대해 항등식이 성립한다고 하면 $\sum_{i=0}^{r-1}\binom{n+i}{i} = \binom{n+r}{r-1}$이다. 그러면 파스칼 항등식에 따라 $\sum_{i=0}^{r}\binom{n+i}{i} = \binom{n+r}{r-1} + \binom{n+r}{r} = \binom{n+r+1}{r}$이다.

---

**연습문제 4.**
반데르몽드 항등식으로 $\sum_{k=0}^{n} \binom{n}{k}^2 = \binom{2n}{n}$임을 보여라.

??? success "연습문제 4 풀이"
    반데르몽드에서 $m = n$, $r = n$으로 두면 $\binom{2n}{n} = \sum_{k=0}^{n}\binom{n}{k}\binom{n}{n-k}$이다. 맞섬에 따라 $\binom{n}{n-k} = \binom{n}{k}$이다. 따라서 $\binom{2n}{n} = \sum_{k=0}^{n}\binom{n}{k}^2$이다.

## 정리하며

이 마당은 직관、정의、핵심 성질、이항 계수 셈하기을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
- Graham, R. L., Knuth, D. E., & Patashnik, O. (1994). *Concrete Mathematics* (2nd ed.). Addison-Wesley. Chapter 5.
