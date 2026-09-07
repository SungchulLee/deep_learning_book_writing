# 파스칼 삼각형

파스칼 삼각형은 칸마다 바로 위 두 칸의 합인 이항 계수의 삼각 배열이다. 보기에 아름다울 뿐 아니라 이항 계수를 효율 좋게 셈하는 길을 주고 알고리즘 살피기, 확률, 수론에 쓰이는 깊은 대수 무늬를 드러낸다.

## 직관

삼각형을 줄마다 세운다. 0줄은 $1$뿐이다. 그다음 칸마다 바로 위 두 수를 더해 만든다(없는 칸은 0으로 본다). $n$줄 $k$번째 칸은 $\binom{n}{k}$과 같다.

## 세우기

```
0줄:                1
Row 1:              1   1
Row 2:            1   2   1
Row 3:          1   3   3   1
Row 4:        1   4   6   4   1
Row 5:      1   5  10  10   5   1
Row 6:    1   6  15  20  15   6   1
```

## 파스칼 항등식

뜻매김하는 되돌이 식은 다음과 같다:

$$
\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k} \quad \text{for } 1 \le k \le n-1
$$

경계 조건은 모든 $n \ge 0$에 대해 $\binom{n}{0} = \binom{n}{n} = 1$이다.

??? note "증명"
    원소 $n$개의 모임 $S$을 생각하고 원소 $x$ 하나를 붙박아 두자. $S$의 $k$원소 부분 모임마다 $x$을 담거나 담지 않는다:

    - **$x$을 담음:** 남은 $k-1$개를 $S \setminus \{x\}$에서 고르므로 부분 모임이 $\binom{n-1}{k-1}$개이다.
    - **$x$을 담지 않음:** $k$개를 모두 $S \setminus \{x\}$에서 고르므로 부분 모임이 $\binom{n-1}{k}$개이다.

    두 경우는 겹치지 않고 빠짐이 없으므로 $\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$이다.

## 파스칼 삼각형에 보이는 성질

**줄의 합.** $n$줄의 모든 칸을 더하면 $2^n$이다:

$$
\sum_{k=0}^{n} \binom{n}{k} = 2^n
$$

이는 $x = y = 1$인 이항 정리에서 따라 나온다.

**번갈아 더한 줄의 합.** $n \ge 1$에 대해:

$$
\sum_{k=0}^{n} (-1)^k \binom{n}{k} = 0
$$

이는 $(1 - 1)^n = 0$에서 따라 나온다.

**맞섬.** 줄마다 앞뒤가 같다: $\binom{n}{k} = \binom{n}{n-k}$.

**대각선의 합(피보나치).** 기울기가 $-1$인 대각선을 따라 더하면 피보나치 수가 나온다:

$$
F_{n+1} = \sum_{k=0}^{\lfloor n/2 \rfloor} \binom{n-k}{k}
$$

**하키 스틱 항등식.** 대각선을 따라 잇단 칸을 더하면:

$$
\sum_{i=r}^{n} \binom{i}{r} = \binom{n+1}{r+1}
$$

??? note "하키 스틱 항등식의 밝힘"
    $n$에 대한 귀납으로 밝힌다. 바탕 경우 $n = r$에서 $\binom{r}{r} = 1 = \binom{r+1}{r+1}$이다. 귀납 걸음:

    $$
    \sum_{i=r}^{n} \binom{i}{r} = \binom{n}{r} + \sum_{i=r}^{n-1} \binom{i}{r} = \binom{n}{r} + \binom{n}{r+1} = \binom{n+1}{r+1}
    $$

    마지막 등식은 파스칼 항등식을 쓴다.

## 나누어떨어짐 성질

**소수 $p$의 $p$줄.** $p$이 소수이면 $1 \le k \le p-1$에 대해 $\binom{p}{k} \equiv 0 \pmod{p}$이다. 곧 $p$줄의 안쪽 칸이 모두 $p$으로 나누어떨어진다.

**뤼카 정리.** 소수 $p$에 대해 $n = \sum n_i p^i$과 $k = \sum k_i p^i$이 밑 $p$ 나타냄이면:

$$
\binom{n}{k} \equiv \prod_{i} \binom{n_i}{k_i} \pmod{p}
$$

## 파스칼 삼각형 세우기

### 온 삼각형

```python
def pascal_triangle(n: int) -> list[list[int]]:
    """0번째 줄부터 n번째 줄까지 파스칼 삼각형을 짓는다.

    시간: O(n^2). 공간: O(n^2).
    """
    triangle = [[1]]
    for i in range(1, n + 1):
        prev = triangle[-1]
        row = [1]
        for j in range(1, i):
            row.append(prev[j - 1] + prev[j])
        row.append(1)
        triangle.append(row)
    return triangle
```

### 줄 하나(공간을 아낌)

```python
def pascal_row(n: int) -> list[int]:
    """자리 O(n)으로 파스칼 삼각형의 n번째 줄을 셈한다.

    지금 되풀이에 필요한 값을 덮어쓰지 않도록
    배열 하나를 오른쪽에서 왼쪽으로 고친다.
    """
    row = [0] * (n + 1)
    row[0] = 1
    for i in range(1, n + 1):
        for j in range(i, 0, -1):
            row[j] += row[j - 1]
    return row


if __name__ == "__main__":
    # === 처음 8줄 찍기 ===
    tri = pascal_triangle(7)
    for i, row in enumerate(tri):
        padding = " " * (7 - i) * 2
        values = "  ".join(f"{v:3d}" for v in row)
        print(f"{padding}{values}")

    # === 줄의 합이 2의 거듭제곱인지 확인 ===
    for i in range(8):
        assert sum(tri[i]) == 2 ** i
    print("\nAll row sums verified: sum(row n) = 2^n")
```

## 복잡도

| 연산 | 시간 | 공간 |
|---|---|---|
| 온 삼각형 세우기(0줄에서 $n$줄까지) | $O(n^2)$ | $O(n^2)$ |
| 줄 하나 셈하기 | $O(n^2)$ | $O(n)$ |
| 미리 세운 표에서 $\binom{n}{k}$ 찾기 | $O(1)$ | 미리 셈한 $O(n^2)$ |

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
- Graham, R. L., Knuth, D. E., & Patashnik, O. (1994). *Concrete Mathematics* (2nd ed.). Addison-Wesley. Chapter 5.

## 연습문제

**연습문제 1.**
파스칼 삼각형의 처음 7줄을 적어라. 줄의 합의 무늬를 짚어라.

??? success "연습문제 1 풀이"
    0줄: 1. 1줄: 1 1. 2줄: 1 2 1. 3줄: 1 3 3 1. 4줄: 1 4 6 4 1. 5줄: 1 5 10 10 5 1. 6줄: 1 6 15 20 15 6 1. 줄의 합: $1, 2, 4, 8, 16, 32, 64 = 2^0, 2^1, \ldots, 2^6$. $n$줄의 합은 $\sum_{k=0}^{n}\binom{n}{k} = 2^n$이다($x = y = 1$인 이항 정리에 따라).

---

**연습문제 2.**
$p$이 소수일 때 파스칼 삼각형 $p$줄의 (양 끝을 뺀) 모든 칸이 $p$으로 나누어떨어짐을 밝혀라.

??? success "연습문제 2 풀이"
    $1 \leq k \leq p - 1$에서 $\binom{p}{k} = p!/(k!(p-k)!)$이다. $p$이 소수이므로 $p$은 $p!$을 나누지만 $k!$이나 $(p-k)!$은 나누지 않는다(둘 다 인수가 모두 $< p$이다). 따라서 $p | \binom{p}{k}$이다. 이 성질이 이항 정리로 페르마의 작은 정리를 밝히는 열쇠이다: $(a+1)^p \equiv a^p + 1 \pmod{p}$.

---

**연습문제 3.**
계승을 곧바로 셈하지 않고 파스칼 삼각형으로 $\binom{8}{3}$을 셈하라.

??? success "연습문제 3 풀이"
    필요한 칸을 세운다: $\binom{5}{0}=1, \binom{5}{1}=5, \binom{5}{2}=10, \binom{5}{3}=10$. $\binom{6}{2}=\binom{5}{1}+\binom{5}{2}=15, \binom{6}{3}=\binom{5}{2}+\binom{5}{3}=20$. $\binom{7}{2}=\binom{6}{1}+\binom{6}{2}=6+15=21, \binom{7}{3}=\binom{6}{2}+\binom{6}{3}=15+20=35$. $\binom{8}{3}=\binom{7}{2}+\binom{7}{3}=21+35=56$.

---

**연습문제 4.**
파스칼 삼각형의 줄 하나를 $O(n)$ 시간과 $O(n)$ 공간에 효율 좋게 셈하는 법을 적어라.

??? success "연습문제 4 풀이"
    되돌이 식 $\binom{n}{k} = \binom{n}{k-1} \cdot (n-k+1)/k$을 쓴다. $\binom{n}{0} = 1$에서 시작한다. $k = 1, \ldots, n$에 대해 $(n - k + 1)/k$을 곱한다. 이는 곱셈 한 번과 나눗셈 한 번으로 앞 칸에서 다음 칸을 셈하므로 칸마다 $O(1)$, 온 시간 $O(n)$이다. 줄을 크기 $n + 1$인 배열에 담는다. 온 삼각형을 세울 필요가 없다. 정수 셈에서는 분수를 피하도록 차례를 지킨다. 곱 $\binom{n}{k} \cdot (n - k)$은 늘 $k + 1$으로 나누어떨어진다.
