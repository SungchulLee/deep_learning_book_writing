# 가장 좋은 이진 찾기 나무

이진 찾기 나무에서 열쇠를 찾는 값은 그 깊이에 달렸다. 어떤 열쇠를 다른 것보다 훨씬 자주 찾는다면 자주 찾는 열쇠를 뿌리 가까이 두어 기대 찾기 값을 줄일 수 있다. **가장 좋은 이진 찾기 나무** 문제는 다가감 잦기를 안다고 할 때 이 기대 값을 가장 작게 하는 나무 짜임을 동적 계획으로 찾는다.

---

## 1. 문제 서술

찾을 낌새가 $p_1, p_2, \dots, p_n$인 열쇠 $n$개 $k_1 < k_2 < \cdots < k_n$과, 못 찾는 일을 나타내며 낌새가 $q_0, q_1, \dots, q_n$인 허깨비 열쇠 $n + 1$개 $d_0, d_1, \dots, d_n$이 주어졌다고 하자. 여기서

$$
\sum_{i=1}^{n} p_i + \sum_{j=0}^{n} q_j = 1
$$

이진 찾기 나무 $T$의 **기대 찾기 값**은 다음과 같다:

$$
E[\text{cost}] = \sum_{i=1}^{n} p_i \cdot (\text{depth}_T(k_i) + 1) + \sum_{j=0}^{n} q_j \cdot (\text{depth}_T(d_j) + 1)
$$

목표는 이 기대 값을 가장 작게 하는 이진 찾기 나무를 찾는 것이다.

---

## 2. 가장 좋은 밑짜임

가장 좋은 이진 찾기 나무의 뿌리가 $k_r$이면, 왼쪽 잔나무($k_1, \dots, k_{r-1}$을 담는다)도 그 열쇠들에 대해 가장 좋은 이진 찾기 나무여야 하고 오른쪽 잔나무도 마찬가지다. 이 가장 좋은 잔얼개 덕에 갈피 다지기로 풀 수 있다.

$e[i, j]$을 열쇠 $k_i, \dots, k_j$(허깨비 열쇠 $d_{i-1}, \dots, d_j$과 함께)에 대한 가장 좋은 이진 찾기 나무의 바라는 값이라 하자. 이 잔문제의 짐은 다음과 같다.

$$
w[i, j] = \sum_{\ell=i}^{j} p_\ell + \sum_{\ell=i-1}^{j} q_\ell
$$

---

## 3. 점화식

열쇠 $k_i, \dots, k_j$의 잔나무 뿌리로 $k_r$을 고르면 값이 $w[i, j]$만큼 는다($k_r$의 자식이 되면서 마디마다 깊이가 1씩 늘기 때문이다).

$$
e[i, j] = \min_{i \le r \le j} \bigl\{e[i, r{-}1] + e[r{+}1, j] + w[i, j]\bigr\}
$$

밑 자리: $e[i, i{-}1] = q_{i-1}$(허깨비 열쇠 $d_{i-1}$만 담는 잔나무).

---

## 4. 구현

```python
"""
동적 계획으로 얻는 가장 좋은 이진 찾기 나무.

열쇠의 다가감 확률이 주어질 때 기대 찾기 값을 가장 작게 하는
이진 찾기 나무 짜임을 O(n^3) 시간에 찾는다.
"""

# === 가장 좋은 이진 찾기 나무 ===

def optimal_bst(
    p: list[float], q: list[float]
) -> tuple[float, list[list[int]]]:
    """가장 좋은 이진 찾기 나무의 값과 뿌리 표 셈하기.

    인수:
        p: 열쇠 k_1..k_n의 찾기 확률(1부터 셈, p[0]은 쓰지 않음).
        q: 허수아비 열쇠 d_0..d_n의 찾기 확률.

    반환값:
        (최소 기대 값, 뿌리 표) 튜플. 여기서 root[i][j]은
        열쇠 k_i..k_j의 가장 좋은 뿌리 번호이다.
    """
    n = len(p) - 1  # p은 1부터 센다

    # e[i][j] = 열쇠 k_i..k_j의 기대 값
    # w[i][j] = 열쇠 k_i..k_j의 전체 확률 무게
    e = [[0.0] * (n + 2) for _ in range(n + 2)]
    w = [[0.0] * (n + 2) for _ in range(n + 2)]
    root = [[0] * (n + 1) for _ in range(n + 1)]

    # 바탕 경우: e[i][i-1] = q[i-1]
    for i in range(1, n + 2):
        e[i][i - 1] = q[i - 1]
        w[i][i - 1] = q[i - 1]

    # 사슬 길이를 늘려 가며 표 채우기
    for length in range(1, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            e[i][j] = float('inf')
            w[i][j] = w[i][j - 1] + p[j] + q[j]

            for r in range(i, j + 1):
                cost = e[i][r - 1] + e[r + 1][j] + w[i][j]
                if cost < e[i][j]:
                    e[i][j] = cost
                    root[i][j] = r

    return e[1][n], root

def print_optimal_bst(root: list[list[int]], i: int, j: int,
                      parent: str = "root") -> None:
    """가장 좋은 이진 찾기 나무의 짜임 찍기."""
    if i > j:
        print(f"  d_{j} is {parent}")
        return
    r = root[i][j]
    print(f"  k_{r} is {parent}")
    print_optimal_bst(root, i, r - 1, f"left child of k_{r}")
    print_optimal_bst(root, r + 1, j, f"right child of k_{r}")

# === 시연 ===

if __name__ == "__main__":
    # CLRS의 보기: 확률이 주어진 열쇠 5개
    p = [0, 0.15, 0.10, 0.05, 0.10, 0.20]  # 1부터 센다
    q = [0.05, 0.10, 0.05, 0.05, 0.05, 0.10]

    cost, root = optimal_bst(p, q)
    print(f"Minimum expected search cost: {cost:.2f}")
    print("Optimal BST structure:")
    print_optimal_bst(root, 1, 5)
```

**출력:**

```
Minimum expected search cost: 2.75
Optimal BST structure:
  k_2 is root
  k_1 is left child of k_2
  d_0 is left child of k_1
  d_1 is right child of k_1
  k_5 is right child of k_2
  k_4 is left child of k_5
  k_3 is left child of k_4
  d_2 is left child of k_3
  d_3 is right child of k_3
  d_4 is right child of k_4
  d_5 is right child of k_5
```

뿌리에 있는 열쇠 $k_2$이 다가감 잦기의 균형을 잡는다. 가장 자주 찾는 열쇠 $k_5$($p_5 = 0.20$)이 깊이 1에 있어 기대 값에 보태는 몫을 가장 작게 한다.

---

## 5. 복잡도

| 항목 | 비용 |
|--------|:----:|
| Time   | $O(n^3)$ |
| Space  | $O(n^2)$ |

겹친 되돌이 셋(길이, 비롯하는 자리, 뿌리 고르기) 때문에 $O(n^3)$ 때가 든다. 크누스의 다듬기는 $\text{root}[i, j-1] \le \text{root}[i, j] \le \text{root}[i+1, j]$임을 살펴 $r$을 찾을 구간을 좁혀 이를 $O(n^2)$으로 줄인다.

---

## 6. 균형 이진 탐색 트리와의 비교

| 전략 | 기대 값 | 보장 |
|----------|:------------:|:---------:|
| 가장 좋은 이진 찾기 나무 | 가능한 최솟값 | 잦기를 알아야 한다 |
| 고른 이진 찾기 나무 | 찾을 때마다 $O(\log n)$ | 잦기를 몰라도 된다 |
| 스플레이 나무 | 고르게 나누어 $O(\log n)$ | 쓰는 무늬에 맞추어 간다 |

가장 좋은 이진 찾기 나무는 붙박이 짜임이다. 다가감 잦기가 때에 따라 바뀌면 스플레이 나무 같은 스스로 고치는 나무가 움직이는 대안이 된다.

---

## 연습문제

**연습문제 1.**
가장 좋은 이진 찾기 나무의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    Optimal Binary Search Tree은 나누어 다스리기 틀을 쓴다. 문제를 더 작은 잔문제로 쪼개고, 되부르며 풀고, 그 결과를 아우른다. 때 복잡도는 잔문제의 크기와 아우르는 값을 다스리는 되돌이 식이 정한다. 흔히 으뜸 정리나 되부름 나무 살피기로 닫힌 꼴의 복잡도를 얻는다. $\square$

---

**연습문제 2.**
가장 좋은 이진 찾기 나무의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    되돌이 식은 그 알고리즘이 어떻게 나누는지에 달려 있다(잔문제의 수 $a$, 크기를 줄이는 값 $b$, 아우르는 값 $f(n)$). 으뜸 정리를 쓴다. $f(n)$을 $n^{\log_b a}$과 견주어 어느 갈래인지 가린다. $f(n) = \Theta(n^{\log_b a})$이면(둘째 갈래) $T(n) = \Theta(n^{\log_b a} \log n)$이다. $\square$

---

**연습문제 3.**
가장 좋은 이진 찾기 나무이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    막무가내로 하는 길은 흔히 $O(n^2)$ 이상이 든다. 나누어 다스리는 길은 되부르며 쪼개어 군더더기 셈을 줄이므로 복잡도가 더 낮다. 들임 크기가 $n = 10^6$이면 $O(n^2) = 10^{12}$과 $O(n \log n) = 2 \times 10^7$의 차이는 $50{,}000$ 곱절이다. $\square$

---

**연습문제 4.**
가장 좋은 이진 찾기 나무의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    밑 자리는 더 나눌 수 없을 만큼 작은 들임을 다룬다(흔히 $n \leq 1$이나 $n \leq 2$). 이때는 옳은 결과를 곧바로 돌려주어야 한다. 밑 자리가 제대로 없으면 되부름이 끝나지 않는다. 밑 자리를 더 크게 잡고($n \leq 10$ 따위) 더 단순한 알고리즘으로 갈아타면 같은 점근 복잡도를 지키면서 되부름 덤을 줄여 참으로 더 빠르게 할 수 있다. $\square$

## 정리하며

이 마당은 문제 서술、가장 좋은 밑짜임、점화식、구현을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 15장: Dynamic Programming.
- Knuth, D. E. (1971). Optimum binary search trees. *Acta Informatica*, 1(1), 14--25.
