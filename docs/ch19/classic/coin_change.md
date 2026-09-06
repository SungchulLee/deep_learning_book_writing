# 동전 거스름(욕심쟁이)

동전 액면 모음으로 거스름돈을 만들 때 자연스러운 욕심쟁이 전략은 남은 액수를 넘지 않는 가장 큰 동전을 늘 고르는 것이다. 이는 표준 화폐 체계(보기로 미국 동전 1, 5, 10, 25센트)에서는 옳게 돌아가지만 아무 액면 모음에서나 되지는 않는다. 욕심쟁이가 언제 되고 언제 어그러지는지 알면 일반 경우에 동적 계획을 쓰게 된다.

## 욕심쟁이 방식

Given denominations $d_1 > d_2 > \cdots > d_k$ and a target amount $A$, the greedy algorithm repeatedly selects the largest denomination that fits:

1. 남은 값 $= A$, 개수 $= 0$으로 둔다.
2. For each denomination $d_i$ (largest first): use $\lfloor \text{remaining} / d_i \rfloor$ coins, update remaining.
3. 남은 값이 $0$이면 개수를 돌려준다. 아니면 (욕심쟁이로는) 풀이가 없다.

## 욕심쟁이가 통할 때

A coin system is called **canonical** if the greedy algorithm always produces the minimum number of coins. The US system $\{1, 5, 10, 25\}$ is canonical.

!!! warning "표준이 아닌 체계에서는 욕심쟁이가 어그러진다"
    For denominations $\{1, 3, 4\}$ and target $6$: greedy picks $4 + 1 + 1 = 3$ coins, but $3 + 3 = 2$ coins is optimal.

## 표준 체계에서 욕심쟁이의 옳음

미국 동전 체계에서 욕심쟁이가 통하는 것은, 더 큰 동전으로 갈음할 수 있는 방식으로 동전을 쓰는 가장 좋은 풀이가 없기 때문이다. 보기로 1센트 다섯 개를 쓰는 가장 좋은 풀이는 없고(5센트로 갈음된다), 10센트 둘과 5센트 하나를 쓰는 가장 좋은 풀이도 없다(25센트로 갈음된다). 이런 눌러 이김 조건이 욕심쟁이의 가장 좋음을 보장한다.

## 일반 풀이: 동적 계획

아무 액면에서든 액수 $i$을 만드는 동전의 최소 개수는 다음을 채운다:

$$
\text{dp}[i] = \min_{c \in \text{coins},\; c \le i} (\text{dp}[i - c] + 1)
$$

with base case $\text{dp}[0] = 0$ and $\text{dp}[i] = \infty$ for amounts that cannot be formed.

## 구현

```python
"""
동전 거스름: 욕심쟁이와 동적 짜기의 견줌.

욕심쟁이가 통하는 때(표준 체계)와 동적 짜기가
필요한 때(아무 액면가)를 보인다.
"""

# === 욕심쟁이 동전 거스름 ===

def coin_change_greedy(coins: list[int], amount: int) -> list[int]:
    """욕심쟁이 전략(큰 것부터)으로 거스름돈을 만든다.

    인수:
        coins: 쓸 수 있는 액면가(내림차순으로 정렬).
        amount: 목표 금액.

    반환값:
        쓴 동전의 목록(표준이 아닌 체계에서는 가장 좋지 않을 수 있다).
    """
    result = []
    remaining = amount
    for coin in sorted(coins, reverse=True):
        while remaining >= coin:
            result.append(coin)
            remaining -= coin
    return result if remaining == 0 else []


# === 동적 짜기 동전 거스름 ===

def coin_change_dp(coins: list[int], amount: int) -> int:
    """목표 금액을 만드는 데 드는 가장 적은 동전 수를 찾는다.

    인수:
        coins: 쓸 수 있는 액면가.
        amount: 목표 금액.

    반환값:
        가장 적은 동전 수, 할 수 없으면 -1.
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for c in coins:
            if c <= i and dp[i - c] + 1 < dp[i]:
                dp[i] = dp[i - c] + 1

    return dp[amount] if dp[amount] != float('inf') else -1


# === 시연 ===

if __name__ == "__main__":
    # 표준 체계: 욕심쟁이가 통한다
    us_coins = [25, 10, 5, 1]
    amount = 63
    greedy_result = coin_change_greedy(us_coins, amount)
    dp_result = coin_change_dp(us_coins, amount)
    print(f"US coins, amount={amount}:")
    print(f"  Greedy: {len(greedy_result)} coins {greedy_result}")
    print(f"  DP:     {dp_result} coins")

    # 표준이 아닌 체계: 욕심쟁이가 어긋난다
    coins = [1, 3, 4]
    amount = 6
    greedy_result = coin_change_greedy(coins, amount)
    dp_result = coin_change_dp(coins, amount)
    print(f"\nCoins {coins}, amount={amount}:")
    print(f"  Greedy: {len(greedy_result)} coins {greedy_result}")
    print(f"  DP:     {dp_result} coins")
```

**출력:**

```
US coins, amount=63:
  Greedy: 6 coins [25, 25, 10, 1, 1, 1]
  DP:     6 coins

Coins [1, 3, 4], amount=6:
  Greedy: 3 coins [4, 1, 1]
  DP:     2 coins
```

For the US system, greedy and DP agree. For $\{1, 3, 4\}$, greedy uses 3 coins while the optimal DP solution uses only 2 coins ($3 + 3$).

## 복잡도

| 알고리즘 | 시간 | 공간 |
|-----------|:----:|:-----:|
| 욕심쟁이 | $O(k)$, $k$ = 액면의 개수 | $O(1)$ |
| DP        | $O(A \cdot k)$ where $A$ = amount | $O(A)$ |

욕심쟁이 방식은 더 빠르지만 표준 체계에서만 옳다. 동적 계획은 늘 최적을 찾지만 액수에 비례하는 시간이 든다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 16장: Greedy Algorithms.
- Kozen, D., & Zaks, S. (1994). Optimal bounds for the change-making problem. *Theoretical Computer Science*, 123(2), 377--388.

## 연습문제

**연습문제 1.**
동전 거스름(욕심쟁이)에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    The greedy choice selects the locally optimal option at each step. For Coin Change (Greedy), this choice satisfies the greedy choice property: there exists an optimal solution that includes this greedy selection. Combined with optimal substructure (the remaining subproblem after the greedy choice is also optimally solvable by the same strategy), the greedy algorithm produces a globally optimal solution. $\square$

---

**연습문제 2.**
동전 거스름(욕심쟁이)이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    Optimal substructure means that an optimal solution to the problem contains optimal solutions to its subproblems. For Coin Change (Greedy), after making the greedy choice, the remaining problem is a smaller instance of the same type. If the subproblem solution were not optimal, we could improve the overall solution by replacing it — contradicting overall optimality. Therefore optimal substructure holds. $\square$

---

**연습문제 3.**
동전 거스름(욕심쟁이)의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    The time complexity depends on the sorting step (if required) and the greedy selection loop. Sorting typically dominates at $O(n \log n)$. The greedy loop processes each element once in $O(n)$. Total: $O(n \log n)$. If the input is pre-sorted, the algorithm runs in $O(n)$. $\square$

---

**연습문제 4.**
(동전 거스름(욕심쟁이)에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    Consider an alternative greedy criterion that does not align with the problem's structure. This alternative may select an element that blocks better future choices. The counterexample demonstrates that the wrong greedy criterion can produce a suboptimal result, highlighting why the specific greedy choice property must be proven for each problem. $\square$
