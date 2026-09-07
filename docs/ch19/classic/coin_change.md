# 동전 거스름(욕심쟁이)

동전 액면 모음으로 거스름돈을 만들 때 자연스러운 욕심쟁이 전략은 남은 액수를 넘지 않는 가장 큰 동전을 늘 고르는 것이다. 이는 표준 화폐 체계(보기로 미국 동전 1, 5, 10, 25센트)에서는 옳게 돌아가지만 아무 액면 모음에서나 되지는 않는다. 욕심쟁이가 언제 되고 언제 어그러지는지 알면 일반 경우에 동적 계획을 쓰게 된다.

## 욕심쟁이 방식

액면 $d_1 > d_2 > \cdots > d_k$과 목표 금액 $A$이 주어지면 욕심쟁이 알고리즘은 들어맞는 가장 큰 액면을 거듭 고른다.

1. 남은 값 $= A$, 개수 $= 0$으로 둔다.
2. 액면 $d_i$마다(큰 것부터) $\lfloor \text{remaining} / d_i \rfloor$개를 쓰고 남은 금액을 고친다.
3. 남은 값이 $0$이면 개수를 돌려준다. 아니면 (욕심쟁이로는) 풀이가 없다.

## 욕심쟁이가 통할 때

욕심쟁이 알고리즘이 늘 가장 적은 동전 수를 내놓으면 그 동전 얼개를 **반듯하다**고 한다. 미국 얼개 $\{1, 5, 10, 25\}$은 반듯하다.

!!! warning "표준이 아닌 체계에서는 욕심쟁이가 어그러진다"
    액면 $\{1, 3, 4\}$과 목표 $6$에서 욕심쟁이는 $4 + 1 + 1 = 3$개를 고르지만 $3 + 3 = 2$개가 가장 좋다.

## 표준 체계에서 욕심쟁이의 옳음

미국 동전 체계에서 욕심쟁이가 통하는 것은, 더 큰 동전으로 갈음할 수 있는 방식으로 동전을 쓰는 가장 좋은 풀이가 없기 때문이다. 보기로 1센트 다섯 개를 쓰는 가장 좋은 풀이는 없고(5센트로 갈음된다), 10센트 둘과 5센트 하나를 쓰는 가장 좋은 풀이도 없다(25센트로 갈음된다). 이런 눌러 이김 조건이 욕심쟁이의 가장 좋음을 보장한다.

## 일반 풀이: 동적 계획

아무 액면에서든 액수 $i$을 만드는 동전의 최소 개수는 다음을 채운다:

$$
\text{dp}[i] = \min_{c \in \text{coins},\; c \le i} (\text{dp}[i - c] + 1)
$$

밑 자리는 $\text{dp}[0] = 0$이고 만들 수 없는 금액에는 $\text{dp}[i] = \infty$이다.

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

미국 얼개에서는 욕심쟁이와 갈피 다지기의 답이 같다. $\{1, 3, 4\}$에서는 욕심쟁이가 동전 3개를 쓰지만 가장 좋은 갈피 다지기 풀이는 2개($3 + 3$)만 쓴다.

## 복잡도

| 알고리즘 | 시간 | 공간 |
|-----------|:----:|:-----:|
| 욕심쟁이 | $O(k)$, $k$ = 액면의 개수 | $O(1)$ |
| 갈피 다지기        | $A$을 금액이라 할 때 $O(A \cdot k)$ | $O(A)$ |

욕심쟁이 방식은 더 빠르지만 표준 체계에서만 옳다. 동적 계획은 늘 최적을 찾지만 액수에 비례하는 시간이 든다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 16장: Greedy Algorithms.
- Kozen, D., & Zaks, S. (1994). Optimal bounds for the change-making problem. *Theoretical Computer Science*, 123(2), 377--388.

## 연습문제

**연습문제 1.**
동전 거스름(욕심쟁이)에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    욕심쟁이 고름은 걸음마다 그 자리에서 가장 좋은 것을 고른다. Coin Change (Greedy)에서는 이 고름이 욕심쟁이 고름 성질을 채운다. 곧 이 욕심쟁이 고름을 담은 가장 좋은 풀이가 있다. 여기에 가장 좋은 아래 짜임(욕심쟁이 고름 뒤 남은 아래 문제도 같은 전략으로 가장 좋게 풀린다)을 더하면 욕심쟁이 알고리즘이 두루 가장 좋은 풀이를 내놓는다. $\square$

---

**연습문제 2.**
동전 거스름(욕심쟁이)이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    가장 좋은 아래 짜임이란 문제의 가장 좋은 풀이가 그 아래 문제의 가장 좋은 풀이를 담는다는 뜻이다. Coin Change (Greedy)에서는 욕심쟁이 고름을 하고 나면 남은 문제가 같은 갈래의 더 작은 사례가 된다. 아래 문제의 풀이가 가장 좋지 않다면 그것을 갈음해 온 풀이를 더 낫게 할 수 있는데, 이는 온 풀이가 가장 좋다는 것과 어긋난다. 그러므로 가장 좋은 아래 짜임이 이루어진다. $\square$

---

**연습문제 3.**
동전 거스름(욕심쟁이)의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    시간 복잡도는 (필요하다면) 줄 세우는 걸음과 욕심쟁이 고름 되돌이에 달렸다. 흔히 줄 세우기가 $O(n \log n)$으로 가장 크다. 욕심쟁이 되돌이는 원소마다 한 번씩 다루어 $O(n)$이다. 모두 $O(n \log n)$이다. 들임이 미리 줄 세워져 있으면 알고리즘은 $O(n)$에 돈다. $\square$

---

**연습문제 4.**
(동전 거스름(욕심쟁이)에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    문제의 짜임과 맞지 않는 다른 욕심쟁이 잣대를 생각해 보자. 이 잣대는 뒤에 더 좋은 고름을 막는 원소를 고를 수 있다. 어긋나는 보기는 잘못된 욕심쟁이 잣대가 가장 좋지 않은 결과를 낼 수 있음을 보이며, 문제마다 그에 맞는 욕심쟁이 고름 성질을 반드시 증명해야 하는 까닭을 드러낸다. $\square$
