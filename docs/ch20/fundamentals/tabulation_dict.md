# 사전으로 하는 표 채우기
$$\begin{array}{llll}
\text{Recursion}&&\text{No Dynamic Programming (No Memoization or Tabulation)}\\
\text{Top Down}&&\text{Dynamic Programming (Memoization)}\\
\text{Bottom Up}&&\text{Dynamic Programming (Tabulation)}
\end{array}$$

```python
import matplotlib.pyplot as plt
import numpy as np
import time

def fib_bottom_up(n):
    memo = {}
    for i in range(n+1):
        if i <= 1:
            memo[i] = i
        else:
            memo[i] = memo[i-1] + memo[i-2]
    return memo[n]

def main():    
    time_record = []
    for n in range(1,100):
        tic = time.time()
        fib_bottom_up(n)
        toc = time.time()
        time_record.append(toc-tic)

    plt.plot(np.arange(1,100), time_record) 
    plt.show()

    
if __name__ == "__main__":
    main()
```

**출력:**
```
<Figure size 432x288 with 1 Axes>
```

# 참고 문헌

CS_Dojo
[youtube](https://www.youtube.com/watch?v=B0NtAFf4bvU&index=6&list=PLBZBJbE_rGRV8D7XZ08LK6z-4zPoWzu5H)
[youtube](https://www.youtube.com/watch?v=5o-kdjv7FD0)

Telusko
[youtube](https://www.youtube.com/watch?v=gfhtaP5Wq7M&index=39&list=PLsyeobzWxl7poL9JTVyndKe62ieoN-MZ3)
[youtube](https://www.youtube.com/watch?v=XkL3SUioNvo&list=PLsyeobzWxl7poL9JTVyndKe62ieoN-MZ3&index=40)
[youtube](https://www.youtube.com/watch?v=TqqQld6m6A0&list=PLsyeobzWxl7poL9JTVyndKe62ieoN-MZ3&index=41)

Abdul_Bari [youtube](https://www.youtube.com/watch?v=5dRGRueKU3M&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=46)

## 연습문제

**연습문제 1.**
동적 짜기의 적어 두기(위에서 아래로)와 표 채우기(아래에서 위로) 방식의 차이를 설명하라.

??? success "연습문제 1 풀이"
    **적어 두기**는 본디 문제에서 시작해 되돌이로 아래 문제로 쪼개며, 결과를 곳간에 담아 겹치는 셈을 피한다. 정말 필요한 아래 문제만 푼다(게으른 값매김). **표 채우기**는 가장 작은 아래 문제부터 본디 문제까지 되풀이로 표를 채우며 모든 아래 문제를 정해진 차례로 푼다. 표 채우기는 되돌이 군더더기와 쌓임 넘침 위험을 피하고, 적어 두기는 흔히 짜기 쉬우며(되돌이 풀이에 곳간만 붙이면 된다) 아래 문제가 다 필요하지 않을 때 더 빠를 수 있다.

---

**연습문제 2.**
적어 두기와 표 채우기 둘 다로 피보나치 차례를 셈하라. 두 방식의 시간 복잡도와 공간 복잡도를 견주어라.

??? success "연습문제 2 풀이"
    ```python
    # 적어 두기
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def fib_memo(n):
        if n <= 1:
            return n
        return fib_memo(n - 1) + fib_memo(n - 2)

    # 표 채우기
    def fib_tab(n):
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
    ```
    둘 다 $O(n)$ 시간과 $O(n)$ 공간에 돈다. 표 채우기는 마지막 두 값만 지녀 $O(1)$ 공간까지 줄일 수 있지만 적어 두기로는 그러기가 더 어렵다.

---

**연습문제 3.**
동적 짜기를 쓰려면 문제가 갖춰야 할 핵심 성질 둘은 무엇인가? 한쪽만 갖추고 다른 쪽은 못 갖춘 문제의 보기를 들어라.

??? success "연습문제 3 풀이"
    **(1) 가장 좋은 아래 짜임**: 가장 좋은 풀이가 아래 문제의 가장 좋은 풀이를 품는다. **(2) 겹치는 아래 문제**: 같은 아래 문제를 되풀이해 푼다. **가장 좋은 아래 짜임은 있으나 겹치는 아래 문제가 없는 보기**: 어울러 정렬. 배열을 가장 좋게 정렬하려면 반쪽씩 가장 좋게 정렬해야 하지만 아래 배열이 저마다 다르므로(겹침이 없다) 동적 짜기가 나누어 이기기보다 나을 것이 없다. **겹치는 아래 문제는 있으나 가장 좋은 아래 짜임이 없는 보기**: 그래프에서 가장 긴 단순 길 찾기(어떤 꼭짓점을 지나는 가장 긴 길이 반드시 그 꼭짓점까지의 가장 긴 길을 쓰지는 않는다).

---

**연습문제 4.**
표의 바로 앞 가로줄(또는 앞선 몇 칸)만 필요할 때 표 채우기 바탕 동적 짜기 풀이의 공간 복잡도를 어떻게 줄이는지 설명하라.

??? success "연습문제 4 풀이"
    되돌이 관계식이 앞선 칸 가운데 정해진 개수에만 기대면 표 전체를 굴리는 창으로 갈음할 수 있다. 피보나치($f(n) = f(n-1) + f(n-2)$)는 변수 둘, 곧 `prev`과 `curr`만 지니면 된다. 가로줄 $i$이 가로줄 $i-1$에만 기대는 고침 거리 같은 2차원 문제는 가로줄 둘을 번갈아 쓴다. `dp[i][w]`이 `dp[i-1][w]`과 `dp[i-1][w-w_i]`에 기대는 0/1 배낭은 $w$을 거꾸로 되풀이해 가로줄 하나로 제자리에서 고친다. 그러면 시간은 $O(n \cdot W)$을 지키면서 공간이 $O(n \cdot W)$에서 $O(W)$로 준다.
