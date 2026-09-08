# 악수 보조정리

악수 보조정리는 그래프 이론에서 가장 먼저 만나는 정리일 때가 많고, 지금도 가장 자주 쓰이는 것 가운데 하나이다. 이름은 잔치에 빗댄 데서 왔다. 곧 모임에서 저마다 몇몇과 악수하면 "악수 끝"의 총 개수는 늘 짝수인데, 악수마다 손이 꼭 둘 들기 때문이다. 이 단순한 세기 따짐은 놀랍도록 힘센 결과를 낳는다. 어떤 그래프가 있을 수 없음을 증명하는 데서부터 망의 [차수](degree.md) 분포를 뜯어보는 데까지 쓰인다.

---

## 1. 진술과 증명

!!! tip "정리: 악수 보조정리"
    아무 무방향 그래프 $G = (V, E)$에 대해 다음이 성립한다,

$$
\sum_{v \in V} \deg(v) = 2|E|
$$

**증명.** 합 $\sum_{v \in V} \deg(v)$을 생각하자. 변 $\{u, w\} \in E$마다 $\deg(u)$에 꼭 1, $\deg(w)$에 꼭 1을 보탠다. 다른 꼭짓점의 차수는 이 변에 영향받지 않는다. 그러므로 변마다 전체 합에 꼭 2을 보태고, 모든 변에 걸쳐 합하면 $2|E|$이 된다. $\square$

!!! example "보조정리 확인하기"
    $\{a, b, c, d\}$ 위에 변 $\{a,b\}, \{b,c\}, \{c,d\}, \{a,d\}, \{a,c\}$을 갖는 그래프를 생각하자. 차수는 $\deg(a)=3$, $\deg(b)=2$, $\deg(c)=3$, $\deg(d)=2$이다. 합은 $3+2+3+2=10=2 \times 5 = 2|E|$이다.

---

## 2. 따름정리: 차수가 홀수인 꼭짓점은 짝수 개

!!! tip "따름정리"
    아무 무방향 그래프에서 차수가 홀수인 꼭짓점의 개수는 짝수이다.

**증명.** $V_{\text{odd}} = \{v \in V : \deg(v) \text{ is odd}\}$이라 하고 $V_{\text{even}} = V \setminus V_{\text{odd}}$이라 하자. 그러면 다음과 같다

$$
\sum_{v \in V_{\text{odd}}} \deg(v) = 2|E| - \sum_{v \in V_{\text{even}}} \deg(v)
$$

오른쪽은 짝수이다(짝수 둘의 차). 왼쪽의 항마다 홀수이므로 항의 개수 $|V_{\text{odd}}|$은 짝수여야 한다. $\square$

이 따름정리는 이를테면 꼭짓점 꼭 3개의 차수가 홀수인 그래프는 없음을 곧바로 알려 준다.

---

## 3. 방향 그래프에서의 대응

방향 그래프에서 방향 변 $(u, v)$마다 $\deg^+(u)$에 1, $\deg^-(v)$에 1을 보탠다. 모든 꼭짓점에 걸쳐 합하면 다음과 같다:

$$
\sum_{v \in V} \deg^+(v) = \sum_{v \in V} \deg^-(v) = |E|
$$

이것이 악수 보조정리의 방향 판이다. 나가는 차수의 합과 들어오는 차수의 합이 같고 둘 다 변의 개수와 같다.

---

## 4. 응용

악수 보조정리는 여러 자리에서 증명 도구로 쓰인다.

### 있음을 따지는 논증

어떤 차수 제약을 갖는 그래프가 있을 수 없음을 보이려면 $\sum \deg(v)$이 홀수가 되는지 살펴라:

- **주장:** 꼭짓점 5개에서 꼭짓점마다 차수가 3인 그래프는 없다.
- **확인:** $\sum \deg(v) = 5 \times 3 = 15$으로 홀수이다. 보조정리에 따라 $2|E|$은 짝수여야 한다. 어긋난다.

### 변 세기

차수 열만 주어져도 이 보조정리는 곧바로 변의 개수를 준다:

$$
|E| = \frac{1}{2}\sum_{v \in V} \deg(v)
$$

### 평균 차수

그래프의 **평균 차수**는 다음과 같다

$$
\bar{d} = \frac{1}{|V|}\sum_{v \in V} \deg(v) = \frac{2|E|}{|V|}
$$

이 관계는 무작위 그래프와 망 모형을 뜯어보는 데 근본이 된다.

---

## 5. 확인 코드

```python
"""
무방향 그래프와 방향 그래프에서 악수 보조정리 확인하기.

차수의 합을 셈해 (무방향에서는) 변 개수의 두 배,
(방향에서는) 변 개수와 같은지 확인한다.
"""

# === 무방향 확인 ===

def verify_handshaking_undirected(adj, n, num_edges):
    """차수의 합이 변 개수의 2배인지 확인한다."""
    degree_sum = sum(len(adj[v]) for v in range(n))
    holds = (degree_sum == 2 * num_edges)
    return degree_sum, holds

# === 방향 확인 ===

def verify_handshaking_directed(adj, n, num_edges):
    """나가는 차수의 합이 변 개수와 같은지 확인한다."""
    out_degree_sum = sum(len(adj[v]) for v in range(n))
    in_degrees = [0] * n
    for u in range(n):
        for v in adj[u]:
            in_degrees[v] += 1
    in_degree_sum = sum(in_degrees)
    holds = (out_degree_sum == num_edges == in_degree_sum)
    return out_degree_sum, in_degree_sum, holds

# === 홀수 차수 세기 ===

def count_odd_degree_vertices(adj, n):
    """차수가 홀수인 꼭짓점을 세고 그 개수가 짝수인지 확인한다."""
    odd_count = sum(1 for v in range(n) if len(adj[v]) % 2 == 1)
    return odd_count, odd_count % 2 == 0

# === 메인 ===

if __name__ == "__main__":
    # 무방향 그래프: 변 5개
    adj_u = [[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]
    deg_sum, ok = verify_handshaking_undirected(adj_u, 4, 5)
    print(f"Undirected: sum(deg) = {deg_sum}, 2*|E| = 10, holds: {ok}")

    odd_count, even_check = count_odd_degree_vertices(adj_u, 4)
    print(f"Odd-degree vertices: {odd_count}, count is even: {even_check}")

    # 방향 그래프: 변 4개
    adj_d = [[1, 2], [2], [0], []]
    out_s, in_s, ok_d = verify_handshaking_directed(adj_d, 4, 3)
    print(f"\nDirected: sum(out-deg) = {out_s}, sum(in-deg) = {in_s}, "
          f"|E| = 3, holds: {ok_d}")
```

**출력:**
```
Undirected: sum(deg) = 10, 2*|E| = 10, holds: True
Odd-degree vertices: 2, count is even: True
Directed: sum(out-deg) = 3, sum(in-deg) = 3, |E| = 3, holds: True
```

---

## 연습문제

**연습문제 1.**
7명이 모인 잔치에서 저마다 꼭 3명과 악수한다. 있을 수 있는 일인가? 악수 보조정리로 뒷받침하여라.

??? success "연습문제 1 풀이"
    악수 그래프에서 사람마다 차수가 3이면 차수의 합은 $7 \times 3 = 21$이다. 악수 보조정리에 따라 $\sum \deg(v) = 2|E|$이며 이는 짝수여야 한다. 21은 홀수이므로 그런 그래프는 없다. $\square$

---

**연습문제 2.**
아무 그래프에서나 차수가 홀수인 꼭짓점의 개수가 짝수임을 증명하여라.

??? success "연습문제 2 풀이"
    차수가 홀수인 꼭짓점의 묶음을 $O$, 짝수인 꼭짓점의 묶음을 $E_v$이라 하자. 악수 보조정리에 따라 $\sum_{v \in O} \deg(v) + \sum_{v \in E_v} \deg(v) = 2|E|$이다. 둘째 합은 짝수이다(짝수의 합). 그러므로 $\sum_{v \in O} \deg(v)$도 짝수이다. 이 합의 항마다 홀수이므로 항의 개수 $|O|$은 짝수여야 한다. $\square$

---

**연습문제 3.**
이어진 그래프 $G$에 꼭짓점 10개가 있다. 일곱 꼭짓점의 차수가 3이고 세 꼭짓점의 차수가 5이다. $G$의 변은 몇 개인가?

??? success "연습문제 3 풀이"
    악수 보조정리에 따라 $2|E| = 7 \times 3 + 3 \times 5 = 21 + 15 = 36$이므로 $|E| = 18$이다. $\square$

---

**연습문제 4.**
악수 보조정리의 방향 판을 증명하여라. 곧 아무 방향 그래프에서나 $\sum_{v} \deg^+(v) = \sum_{v} \deg^-(v) = |E|$임을 보여라.

??? success "연습문제 4 풀이"
    방향 변 $(u, v)$마다 $\deg^+(u)$(꼬리의 나가는 차수)에 꼭 1, $\deg^-(v)$(머리의 들어오는 차수)에 꼭 1을 보탠다. 모든 변에 걸쳐 합하면 변마다 $\sum \deg^+(v)$에 한 번, $\sum \deg^-(v)$에 한 번 세어진다. 그러므로 두 합 모두 $|E|$과 같다. $\square$

---

**연습 5.**
이항 계수 공식을 쓰지 않고 악수 보조정리로 $K_5$의 변이 꼭 10개임을 증명하여라.

??? success "연습 5의 풀이"
    $K_5$에서 꼭짓점마다 나머지 4개 모두에 이웃하므로 꼭짓점 5개 저마다 $\deg(v) = 4$이다. 악수 보조정리에 따라 $2|E| = \sum_{v=1}^{5} 4 = 20$이므로 $|E| = 10$이다. $\square$

## 정리하며

이 마당은 진술과 증명、따름정리: 차수가 홀수인 꼭짓점은 짝수 개、방향 그래프에서의 대응、응용을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 22장.
- West, D. B. (2001). *Introduction to Graph Theory* (2nd ed.). Prentice Hall. 명제 1.3.3.
