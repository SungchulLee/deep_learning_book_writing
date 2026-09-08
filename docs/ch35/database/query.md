# 물음 다듬기

SQL 물음 하나를 치르는 길은 여럿이고, 길마다 빠르기가 크게 다르다. **물음 다듬기**는 데이터베이스 다룸 얼개(DBMS)가 무엇을 바라는지만 적은 SQL 글월을 잘 듣는 치름 꾀로 바꾸는 일이다. 다듬이는 여러 꾀를 살피고, 자료에 대한 셈속으로 저마다의 비용을 어림한 뒤, 가장 싼 것을 고른다. 물음 다듬기를 알면 데이터베이스를 꾸미는 이도, 프로그램을 짓는 이도 다듬이가 잘 다룰 수 있는 물음을 쓰게 된다.

---

## 1. SQL에서 치름 꾀까지

물음은 치러지기 앞서 여러 마디를 거친다.

1. **가름**: SQL 글을 가름 나무로 쪼갠다.
2. **속뜻 꾀**: 가름 나무를 관계 셈법 식으로 바꾼다.
3. **다듬기**: 뜻이 같은 여러 속뜻 꾀를 살피고 가장 싼 몸 꾀를 고른다.
4. **치름**: 고른 몸 꾀를 갈무리 얼개 위에서 돌린다.

다듬이가 맡는 곳은 셋째 마디, 곧 어림 비용이 가장 낮은 꾀를 찾는 일이다.

---

## 2. 관계 셈법의 같은 값 규칙

다듬이는 셈법의 같은 값 규칙으로 속뜻 꾀를 고쳐 쓴다. 으뜸 규칙은 이렇다.

**고르기 밀어내리기** -- 사이 결과의 크기를 줄이도록 거르기를 되도록 일찍 건다.

$$
\sigma_{\theta}(R \bowtie S) \equiv \sigma_{\theta}(R) \bowtie S
$$

여기서 $\theta$은 $R$의 속성만 쓴다.

**추리기 밀어내리기** -- 쓰지 않을 칸을 일찍 버린다.

$$
\pi_L(R \bowtie S) \equiv \pi_L(\pi_{L_1}(R) \bowtie \pi_{L_2}(S))
$$

여기서 $L_1$과 $L_2$은 맞이음 칸과 $L$에 든 칸을 아우른다.

**맞이음의 자리 바꿈과 묶음 바꿈**:

$$
R \bowtie S \equiv S \bowtie R
$$

$$
(R \bowtie S) \bowtie T \equiv R \bowtie (S \bowtie T)
$$

이 덕에 다듬이가 맞이음 차례를 마음대로 바꿀 수 있고, 맞이음 차례가 꾀의 비용을 판치므로 이는 아주 중요하다.

---

## 3. 비용 어림하기

다듬이는 후보 꾀마다 다음을 바탕으로 비용을 매긴다.

- **들고남 비용**: 원반 쪽을 읽고 쓴 횟수(흔히 판치는 몫).
- **CPU 비용**: 견줌과 해시 연산.
- **그물 비용**: 흩은 데이터베이스에서 자료 옮김.

### 고르기 몫

어떤 가름의 **고르기 몫**은 그것을 채우는 행의 비율이다. 가름 $\sigma_{A=v}(R)$에 대해

$$
\text{sel}(A = v) = \frac{1}{V(A, R)}
$$

여기서 $V(A, R)$은 표 $R$에서 속성 $A$의 서로 다른 값의 수다(값이 고루 퍼졌다고 여긴다).

너비 가름 $\sigma_{A \leq v}(R)$에 대해서는

$$
\text{sel}(A \leq v) = \frac{v - \min(A)}{\max(A) - \min(A)}
$$

### 낱수 어림하기

맞이음의 어림 내놓기 크기는 다음과 같다.

$$
\lvert R \bowtie_A S \rvert = \frac{\lvert R \rvert \cdot \lvert S \rvert}{\max(V(A, R),\; V(A, S))}
$$

낱수를 옳게 어림하는 일이 아주 중요하다. 어긋남이 여러 맞이음을 거치며 불어나면 다듬이가 가장 나은 꾀보다 자릿수만큼 느린 꾀를 고르게 된다.

!!! warning "어림의 어긋남"
    참 자료가 고루 퍼졌다는 여김을 따르는 일은 드물다. 잦기 그림, 뽑아보기, 밑그림이 어림을 낫게 하지만, 낱수 어림은 여전히 물음 다듬기에서 가장 어려운 문제 가운데 하나다.

---

## 4. 맞이음 차례 다듬기

표 $n$개를 맞이음하는 물음에서 있을 수 있는 맞이음 차례의 수는 다음과 같다.

$$
\frac{(2(n-1))!}{(n-1)!}
$$

이는 $n$째 카탈란 수이고 곱절보다 빠르게 커진다. 다듬이는 크게 두 가지 꾀를 쓴다.

**갈피 다지기(System R 결)**: 아래에서 위로 가장 나은 꾀를 쌓는다. 표의 부분 모둠마다, 그 모둠을 둘로 가르는 모든 길을 살펴 가장 싼 맞이음 길을 찾는다. 표 $n$개에 걸리는 때는 $O(3^n)$이고 $n \leq 15$ 남짓까지 쓸 만하다.

**욕심꾸러기 / 어림 꾀**: 비용이 가장 낮은 두 표 맞이음에서 시작해, 다음으로 싼 표를 거듭 붙인다. 빠르지만 온 세상에서 가장 나은 꾀를 놓칠 수 있다.

??? example "표 넷의 갈피 다지기 맞이음 늘어놓기"
    표 $A, B, C, D$이 있을 때 다듬이는 이렇게 살핀다.

    - 두 표 맞이음 모두: $A \bowtie B$, $A \bowtie C$, ..., $C \bowtie D$(짝 6개)
    - 가장 나은 두 표 꾀에서 쌓은 세 표 꾀 모두
    - 가장 나은 세 표 꾀에서 쌓은 마지막 네 표 꾀

    작은 문제마다 한 번만 풀어 갈무리하므로 같은 셈을 되풀이하지 않는다.

---

## 5. 몸 꾀 고르기

속뜻 꾀와 맞이음 차례가 정해지면 다듬이가 **몸 다룸꾼**을 고른다.

| 속뜻 다룸꾼 | 몸 고를 거리 |
|-----------------|-----------------|
| 고르기 ($\sigma$) | 차례로 훑기, 색인 훑기, 비트 그림 훑기 |
| 맞이음 ($\bowtie$) | 겹돌기, 줄 세워 합치기, 해시 맞이음 |
| 줄 세우기 | 바깥 합치기 줄 세우기, 기억 안 빠른 줄 세우기 |
| 뭉뚱그리기 | 해시 뭉뚱그리기, 줄 세우기 바탕 뭉뚱그리기 |

다듬이는 다룸꾼마다의 들고남 비용 꼴을 모아 꾀 하나의 온 비용을 어림한다.

---

## 6. 색인과 물음 빠르기

색인은 물음 꾀를 크게 바꾼다.

- **B 나무 색인**은 같음 물음과 너비 물음을 받친다. 찾기 $O(\log n)$.
- **해시 색인**은 같음 물음만 받는다. 찾기 어림 $O(1)$.
- **덮개 색인**은 물음에 드는 칸을 모두 담아 표 자체에 닿지 않아도 되게 한다.

다듬이는 고르기 몫을 보고 색인을 쓸지 정한다. 어림잡아, 고르기 몫이 10~15% 아래일 때 색인 훑기가 이롭다. 그보다 덜 고르는 가름이면 차례로 훑기가 빠른데, 이는 차례로 하는 들고남이 빠르기 때문이다.

---

## 7. 짜보기

```python
"""
물음 다듬기 -- 고르기 몫 어림과 맞이음 차례 늘어놓기.

표가 적을 때 낱수 어림과 갈피 다지기 바탕 맞이음 차례 다듬기를
보인다.
"""

from itertools import combinations

# === 고르기 몫 어림 =========================================================

def selectivity_equality(n_distinct: int) -> float:
    """값이 고루 퍼졌다는 여김 아래 같음 가름의 고르기 몫을 어림한다.

    Args:
        n_distinct: 그 속성의 서로 다른 값의 수.

    Returns:
        A = v를 채우는 행의 어림 비율.
    """
    if n_distinct <= 0:
        return 1.0
    return 1.0 / n_distinct

def estimate_join_cardinality(card_r: int, card_s: int,
                               distinct_r: int, distinct_s: int) -> float:
    """같음 맞이음의 내놓기 낱수를 어림한다.

    |R join S| = |R| * |S| / max(V(A,R), V(A,S)) 꼴을 쓴다.
    """
    max_distinct = max(distinct_r, distinct_s)
    if max_distinct == 0:
        return 0.0
    return (card_r * card_s) / max_distinct

# === 맞이음 차례 다듬기(갈피 다지기) ========================================

def dp_join_order(tables: dict[str, int],
                  join_costs: dict[tuple[str, str], float]) -> tuple[float, list]:
    """갈피 다지기로 가장 나은 맞이음 차례를 찾는다.

    Args:
        tables: 표 이름에서 낱수로 가는 짝지음.
        join_costs: (표_i, 표_j)에서 맞이음 비용으로 가는 짝지음.
                    빠진 짝은 곧바로 맞이을 수 없다고 여긴다.

    Returns:
        (가장 작은 온 비용, 맞이음 차례) 짝.
    """
    table_names = sorted(tables.keys())
    n = len(table_names)
    name_to_idx = {name: i for i, name in enumerate(table_names)}

    # dp[frozenset] = (비용, 낱수, 맞이음 차례)
    dp: dict[frozenset, tuple[float, int, list]] = {}

    # 밑자리: 표 하나
    for name in table_names:
        dp[frozenset([name])] = (0, tables[name], [name])

    # 크기를 늘려 가며 부분 모둠을 늘어놓는다
    for size in range(2, n + 1):
        for subset in combinations(table_names, size):
            fs = frozenset(subset)
            best = (float("inf"), 0, [])

            # 부분 모둠을 비지 않은 둘로 가르는 모든 길을 해 본다
            for split_size in range(1, size):
                for left in combinations(subset, split_size):
                    left_set = frozenset(left)
                    right_set = fs - left_set
                    if left_set not in dp or right_set not in dp:
                        continue

                    l_cost, l_card, l_seq = dp[left_set]
                    r_cost, r_card, r_seq = dp[right_set]

                    # 두 쪽 사이에 맞이음 모서리가 있는지 살핀다
                    join_cost = 0
                    for lt in left_set:
                        for rt in right_set:
                            key = (min(lt, rt), max(lt, rt))
                            if key in join_costs:
                                join_cost = join_costs[key]
                                break
                        if join_cost > 0:
                            break

                    if join_cost == 0:
                        continue

                    total = l_cost + r_cost + join_cost
                    if total < best[0]:
                        out_card = l_card + r_card  # 단순하게 잡음
                        best = (total, out_card, l_seq + r_seq)

            if best[0] < float("inf"):
                dp[fs] = best

    full_set = frozenset(table_names)
    if full_set in dp:
        cost, _, sequence = dp[full_set]
        return cost, sequence
    return float("inf"), []

# === 메인 ===================================================================

if __name__ == "__main__":
    # 고르기 몫 보기
    print("=== 고르기 몫 어림 ===")
    print(f"같음 (서로 다른 값 100개): {selectivity_equality(100):.4f}")
    print(f"맞이음 낱수 (1000 x 500, 서로 다른 값 50 대 80): "
          f"{estimate_join_cardinality(1000, 500, 50, 80):.0f}")
    print()

    # 맞이음 차례 다듬기
    print("=== 맞이음 차례 다듬기(갈피 다지기) ===")
    tables = {"A": 1000, "B": 5000, "C": 200, "D": 3000}
    costs = {
        ("A", "B"): 150,
        ("A", "C"): 30,
        ("B", "D"): 200,
        ("C", "D"): 80,
    }
    best_cost, order = dp_join_order(tables, costs)
    print(f"표: {tables}")
    print(f"가장 나은 맞이음 차례: {' -> '.join(order)}")
    print(f"어림 비용: {best_cost}")
```

**출력:**

```
=== 고르기 몫 어림 ===
같음 (서로 다른 값 100개): 0.0100
맞이음 낱수 (1000 x 500, 서로 다른 값 50 대 80): 6250

=== 맞이음 차례 다듬기(갈피 다지기) ===
표: {'A': 1000, 'B': 5000, 'C': 200, 'D': 3000}
가장 나은 맞이음 차례: C -> D -> A -> B
어림 비용: 260
```

---

## 연습문제

**연습문제 1.**
관계형 데이터베이스에서 물음 다듬기의 큰 마디 셋, 곧 가름, 속뜻 다듬기, 몸 다듬기를 밝혀라.

??? success "연습문제 1 풀이"
    (1) **가름**: SQL 글월을 뜻 나무(AST)로 쪼개고 밑그림에 견주어 옳은지 살핀다(표 이름, 칸 갈래, 매인 조건). 그 열매가 관계 셈법 다룸꾼으로 된 속뜻 물음 나무다. (2) **속뜻 다듬기**: 같은 값 규칙으로 물음 나무를 바꾸어 어림 비용을 낮춘다. 으뜸 바꿈은 고르기를 맞이음 아래로 밀기(사이 크기를 줄인다), 맞이음 차례 바꾸기(차례마다 비용이 크게 다르다), 군더더기 추리기 없애기다. 다듬이는 뜻이 같은 꾀를 살피고 못한 것을 쳐낸다. (3) **몸 다듬기**: 속뜻 다룸꾼마다 몸으로 짜는 길을 매긴다(보기: 해시 맞이음 대 줄 세워 합치기 맞이음, 색인 훑기 대 차례로 훑기). 다듬이는 셈속(표 크기, 색인 고르기 몫, 잦기 그림 퍼짐)으로 어우름마다의 비용을 어림하고 가장 싼 꾀를 고른다. $\square$

---

**연습문제 2.**
물음 `SELECT * FROM orders WHERE customer_id = 42 AND total > 100`에는 가름이 둘이다. 다듬이가 거르기를 거는 차례를 어떻게 정하는지 밝혀라.

??? success "연습문제 2 풀이"
    다듬이는 가름마다 고르기 몫을 어림한다. `customer_id = 42`은 (손님이 1000명이면) 행의 0.1%쯤에 맞고, `total > 100`은 30%쯤에 맞을 수 있다. 더 잘 고르는 가름(customer_id = 42)을 먼저 걸면 사이 결과가 훨씬 세게 줄어든다. `customer_id`에 색인이 있으면 다듬이는 색인 훑기로 맞는 $\approx 0.1\%$의 행을 곧바로 집어 온 다음 `total > 100` 거르기를 건다. 색인이 없으면 차례로 훑는 동안 두 가름을 함께 건다. 어림 열매 크기는 $\lvert orders \rvert \times 0.001 \times 0.3$이다. 다듬이는 여러 칸에 걸친 셈속이나 잦기 그림이 달리 말하지 않는 한 가름끼리 서로 얽히지 않는다고 여기는데(손님의 씀씀이 결이 얽혀 있으면 어긋날 수 있다), 이것이 어긋남의 씨앗이 된다. $\square$

---

**연습문제 3.**
어떤 물음 꾀가 겹돌기 맞이음을 쓰는데 EXPLAIN을 보니 다듬이가 해시 맞이음을 골랐다. 어떤 몫이 이런 고름을 이끌었겠는가?

??? success "연습문제 3 풀이"
    다듬이는 다음을 바탕으로 해시 맞이음이 싸다고 어림했을 것이다. (1) 안쪽 표가 버퍼 못에 담기에는 너무 커서 겹돌기의 되풀이 훑기가 감당 못 할 만큼 비싸다. (2) 안쪽 표의 맞이음 칸에 색인이 없어 겹돌기의 더듬기마다 차례로 훑기가 든다. (3) 맞이음이 같음 맞이음이다(해시 맞이음은 같음 가름에만 듣는다). (4) 해시를 걸고 나면 작은 표가 기억에 담겨 표마다 한 번씩만 오가면 된다. 다듬이는 어림 들고남 비용을 견준다. 색인이 없으면 겹돌기가 $O(\lvert R \rvert \times \lvert S \rvert)$인데 해시 맞이음은 $O(\lvert R \rvert + \lvert S \rvert)$이다. $\lvert R \rvert = 10{,}000$쪽, $\lvert S \rvert = 5{,}000$쪽이면 겹돌기가 $50 \times 10^6$번, 해시 맞이음이 $15{,}000$번이다. $\square$

---

**연습문제 4.**
낱수 어림의 어긋남을 밝히고, 그것이 어떻게 다듬이로 하여금 못한 꾀를 고르게 하는지 밝혀라. 손에 잡히는 보기를 들어라.

??? success "연습문제 4 풀이"
    다듬이는 셈속(잦기 그림, 서로 다른 값의 수)으로 마디마다 행의 수(낱수)를 어림한다. 이 어림이 틀리면 비용 꼴이 잘못된 꾀를 고른다. 보기로 `orders`과 `products`을 `WHERE product.category = 'electronics'` 거르기와 함께 맞이음하는 물음을 보자. 다듬이는 (고루 퍼졌다는 여김으로) 전자 제품이 1000개라고 어림하고, 걸러 낸 제품으로 해시 표를 쌓는 해시 맞이음을 꾀한다. 그런데 실제로는 전자 제품이 100,000개다(퍼짐이 한쪽으로 쏠렸다). 해시 표가 기억을 넘어 원반으로 흘러넘치고, "싸다"던 해시 맞이음이 비싸진다. 큰 사이 열매를 잘 견디는 줄 세워 합치기 맞이음이 더 나은 꾀였을 것이다. 어긋남의 흔한 씨앗은 얽힌 칸을 서로 남남으로 여김, 낡은 셈속, 쏠린 퍼짐을 고른 것으로 여김, 그리고 여러 맞이음을 거치며 어긋남이 곱절로 쌓임이다. $\square$

---

**연습문제 5.**
요즘 물음 다듬이는 "맞춰 가는 물음 치름"을 쓴다. 이 길을 밝히고 낱수 어림의 어긋남을 어떻게 눅이는지 밝혀라.

??? success "연습문제 5 풀이"
    맞춰 가는 물음 치름은 치르는 동안 참 낱수를 지켜보며 꾀를 그때그때 고친다. 치르기 앞서 꾀 하나를 못 박는 대신, 다듬이가 참 행의 수와 어림을 견주는 "살핌 자리"를 끼워 넣는다. 참 수가 어림에서 크게 벌어지면(보기로 바란 것보다 열 곱절 많으면) 남은 몫을 다시 다듬는다. 보기로 Spark AQE에서는 섞기 마디 뒤에 어떤 나눔이 바란 것보다 백 곱절 크다는 것을 보고, 그 나눔에 한해 뿌리기 맞이음을 줄 세워 합치기 맞이음으로 바꾸거나 잔 나눔을 뭉쳐 군더더기를 줄인다. 꾀가 어림에만 기대지 않고 참에 맞추어 가므로 낱수 어긋남이 눅는다. 값으로 지켜봄과 다시 다듬음에 드는 품이 붙지만, 크게 못한 꾀를 끝까지 치르는 것에 견주면 흔히 작다. $\square$

## 정리하며

이 마당은 SQL에서 치름 꾀까지、관계 셈법의 같은 값 규칙、비용 어림하기、맞이음 차례 다듬기을 차례로 짚었다.

**살펴볼 거리**

- Selinger, P. G. et al. "Access Path Selection in a Relational Database Management System." *SIGMOD*, 1979
- [Database System Concepts (Silberschatz, Korth, Sudarshan)](https://www.db-book.com/)
- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)
