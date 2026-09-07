# 해밀턴 경로와 순환

오일러 회로가 **변**마다 정확히 한 번씩 들르는 데 견주어, 해밀턴 순환은 **꼭짓점**마다 정확히 한 번씩 들른다. 문제 진술의 이 사소해 보이는 바뀜이 엄청난 결과를 낳는다. 곧 그래프에 해밀턴 순환이 있는지 정하는 것은 NP-완전이며, 이는 효율적인 일반 알고리즘이 알려져 있지 않다는 뜻이다. 그럼에도 있음을 보장하는 충분조건이 여럿 있고, 실전에서 쓰이는 되돌아가기 알고리즘이 알맞은 크기의 보기를 푼다.

## 정의

**해밀턴 경로.** 꼭짓점마다 정확히 한 번씩 들르는 $G = (V, E)$의 단순 경로.

**해밀턴 돌이.** 꼭짓점마다 꼭 한 번씩 들르고 처음 꼭짓점으로 돌아오는 단순 돌이. 같은 말로, $(w, v) \in E$일 때 $v$에서 $w$으로 가는 해밀턴 길이다.

**해밀턴 그래프.** 해밀턴 순환이 있는 그래프.

## 오일러와의 견줌

| 성질 | 오일러 | 해밀턴 |
|----------|:--------:|:-----------:|
| 무엇마다 들르나 | 변 | 꼭짓점 |
| 있는지 살피기 | 차수 홀짝으로 $O(V + E)$ | NP-완전 |
| 효율적인 특징지음 | 있다(차수 정리) | 알려진 특징지음 없음 |

## 충분조건

해밀턴성에 대한 간단한 필요충분조건은 알려져 있지 않지만, 고전적인 정리 몇이 충분조건을 준다.

!!! note "디랙 정리(1952)"
    $G$이 꼭짓점 $n \ge 3$개인 단순 그래프이고 꼭짓점마다 $\deg(v) \ge n/2$을 채우면 $G$은 해밀턴 그래프다.

!!! note "오레 정리(1960)"
    $G$이 꼭짓점 $n \ge 3$개인 단순 그래프이고 이웃하지 않은 꼭짓점 짝 $u, v$마다 $\deg(u) + \deg(v) \ge n$이면 $G$은 해밀턴 그래프다.

오레 정리는 디랙 정리를 넓힌 것이다. 꼭짓점마다 자릿수가 적어도 $n/2$이면 어떤 꼭짓점 짝도 $\deg(u) + \deg(v) \ge n$을 채운다.

## 복잡도

해밀턴 순환 문제는 카프가 처음 내놓은 21가지 NP-완전 문제(1972) 가운데 하나이다. 다음으로 제한해도 여전히 NP-완전이다:

- 최대 차수가 3인 평면 그래프.
- 두 쪽 그래프.
- 격자 그래프.

이제까지 알려진 가장 좋은 딱 맞는 알고리즘은 부분 묶음에 대한 갈피 다지기(헬드-카프 알고리즘)로 $O^*(2^n)$ 때에 돌며, 손쉬운 $O(n!)$ 길보다 낫다.

## 되돌아가기 알고리즘

되돌아가기 방식은 꼭짓점을 하나씩 붙여 경로를 세우며, 올바른 해밀턴 경로나 순환으로 이어질 수 없는 가지를 쳐 낸다. 가장 나쁜 경우 지수 시간이지만 작거나 알맞은 크기의 그래프에는 쓸 만하다.

```python
"""
되돌아가기로 해밀턴 순환 찾기.

꼭짓점을 하나씩 붙여 경로를 세우고 올바르지 않은 뻗음을 쳐 내며
해밀턴 순환이 있는지 살핀다.
"""

# === 되돌아가기 해밀턴 순환 ===

def hamiltonian_cycle(n: int, edges: list[tuple[int, int]]) -> list[int] | None:
    """있으면 해밀턴 순환을 찾는다.

    인수:
        n: 꼭짓점의 개수(0부터 셈).
        edges: 방향 없는 변의 목록.

    반환값:
        순환을 이루는 꼭짓점의 목록. 순환이 없으면 None.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    path = [0]
    visited = {0}

    def backtrack() -> bool:
        if len(path) == n:
            # 시작점으로 돌아올 수 있는지 살피기
            return 0 in adj[path[-1]]

        last = path[-1]
        for neighbor in sorted(adj[last]):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                if backtrack():
                    return True
                path.pop()
                visited.remove(neighbor)
        return False

    if backtrack():
        return path + [path[0]]
    return None


# === 시연 ===

if __name__ == "__main__":
    # 완전 그래프 K4
    k4_edges = [(i, j) for i in range(4) for j in range(i+1, 4)]
    result = hamiltonian_cycle(4, k4_edges)
    print(f"K4 Hamiltonian cycle: {result}")

    # 경로 그래프: 0-1-2-3(해밀턴 순환 없음)
    path_edges = [(0,1),(1,2),(2,3)]
    result = hamiltonian_cycle(4, path_edges)
    print(f"Path graph cycle: {result}")

    # 피터슨 그래프(해밀턴 경로는 있으나 순환은 없다)
    petersen = [
        (0,1),(1,2),(2,3),(3,4),(4,0),  # 바깥 순환
        (0,5),(1,6),(2,7),(3,8),(4,9),  # 바큇살
        (5,7),(7,9),(9,6),(6,8),(8,5),  # 안쪽 오각별
    ]
    result = hamiltonian_cycle(10, petersen)
    print(f"Petersen graph cycle: {result}")
```

**출력:**

```
K4 Hamiltonian cycle: [0, 1, 2, 3, 0]
Path graph cycle: None
Petersen graph cycle: None
```

온 그래프 $K_4$은 디랙 조건을 채우고($\deg(v) = 3 \ge 4/2$) 참으로 해밀턴 돌이를 지닌다. 꼭짓점 4개짜리 길 그래프에는 돌이가 아예 없다. 페테르센 그래프는 해밀턴 길은 있으나 해밀턴 돌이는 없는 것으로 이름났다.

## 동적 계획 방식

헬드-카프 알고리즘은 비트 가림 갈피 다지기로 해밀턴 길을 $O(n^2 \cdot 2^n)$ 때와 $O(n \cdot 2^n)$ 자리에 찾는다. 다음과 같이 매긴다.

$$
\text{dp}[S][v] = \text{True if there is a path visiting exactly the vertices in } S \text{ and ending at } v
$$

점화식은 다음과 같다.

$$
\text{dp}[S][v] = \bigvee_{u \in S \setminus \{v\},\; (u,v) \in E} \text{dp}[S \setminus \{v\}][u]
$$

비롯하는 꼭짓점과 이웃한 어떤 $v$에 대해 $\text{dp}[\{0, 1, \dots, n{-}1\}][v] = \text{True}$이면 해밀턴 돌이가 있다.

## 참고 문헌

- Karp, R. M. (1972). Reducibility among combinatorial problems. In *Complexity of Computer Computations*, pp. 85--103.
- Ore, O. (1960). Note on Hamilton circuits. *The American Mathematical Monthly*, 67(1), 55.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 34장: NP-Completeness.

## 연습문제

**연습문제 1.**
오일러 회로를 살피는 것은 다항 시간인데 해밀턴 순환이 있는지 정하는 것은 왜 NP-완전인지 설명하여라.

??? success "연습문제 1 풀이"
    오일러 돌이에는 단순한 가름 조건(자릿수가 모두 짝수 + 이어져 있음)이 있어 $O(V + E)$에 살펴볼 수 있다. 해밀턴 돌이에는 그런 단순한 가름 조건이 없다. 해밀턴 돌이 문제는 NP 완전이다(3-SAT이나 꼭짓점 덮개로 되돌려 밝혔다). 곧 다항식 때 알고리즘이 알려져 있지 않고, 알려진 길은 모두 가장 나쁠 때 지수 때가 든다($O(n! \cdot n)$ 막무가내, $O(2^n \cdot n^2)$ 갈피 다지기 따위). $\square$

---

**연습문제 2.**
디랙 정리를 말하고 그것으로 $K_6$에 해밀턴 순환이 있는지 정하여라.

??? success "연습문제 2 풀이"
    **디랙 정리**: $G$이 꼭짓점 $n \geq 3$개인 단순 그래프이고 꼭짓점마다 자릿수가 $\geq n/2$이면 $G$에는 해밀턴 돌이가 있다. $K_6$에서는 꼭짓점마다 자릿수가 5다. $5 \geq 6/2 = 3$이므로 디랙 조건이 채워지고 $K_6$에는 해밀턴 돌이가 있다. (참으로 $K_n$에는 모든 $n \geq 3$에 대해 해밀턴 돌이가 있다.) $\square$

---

**연습문제 3.**
해밀턴 돌이 문제를 푸는 $O(2^n \cdot n^2)$ 갈피 다지기 알고리즘을 밝혀라.

??? success "연습문제 3 풀이"
    비트 가림 갈피 다지기를 쓴다. 묶음 $S$의 꼭짓점만 꼭 들르고 꼭짓점 $v$에서 끝나는 길이 있으면 $dp[S][v]$ = True이라 하자. 밑 자리: $dp[\{0\}][0] = \text{True}$(꼭짓점 0에서 비롯한다). 넘어감: 어떤 $u \in S$에 대해 $dp[S][u] = \text{True}$이고 $(u, v)$이 이음이면 $dp[S \cup \{v\}][v] = \text{True}$이다. 답: 어떤 $v$에 대해 $dp[\{0,\ldots,n-1\}][v] \land (v, 0) \in E$이다. 부분 묶음이 $2^n$개, $v$의 고름이 $n$가지, $u$의 고름이 $n$가지이므로 $O(2^n \cdot n^2)$이다. $\square$

---

**연습문제 4.**
떠돌이 장사꾼 문제(TSP)는 무게가 가장 작은 해밀턴 순환을 묻는다. 이것은 해밀턴 순환 문제와 어떻게 이어지는가?

??? success "연습문제 4 풀이"
    떠돌이 장사꾼 문제는 가장 좋게 하기 갈래다. 곧 짐의 합이 가장 작은 해밀턴 돌이를 찾는다. 이 문제의 판단 갈래("짐이 $\leq k$인 해밀턴 돌이가 있는가?")는 NP 완전이다. 떠돌이 장사꾼 문제를 다항식 때에 풀 수 있다면, 있는 이음의 짐을 1, 없는 이음의 짐을 $\infty$으로 둔 그래프에서 가장 작은 순회의 짐이 마디 있는지 살펴 해밀턴 돌이 문제를 풀 수 있다. 그러므로 떠돌이 장사꾼 문제는 적어도 해밀턴 돌이 문제만큼 어렵다. $\square$
