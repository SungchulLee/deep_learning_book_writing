# 표현 방식의 견줌

알맞은 그래프 표현을 고르는 일은 그래프 알고리즘을 짤 때 가장 먼저 하는, 그리고 가장 크게 좌우하는 결정 가운데 하나이다. 이웃 행렬, 이웃 목록, 변 목록은 저마다 공간 씀씀이와 연산 속도에서 다른 주고받음을 준다. 가장 좋은 고름은 그래프의 빽빽함, 알고리즘이 가장 자주 하는 연산, 그리고 그래프가 붙박이인지 움직이는지에 달렸다.

---

## 1. 성긴 그래프와 빽빽한 그래프

성김과 빽빽함의 갈림이 표현 고르기를 이끈다.

꼭짓점이 $|V|$개인 그래프는 변을 많아야 $O(|V|^2)$개 가질 수 있다. $|E| = O(|V|)$이거나 더 일반으로 $|E| \ll |V|^2$이면 그래프가 **성기다**고 한다. $|E| = \Theta(|V|^2)$이면 **빽빽하다**고 한다.

실제 세상의 그래프는 대부분 성기다. 사회 망, 도로 지도, 생물 망에서는 보통 꼭짓점마다 다른 모든 꼭짓점 가운데 아주 적은 몫에만 이어져 있다.

---

## 2. 연산 복잡도 비교

다음 표는 흔한 연산에 걸쳐 표준 표현 셋을 견준다.

| 연산 | 이웃 행렬 | 이웃 목록 | 변 목록 |
|---|---|---|---|
| **공간** | $O(V^2)$ | $O(V + E)$ | $O(E)$ |
| **변 $(u,v)$ 확인** | $O(1)$ | $O(\deg(u))$ | $O(E)$ |
| **$u$의 이웃 훑기** | $O(V)$ | $O(\deg(u))$ | $O(E)$ |
| **모든 변 훑기** | $O(V^2)$ | $O(V + E)$ | $O(E)$ |
| **변 더하기** | $O(1)$ | $O(1)$ | $O(1)$ |
| **변 지우기** | $O(1)$ | $O(\deg(u))$ | $O(E)$ |
| **꼭짓점 더하기** | $O(V)$(크기 바꾸기) | $O(1)$ | $O(1)$ |
| **빽빽한 그래프에 맞나** | 잘 맞음 | 헤픔 | 헤픔 |

---

## 3. 자세한 분석

### 이웃 행렬

[이웃 행렬](matrix.md)은 변 $(i,j)$이 있으면 $A[i][j] = 1$(또는 변의 무게)인 $|V| \times |V|$ 행렬 $A$을 저장한다.

**좋은 점:**

- 변이 있는지 묻는 데 $O(1)$ — $A[i][j]$을 되풀이해 살피는 플로이드-워셜 같은 알고리즘에 아주 중요하다.
- 2차원 배열로 구현이 단순하다.
- 행렬 연산(곱하기, 이행 닫힘)을 곧바로 쓸 수 있다.

**나쁜 점:**

- 변의 개수와 상관없이 공간이 $O(V^2)$이라 성긴 그래프에는 헤프다.
- 꼭짓점의 이웃이 적어도 이웃을 훑는 데 늘 $O(V)$이 든다.
- 꼭짓점을 더하려면 행렬 전체의 크기를 바꿔야 한다.

**가장 알맞은 곳:** 빽빽한 그래프, 행렬 연산을 쓰는 알고리즘, $O(V^2)$ 공간을 감당할 수 있는 작은 그래프.

### 이웃 목록

[이웃 목록](list.md)은 꼭짓점 $u$마다 $u$에 이웃한 꼭짓점의 목록을 저장한다(변의 무게를 함께 둘 수도 있다).

**좋은 점:**

- 공간이 $O(V + E)$으로 실제 그래프 크기에 비례한다.
- $u$의 이웃을 훑는 데 $O(\deg(u))$이 들며 이것이 가장 좋다.
- 그래프 돌아보기 알고리즘(BFS, DFS) 대부분이 이웃 목록을 자연스럽게 훑으므로 이것이 기본 고름이 된다.

**나쁜 점:**

- 변이 있는지 묻는 데 최악의 경우 $O(\deg(u))$이 든다(목록 대신 해시 집합을 쓰면 $O(1)$으로 나아질 수 있다).
- 행렬보다 구현이 살짝 더 복잡하다.

**가장 알맞은 곳:** 성긴 그래프(가장 흔한 경우), BFS/DFS 기반 알고리즘, 움직이는 그래프.

### 변 목록

[변 목록](edge_list.md)은 변을 튜플 $(u, v)$이나 $(u, v, w)$의 납작한 목록으로 저장한다.

**좋은 점:**

- 공간이 $O(E)$으로, $E \ll V$일 때 가장 간결한 표현이다.
- 모든 변을 훑기 쉬워 크러스컬 알고리즘을 비롯한 변 다루기 알고리즘에 자연스럽다.
- 변을 무게로 정렬하기 쉽다.

**나쁜 점:**

- 이웃을 묻거나 변이 있는지 살피려면 $O(E)$번 훑어야 한다.
- 이웃함을 되풀이해 묻는 알고리즘에는 알맞지 않다.

**가장 알맞은 곳:** 변 중심 알고리즘(크러스컬, 벨먼-포드), 입출력 꼴, 아주 성긴 그래프.

---

## 4. 고르기 길잡이

```python
"""
그래프 표현 고르기 길잡이.

그래프의 빽빽함과 주로 필요한 연산에 따라 어느 표현을
고를지 보인다.
"""

# === 표현 고르기 ===

def recommend_representation(n_vertices, n_edges, primary_ops):
    """
    그래프의 성질과 필요한 연산에 따라
    그래프 표현을 추천한다.

    매개변수:
        n_vertices: 꼭짓점의 개수
        n_edges: 변의 개수
        primary_ops: 주로 필요한 연산의 목록
            ('edge_query', 'neighbor_iter', 'all_edges', 'matrix_ops')
    """
    density = n_edges / max(1, n_vertices * (n_vertices - 1) / 2)
    recommendations = []

    if 'matrix_ops' in primary_ops:
        recommendations.append(("Adjacency Matrix",
                                "matrix operations required"))
    elif 'edge_query' in primary_ops and density > 0.5:
        recommendations.append(("Adjacency Matrix",
                                f"dense graph ({density:.1%}), "
                                f"O(1) edge queries"))
    elif 'all_edges' in primary_ops and 'neighbor_iter' not in primary_ops:
        recommendations.append(("Edge List",
                                "only need to iterate all edges"))
    else:
        recommendations.append(("Adjacency List",
                                f"sparse graph ({density:.1%}), "
                                f"efficient neighbor iteration"))

    return recommendations

# === 공간 견줌 ===

def compare_space(n_vertices, n_edges):
    """표현 셋의 공간 씀씀이를 견준다."""
    matrix_space = n_vertices ** 2
    adj_list_space = n_vertices + 2 * n_edges  # 무방향
    edge_list_space = 2 * n_edges

    return {
        "Adjacency Matrix": matrix_space,
        "Adjacency List": adj_list_space,
        "Edge List": edge_list_space,
    }

# === 메인 ===

if __name__ == "__main__":
    # 성긴 그래프: 꼭짓점 1000개, 변 3000개
    print("=== Sparse Graph (V=1000, E=3000) ===")
    space = compare_space(1000, 3000)
    for name, s in space.items():
        print(f"  {name}: {s:,} entries")
    recs = recommend_representation(1000, 3000, ['neighbor_iter'])
    print(f"  Recommendation: {recs[0][0]} ({recs[0][1]})")

    # 빽빽한 그래프: 꼭짓점 100개, 변 4000개
    print("\n=== Dense Graph (V=100, E=4000) ===")
    space = compare_space(100, 4000)
    for name, s in space.items():
        print(f"  {name}: {s:,} entries")
    recs = recommend_representation(100, 4000, ['edge_query'])
    print(f"  Recommendation: {recs[0][0]} ({recs[0][1]})")

    # 변 중심: 크러스컬의 MST
    print("\n=== Edge-Centric (Kruskal's MST) ===")
    recs = recommend_representation(1000, 5000, ['all_edges'])
    print(f"  Recommendation: {recs[0][0]} ({recs[0][1]})")
```

**출력:**
```
=== Sparse Graph (V=1000, E=3000) ===
  Adjacency Matrix: 1,000,000 entries
  Adjacency List: 7,000 entries
  Edge List: 6,000 entries
  Recommendation: Adjacency List (sparse graph (0.6%), efficient neighbor iteration)
=== Dense Graph (V=100, E=4000) ===
  Adjacency Matrix: 10,000 entries
  Adjacency List: 8,100 entries
  Edge List: 8,000 entries
  Recommendation: Adjacency Matrix (dense graph (80.8%), O(1) edge queries)
=== Edge-Centric (Kruskal's MST) ===
  Recommendation: Edge List (only need to iterate all edges)
```

---

## 5. 혼합형 접근

실전에서는 여러 표현의 좋은 점을 합친 섞음 전략을 쓰기도 한다:

- **해시 집합을 쓴 이웃 목록.** 꼭짓점마다의 이웃 목록을 해시 집합으로 바꾸면 $O(V + E)$ 공간을 지키면서 변이 있는지 $O(1)$에 물을 수 있다.
- **눌린 성긴 행(CSR).** 이웃 목록을 이어 붙은 배열에 저장해 캐시에 상냥하게 이웃을 훑는다. 성능이 중요한 그래프 라이브러리에서 흔하다.
- **속뜻 표현.** 규칙으로 정해지는 그래프(격자, 놀이 상태)에서는 저장하지 않고 이웃을 그때그때 셈한다. [속뜻 그래프](implicit.md)를 보아라.

---

## 연습문제

**연습문제 1.**
사회 망에 이용자가 백만 명 있고 이용자마다 벗이 평균 200명이다. 이웃 행렬과 이웃 목록의 기억 공간 씀씀이를 견주어라. 어느 표현이 알맞은가?

??? success "연습문제 1 풀이"
    $V = 10^6$이고 $E \approx 10^6 \times 200 / 2 = 10^8$이라고 하자(벗 관계를 한 번씩만 센다). 이웃 행렬은 항목 $V^2 = 10^{12}$개가 필요하며 항목마다 1바이트면 1 TB쯤이다. 이웃 목록은 항목 $O(V + 2E) = O(10^6 + 2 \times 10^8) \approx 2 \times 10^8$개, 곧 가리개나 정수 2억 개쯤으로 몇 GB이다. 이 그래프는 몹시 성기므로($E \ll V^2$) 이웃 목록이 뚜렷이 낫다. $\square$

---

**연습문제 2.**
아래 연산마다 어느 표현(이웃 행렬, 이웃 목록, 변 목록)이 최악의 경우 시간 복잡도가 가장 좋은지 밝혀라. (a) 변 $(u, v)$이 있는지 살피기, (b) $v$의 모든 이웃 훑기, (c) 그래프의 모든 변 훑기.

??? success "연습문제 2 풀이"
    (a) **이웃 행렬**: $M[u][v]$을 $O(1)$에 찾는다. 이웃 목록은 $O(\deg(v))$번 훑어야 하고, 변 목록은 $O(E)$번 훑어야 한다.

    (b) **이웃 목록**: 이웃 목록을 훑어 $O(\deg(v))$이다. 이웃 행렬은 $O(V)$이 들고(행 전체를 훑는다), 변 목록은 $O(E)$이 든다.

    (c) **변 목록**: 모든 변이 차례로 저장되어 있어 $O(E)$이다. 이웃 목록도 모든 목록을 훑어 $O(V + E)$을 이룬다. 이웃 행렬은 $O(V^2)$이 든다. $\square$

---

**연습문제 3.**
움직이는 그래프에서 변을 되풀이해 더하고 지워야 한다. 이웃 행렬을 쓰는 것과 이웃 저장에 해시 집합을 쓴 이웃 목록을 쓰는 것의 주고받음을 이야기하여라.

??? success "연습문제 3 풀이"
    **이웃 행렬**: 변을 더하거나 지우는 데 $O(1)$이다. $M[u][v]$을 $1$이나 $0$으로 놓으면 된다. 빽빽함과 상관없이 공간은 $O(V^2)$이다. 변이 있는지 살피는 것도 $O(1)$이다.

    **해시 집합을 쓴 이웃 목록**: 변을 더하는 데 고르게 나눠 $O(1)$이다(해시 집합에 넣기). 변을 지우는 것도 고르게 나눠 $O(1)$이다. 변이 있는지 살피는 것은 기대 $O(1)$이다. 공간은 $O(V + E)$으로 성긴 그래프에 훨씬 낫다.

    성기고 움직이는 그래프에는 해시 집합 이웃 목록이 낫다. 빽빽한 그래프이거나 최악의 경우에도 $O(1)$이 보장되어야 한다면 이웃 행렬이 더 단순하다. $\square$

---

**연습문제 4.**
변의 개수와 상관없이 이웃 행렬을 이웃 목록으로 바꾸는 데 $\Theta(V^2)$ 시간이 듦을 증명하여라.

??? success "연습문제 4 풀이"
    이웃 행렬에는 항목이 $V^2$개 있다(무방향이면 대각선을 뺀 $V(V-1)/2$개). 이웃 목록을 쌓으려면 어떤 변이 있는지 알려고 항목을 모두 살펴야 한다. 그래프에 변이 0개여도 변이 없음을 확인하려면 항목 $V^2$개를 모두 들여다봐야 한다. 그러므로 바꾸는 데 $\Theta(V^2)$ 시간이 든다. 거꾸로 이웃 목록을 행렬로 바꾸는 데도 $V \times V$ 행렬을 첫걸음 잡아야 하므로 $\Theta(V^2)$ 시간이 든다. $\square$

## 정리하며

이 마당은 성긴 그래프와 빽빽한 그래프、연산 복잡도 비교、자세한 분석、고르기 길잡이을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 22장.
