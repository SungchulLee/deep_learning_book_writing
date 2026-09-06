# 매트로이드

욕심쟁이 알고리즘은 어떤 문제(최소 뻗은 나무, 구간 일정 짜기)에서는 가장 좋게 돌아가지만 다른 문제(떠돌이 장사꾼, 일반 배낭)에서는 어그러진다. 어떤 짜임의 성질이 이 둘을 가르는가? **매트로이드**는 욕심쟁이 알고리즘이 가장 좋은 풀이를 찾음이 보장되는 때를 정확히 담아내는 추상 조합 짜임이다. 문제가 매트로이드 짜임을 지니면, 늘 쓸 수 있는 가장 좋은 원소를 취하는 단순한 전략이 전체 최적을 낸다.

## 엄밀한 정의

A **matroid** is a pair $M = (S, \mathcal{I})$ where $S$ is a finite ground set and $\mathcal{I} \subseteq 2^S$ is a family of subsets (called **independent sets**) satisfying three axioms:

**Axiom 1 (Non-emptiness).** $\emptyset \in \mathcal{I}$.

**Axiom 2 (Hereditary property).** If $B \in \mathcal{I}$ and $A \subseteq B$, then $A \in \mathcal{I}$. Every subset of an independent set is independent.

**Axiom 3 (Exchange property).** If $A, B \in \mathcal{I}$ and $|A| < |B|$, then there exists $x \in B \setminus A$ such that $A \cup \{x\} \in \mathcal{I}$.

맞바꿈 성질이 결정적인 공리이다. 벡터 공간의 모든 기저가 같은 차원을 갖듯, 이 성질은 **가장 큰** 서로 얽히지 않는 모음(**기저**라 한다)이 모두 같은 크기를 가짐을 보장한다.

## 말

- **Independent set.** A member of $\mathcal{I}$.
- **Dependent set.** A subset of $S$ not in $\mathcal{I}$.
- **회로.** 가장 작은 얽힌 모음(아무 원소나 없애면 얽히지 않게 된다).
- **기저.** 가장 큰 얽히지 않는 모음.
- **Rank.** The rank of a set $A \subseteq S$ is the size of the largest independent subset of $A$: $r(A) = \max\{|B| : B \subseteq A,\; B \in \mathcal{I}\}$.

## 보기

### 고른 매트로이드

$U_{k,n} = (S, \mathcal{I})$ where $|S| = n$ and $\mathcal{I} = \{A \subseteq S : |A| \le k\}$. Every subset of size at most $k$ is independent. The bases are all subsets of size exactly $k$.

### 선형(벡터) 매트로이드

Let $S$ be a set of vectors in $\mathbb{R}^d$. Define $\mathcal{I}$ as the collection of linearly independent subsets of $S$. The exchange property follows from the Steinitz exchange lemma in linear algebra.

### 그래프 매트로이드

Given a graph $G = (V, E)$, let $S = E$ and $\mathcal{I} = \{F \subseteq E : F \text{ is acyclic}\}$. The independent sets are forests, the bases are spanning trees, and the circuits are simple cycles. This matroid underlies the correctness of Kruskal's algorithm.

### 나눔 매트로이드

Let $S = S_1 \cup S_2 \cup \cdots \cup S_k$ be a partition. Given bounds $b_1, \dots, b_k$, define $\mathcal{I} = \{A \subseteq S : |A \cap S_i| \le b_i \text{ for all } i\}$.

## 핵심 성질

!!! note "모든 기저는 크기가 같다"
    어떤 매트로이드에서도 기저는 모두 크기가 같다. 이는 맞바꿈 성질에서 곧바로 따라 나온다. 곧 기저 $B_1$과 $B_2$의 크기가 다르다면 작은 쪽을 늘릴 수 있어 가장 큼에 어긋난다.

!!! note "매트로이드 쌍대성"
    Given matroid $M = (S, \mathcal{I})$, the **dual matroid** $M^* = (S, \mathcal{I}^*)$ where $B^*$ is a base of $M^*$ if and only if $S \setminus B^*$ is a base of $M$. The dual of a graphic matroid is called a **cographic matroid**.

## 확인

```python
"""
매트로이드 공리 확인.

주어진 모임의 집안이 매트로이드의 세 공리, 곧 비지 않음,
물려받는 성질, 맞바꿈 성질을 채우는지 살핀다.
"""

from itertools import combinations

# === 매트로이드 살피개 ===

def is_matroid(ground_set: set, independent: list[frozenset]) -> bool:
    """(바탕 모임, 홀로서기 모임)이 매트로이드를 이루는지 살핀다.

    인수:
        ground_set: 유한한 바탕 모임 S.
        independent: 홀로서기 모임의 목록(frozenset으로).

    반환값:
        세 매트로이드 공리를 채우면 참.
    """
    ind_set = set(independent)

    # 공리 1: 비지 않음
    if frozenset() not in ind_set:
        print("Fails Axiom 1: empty set not independent")
        return False

    # 공리 2: 물려받는 성질
    for s in independent:
        for size in range(len(s)):
            for subset in combinations(s, size):
                if frozenset(subset) not in ind_set:
                    print(f"Fails Axiom 2: {set(subset)} not independent")
                    return False

    # 공리 3: 맞바꿈 성질
    for a in independent:
        for b in independent:
            if len(a) < len(b):
                found = False
                for x in b - a:
                    if frozenset(a | {x}) in ind_set:
                        found = True
                        break
                if not found:
                    print(f"Fails Axiom 3: {set(a)}, {set(b)}")
                    return False

    return True


# === 시연 ===

if __name__ == "__main__":
    # 고른 매트로이드 U_{2,3}
    S = {1, 2, 3}
    I = [frozenset(), frozenset({1}), frozenset({2}), frozenset({3}),
         frozenset({1,2}), frozenset({1,3}), frozenset({2,3})]
    print(f"U(2,3) is matroid: {is_matroid(S, I)}")

    # 매트로이드가 아님: {1,2}와 {3,4}는 홀로서기이나 {1,3}은 아니다
    S2 = {1, 2, 3, 4}
    I2 = [frozenset(), frozenset({1}), frozenset({2}), frozenset({3}),
          frozenset({4}), frozenset({1,2}), frozenset({3,4})]
    print(f"Non-matroid check: {is_matroid(S2, I2)}")
```

**출력:**

```
U(2,3) is matroid: True
Fails Axiom 3: {1}, {3, 4}
Non-matroid check: False
```

The uniform matroid $U_{2,3}$ satisfies all three axioms. The second example fails the exchange property: $\{1\}$ and $\{3,4\}$ are independent with $|\{1\}| < |\{3,4\}|$, but neither $\{1,3\}$ nor $\{1,4\}$ is independent.

## 참고 문헌

- Whitney, H. (1935). On the abstract properties of linear dependence. *American Journal of Mathematics*, 57(3), 509--533.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 16장: Greedy Algorithms.
- Oxley, J. G. (2011). *Matroid Theory* (2nd ed.). Oxford University Press.

## 연습문제

**연습문제 1.**
매트로이드에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    The greedy choice selects the locally optimal option at each step. For Matroids, this choice satisfies the greedy choice property: there exists an optimal solution that includes this greedy selection. Combined with optimal substructure (the remaining subproblem after the greedy choice is also optimally solvable by the same strategy), the greedy algorithm produces a globally optimal solution. $\square$

---

**연습문제 2.**
매트로이드이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    Optimal substructure means that an optimal solution to the problem contains optimal solutions to its subproblems. For Matroids, after making the greedy choice, the remaining problem is a smaller instance of the same type. If the subproblem solution were not optimal, we could improve the overall solution by replacing it — contradicting overall optimality. Therefore optimal substructure holds. $\square$

---

**연습문제 3.**
매트로이드의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    The time complexity depends on the sorting step (if required) and the greedy selection loop. Sorting typically dominates at $O(n \log n)$. The greedy loop processes each element once in $O(n)$. Total: $O(n \log n)$. If the input is pre-sorted, the algorithm runs in $O(n)$. $\square$

---

**연습문제 4.**
(매트로이드에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    Consider an alternative greedy criterion that does not align with the problem's structure. This alternative may select an element that blocks better future choices. The counterexample demonstrates that the wrong greedy criterion can produce a suboptimal result, highlighting why the specific greedy choice property must be proven for each problem. $\square$
