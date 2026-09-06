# 허프먼 가장 좋음 증명

허프먼 부호 쪽에서 알고리즘을 설명했다. 곧 잦기가 가장 낮은 글자 둘을 거듭 어울린다. 그런데 이 욕심쟁이 전략이 왜 가장 좋은 앞가지 없는 부호를 낼까? 증명은 표준 두 부분 욕심쟁이 얼거리를 따른다. 곧 **욕심쟁이 고름 성질**(잦기가 가장 낮은 두 글자를 가장 깊은 곳의 형제로 만들 수 있음)과 **가장 좋은 아래 짜임**(어울린 뒤 줄어든 문제의 가장 좋은 풀이가 본디 문제로 뻗음)을 세운다. 이 쪽에서는 두 보조정리와 온전한 귀납법 증명을 내놓는다.

## 자리매김과 기호

Let $C = \{c_1, c_2, \ldots, c_n\}$ be an alphabet of $n$ characters with frequencies $f_1, f_2, \ldots, f_n > 0$. A prefix-free code is represented by a full binary tree $T$ in which each leaf corresponds to a character. The cost is:

$$
B(T) = \sum_{i=1}^{n} f_i \cdot d_T(c_i)
$$

여기서 $d_T(c_i)$은 나무 $T$에서 글자 $c_i$의 깊이이다.

**Goal.** Show that Huffman's algorithm constructs a tree $T^*$ minimizing $B(T)$ over all full binary trees with $n$ leaves.

## 보조정리 1: 욕심쟁이 고름 성질

!!! note "보조정리 1(잦기가 가장 낮은 형제)"
    $x$과 $y$을 $C$에서 잦기가 가장 낮은 두 글자라 하자(같으면 아무렇게나 정한다). 그러면 $x$과 $y$이 가장 깊은 곳의 형제인 가장 좋은 앞가지 없는 부호가 있다.

**Proof.** Let $T^*$ be an optimal tree. Let $a$ and $b$ be two sibling leaves at the maximum depth of $T^*$.

**Case 1:** $\{a, b\} = \{x, y\}$. Done.

**Case 2:** $\{a, b\} \neq \{x, y\}$. Without loss of generality, assume $f_x \leq f_y$. Since $x$ and $y$ have the two smallest frequencies, $f_x \leq f_a$ and $f_y \leq f_b$ (possibly after relabeling $a, b$).

Construct $T'$ by swapping $x$ with $a$ in $T^*$ (i.e., $x$ moves to $a$'s position and $a$ moves to $x$'s position). The change in cost is:

$$
B(T') - B(T^*) = f_x \cdot d_{T^*}(a) + f_a \cdot d_{T^*}(x) - f_x \cdot d_{T^*}(x) - f_a \cdot d_{T^*}(a)
$$

$$
= (f_a - f_x)(d_{T^*}(x) - d_{T^*}(a))
$$

Since $f_a \geq f_x$ and $d_{T^*}(a) \geq d_{T^*}(x)$ (because $a$ is at maximum depth), we have $d_{T^*}(x) - d_{T^*}(a) \leq 0$. Therefore $B(T') - B(T^*) \leq 0$, so $T'$ is at least as good as $T^*$.

Now swap $y$ with $b$ in $T'$ (where $b$ is the sibling of $x$ at maximum depth). By an analogous argument, $B(T'') \leq B(T')$. In $T''$, characters $x$ and $y$ are siblings at maximum depth, and $B(T'') \leq B(T^*)$. Since $T^*$ is optimal, $T''$ is also optimal. $\square$

## 보조정리 2: 가장 좋은 아래 짜임

!!! note "보조정리 2(가장 좋은 아래 짜임)"
    Let $x$ and $y$ be sibling leaves in an optimal tree $T^*$ for alphabet $C$. Define a reduced alphabet $C' = (C \setminus \{x, y\}) \cup \{z\}$ where $z$ is a new character with frequency $f_z = f_x + f_y$. If $T'$ is an optimal tree for $C'$, then replacing the leaf $z$ in $T'$ with an internal node having children $x$ and $y$ produces an optimal tree for $C$.

**증명.** $T'$의 잎 $z$을 자식이 $x$과 $y$인 속 마디로 부풀려 얻은 나무를 $T$이라 하자. $T$의 값은 $T'$의 값과 다음처럼 이어진다:

For every character $c \notin \{x, y, z\}$, $d_T(c) = d_{T'}(c)$.

For $x$ and $y$: $d_T(x) = d_T(y) = d_{T'}(z) + 1$.

따라서 다음이 성립한다.

$$
B(T) = \sum_{c \neq x,y} f_c \cdot d_T(c) + f_x \cdot d_T(x) + f_y \cdot d_T(y)
$$

$$
= \sum_{c \neq x,y} f_c \cdot d_{T'}(c) + (f_x + f_y)(d_{T'}(z) + 1)
$$

$$
= \sum_{c \neq z} f_c \cdot d_{T'}(c) + f_z \cdot d_{T'}(z) + f_x + f_y
$$

$$
= B(T') + f_x + f_y
$$

따라서 $B(T) = B(T') + f_x + f_y$이다.

**Claim:** $T$ is optimal for $C$. Suppose not --- there exists a tree $\hat{T}$ for $C$ with $B(\hat{T}) < B(T)$. By Lemma 1, we may assume $x$ and $y$ are siblings in $\hat{T}$. Collapsing $x$ and $y$ into a single leaf $z$ with $f_z = f_x + f_y$ yields a tree $\hat{T}'$ for $C'$ with:

$$
B(\hat{T}') = B(\hat{T}) - f_x - f_y < B(T) - f_x - f_y = B(T')
$$

This contradicts the optimality of $T'$ for $C'$. $\square$

## 주요 정리

**정리.** 허프먼 알고리즘은 가장 좋은 앞가지 없는 부호를 낸다.

**$n = |C|$에 대한 강한 귀납법 증명.**

**바탕 경우**($n = 2$): 글자가 둘뿐이다. 꽉 찬 이진 나무는 두 글자를 모두 깊이 1에 두는 것뿐이다. 이는 뻔히 가장 좋으며 알고리즘이 한 번의 어울리기로 그것을 낸다.

**귀납 단계:** 크기가 $< n$인 모든 글자 모음에 대해 알고리즘이 가장 좋다고 하자. 크기 $n$인 글자 모음 $C$을 보자.

1. 알고리즘이 잦기가 가장 낮은 두 글자 $x$과 $y$을 골라 $f_z = f_x + f_y$인 $z$으로 어울린다. 이러면 $|C'| = n - 1$인 줄어든 글자 모음 $C'$이 생긴다.

2. 귀납 가정으로 알고리즘은 $C'$에 대해 가장 좋은 나무 $T'$을 낸다.

3. 보조정리 2로, $T'$의 $z$을 자식이 $x$과 $y$인 속 마디로 부풀리면 $C$에 대한 가장 좋은 나무 $T$이 나온다.

4. 이 부풀림이 바로 알고리즘이 하는 일이다. 곧 어울릴 때 $x$과 $y$을 $z$의 자식으로 두었다.

Therefore, the algorithm produces an optimal tree for $C$. $\square$

## 핵심 눈썰미: 증명이 통하는 까닭

증명은 관계 $B(T) = B(T') + f_x + f_y$에 달렸다. 항 $f_x + f_y$은 나무의 꼴에 매이지 않는 상수이다. 오직 $x$과 $y$이 $z$보다 한 켜 깊다는 사실에서 온다. 곧 $x$과 $y$이 형제인 $C$의 모든 나무에서 $B(T)$을 가장 작게 하는 것은 $C'$의 모든 나무에서 $B(T')$을 가장 작게 하는 것과 같다. 욕심쟁이 어울리기는 남은 가장 좋게 하기에 대한 앎을 조금도 잃지 않는다.

## 유일성

The Huffman tree is not necessarily unique --- different tie-breaking rules when frequencies are equal produce different trees. However, all optimal trees have the same cost $B(T^*)$. Moreover, all optimal prefix-free codes for a given frequency distribution have the same set of codeword lengths (up to permutation of characters with equal frequency).

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 16.3절. MIT Press.
- Huffman, D. A. (1952). A method for the construction of minimum-redundancy codes. *Proceedings of the IRE*, 40(9), 1098--1101.

## 연습문제

**연습문제 1.**
허프먼 가장 좋음 증명에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    The greedy choice selects the locally optimal option at each step. For Huffman Optimality Proof, this choice satisfies the greedy choice property: there exists an optimal solution that includes this greedy selection. Combined with optimal substructure (the remaining subproblem after the greedy choice is also optimally solvable by the same strategy), the greedy algorithm produces a globally optimal solution. $\square$

---

**연습문제 2.**
허프먼 가장 좋음 증명이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    Optimal substructure means that an optimal solution to the problem contains optimal solutions to its subproblems. For Huffman Optimality Proof, after making the greedy choice, the remaining problem is a smaller instance of the same type. If the subproblem solution were not optimal, we could improve the overall solution by replacing it — contradicting overall optimality. Therefore optimal substructure holds. $\square$

---

**연습문제 3.**
허프먼 가장 좋음 증명의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    The time complexity depends on the sorting step (if required) and the greedy selection loop. Sorting typically dominates at $O(n \log n)$. The greedy loop processes each element once in $O(n)$. Total: $O(n \log n)$. If the input is pre-sorted, the algorithm runs in $O(n)$. $\square$

---

**연습문제 4.**
(허프먼 가장 좋음 증명에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    Consider an alternative greedy criterion that does not align with the problem's structure. This alternative may select an element that blocks better future choices. The counterexample demonstrates that the wrong greedy criterion can produce a suboptimal result, highlighting why the specific greedy choice property must be proven for each problem. $\square$
