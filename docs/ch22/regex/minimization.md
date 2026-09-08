# 결정 유한 자동 기계 줄이기
결정 유한 자동 기계 줄이기는 결정 유한 자동 기계를 그와 같은 가장 작은 것으로 줄인다. 두 결정 유한 자동 기계가 같은 말을 받아들이면 서로 같다. 주어진 정규 말마다 줄인 결정 유한 자동 기계는 (상태 이름을 빼고) 하나뿐이다.

---

## 1. 홉크로프트 알고리즘

가장 효율 좋은 결정 유한 자동 기계 줄이기 알고리즘은 홉크로프트 알고리즘(1971)이며 $n$이 상태 수일 때 $O(n \log n)$ 시간에 돈다. 상태의 나눔을 되풀이해 잘게 다듬어 돈다.

### 핵심 생각

$\delta^*(p, w)$과 $\delta^*(q, w)$ 가운데 꼭 하나만 받아들이는 상태가 되는 글줄 $w$이 있으면 상태 $p$과 $q$은 **가를 수 있다**. 줄인 결정 유한 자동 기계는 가를 수 없는 상태를 모두 합친다.

### 알고리즘(무어 방법)

더 단순한 무어 방법은 상태를 되풀이해 나눈다:

1. 무리 둘로 시작한다. 곧 받아들이는 상태 $F$과 받아들이지 않는 상태 $Q \setminus F$이다.
2. 무리마다 그 안의 두 상태가 어떤 들임 글자에서 다른 무리로 옮아가면 무리를 쪼갠다.
3. 더 쪼갤 무리가 없을 때까지 되풀이한다.

$$
\text{상태 } p, q \text{ 가 같다} \iff \forall a \in \Sigma: \delta(p,a) \text{ 와 } \delta(q,a) \text{ 가 같은 무리에 있다}
$$

```python
def minimize_dfa(states, alphabet, transitions, start, accept_states):
    """나눔 잘게 다듬기로 결정 유한 자동 기계를 줄인다."""
    # 닿을 수 없는 상태를 없앤다
    reachable = set()
    queue = [start]
    reachable.add(start)
    while queue:
        s = queue.pop()
        for a in alphabet:
            t = transitions.get((s, a))
            if t is not None and t not in reachable:
                reachable.add(t)
                queue.append(t)
    states = states & reachable
    accept_states = accept_states & reachable

    # 첫 나눔
    non_accept = states - accept_states
    partition = []
    if accept_states:
        partition.append(frozenset(accept_states))
    if non_accept:
        partition.append(frozenset(non_accept))

    def state_to_group(s, part):
        for i, group in enumerate(part):
            if s in group:
                return i
        return -1

    changed = True
    while changed:
        changed = False
        new_partition = []
        for group in partition:
            splits = {}
            for s in group:
                signature = tuple(
                    state_to_group(transitions.get((s, a)), partition)
                    if transitions.get((s, a)) is not None else -1
                    for a in sorted(alphabet)
                )
                splits.setdefault(signature, set()).add(s)
            if len(splits) > 1:
                changed = True
            for sub in splits.values():
                new_partition.append(frozenset(sub))
        partition = new_partition

    # 줄인 결정 유한 자동 기계를 세운다
    group_rep = {}
    for group in partition:
        rep = min(group)
        for s in group:
            group_rep[s] = rep

    new_states = {group_rep[s] for s in states}
    new_trans = {}
    for (s, a), t in transitions.items():
        if s in states and t in states:
            new_trans[(group_rep[s], a)] = group_rep[t]
    new_start = group_rep[start]
    new_accept = {group_rep[s] for s in accept_states}

    return new_states, new_trans, new_start, new_accept

# 예
states = {0, 1, 2, 3, 4}
alphabet = {'a', 'b'}
transitions = {
    (0, 'a'): 1, (0, 'b'): 2,
    (1, 'a'): 1, (1, 'b'): 3,
    (2, 'a'): 1, (2, 'b'): 2,
    (3, 'a'): 1, (3, 'b'): 4,
    (4, 'a'): 1, (4, 'b'): 2,
}
start = 0
accept = {4}

new_states, new_trans, new_start, new_accept = minimize_dfa(
    states, alphabet, transitions, start, accept
)
print(f"Original states: {len(states)}")
print(f"Minimized states: {len(new_states)}")
# 상태 0과 2는 같다(둘 다 받아들이지 않고 몸가짐이 같다)
```

**출력:**

```
Original states: 5
Minimized states: 4
```

---

## 2. 복잡도

| 알고리즘 | 시간 | 공간 |
|-----------|------|-------|
| 무어 방법 | $O(n^2 \cdot |\Sigma|)$ | $O(n)$ |
| 홉크로프트 알고리즘 | $O(n \log n \cdot |\Sigma|)$ | $O(n)$ |

---

## 3. 마이힐-네로드 정리

결정 유한 자동 기계 줄이기의 이론 바탕은 **마이힐-네로드 정리**이다. 이는 말 $L$의 최소 결정 유한 자동 기계의 상태 수가 다음으로 정의되는 오른쪽 불변 동치 관계 $\equiv_L$의 동치류 수와 같다고 말한다:

$$
x \equiv_L y \iff \forall z \in \Sigma^*: (xz \in L \Leftrightarrow yz \in L)
$$

이로써 줄인 결정 유한 자동 기계가 하나뿐임이 보장된다.

# 참고 문헌

[Hopcroft - An n log n Algorithm for Minimizing States in a Finite Automaton (1971)](https://doi.org/10.1016/B978-0-12-417750-5.50022-1)

[Introduction to Automata Theory - Hopcroft, Motwani, Ullman, Chapter 4](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-automata-theory-languages-and-computation/P200000003517)

---

## 연습문제

**연습문제 1.**
결정 유한 자동 기계와 비결정 유한 자동 기계의 차이를 설명하라. 정규 표현에서 비결정 유한 자동 기계를 세우기가 왜 더 쉬운가?

??? success "연습문제 1 풀이"
    **결정 유한 자동 기계**는 (상태, 기호) 짝마다 옮아감이 꼭 하나인 반면 **비결정 유한 자동 기계**는 옮아감이 여럿(또는 없음)일 수 있고 들임을 쓰지 않는 엡실론 옮아감도 있다. 비결정 유한 자동 기계를 세우기 쉬운 까닭은 이렇다. (1) 톰프슨 세우기가 단순한 아우름 규칙(잇기, 고르기, 클레이니 별)으로 정규 표현 짜임에서 곧바로 비결정 유한 자동 기계를 낸다. (2) 엡실론 옮아감이 고를 수 있는 것과 되풀이되는 아래 표현을 자연스럽게 나타낸다. (3) 길이 $m$인 정규 표현에서 나온 비결정 유한 자동 기계의 상태가 $O(m)$개이다. 그와 같은 결정 유한 자동 기계는 상태가 $2^{O(m)}$개까지 될 수 있으나 실전에서는 흔히 훨씬 작다.

---

**연습문제 2.**
정규 표현 `a(b|c)*d`의 비결정 유한 자동 기계를 톰프슨 세우기로 만들어라.

??? success "연습문제 2 풀이"
    걸음마다 보자. (1) `a`의 자동 기계: 상태 $q_0 \xrightarrow{a} q_1$. (2) `b`의 자동 기계: $q_2 \xrightarrow{b} q_3$. (3) `c`의 자동 기계: $q_4 \xrightarrow{c} q_5$. (4) `b|c`의 자동 기계: $q_2$과 $q_4$으로 가는 $\epsilon$ 옮아감을 가진 새 시작 $q_6$, $q_3$과 $q_5$에서 오는 $\epsilon$ 옮아감을 가진 새 받아들임 $q_7$. (5) `(b|c)*`의 자동 기계: $q_6$과 새 받아들임 $q_9$으로 가는 $\epsilon$을 가진 새 시작 $q_8$, 그리고 $q_7$에서 $q_6$과 $q_9$으로 가는 $\epsilon$. (6) `d`의 자동 기계: $q_{10} \xrightarrow{d} q_{11}$. (7) 셋을 $\epsilon$ 옮아감으로 잇는다. 모두 상태 12개이다.

---

**연습문제 3.**
비결정 유한 자동 기계를 결정 유한 자동 기계로 바꾸는 부분 모임 세우기 알고리즘을 적어라. 최악의 경우 복잡도는 무엇인가?

??? success "연습문제 3 풀이"
    이 알고리즘은 비결정 유한 자동 기계 상태의 모임으로 결정 유한 자동 기계 상태를 세운다. (1) 시작 상태는 비결정 유한 자동 기계 시작 상태의 $\epsilon$ 닫힘이다. (2) 상태 $S$과 들임 기호 $a$마다 $\text{move}(S, a) = \{q' : q \in S, q \xrightarrow{a} q'\}$을 셈하고 그 $\epsilon$ 닫힘을 취해 다음 상태를 얻는다. (3) 비결정 유한 자동 기계의 받아들이는 상태를 하나라도 품으면 그 상태는 받아들인다. (4) 새 상태가 나오지 않을 때까지 되풀이한다. 최악의 경우: 상태 $n$개인 비결정 유한 자동 기계가 상태 $2^n$개인 결정 유한 자동 기계를 낼 수 있다(예컨대 정규 표현 `(a|b)*a(a|b)^{n-1}`). 실전에서는 대개 훨씬 작다.

---

**연습문제 4.**
결정 유한 자동 기계 줄이기란 무엇이며 왜 쓸모 있는가? 홉크로프트 알고리즘을 큰 틀에서 적어라.

??? success "연습문제 4 풀이"
    결정 유한 자동 기계 줄이기는 같은 상태(앞으로 올 모든 들임에 대해 몸가짐이 똑같은 상태)를 합쳐 같은 말을 알아보는 가장 작은 결정 유한 자동 기계를 낸다. **홉크로프트 알고리즘**: (1) 상태를 받아들이는 무리와 받아들이지 않는 무리로 나눈다. (2) 되풀이해 잘게 다듬는다. 곧 무리와 들임 기호마다 그 무리의 옮아감이 다른 무리로 가면 쪼갠다. (3) 더 쪼갤 무리가 없을 때까지 되풀이한다. 최소 결정 유한 자동 기계는 동치류마다 상태 하나를 갖는다. 시간: 상태 $n$개에 $O(n \log n)$. 줄이기는 어휘 분석기 짜기에서 기억 공간 줄이기, 정규 표현의 표준 견줌, 본 찾기 자동 기계 다듬기에 쓸모 있다.

## 정리하며

이 마당은 홉크로프트 알고리즘、복잡도、마이힐-네로드 정리을 차례로 짚었다.
