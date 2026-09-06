# 비결정에서 결정 유한 자동 기계로
부분 모임 세우기(멱집합 세우기라고도 한다)는 비결정 유한 자동 기계를 그와 같은 결정 유한 자동 기계로 바꾼다. 상태 $n$개인 비결정 유한 자동 기계에서 나온 결정 유한 자동 기계의 상태는 많아야 $2^n$개이지만 실전에서는 대개 훨씬 작다.

## 알고리즘: 부분 모임 세우기

결정 유한 자동 기계의 상태마다 비결정 유한 자동 기계 상태의 모임에 맞닿는다. 시작 상태는 비결정 유한 자동 기계 시작 상태의 엡실론 닫힘이다. 상태 $S$과 들임 글자 $a$마다 옮아감은 $\bigcup_{s \in S} \delta(s, a)$의 엡실론 닫힘으로 간다.

$$
\text{DFA state } D = \{q_1, q_2, \ldots, q_k\} \subseteq Q_{\text{NFA}}
$$

$$
\delta_{\text{DFA}}(D, a) = \varepsilon\text{-closure}\!\left(\bigcup_{q \in D} \delta_{\text{NFA}}(q, a)\right)
$$

$D \cap F_{\text{NFA}} \neq \emptyset$이면 상태 $D$은 받아들인다.

```python
from collections import deque

def epsilon_closure(states, nfa_epsilon):
    closure = set(states)
    stack = list(states)
    while stack:
        s = stack.pop()
        for t in nfa_epsilon.get(s, []):
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return frozenset(closure)

def subset_construction(nfa_start, nfa_accept_states, nfa_transitions, nfa_epsilon, alphabet):
    """부분 모임 세우기로 비결정 유한 자동 기계를 결정 유한 자동 기계로 바꾼다."""
    start = epsilon_closure({nfa_start}, nfa_epsilon)
    dfa_states = {start}
    dfa_trans = {}
    dfa_accept = set()
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current & nfa_accept_states:
            dfa_accept.add(current)
        for a in alphabet:
            next_nfa = set()
            for s in current:
                next_nfa.update(nfa_transitions.get((s, a), set()))
            next_state = epsilon_closure(next_nfa, nfa_epsilon)
            if next_state:
                dfa_trans[(current, a)] = next_state
                if next_state not in dfa_states:
                    dfa_states.add(next_state)
                    queue.append(next_state)

    return start, dfa_accept, dfa_trans

# 정규 표현 (a|b)*abb의 보기 자동 기계
nfa_transitions = {
    (2, 'a'): {3}, (4, 'b'): {5},
    (7, 'a'): {8}, (8, 'b'): {9}, (9, 'b'): {10},
}
nfa_epsilon = {0: [1, 7], 1: [2, 4], 3: [6], 5: [6], 6: [1, 7]}
alphabet = {'a', 'b'}

start, accept, trans = subset_construction(0, {10}, nfa_transitions, nfa_epsilon, alphabet)
print(f"DFA start: {sorted(start)}")
print(f"Number of DFA states: {len({start} | {v for v in trans.values()})}")
```

## 복잡도 분석

- **시간:** $n$이 비결정 유한 자동 기계 상태 수일 때 최악의 경우 $O(2^n \cdot |\Sigma|)$.
- **공간:** 최악의 경우 결정 유한 자동 기계 상태가 $O(2^n)$개.
- **실전에서:** 닿을 수 있는 상태의 수는 대개 $2^n$보다 훨씬 작다.

## 최악의 경우 보기

말 $L_k = \{w \in \{a,b\}^* : \text{끝에서 } k\text{번째 기호가 } a\}$은 비결정 유한 자동 기계로는 상태가 $O(k)$개면 되지만 결정 유한 자동 기계로는 $2^k$개가 필요하다. 지수로 부풀어 오르는 것이 빡빡함을 보여 준다.

# 참고 문헌

[Introduction to Automata Theory - Hopcroft, Motwani, Ullman, Chapter 2](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-automata-theory-languages-and-computation/P200000003517)

[Subset Construction - Wikipedia](https://en.wikipedia.org/wiki/Powerset_construction)

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
