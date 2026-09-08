# 비결정 유한 자동 기계 세우기
비결정 유한 자동 기계(NFA)는 정규 표현 찾기의 바탕 모델이다. 유한한 상태 모임, 글자 모임, 상태와 기호를 상태 여럿에 옮길 수 있는(엡실론이나 빈 옮아감도 포함) 옮아감 함수, 시작 상태, 받아들이는 상태 모임으로 이루어진다.

---

## 1. 엄밀한 정의

비결정 유한 자동 기계는 5쌍 $(Q, \Sigma, \delta, q_0, F)$이며 여기서:

$$
\begin{aligned}
Q &: \text{유한한 상태 모임}\\
\Sigma &: \text{들임 글자 모임}\\
\delta &: Q \times (\Sigma \cup \{\varepsilon\}) \to \mathcal{P}(Q) \quad \text{(옮아감 함수)}\\
q_0 &\in Q : \text{시작 상태}\\
F &\subseteq Q : \text{받아들이는 상태 모임}
\end{aligned}
$$

비결정 유한 자동 기계가 글줄 $w$을 받아들이는 것은 (엡실론 옮아감을 아무렇게나 섞어) $w$을 이루며 $q_0$에서 $F$의 어떤 상태로 가는 길이 하나 이상 있을 때이다.

---

## 2. 정규 표현에서 비결정 유한 자동 기계로

정규 표현 $r$은 모두 아우르는(귀납) 방식으로 그와 같은 비결정 유한 자동 기계로 바꿀 수 있다. 기본 연산 셋은 다음과 같다:

1. **글자 하나 $a$:** 상태 둘 $s_0 \xrightarrow{a} s_1$을 만든다.
2. **잇기 $r_1 r_2$:** $N(r_1)$의 받아들이는 상태를 $\varepsilon$ 옮아감으로 $N(r_2)$의 시작 상태에 잇는다.
3. **합치기 $r_1 | r_2$:** 두 자동 기계로 가는 $\varepsilon$ 옮아감을 가진 새 시작 상태를 만들고, 두 받아들이는 상태를 $\varepsilon$으로 새 받아들이는 상태에 잇는다.
4. **클레이니 별 $r^*$:** 되풀이와 건너뛰기를 위한 $\varepsilon$ 옮아감을 더한다.

```python
class State:
    _id_counter = 0
    def __init__(self, is_accept=False):
        self.id = State._id_counter
        State._id_counter += 1
        self.is_accept = is_accept
        self.transitions = {}  # 글자 -> 상태 목록
        self.epsilon = []      # 엡실론 옮아감

    def add_transition(self, char, state):
        self.transitions.setdefault(char, []).append(state)

    def add_epsilon(self, state):
        self.epsilon.append(state)

class NFA:
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

def char_nfa(c):
    start = State()
    accept = State(is_accept=True)
    start.add_transition(c, accept)
    return NFA(start, accept)

def concat_nfa(n1, n2):
    n1.accept.is_accept = False
    n1.accept.add_epsilon(n2.start)
    return NFA(n1.start, n2.accept)

def union_nfa(n1, n2):
    start = State()
    accept = State(is_accept=True)
    start.add_epsilon(n1.start)
    start.add_epsilon(n2.start)
    n1.accept.is_accept = False
    n1.accept.add_epsilon(accept)
    n2.accept.is_accept = False
    n2.accept.add_epsilon(accept)
    return NFA(start, accept)

def star_nfa(n1):
    start = State()
    accept = State(is_accept=True)
    start.add_epsilon(n1.start)
    start.add_epsilon(accept)
    n1.accept.is_accept = False
    n1.accept.add_epsilon(n1.start)
    n1.accept.add_epsilon(accept)
    return NFA(start, accept)
```

---

## 3. 비결정 유한 자동 기계 흉내내기

비결정 유한 자동 기계가 글줄을 받아들이는지 살피려면 지금 상태의 모임을(엡실론 닫힘까지 넣어) 좇는다:

```python
def epsilon_closure(states):
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for t in s.epsilon:
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return closure

def nfa_accepts(nfa, text):
    current = epsilon_closure({nfa.start})
    for ch in text:
        next_states = set()
        for state in current:
            for target in state.transitions.get(ch, []):
                next_states.add(target)
        current = epsilon_closure(next_states)
    return any(s.is_accept for s in current)
```

---

## 4. 복잡도

- **세우기:** 길이 $r$인 정규 표현에 상태와 옮아감이 $O(r)$개.
- **흉내내기:** 걸음마다 상태 $O(r)$개를 들를 수 있으므로 길이 $n$인 글줄마다 $O(n \cdot r)$.

# 참고 문헌

[Introduction to Automata Theory, Languages, and Computation - Hopcroft, Motwani, Ullman](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-automata-theory-languages-and-computation/P200000003517)

[Regular Expression Matching Can Be Simple And Fast - Russ Cox](https://swtch.com/~rsc/regexp/regexp1.html)

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

이 마당은 엄밀한 정의、정규 표현에서 비결정 유한 자동 기계로、비결정 유한 자동 기계 흉내내기、복잡도을 차례로 짚었다.
