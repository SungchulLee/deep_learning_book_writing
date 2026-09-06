# 톰프슨 세우기
톰프슨 세우기는 어떤 정규 표현이든 그와 같은 비결정 유한 자동 기계로 바꾸는 짜임새 있는 알고리즘이다. 1968년 켄 톰프슨이 내놓았으며 길이 $r$인 정규 표현에서 상태가 많아야 $2r$개인 자동 기계를 내고 상태마다 나가는 옮아감이 많아야 둘이다.

## 알고리즘

이 세우기는 정규 표현의 짜임을 그대로 따라가는 되돌이이다. 아래 자동 기계마다 시작 상태와 받아들이는 상태가 꼭 하나씩이다.

### 바탕 경우

**빈 글줄 $\varepsilon$:** $q_0 \xrightarrow{\varepsilon} q_f$

**글자 하나 $a$:** $q_0 \xrightarrow{a} q_f$

### 귀납 경우

**잇기 $r_1 r_2$:** $N(r_1)$의 받아들이는 상태를 $N(r_2)$의 시작 상태와 합친다.

**합치기 $r_1 | r_2$:** 두 아래 자동 기계로 가는 $\varepsilon$ 옮아감을 가진 새 시작 상태를 두고, 두 받아들이는 상태를 $\varepsilon$으로 새 받아들이는 상태에 잇는다.

**클레이니 별 $r_1^*$:** 새 시작 상태와 받아들이는 상태를 두고 0번 이상 되풀이할 수 있는 $\varepsilon$ 옮아감을 둔다.

## 구현

```python
class State:
    _counter = 0
    def __init__(self):
        self.id = State._counter
        State._counter += 1
        self.char_trans = {}
        self.epsilon = []
        self.is_accept = False

class Fragment:
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

def thompson(postfix):
    """뒤 표기 정규 표현에서 비결정 유한 자동 기계를 세운다. 연산: .(잇기), |(합치기), *(별)"""
    stack = []
    for ch in postfix:
        if ch == '.':
            f2 = stack.pop()
            f1 = stack.pop()
            f1.accept.epsilon.append(f2.start)
            f1.accept.is_accept = False
            stack.append(Fragment(f1.start, f2.accept))
        elif ch == '|':
            f2 = stack.pop()
            f1 = stack.pop()
            start = State()
            accept = State()
            accept.is_accept = True
            start.epsilon = [f1.start, f2.start]
            f1.accept.epsilon.append(accept)
            f1.accept.is_accept = False
            f2.accept.epsilon.append(accept)
            f2.accept.is_accept = False
            stack.append(Fragment(start, accept))
        elif ch == '*':
            f = stack.pop()
            start = State()
            accept = State()
            accept.is_accept = True
            start.epsilon = [f.start, accept]
            f.accept.epsilon = [f.start, accept]
            f.accept.is_accept = False
            stack.append(Fragment(start, accept))
        else:
            start = State()
            accept = State()
            accept.is_accept = True
            start.char_trans[ch] = accept
            stack.append(Fragment(start, accept))
    return stack.pop()

def infix_to_postfix(regex):
    """가운데 표기 정규 표현을 또렷한 잇기 연산을 넣어 뒤 표기로 바꾼다."""
    output = []
    for i, ch in enumerate(regex):
        output.append(ch)
        if ch not in ('(', '|') and i + 1 < len(regex):
            next_ch = regex[i + 1]
            if next_ch not in (')', '|', '*'):
                output.append('.')
    expr = ''.join(output)

    precedence = {'*': 3, '.': 2, '|': 1}
    result = []
    ops = []
    for ch in expr:
        if ch == '(':
            ops.append(ch)
        elif ch == ')':
            while ops and ops[-1] != '(':
                result.append(ops.pop())
            ops.pop()
        elif ch in precedence:
            while (ops and ops[-1] != '(' and
                   ops[-1] in precedence and
                   precedence[ops[-1]] >= precedence[ch]):
                result.append(ops.pop())
            ops.append(ch)
        else:
            result.append(ch)
    while ops:
        result.append(ops.pop())
    return ''.join(result)

# 보기: 정규 표현 "a(b|c)*d"
postfix = infix_to_postfix("a(b|c)*d")
print(f"Postfix: {postfix}")
# 내놓기: 뒤 표기: abc|*.d.

nfa = thompson(postfix)
print(f"Start state: {nfa.start.id}, Accept state: {nfa.accept.id}")
```

## 성질

1. 길이 $r$인 정규 표현에 **상태가 많아야 $2r$개**이다(연산마다 새 상태를 많아야 2개 만든다).
2. **상태마다 나가는 옮아감이 많아야 둘**이다(글자 옮아감 하나이거나 엡실론 옮아감 최대 둘).
3. **받아들이는 상태에는 나가는 옮아감이 없다.**
4. **세우기는 선형이다:** 시간과 공간이 $O(r)$.

# 참고 문헌

[Thompson - Regular Expression Search Algorithm (1968)](https://doi.org/10.1145/363347.363387)

[Regular Expression Matching Can Be Simple And Fast - Russ Cox](https://swtch.com/~rsc/regexp/regexp1.html)

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
