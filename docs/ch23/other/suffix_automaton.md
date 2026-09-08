# 뒷가지 자동 기계

**뒷가지 자동 기계**(방향 있는 돌기 없는 낱말 그래프, DAWG라고도 한다)는 주어진 글줄의 뒷가지만을 정확히 받아들이는 가장 작은 결정 유한 자동 기계이다. $O(n)$ 시간과 공간에 세울 수 있고 부분 글줄 묻기, 서로 다른 부분 글줄 세기, 최장 공통 부분 글줄 찾기를 모두 효율 좋게 되게 한다.

---

## 1. 핵심 성질

길이 $n$인 글줄 $s$에 대해:

- 뒷가지 자동 기계의 상태는 많아야 $2n - 1$개, 옮아감은 많아야 $3n - 4$개이다.
- 글줄 $w$을 받아들일 필요충분조건은 $w$이 $s$의 부분 글줄인 것이다.
- 상태마다 $s$에서 끝나는 자리의 모임이 같은 부분 글줄의 **동치류**를 뜻한다.

---

## 2. 끝자리 모임과 동치류

$\text{endpos}(w)$을 $s$에서 부분 글줄 $w$이 끝나는 자리의 모임이라 정하자. $\text{endpos}(u) = \text{endpos}(v)$이면 부분 글줄 $u$과 $v$은 같은 동치류에 든다.

뒷가지 자동 기계의 상태마다 동치류 하나에 맞닿는다. 상태의 **뒷가지 이음**은 다른 동치류에 드는 가장 긴 진뒷가지를 뜻하는 상태를 가리킨다.

---

## 3. 온라인 세우기

뒷가지 자동 기계는 한 번에 글자 하나씩 더하며 조금씩 세운다. $s[0 \ldots i-1]$의 자동 기계에 글자 $c$을 더할 때:

1. 넓힌 글줄의 새 상태 `cur`을 만든다.
2. 앞선 온전한 글줄을 뜻하는 상태에서 시작해 뒷가지 이음을 따라가며 `cur`으로 가는 글자 $c$ 옮아감을 더한다.
3. 어떤 조상에서 글자 $c$ 옮아감이 이미 있으면 목표 상태를 베껴 `cur`의 뒷가지 이음을 다룬다.

---

## 4. 파이썬 구현

```python
"""
뒷가지 자동 기계(DAWG) — O(n)에 하는 온라인 세우기.

글줄의 뒷가지를 모두 받아들이는 가장 작은 결정 유한 자동 기계를 세운다.
부분 글줄 살피기, 서로 다른 부분 글줄 세기,
최장 공통 부분 글줄 찾기를 받친다.
"""

# === 상태 갈래 ===

class State:
    """뒷가지 자동 기계의 상태."""

    def __init__(self) -> None:
        self.length = 0       # 이 동치류에서 가장 긴 글줄의 길이
        self.link = -1        # 뒷가지 이음
        self.transitions: dict[str, int] = {}
        self.count = 0        # 끝자리 모임이 바뀌는 횟수

# === 뒷가지 자동 기계 ===

class SuffixAutomaton:
    """온라인 뒷가지 자동 기계 세우기."""

    def __init__(self) -> None:
        init_state = State()
        init_state.length = 0
        init_state.link = -1
        self.states = [init_state]
        self.last = 0  # 지금 온전한 글줄의 상태 번호

    def extend(self, c: str) -> None:
        """뒷가지 자동 기계에 글자를 더한다."""
        cur = len(self.states)
        new_state = State()
        new_state.length = self.states[self.last].length + 1
        new_state.count = 1
        self.states.append(new_state)

        p = self.last
        while p != -1 and c not in self.states[p].transitions:
            self.states[p].transitions[c] = cur
            p = self.states[p].link

        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].transitions[c]
            if self.states[p].length + 1 == self.states[q].length:
                self.states[cur].link = q
            else:
                # 상태 q을 베낀다
                clone = len(self.states)
                cloned = State()
                cloned.length = self.states[p].length + 1
                cloned.link = self.states[q].link
                cloned.transitions = dict(self.states[q].transitions)
                self.states.append(cloned)

                while p != -1 and self.states[p].transitions.get(c) == q:
                    self.states[p].transitions[c] = clone
                    p = self.states[p].link

                self.states[q].link = clone
                self.states[cur].link = clone

        self.last = cur

    def build(self, s: str) -> None:
        """글줄 s의 뒷가지 자동 기계를 세운다."""
        for c in s:
            self.extend(c)

    def contains(self, pattern: str) -> bool:
        """본이 본디 글줄의 부분 글줄인지 살핀다."""
        cur = 0
        for c in pattern:
            if c not in self.states[cur].transitions:
                return False
            cur = self.states[cur].transitions[c]
        return True

    def count_distinct_substrings(self) -> int:
        """비지 않은 서로 다른 부분 글줄의 수를 센다."""
        total = 0
        for i in range(1, len(self.states)):
            state = self.states[i]
            link_len = self.states[state.link].length if state.link >= 0 else 0
            total += state.length - link_len
        return total

# === 최장 공통 부분 글줄 ===

def longest_common_substring(s1: str, s2: str) -> str:
    """뒷가지 자동 기계로 최장 공통 부분 글줄을 찾는다."""
    sa = SuffixAutomaton()
    sa.build(s1)

    cur = 0
    cur_len = 0
    best_len = 0
    best_end = 0

    for i, c in enumerate(s2):
        while cur != 0 and c not in sa.states[cur].transitions:
            cur = sa.states[cur].link
            cur_len = sa.states[cur].length

        if c in sa.states[cur].transitions:
            cur = sa.states[cur].transitions[c]
            cur_len += 1
        else:
            cur = 0
            cur_len = 0

        if cur_len > best_len:
            best_len = cur_len
            best_end = i

    return s2[best_end - best_len + 1:best_end + 1]

# === 메인 ===

if __name__ == "__main__":
    s = "abcbc"
    sa = SuffixAutomaton()
    sa.build(s)

    print(f"String: '{s}'")
    print(f"States: {len(sa.states)}")
    print(f"Distinct substrings: {sa.count_distinct_substrings()}")

    for pattern in ["abc", "bcb", "cb", "xyz"]:
        print(f"  Contains '{pattern}': {sa.contains(pattern)}")

    lcs = longest_common_substring("abcdef", "zbcdf")
    print(f"\nLCS of 'abcdef' and 'zbcdf': '{lcs}'")
    # 내임:
    # 글줄: 'abcbc'
    # 상태: 8
    # 서로 다른 부분 글줄: 12
    #   'abc'을 품음: True
    #   'bcb'을 품음: True
    #   'cb'을 품음: True
    #   'xyz'을 품음: False
    #
    # 'abcdef'과 'zbcdf'의 최장 공통 부분 글줄: 'bcd'
```

**출력:**

```
String: 'abcbc'
States: 8
Distinct substrings: 12
  Contains 'abc': True
  Contains 'bcb': True
  Contains 'cb': True
  Contains 'xyz': False

LCS of 'abcdef' and 'zbcdf': 'bcd'
```

---

## 5. 복잡도

| 연산 | 시간 | 공간 |
|---|---|---|
| 세우기 | $O(n)$ | 상태 $O(n)$개, 옮아감 $O(n |\Sigma|)$개 |
| 부분 글줄 살피기 | $O(|w|)$ | — |
| 서로 다른 부분 글줄 세기 | $O(n)$ | — |
| 최장 공통 부분 글줄 | $O(n + m)$ | $O(n)$ |

---

## 연습문제

**연습문제 1.**
뒷가지 자동 기계의 핵심 자료 짜임이나 개념과 그 으뜸 쓰임새를 설명하라.

??? success "연습문제 1 풀이"
    뒷가지 자동 기계은 글줄이나 차례 자료를 미리 다듬고 묻는 효율 좋은 길을 준다. 으뜸 쓰임새는 부분 글줄, 본, 들임의 짜임 성질에 대한 되풀이되는 물음에 답하는 것이다. 미리 다듬기가 다룰 만한 시간에 자료 짜임을 세우고 나면 맨바닥에서 다시 다듬는 것보다 훨씬 빠르게 물음에 답할 수 있다. $\square$

---

**연습문제 2.**
뒷가지 자동 기계을 세우는 시간 복잡도는 무엇인가? 으뜸 연산의 묻기 시간은 무엇인가?

??? success "연습문제 2 풀이"
    세우는 시간은 쓰는 알고리즘에 달렸다. 흔한 한계는 $n$이 들임 크기일 때 $O(n)$에서 $O(n \log n)$ 사이이다. 묻기는 흔히 본 찾기에 $O(m)$($m$은 물음 길이), 미리 셈한 성질에 $O(1)$이 든다. 공간 복잡도는 흔히 $O(n)$이거나 $\sigma$이 글자 모임의 크기일 때 $O(n\sigma)$이다. $\square$

---

**연습문제 3.**
뒷가지 자동 기계을 더 단순한 다른 방식과 견주어라. 더 정교한 짜임은 언제 값어치가 있는가?

??? success "연습문제 3 풀이"
    더 단순한 방식(예컨대 막무가내 훑기나 정렬)은 묻기 시간이 더 길지만 세우는 군더더기가 적다. 정교한 짜임은 다음일 때 값어치가 있다. (1) 같은 자료에 물음을 많이 던져 세우는 값이 고르게 나뉠 때, (2) 묻기 시간이 결정적일 때(실시간 쓰임새), (3) 자료가 커서 점근 나아짐이 실전에서 중요할 때이다. 작은 자료에 물음을 한 번 던지는 경우에는 상수 인수가 작은 단순한 방식이 더 빠를 수 있다. $\square$

---

**연습문제 4.**
들임 글줄 "banana"에 대해 뒷가지 자동 기계을 세우는 것을 좇아라. 중간 걸음을 보여라.

??? success "연습문제 4 풀이"
    "banana"($n = 6$)에 대해: 글줄을 글자마다(또는 뒷가지마다) 처리하며 자료 짜임을 조금씩 세운다. 마지막 짜임은 뒷가지 "banana", "anana", "nana", "ana", "na", "a"을 모두 담는다. 결과의 핵심 성질을 확인할 수 있다. 곧 공통 앞가지를 나눠 쓰고, 뒷가지 차례가 지켜지며, 부분 글줄에 대한 모든 물음을 그 짜임에서 답할 수 있다. $\square$

## 정리하며

이 마당은 핵심 성질、끝자리 모임과 동치류、온라인 세우기、파이썬 구현을 차례로 짚었다.

**참고 문헌**

- Blumer, A., Blumer, J., Haussler, D., McConnell, R., & Ehrenfeucht, A. (1987). Complete inverted files for efficient text retrieval and analysis. *Journal of the ACM*, 34(3), 578-595.
