# 매트로이드

욕심쟁이 알고리즘은 어떤 문제(최소 뻗은 나무, 구간 일정 짜기)에서는 가장 좋게 돌아가지만 다른 문제(떠돌이 장사꾼, 일반 배낭)에서는 어그러진다. 어떤 짜임의 성질이 이 둘을 가르는가? **매트로이드**는 욕심쟁이 알고리즘이 가장 좋은 풀이를 찾음이 보장되는 때를 정확히 담아내는 추상 조합 짜임이다. 문제가 매트로이드 짜임을 지니면, 늘 쓸 수 있는 가장 좋은 원소를 취하는 단순한 전략이 전체 최적을 낸다.

---

## 1. 엄밀한 정의

**매트로이드**는 짝 $M = (S, \mathcal{I})$이며, $S$은 마디 있는 밑 모음이고 $\mathcal{I} \subseteq 2^S$은 다음 세 공리를 채우는 부분 모음의 무리(**홀로 선 모음**이라 한다)다.

**공리 1(비지 않음).** $\emptyset \in \mathcal{I}$.

**공리 2(물림 성질).** $B \in \mathcal{I}$이고 $A \subseteq B$이면 $A \in \mathcal{I}$이다. 홀로 선 모음의 부분 모음은 모두 홀로 서 있다.

**공리 3(맞바꿈 성질).** $A, B \in \mathcal{I}$이고 $|A| < |B|$이면 $A \cup \{x\} \in \mathcal{I}$인 $x \in B \setminus A$이 있다.

맞바꿈 성질이 결정적인 공리이다. 벡터 공간의 모든 기저가 같은 차원을 갖듯, 이 성질은 **가장 큰** 서로 얽히지 않는 모음(**기저**라 한다)이 모두 같은 크기를 가짐을 보장한다.

---

## 2. 말

- **홀로 선 모음.** $\mathcal{I}$의 원소.
- **매인 모음.** $\mathcal{I}$에 들지 않는 $S$의 부분 모음.
- **회로.** 가장 작은 얽힌 모음(아무 원소나 없애면 얽히지 않게 된다).
- **기저.** 가장 큰 얽히지 않는 모음.
- **계수.** 모음 $A \subseteq S$의 계수는 $A$의 홀로 선 부분 모음 가운데 가장 큰 것의 크기다. $r(A) = \max\{|B| : B \subseteq A,\; B \in \mathcal{I}\}$이다.

---

## 3. 보기

### 고른 매트로이드

$U_{k,n} = (S, \mathcal{I})$이며 $|S| = n$이고 $\mathcal{I} = \{A \subseteq S : |A| \le k\}$이다. 크기가 많아야 $k$인 부분 모음은 모두 홀로 서 있다. 밑틀은 크기가 꼭 $k$인 부분 모음 모두다.

### 선형(벡터) 매트로이드

$S$을 $\mathbb{R}^d$ 안의 벡터 모음이라 하자. $\mathcal{I}$을 $S$의 선형으로 홀로 선 부분 모음의 무리라 하자. 맞바꿈 성질은 선형대수의 슈타이니츠 맞바꿈 보조정리에서 따라 나온다.

### 그래프 매트로이드

그래프 $G = (V, E)$이 주어졌을 때 $S = E$이라 하고 $\mathcal{I} = \{F \subseteq E : F \text{ is acyclic}\}$이라 하자. 홀로 선 모음은 숲이고 밑틀은 뻗는 나무이며 회로는 단순 순환이다. 이 매트로이드가 크러스컬 알고리즘의 옳음을 받친다.

### 나눔 매트로이드

$S = S_1 \cup S_2 \cup \cdots \cup S_k$을 가름이라 하자. 테두리 $b_1, \dots, b_k$이 주어지면 $\mathcal{I} = \{A \subseteq S : |A \cap S_i| \le b_i \text{ for all } i\}$이라 둔다.

---

## 4. 핵심 성질

!!! note "모든 기저는 크기가 같다"
    어떤 매트로이드에서도 기저는 모두 크기가 같다. 이는 맞바꿈 성질에서 곧바로 따라 나온다. 곧 기저 $B_1$과 $B_2$의 크기가 다르다면 작은 쪽을 늘릴 수 있어 가장 큼에 어긋난다.

!!! note "매트로이드 쌍대성"
    매트로이드 $M = (S, \mathcal{I})$이 주어졌을 때 **짝 매트로이드** $M^* = (S, \mathcal{I}^*)$은 $S \setminus B^*$이 $M$의 밑틀인 것과 $B^*$이 $M^*$의 밑틀인 것이 같은 뜻이 되도록 매긴다. 그래프 매트로이드의 짝을 **짝 그래프 매트로이드**라 한다.

---

## 5. 확인

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

고른 매트로이드 $U_{2,3}$은 세 공리를 모두 채운다. 둘째 보기는 맞바꿈 성질을 채우지 못한다. $\{1\}$과 $\{3,4\}$이 홀로 서 있고 $|\{1\}| < |\{3,4\}|$인데 $\{1,3\}$도 $\{1,4\}$도 홀로 서 있지 않다.

---

## 연습문제

**연습문제 1.**
매트로이드에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    욕심쟁이 고름은 걸음마다 그 자리에서 가장 좋은 것을 고른다. Matroids에서는 이 고름이 욕심쟁이 고름 성질을 채운다. 곧 이 욕심쟁이 고름을 담은 가장 좋은 풀이가 있다. 여기에 가장 좋은 아래 짜임(욕심쟁이 고름 뒤 남은 아래 문제도 같은 전략으로 가장 좋게 풀린다)을 더하면 욕심쟁이 알고리즘이 두루 가장 좋은 풀이를 내놓는다. $\square$

---

**연습문제 2.**
매트로이드이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    가장 좋은 아래 짜임이란 문제의 가장 좋은 풀이가 그 아래 문제의 가장 좋은 풀이를 담는다는 뜻이다. Matroids에서는 욕심쟁이 고름을 하고 나면 남은 문제가 같은 갈래의 더 작은 사례가 된다. 아래 문제의 풀이가 가장 좋지 않다면 그것을 갈음해 온 풀이를 더 낫게 할 수 있는데, 이는 온 풀이가 가장 좋다는 것과 어긋난다. 그러므로 가장 좋은 아래 짜임이 이루어진다. $\square$

---

**연습문제 3.**
매트로이드의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    시간 복잡도는 (필요하다면) 줄 세우는 걸음과 욕심쟁이 고름 되돌이에 달렸다. 흔히 줄 세우기가 $O(n \log n)$으로 가장 크다. 욕심쟁이 되돌이는 원소마다 한 번씩 다루어 $O(n)$이다. 모두 $O(n \log n)$이다. 들임이 미리 줄 세워져 있으면 알고리즘은 $O(n)$에 돈다. $\square$

---

**연습문제 4.**
(매트로이드에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    문제의 짜임과 맞지 않는 다른 욕심쟁이 잣대를 생각해 보자. 이 잣대는 뒤에 더 좋은 고름을 막는 원소를 고를 수 있다. 어긋나는 보기는 잘못된 욕심쟁이 잣대가 가장 좋지 않은 결과를 낼 수 있음을 보이며, 문제마다 그에 맞는 욕심쟁이 고름 성질을 반드시 증명해야 하는 까닭을 드러낸다. $\square$

## 정리하며

이 마당은 엄밀한 정의、말、보기、핵심 성질을 차례로 짚었다.

**참고 문헌**

- Whitney, H. (1935). On the abstract properties of linear dependence. *American Journal of Mathematics*, 57(3), 509--533.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 16장: Greedy Algorithms.
- Oxley, J. G. (2011). *Matroid Theory* (2nd ed.). Oxford University Press.
