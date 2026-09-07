# 갈래 여 NP

NP는 "예" 답에 짧고 효율 좋게 살필 수 있는 밝힘이 있는 문제를 담는다. 그러면 "아니오" 답 쪽에 짧은 밝힘이 있는 문제는 어떤가? 보기로 부울 식이 *만족 불가능*함을 밝히려면 *어떤* 매김도 통하지 않음을 보여야 한다. 자연스러운 증서 얼개가 뒤집힌 것이다. 갈래 **여 NP**은 이 여 관점을 갈라 적어 주며, NP와 여 NP가 같은지의 물음은 복잡도 이론의 근본 열린 문제 가운데 하나이다.

## 직관

그래프 $G$에 대한 서로 이어진 두 물음을 살펴보자:

- "$G$에 해밀턴 돌이가 있는가?" -- 예 답은 살피기 쉽다. 그 돌이를 보이면 된다. 이 문제는 NP에 든다.
- "$G$에 해밀턴 돌이가 *없는가*?" -- 이 여 물음의 예 답은 해밀턴 돌이가 *하나도* 없다는 뜻인데, 없음을 어떻게 효율 좋게 증명하는가?

여 NP는 *아니오* 사례(같은 말로 여 문제의 예 사례)에 효율 좋은 증서가 있는 문제를 담는다. NP와 여 NP가 다르다면 어떤 문제에는 있음을 밝히는 것과 없음을 밝히는 것 사이에 본디 맞섬이 깨져 있는 것이다.

## 엄밀한 정의

말 $L \subseteq \Sigma^*$의 **여 말**은 $\overline{L} = \Sigma^* \setminus L$이다.

갈래 **여 NP**은 다음과 같이 뜻매김된다:

$$
\mathbf{co\text{-}NP} = \{ L \subseteq \Sigma^* \mid \overline{L} \in \mathbf{NP} \}
$$

같은 말로 $L \in \mathbf{co\text{-}NP}$일 필요충분조건은 다음을 만족하는 다항 시간 살피개 $V$과 다항식 $p$이 있는 것이다:

$$
x \notin L \iff \exists\, c,\; |c| \leq p(|x|),\; V(x, c) = \text{accept}
$$

달리 말해 $L$의 *아니오* 사례에는 짧은 증서가 있다. 고쳐 쓰면 예 사례에서는 있을 수 있는 *모든* 증서가 물리쳐져야 한다:

$$
x \in L \iff \forall\, c,\; |c| \leq p(|x|),\; V(x, c) = \text{reject}
$$

이 모든 한정 기호("모든 것에 대해")가 여 NP를 NP의 있음 한정 기호("어떤 것이 있어")와 갈라 준다.

## P 및 NP과의 관계

### P은 둘 다에 담긴다

P은 여 연산에 닫혀 있으므로(받아들임과 물리침을 뒤집기만 하면 된다) P의 모든 말은 NP와 여 NP에 모두 든다:

$$
\mathbf{P} \subseteq \mathbf{NP} \cap \mathbf{co\text{-}NP}
$$

### 열린 물음

$\mathbf{NP} = \mathbf{co\text{-}NP}$인지는 알려져 있지 않다. 널리 받아들여지는 추측은 둘이 다르다는 것이다:

$$
\mathbf{NP} \neq \mathbf{co\text{-}NP} \quad \text{(conjectured)}
$$

**따름**: $\mathbf{NP} \neq \mathbf{co\text{-}NP}$이면 $\mathbf{P} \neq \mathbf{NP}$이다(P이 둘 다에 담기기 때문이다).

??? info "NP = 여 NP는 무엇을 뜻하는가?"
    $\mathbf{NP} = \mathbf{co\text{-}NP}$이면 모든 NP 완전 문제의 여 문제도 NP에 든다. 이는 UNSAT(만족 불가능성)에 다항 길이 증서가 있다는 뜻이며, 밝힘 복잡도에 큰 뜻을 지니는 돌파 결과가 될 것이다.

### 켜

$$
\mathbf{P} \subseteq \mathbf{NP} \cap \mathbf{co\text{-}NP} \subseteq \mathbf{NP} \cup \mathbf{co\text{-}NP} \subseteq \mathbf{PSPACE}
$$

이 담김 가운데 어느 것이라도 엄격한지는 알려져 있지 않다.

## 대표 보기

| NP 문제 | 여 NP 여 문제 | "아니오"의 증서 |
|-----------|------------------|---------------------|
| SAT | UNSAT(항진식) | 알려진 짧은 증서 없음 |
| 해밀턴 돌이 | 해밀턴 돌이 없음 | 알려진 짧은 증서 없음 |
| 합성수 | 소수 | 소수 증서(프랫) |
| 그래프 3색 칠하기 | 3색 칠할 수 없음 | 알려진 짧은 증서 없음 |

### NP와 여 NP에 모두 드는 문제

어떤 문제는 P에 든다고 알려지지 않은 채로 $\mathbf{NP} \cap \mathbf{co\text{-}NP}$에 든다고 알려져 있다:

- **인수 분해**(가름 판): $n$과 $k$이 주어질 때 $n$에 $\leq k$인 인수가 있는가? 인수 하나가 예 증서 노릇을 하고 온전한 인수 분해가 아니오 증서 노릇을 한다.
- **선형 계획**(가능성): (타원체 방법으로) P에 든다고 알려져 있으므로 둘 다에 든다.
- **소수 판정**: 이제 P에 든다고 알려져 있지만(AKS) 효율 좋은 알고리즘이 나오기 전에는 $\mathbf{NP} \cap \mathbf{co\text{-}NP}$의 대표 보기였다.

??? example "소수 판정의 프랫 증서"
    AKS(2002) 이전에 소수 판정은 $\mathbf{NP} \cap \mathbf{co\text{-}NP}$에 든다고 알려져 있었다. **프랫 증서**는 원시근과 $p - 1$의 소인수에 대한 되돌이 소수 밝힘을 보여 어떤 수가 소수임을 밝힌다. 이는 다항 시간에 살필 수 있는 다항 길이 증서를 주어 소수를 NP에(따라서 합성수를 여 NP에) 놓는다.

## 여 NP 완전성

다음이면 말 $L$은 **여 NP 완전**이다:

1. $L \in \mathbf{co\text{-}NP}$이고
2. 여 NP의 모든 말이 다항 시간에 $L$으로 줄여진다.

같은 말로 $L$이 여 NP 완전일 필요충분조건은 $\overline{L}$이 NP 완전인 것이다.

**여 NP 완전 문제의 보기:**

- **항진식**: 부울 식이 *모든* 매김에서 참인가?
- **UNSAT**: 부울 식이 만족 불가능한가?
- **타당성**: 일차 논리식이 (유한 구조 위에서) 타당한가?

**정리.** 어떤 여 NP 완전 문제가 NP에 들면 $\mathbf{NP} = \mathbf{co\text{-}NP}$이다.

## 시사하는 것

NP 대 여 NP 물음은 여러 마당과 이어진다:

- **밝힘 복잡도**: $\mathbf{NP} \neq \mathbf{co\text{-}NP}$이면 어떤 밝힘 얼개에서도 다항을 넘는 길이의 밝힘이 필요한 항진식이 있다.
- **암호**: 많은 암호 가정이 $\mathbf{NP} \neq \mathbf{co\text{-}NP}$이라는 추측에 은근히 기댄다.
- **프로그램 살피기**: 프로그램이 옳음을 밝히려면 *어떤* 실행 길도 어긋남으로 이어지지 않음을 보여야 하는데 이는 여 NP 꼴의 글월이다.

## 참고 문헌

- Sipser, M. *Introduction to the Theory of Computation*. Cengage Learning.
- Arora, S. and Barak, B. *Computational Complexity: A Modern Approach*. Cambridge University Press.
- Pratt, V. "Every Prime has a Succinct Certificate." *SIAM Journal on Computing*, 1975.

## 연습문제

**연습문제 1.**
항진식(부울 식이 모든 매김에서 참인지 가리기)이 여 NP 완전임을 보여라.

??? success "연습문제 1 풀이"
    항진식은 여 NP에 든다. 식 $\phi$이 항진식일 필요충분조건은 거짓으로 만드는 매김이 없는 것이다. "아니오" 증서는 거짓으로 만드는 매김이며 다항 시간에 살필 수 있으므로 항진식 $\in$ 여 NP이다.

    여 NP 어려움은 여 SAT(SAT의 여 문제)에서 줄인다. 식 $\psi$이 주어지면 $\phi = \neg \psi$이라 두자. 그러면 $\psi \in \text{co-SAT}$(즉 $\psi$이 만족 불가능)일 필요충분조건은 $\neg\psi$이 항진식인 것이고 이는 $\phi \in \text{TAUTOLOGY}$과 같다. 여 SAT은 (NP 완전 문제 SAT의 여 문제로서) 여 NP 완전이므로 항진식은 여 NP 어려움이다. 둘을 합치면 항진식은 여 NP 완전이다.

---

**연습문제 2.**
$\mathbf{P} = \mathbf{NP}$일 필요충분조건이 $\mathbf{P} = \mathbf{co\text{-}NP}$임을 밝혀라.

??? success "연습문제 2 풀이"
    앞 방향: $\mathbf{P} = \mathbf{NP}$이면 $\mathbf{P}$이 여 연산에 닫혀 있으므로($\mathbf{P} = \mathbf{co\text{-}P}$) $\mathbf{co\text{-}NP} = \mathbf{co\text{-}P} = \mathbf{P} = \mathbf{NP}$이고 따라서 $\mathbf{P} = \mathbf{co\text{-}NP}$이다.

    뒤 방향: $\mathbf{P} = \mathbf{co\text{-}NP}$이면 SAT $\in \mathbf{NP}$이므로 여 SAT $\in \mathbf{co\text{-}NP} = \mathbf{P}$이다. 그런데 여 SAT $\in \mathbf{P}$은 SAT $\in \mathbf{P}$을 뜻한다(P은 여 연산에 닫혀 있다). SAT이 NP 완전이고 SAT $\in \mathbf{P}$이므로 모든 NP 문제가 다항 시간에 SAT으로 줄여져 $\mathbf{NP} \subseteq \mathbf{P}$이다. $\mathbf{P} \subseteq \mathbf{NP}$과 합치면 $\mathbf{P} = \mathbf{NP}$을 얻는다.

---

**연습문제 3.**
NP와 여 NP에 모두 들지만 P에 든다고 알려지지 않은 문제의 보기를 들어라. 그런 문제가 NP 완전일 법하지 않은 까닭을 밝혀라.

??? success "연습문제 3 풀이"
    인수 분해(정수 $N$과 $k$이 주어질 때 $N$에 $\leq k$인 인수가 있는가?)는 NP $\cap$ 여 NP에 든다. 인수 $d \leq k$이 증서 노릇을 하므로 NP에 든다. $N$의 온전한 소인수 분해가 $\leq k$인 인수가 없다는 증서 노릇을 하므로(AKS 소수 시험과 곱셈으로 살필 수 있다) 여 NP에 든다.

    인수 분해가 NP 완전이라면 모든 NP 문제가 그것으로 줄여지고 여 NP가 NP 완전 문제를 담는다는 뜻이 된다. 이는 $\mathbf{NP} \subseteq \mathbf{co\text{-}NP}$을 뜻하고 맞섬에 따라 $\mathbf{NP} = \mathbf{co\text{-}NP}$이 된다. 이는 다항 켜의 첫 켜를 무너뜨리는데 그런 일은 일어나지 않으리라 널리 믿어진다.

---

**연습문제 4.**
$\mathbf{NP} \neq \mathbf{co\text{-}NP}$이면 어떤 NP 완전 문제도 여 NP에 들지 않음을 보여라.

??? success "연습문제 4 풀이"
    $L$이 NP 완전이고 $L \in \mathbf{co\text{-}NP}$이라 하자. 어떤 말 $A \in \mathbf{NP}$에 대해서도 $L$이 NP 완전이므로 $x \in A \iff f(x) \in L$인 다항 시간 줄임 $f$이 있다. 이 줄임은 $\bar{A} \leq_p \bar{L}$도 보인다. $L \in \mathbf{co\text{-}NP}$이므로 $\bar{L} \in \mathbf{NP}$이다. 그러면 $\bar{A}$이 NP 문제로 줄여져 $\bar{A} \in \mathbf{NP}$이고 이는 $A \in \mathbf{co\text{-}NP}$을 뜻한다. $A$이 아무거나였으므로 $\mathbf{NP} \subseteq \mathbf{co\text{-}NP}$이고 맞섬에 따라 $\mathbf{NP} = \mathbf{co\text{-}NP}$이 된다. 이는 가정과 어긋나므로 어떤 NP 완전 문제도 여 NP에 들지 않는다.
