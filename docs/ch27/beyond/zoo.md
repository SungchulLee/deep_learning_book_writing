# 복잡도 동물원

복잡도 이론은 수백 개의 복잡도 갈래를 밝혀냈으며 저마다 다른 셈 모델이나 밑천 제약을 담는다. 이 쪽은 P과 NP 너머의 주요 갈래, 그 담김 관계, 그리고 이들을 잇는 핵심 열린 물음을 안내한다. 셈 풍경의 지도라고 여기면 된다.

## 시간 바탕 갈래

### 정해진 시간

| 갈래 | 뜻매김 | 핵심 문제 |
|-------|-----------|-------------|
| **P** | DTIME($n^{O(1)}$) | 줄 세우기, 최단 길, 짝짓기 |
| **EXPTIME** | DTIME($2^{n^{O(1)}}$) | 넓힌 서양 장기 |
| **2-EXPTIME** | DTIME($2^{2^{n^{O(1)}}}$) | 어떤 같음 문제 |

알려진 갈라짐: P $\neq$ EXPTIME(시간 켜 정리에 따라).

### 정해지지 않은 시간

| 갈래 | 뜻매김 | 핵심 문제 |
|-------|-----------|-------------|
| **NP** | NTIME($n^{O(1)}$) | SAT, 덩어리, 떠돌이 장수 가름 |
| **여 NP** | NP 문제의 여 문제 | 항진식, 만족 불가능성 |
| **NEXPTIME** | NTIME($2^{n^{O(1)}}$) | 간추린 회로 SAT |

알려진 것: NP $\subseteq$ EXPTIME. 알려지지 않은 것: NP $\neq$ 여 NP인지.

## 공간 바탕 갈래

| 갈래 | 뜻매김 | 핵심 문제 |
|-------|-----------|-------------|
| **L** | DSPACE($O(\log n)$) | 방향 없는 이어짐(라인골드) |
| **NL** | NSPACE($O(\log n)$) | 방향 있는 이어짐 |
| **PSPACE** | DSPACE($n^{O(1)}$) | TQBF, 넓힌 지리 놀이 |

사비치 정리에 따라 NSPACE($s$) $\subseteq$ DSPACE($s^2$)이므로 NL $\subseteq$ L$^2$ $\subseteq$ P이고 NPSPACE = PSPACE이다.

## 마구잡이 갈래

| 갈래 | 어긋남 갈래 | 뜻매김 |
|-------|-----------|-----------|
| **ZPP** | 어긋남 없음 | 기댓값 다항 시간, 늘 옳다 |
| **RP** | 한쪽 | 거짓 양성 없음, $\Pr[\text{accept} \mid x \in L] \geq 1/2$ |
| **여 RP** | 한쪽 | 거짓 음성 없음 |
| **BPP** | 양쪽 | $\Pr[\text{correct}] \geq 2/3$ |

**담김:** P $\subseteq$ ZPP = RP $\cap$ 여 RP $\subseteq$ RP $\subseteq$ BPP.

**추측:** BPP = P(다항 시간 셈에 마구잡이는 꼭 필요하지 않다).

## 회로 갈래

| 갈래 | 뜻매김 |
|-------|-----------|
| **NC** | 로그 거듭 깊이, 다항 크기 회로(효율 좋은 나란한 셈) |
| **AC$^0$** | 상수 깊이, 다항 크기, 들임 개수 제한 없음 |
| **TC$^0$** | 문턱 문을 갖춘 AC$^0$ |
| **NC$^1$** | $O(\log n)$ 깊이, 들임 개수 가둠 |
| **P/poly** | 다항 크기 회로(도움말 딸림) |

**알려진 것:** AC$^0 \subsetneq$ TC$^0 \subseteq$ NC$^1 \subseteq$ L $\subseteq$ NL $\subseteq$ NC$^2 \subseteq$ P $\subseteq$ P/poly.

갈래 P/poly은 (도움말을 거쳐) 가릴 수 없는 문제도 담으므로 P/poly $\not\subseteq$ NP일 수 있다. 그러나 NP $\subseteq$ P/poly이면 다항 켜가 무너진다(카프-립튼 정리).

## 세기 갈래와 함수 갈래

| 갈래 | 뜻매김 |
|-------|-----------|
| **#P** | 정해지지 않은 튜링 기계의 받아들이는 길 세기 |
| **FP** | 다항 시간에 셈할 수 있는 함수 |
| **GapP** | #P 함수 둘의 차 |
| **PP** | 셈 길의 많은 쪽이 받아들임 |

**도다 정리:** PH $\subseteq$ P$^{\text{#P}}$.

**PP과 BPP:** BPP $\subseteq$ PP이지만 PP이 훨씬 힘세다(PP은 다항 시간 줄임에서 PP 완전이고 PH $\subseteq$ P$^{\text{PP}}$이다).

## 주고받는 밝힘 갈래

| 갈래 | 뜻매김 |
|-------|-----------|
| **IP** | 다항 시간 살피개를 가진 주고받는 밝힘 |
| **AM** | 아서-멀린 규약(드러난 동전) |
| **MA** | 멀린-아서(멀린이 밝힘을 보내고 아서가 확률로 살핀다) |
| **MIP** | 서로 얽매이지 않은 밝히개 여럿 |

**이정표가 되는 결과:**

- IP = PSPACE(샤미르, 1992)
- AM $\subseteq$ PH이다(AM $\subseteq \Pi_2^p$)
- MIP = NEXPTIME(바바이, 포트나우, 룬드, 1991)
- MIP* = RE(지 외, 2020) --- 얽힌 밝히개를 쓰면 이 갈래는 되돌이로 늘어놓을 수 있는 말과 같아진다

## 큰 담김 그림

$$
\text{L} \subseteq \text{NL} \subseteq \text{P} \subseteq \text{NP} \cap \text{co-NP} \subseteq \text{NP} \cup \text{co-NP} \subseteq \text{PH} \subseteq \text{PSPACE} \subseteq \text{EXPTIME}
$$

덧붙여:

$$
\text{P} \subseteq \text{BPP} \subseteq \text{PSPACE}
$$

$$
\text{NP} \subseteq \text{P}^{\text{\#P}} \subseteq \text{PSPACE}
$$

## 주요 열린 문제

| 물음 | 상태 | 뜻 |
|----------|--------|-------------|
| P $\stackrel{?}{=}$ NP | 1971년부터 열림 | 셈 과학에서 가장 중요한 열린 문제 |
| NP $\stackrel{?}{=}$ 여 NP | 열림 | 같으면 PH이 무너진다 |
| P $\stackrel{?}{=}$ PSPACE | 열림 | 알려진 것: P $\neq$ PSPACE(공간 켜) |
| BPP $\stackrel{?}{=}$ P | 같다고 추측됨 | 마구잡이가 꼭 필요하지 않음을 보일 것이다 |
| L $\stackrel{?}{=}$ P | 열림 | 공간과 시간에 대한 근본 물음 |
| NP $\stackrel{?}{\subseteq}$ P/poly | 거짓이라 믿어짐 | PH이 무너질 것이다(카프-립튼) |

!!! warning "가로막에 대한 알림"
    가로막 결과 셋이 P 대 NP의 밝힘 재주를 제한한다. **상대화**(베이커-길-솔로베이), **자연스러운 밝힘**(라즈보로프-루디치), **대수화**(애런슨-위그더슨)이다. P $\neq$ NP의 어떤 밝힘도 이 셋을 모두 비켜 가야 한다.

??? example "보기: 동물원 둘러보기"
    **물음:** 그래프 같은 꼴(GI)은 어디에 놓이는가?

    - GI $\in$ NP이다(자리바꿈이 증인 노릇을 한다).
    - GI은 NP 완전일 법하지 않다. 만약 그렇다면 PH이 무너진다(보파나-호스타드-자코스).
    - GI $\in$ 여 AM이므로 GI은 켜에서 "낮은" 자리에 있다.
    - 바바이(2016)는 GI $\in$ 준다항 시간 $2^{O((\log n)^c)}$임을 보였다.

    GI은 P과 NP 완전 사이, 중간 어려움의 문제가 있다고 믿어지는 자리에 놓인다(P $\neq$ NP이면 라드너 정리가 그런 문제가 있음을 보장한다).

## 참고 문헌

- Sipser, M. (2012). *Introduction to the Theory of Computation* (3rd ed.). Cengage Learning.
- Arora, S., & Barak, B. (2009). *Computational Complexity: A Modern Approach*. Cambridge University Press.
- Aaronson, S. (2016). P $\stackrel{?}{=}$ NP. In *Open Problems in Mathematics*, Springer.

## 연습문제

**연습문제 1.**
다음 복잡도 갈래를 담김 차례로 놓아라: P, NP, BPP, PSPACE, EXP, 여 NP, PH. 담김 그림을 그리고 어느 담김이 엄격하다고 알려져 있는지 밝혀라.

??? success "연습문제 1 풀이"
    알려진 담김 사슬은 다음과 같다:

    $\mathbf{P} \subseteq \mathbf{NP} \subseteq \mathbf{PH} \subseteq \mathbf{PSPACE} \subseteq \mathbf{EXP}$

    덧붙여 $\mathbf{P} \subseteq \mathbf{co\text{-}NP} \subseteq \mathbf{PH}$이고 $\mathbf{P} \subseteq \mathbf{BPP} \subseteq \mathbf{PSPACE}$이다.

    알려진 엄격한 담김: $\mathbf{P} \neq \mathbf{EXP}$(시간 켜 정리에 따라)과 $\mathbf{NP} \neq \mathbf{EXPTIME}$이다. $\mathbf{P} \neq \mathbf{NP}$, $\mathbf{NP} \neq \mathbf{co\text{-}NP}$, $\mathbf{NP} \neq \mathbf{PSPACE}$, $\mathbf{BPP} \neq \mathbf{P}$인지는 열려 있다. $\mathbf{P} \neq \mathbf{EXP}$이므로 사슬 $\mathbf{P} \subseteq \mathbf{NP} \subseteq \mathbf{PSPACE} \subseteq \mathbf{EXP}$의 담김 가운데 적어도 하나는 엄격해야 하지만 어느 것인지는 모른다.

---

**연습문제 2.**
라드너 정리를 적고 NP의 얼개에 대해 그것이 지니는 뜻을 밝혀라. 왜 NP 중간 문제가 있음을 뜻하는가?

??? success "연습문제 2 풀이"
    라드너 정리(1975)는 이렇게 말한다. $\mathbf{P} \neq \mathbf{NP}$이면 NP 완전이 아닌 말 $L \in \mathbf{NP} \setminus \mathbf{P}$이 있다. 그런 말을 NP 중간이라 부른다.

    밝힘은 대각선 재주를 써서 SAT 사례를 "채워 넣어" $L$을 세운다. 이 말은 SAT처럼 굴다가(P에 들지 않게 한다) 쉬워지기를 번갈아 한다(SAT이 그것으로 줄여지지 않게 하여 NP 완전이 아니게 한다). 채워 넣는 빠르기는 다항 시간 알고리즘과 다항 시간 줄임이 저마다 결국 꺾이도록 조심스레 고른다.

    그 뜻은 $\mathbf{P} \neq \mathbf{NP}$이라 가정하면 NP의 풍경이 "쉬움"(P)과 "가장 어려움"(NP 완전)의 단순한 두 갈래가 아니라는 것이다. NP 안에는 서로 다른 어려움의 켜가 끝없이 많아야 한다. NP 중간 문제의 후보로는 그래프 같은 꼴과 인수 분해가 있다.

---

**연습문제 3.**
다항 켜(PH)를 밝혀라. 일반적인 $k$에 대해 $\Sigma_k^p$과 $\Pi_k^p$을 뜻매김하고, PH이 $k$번째 켜로 무너지면(즉 $\Sigma_k^p = \Sigma_{k+1}^p$이면) $\mathbf{PH} = \Sigma_k^p$임을 보여라.

??? success "연습문제 3 풀이"
    다항 켜는 귀납으로 뜻매김된다:

    - $\Sigma_0^p = \Pi_0^p = \mathbf{P}$
    - $\Sigma_{k+1}^p = \mathbf{NP}^{\Sigma_k^p}$($\Sigma_k^p$ 신탁을 가진 NP)
    - $\Pi_{k+1}^p = \mathbf{co\text{-}NP}^{\Sigma_k^p}$
    - $\mathbf{PH} = \bigcup_{k \geq 0} \Sigma_k^p$

    따라서 $\Sigma_1^p = \mathbf{NP}$, $\Pi_1^p = \mathbf{co\text{-}NP}$, $\Sigma_2^p = \mathbf{NP}^{\mathbf{NP}}$ 등이다.

    $\Sigma_k^p = \Sigma_{k+1}^p$이면 $\Pi_{k+1}^p = \text{co-}\Sigma_{k+1}^p = \text{co-}\Sigma_k^p = \Pi_k^p \subseteq \Sigma_k^p$이다. 그러면 $\Sigma_{k+2}^p = \mathbf{NP}^{\Sigma_{k+1}^p} = \mathbf{NP}^{\Sigma_k^p} = \Sigma_{k+1}^p = \Sigma_k^p$이다. 귀납에 따라 더 높은 켜가 모두 $\Sigma_k^p$과 같으므로 $\mathbf{PH} = \Sigma_k^p$이다.

---

**연습문제 4.**
BQP이 무엇이며 복잡도 동물원에서 어디에 놓이는지 밝혀라. 쇼어 알고리즘이 BQP과 고전 복잡도 갈래의 관계에 왜 중요한가?

??? success "연습문제 4 풀이"
    $\mathbf{BQP}$(어긋남이 가둬진 양자 다항 시간)은 어긋날 확률이 많아야 $1/3$인 채로 양자 셈틀이 다항 시간에 풀 수 있는 가름 문제의 갈래이다. BPP의 양자 판이다.

    알려진 담김: $\mathbf{BPP} \subseteq \mathbf{BQP} \subseteq \mathbf{PSPACE}$이다. 첫째 담김은 양자 셈틀이 고전 확률 셈을 흉내 낼 수 있기에 성립한다. 둘째는 양자 셈을 (진폭을 좇아) 고전으로 다항 공간에서 흉내 낼 수 있기에 성립한다.

    쇼어 알고리즘은 정수를 양자 다항 시간에 인수 분해하여 인수 분해를 BQP에 놓는다. 인수 분해가 BPP에 든다고도, NP 완전이라고도 알려져 있지 않으므로 이는 $\mathbf{BQP} \not\subseteq \mathbf{BPP}$(양자 빨라짐이 실재한다)의 증거가 되며 BQP과 NP을 견줄 수 없음을 시사한다. BQP은 P에 없을 수도 있는 문제를 풀 수 있지만 BQP이 NP 전체를 담는다는 증거는 없다.

---

**연습 5.**
갈래 IP(주고받는 밝힘)을 적고 놀라운 결과 IP = PSPACE을 밝혀라. 이 결과가 직관에 어긋나는 까닭은 무엇인가?

??? success "연습 5의 풀이"
    $\mathbf{IP}$은 주고받는 밝힘 얼개로 가를 수 있는 말의 갈래이다. 다항 시간 확률 살피개가 전능한 밝히개와 다항 번의 바퀴에 걸쳐 주고받는다. 살피개는 올바른 밝힘을 확률 $\geq 2/3$으로 받아들이고 그른 주장을 확률 $\geq 2/3$으로 물리친다.

    샤미르(1992) 정리는 $\mathbf{IP} = \mathbf{PSPACE}$이라 한다. $\mathbf{IP} \subseteq \mathbf{PSPACE}$ 쪽은 밝히개의 모든 셈속을 늘어놓아 살피개의 가장 좋은 셈속을 PSPACE에서 셈할 수 있기에 따라 나온다. 놀라운 쪽은 $\mathbf{PSPACE} \subseteq \mathbf{IP}$이다. 밝힘은 부울 식의 셈 꼴로 바꾸기와 합 살피기 규약을 써서 PSPACE 완전 문제 TQBF에 주고받는 밝힘이 있음을 보인다.

    이는 주고받기와 마구잡이가 함께라면 다항 공간만큼 힘세다는 말이므로 직관에 어긋난다. 셈이 제한된 살피개가 밝히개를 믿지 않고도 그와 말을 주고받아 NP을 훨씬 넘어선 문제의 주장을 살필 수 있다는 것이다.
