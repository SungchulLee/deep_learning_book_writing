# 확장 마스터 정리

표준 [마스터 정리](master.md)는 $T(n) = aT(n/b) + f(n)$ 형태의 점화식을 세 경우로 풀지만 틈을 남긴다. $f(n)$이 분수령 함수 $n^{\log_b a}$에 "가까워서" 로그 인자만큼만 차이가 날 때는 표준 정리를 적용할 수 없을 수 있다. 확장 마스터 정리는 $\log n$의 거듭제곱을 포함하는 통행료 함수를 다루는 경우를 추가하여 이 틈을 메운다. 그래서 기본판의 틈으로 빠지는 실용적인 점화식들에 대해 즐겨 쓰는 도구가 된다.

## 표준 마스터 정리 복습

참고를 위해, 표준 마스터 정리는 다음을 다룬다.

$$
T(n) = aT(n/b) + f(n)
$$

$a \geq 1$, $b > 1$이며 $f(n)$을 임계 함수 $n^{\log_b a}$와 비교한다.

- **경우 1**: 어떤 $\epsilon > 0$에 대해 $f(n) = O(n^{\log_b a - \epsilon})$이면 $T(n) = \Theta(n^{\log_b a})$이다
- **경우 2**: $f(n) = \Theta(n^{\log_b a})$이면 $T(n) = \Theta(n^{\log_b a} \log n)$이다
- **경우 3**: 어떤 $\epsilon > 0$에 대해 $f(n) = \Omega(n^{\log_b a + \epsilon})$이고 정칙 조건이 성립하면 $T(n) = \Theta(f(n))$이다

$k \neq 0$에 대해 $f(n) = \Theta(n^{\log_b a} \log^k n)$일 때 틈이 생긴다. 표준 경우 2는 $k = 0$만 다루고, 경우 1과 3은 다항식적 차이를 요구한다. 확장 정리가 이를 해결한다.

## 확장된 경우 2

핵심 확장은 표준 경우 2를 통행료 함수의 로그 인자를 수용하는 더 일반적인 판본으로 대체하는 것이다.

!!! note "확장 마스터 정리(경우 2의 일반화)"
    $a \geq 1$, $b > 1$이고 어떤 상수 $k \geq 0$에 대해 $f(n) = \Theta(n^{\log_b a} \log^k n)$인 $T(n) = aT(n/b) + f(n)$이 주어지면

    $$
    T(n) = \Theta(n^{\log_b a} \log^{k+1} n)
    $$

$k = 0$이면 이는 표준 경우 2인 $T(n) = \Theta(n^{\log_b a} \log n)$으로 환원된다.

### 직관

재귀 트리의 각 층에서 전체 일은 (층에 맞게 조정된) $\Theta(n^{\log_b a} \log^k n)$이다. 층이 $\Theta(\log n)$개 있고 로그 인자가 층에 걸쳐 누적되어 결과에 $\log n$의 거듭제곱이 하나 더해진다.

## 완전한 확장 마스터 정리

완전한 확장판은 로그보다 작은 틈을 포함해 $f(n)$과 $n^{\log_b a}$ 사이의 모든 관계를 덮는다.

!!! note "완전한 확장 마스터 정리"
    $a \geq 1$, $b > 1$인 $T(n) = aT(n/b) + f(n)$이 주어지면 다음과 같다.

    **경우 1**(재귀적 일이 지배): 어떤 $\epsilon > 0$에 대해 $f(n) = O(n^{\log_b a - \epsilon})$이면

    $$
    T(n) = \Theta(n^{\log_b a})
    $$

    **경우 2**(로그 인자가 있는 균형): 어떤 $k \geq 0$에 대해 $f(n) = \Theta(n^{\log_b a} \log^k n)$이면

    $$
    T(n) = \Theta(n^{\log_b a} \log^{k+1} n)
    $$

    **경우 3**(통행료 함수가 지배): 어떤 $\epsilon > 0$에 대해 $f(n) = \Omega(n^{\log_b a + \epsilon})$이고, 어떤 $c < 1$과 충분히 큰 모든 $n$에 대해 $a f(n/b) \leq c f(n)$이면

    $$
    T(n) = \Theta(f(n))
    $$

어떤 문헌은 $k < 0$(로그의 역수 인자)과 다항식보다 작은 틈까지 다루는 더 정교한 판본을 제시하지만, 위의 세 경우가 실무에서 마주치는 사실상 모든 점화식을 덮는다.

### 음의 로그 거듭제곱 다루기

$k > 0$에 대해 $f(n) = \Theta(n^{\log_b a} / \log^k n)$(동등하게 $f(n) = \Theta(n^{\log_b a} \log^{-k} n)$)인 경우를 다루는 추가 확장도 있다.

- $k > 1$이면 적분이 수렴하고 $T(n) = \Theta(n^{\log_b a})$이다
- $k = 1$이면 $T(n) = \Theta(n^{\log_b a} \log \log n)$이다
- $0 < k < 1$이면 $T(n) = \Theta(n^{\log_b a} \log^{1-k} n)$이다

이 하위 경우들은 실무에서 거의 필요하지 않지만 완결성을 위해 중요하다.

## 풀이 예제

### 예제 1: 로그 통행료

다음을 생각하자.

$$
T(n) = 2T(n/2) + n \log n
$$

여기서 $a = 2$, $b = 2$이므로 $\log_b a = 1$이다. 통행료 함수는 $f(n) = n \log n = n^1 \cdot \log^1 n$이므로 $k = 1$인 $f(n) = \Theta(n^{\log_b a} \log^k n)$이다.

확장된 경우 2에 의해

$$
T(n) = \Theta(n \log^{1+1} n) = \Theta(n \log^2 n)
$$

$f(n) = n \log n$이 $n^{\log_b a} = n$보다 다항식적으로 작지도 크지도 않으므로 표준 마스터 정리로는 이 점화식을 다룰 수 없다.

### 예제 2: 특수한 경우로서의 표준 경우 2

다음을 생각하자.

$$
T(n) = 4T(n/2) + n^2
$$

여기서 $a = 4$, $b = 2$이므로 $\log_b a = 2$이다. 통행료 함수는 $f(n) = n^2 = n^{\log_b a} \log^0 n$이므로 $k = 0$이다.

확장된 경우 2에 의해

$$
T(n) = \Theta(n^2 \log^{0+1} n) = \Theta(n^2 \log n)
$$

이는 표준 경우 2가 주는 것과 정확히 같아 일관성이 확인된다.

### 예제 3: 더 높은 로그 거듭제곱

다음을 생각하자.

$$
T(n) = 2T(n/2) + n \log^3 n
$$

여기서 $a = 2$, $b = 2$, $\log_b a = 1$이고 $f(n) = n \log^3 n = n^1 \cdot \log^3 n$이므로 $k = 3$이다.

확장된 경우 2에 의해

$$
T(n) = \Theta(n \log^4 n)
$$

### 예제 4: 경우 2가 아닌 점화식(검증)

다음을 생각하자.

$$
T(n) = 9T(n/3) + n
$$

여기서 $a = 9$, $b = 3$, $\log_b a = 2$이고 $f(n) = n$이다. $\epsilon = 1$에 대해 $f(n) = O(n^{2 - \epsilon})$이므로 이는 경우 1에 해당한다.

$$
T(n) = \Theta(n^2)
$$

확장 정리의 경우 1은 표준 정리의 경우 1과 같다.

## 어느 정리를 쓸 것인가

| $n^{\log_b a}$에 대한 통행료 함수 $f(n)$ | 표준 마스터 정리 | 확장 마스터 정리 |
|--------------------------------------------------|----------------|-----------------|
| $f(n) = O(n^{\log_b a - \epsilon})$ | 경우 1 적용 | 경우 1(동일) |
| $f(n) = \Theta(n^{\log_b a})$ | 경우 2 적용 | $k=0$인 경우 2 |
| $f(n) = \Theta(n^{\log_b a} \log^k n)$, $k > 0$ | 적용 불가 | 경우 2 적용 |
| 정칙 조건과 함께 $f(n) = \Omega(n^{\log_b a + \epsilon})$ | 경우 3 적용 | 경우 3(동일) |
| $f(n)$이 로그보다 작은 틈에 있음 | 적용 불가 | 적용될 수 있음(음의 $k$ 경우 참고) |

## 확장된 경우 2의 증명 개요

증명은 재귀 트리 분석을 따른다. 재귀 트리의 $j$층($j = 0, 1, \ldots, \log_b n - 1$)에는 $a^j$개의 노드가 있고 각각 크기 $n/b^j$인 부분문제를 처리한다. $j$층의 일은 다음과 같다.

$$
a^j \cdot f\!\left(\frac{n}{b^j}\right) = a^j \cdot \Theta\!\left(\left(\frac{n}{b^j}\right)^{\log_b a} \log^k \frac{n}{b^j}\right)
$$

$a^j / (b^j)^{\log_b a} = a^j / a^j = 1$이므로 이는 다음으로 간단해진다.

$$
\Theta\!\left(n^{\log_b a} \log^k \frac{n}{b^j}\right) = \Theta\!\left(n^{\log_b a} (\log n - j \log b)^k\right)
$$

$\log_b n$개의 모든 층에 대해 더하면

$$
T(n) = \Theta\!\left(n^{\log_b a} \sum_{j=0}^{\log_b n - 1} (\log n - j \log b)^k\right)
$$

이 합은 (다항식 거듭제곱을 더할 때의 표준적인 결과로) $\Theta(\log^{k+1} n)$이므로

$$
T(n) = \Theta(n^{\log_b a} \log^{k+1} n)
$$

## 다른 주제와의 연결

- **[마스터 정리](master.md)**: 확장 정리가 일반화하는 표준판
- **[Akra-Bazzi 방법](akra_bazzi.md)**: 적분 계산을 사용하는 더욱 일반적인 접근
- **[재귀 트리 방법](recursion_tree.md)**: 증명의 바탕이 되는 기하학적 직관을 제공한다
- **[분할 정복으로부터의 점화식](divide_conquer.md)**: 이 정리들이 푸는 점화식을 유도하는 방법

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 4. MIT Press.
- Leighton, T. (1996). Notes on better master theorems for divide-and-conquer recurrences. MIT CSAIL.
- Roura, S. (2001). An improved master theorem for divide-and-conquer recurrences. *Automata, Languages and Programming*, LNCS 2076, 449-459.


## 연습문제

**연습문제 1.**
확장 마스터 정리에서 다룬 점화식 풀이 기법을 점화식 $T(n) = 2T(n/2) + n$에 적용하라.

??? success "연습문제 1 풀이"
    이 절에서 설명한 방법을 사용한다. 핵심 매개변수를 찾고 기법을 적용하면 $T(n) = \Theta(n \log n)$을 얻는다. 이것이 병합 정렬의 점화식이며, 일이 층마다 고르게 분포되는 균형 잡힌 경우를 나타낸다.

---

**연습문제 2.**
확장 마스터 정리를 사용하여 $T(n) = 4T(n/2) + n$을 풀어라. 어느 경우에 해당하는가?

??? success "연습문제 2 풀이"
    $a = 4, b = 2, \log_b a = 2$이다. $f(n) = n = O(n^{2-1})$이다. 재귀 비용이 지배하므로 $T(n) = \Theta(n^2)$이다.

---

**연습문제 3.**
길이 $n$인 시퀀스를 두 절반으로 나누어 각각을 재귀적으로 처리한 뒤 $O(n)$의 교차 어텐션으로 결합하는 트랜스포머 층의 점화식을 쓰고 풀어라.

??? success "연습문제 3 풀이"
    $T(n) = 2T(n/2) + O(n)$이다. 이는 $T(n) = \Theta(n \log n)$을 주며 병합 정렬과 같다. 실제로 트랜스포머는 이런 재귀 구조를 쓰지 않지만, (Longformer 같은) 계층적 어텐션 기법이 이를 근사한다.

---

**연습문제 4.**
확장 마스터 정리에 나오는 점화식의 해를 치환 방법으로 검증하라. 귀납 가정을 서술하고 증명을 수행하라.

??? success "연습문제 4 풀이"
    이 절의 기법으로 닫힌 형태를 추측한다. 모든 $k < n$에 대해 $T(k) \leq ck^p$(또는 적절한 형태)를 가정한다. 이를 점화식에 대입하여 $T(n) \leq cn^p$임을 검증한다. 기저 사례는 따로 처리한다. $\square$
