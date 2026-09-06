# 카운트-민 스케치

작고 붙박인 기억만으로 거대한 자료 흐름 속 어떤 것의 잦음을 어떻게 어림할 수 있을까? Cormode와 Muthukrishnan이 2005년에 내놓은 **Count-Min 스케치**(CMS)가 우아한 확률 자료 얼개로 이 물음에 답한다. 짝마다 독립인 흩기 함수와 함께 두 차원 셈틀 배열을 써서 한쪽 어긋남만 보장한다. 어림이 결코 모자라게 세지 않으며 넘치게 세는 정도가 높은 확률로 가둬진다. CMS은 자료 바탕, 그물 지켜보기, 기계 배움 시스템에 쓰이며 가장 널리 펼쳐진 흐름 자료 얼개 가운데 하나가 되었다.

## 자료 구조

Count-Min 스케치는 0으로 시작하는 $d \times w$ 셈틀 배열 $C[1 \ldots d][1 \ldots w]$과 짝마다 독립인 흩기 함수 $h_1, h_2, \ldots, h_d : [U] \to [w]$ $d$개로 이루어진다.

잡은 다음과 같이 고른다:

$$
w = \left\lceil \frac{e}{\epsilon} \right\rceil, \quad d = \left\lceil \ln \frac{1}{\delta} \right\rceil
$$

여기서 $\epsilon$은 바라는 맞음 정도이고 $\delta$은 어긋날 확률이다.

**온 공간**: 셈틀 $O\left(\frac{1}{\epsilon} \log \frac{1}{\delta}\right)$개에 흩기 함수용 $O(d \log U)$비트를 더한 것.

## 연산

### 갱신

원소 $a$이 흐름에 들어오면 줄마다 해당 셈틀을 하나 올린다:

$$
C[i][h_i(a)] \leftarrow C[i][h_i(a)] + 1 \quad \text{for } i = 1, 2, \ldots, d
$$

무게 $c$의 무게 매긴 고침에서는 1 대신 $c$을 더한다.

### 점 묻기

것 $j$의 잦음 $f_j$을 어림하려면 모든 줄을 통틀어 가장 작은 셈틀 값을 돌려준다:

$$
\hat{f}_j = \min_{i=1}^{d} C[i][h_i(j)]
$$

최소 연산이 흩기 부딪침의 영향을 누그러뜨린다. 어떤 줄은 부딪침 탓에 셈이 부풀 수 있지만 적어도 한 줄은 참 잦음에 가까운 셈을 가질 법하다.

## 맞음 보장

**정리.** 어떤 것 $j$에 대해서도 Count-Min 스케치의 어림은 다음을 만족한다:

1. **모자라게 세지 않음**: 늘 $\hat{f}_j \geq f_j$이다.
2. **넘치게 셈이 가둬짐**: $n$이 흐름 길이일 때 $P(\hat{f}_j > f_j + \epsilon n) < \delta$이다.

*밝힘.*

(1) 셈틀 $C[i][h_i(j)]$마다 것 $j$ 자신에게서 온 $f_j$에 같은 통으로 흩어진 다른 것들의 음이 아닌 이바지가 쌓인다. 그러므로 모든 $i$에 대해 $C[i][h_i(j)] \geq f_j$이고 최소도 적어도 $f_j$이다.

(2) 붙박인 줄 $i$에 대해 **부딪침 잡소리**를 뜻매김한다:

$$
X_i = C[i][h_i(j)] - f_j = \sum_{k \neq j} f_k \cdot \mathbf{1}[h_i(k) = h_i(j)]
$$

$h_i$의 짝마다 독립성에서:

$$
\mathbb{E}[X_i] = \sum_{k \neq j} f_k \cdot \frac{1}{w} \leq \frac{n}{w} = \frac{\epsilon n}{e} \leq \frac{\epsilon n}{e}
$$

마르코프 부등식에서:

$$
P(X_i > \epsilon n) \leq \frac{\mathbb{E}[X_i]}{\epsilon n} \leq \frac{1}{e}
$$

흩기 함수 $d$개가 독립이므로:

$$
P(\hat{f}_j - f_j > \epsilon n) = P(\min_i X_i > \epsilon n) = \prod_{i=1}^{d} P(X_i > \epsilon n) \leq \left(\frac{1}{e}\right)^d = e^{-d} \leq \delta
$$

$\square$

!!! tip "조심스러운 고침"
    **조심스러운 고침**은 무턱대고 1을 더하는 대신 $C[i][h_i(a)]$을 $\max(C[i][h_i(a)], \hat{f}_a + 1)$까지만 올린다. 가장 나쁜 경우의 보장을 바꾸지 않으면서 실제로 넘치게 셈을 줄인다.

## 응용

### 큰손 찾기

잦음이 $\epsilon n$을 넘는 것을 모두 찾으려면:

1. Count-Min 스케치를 지닌다.
2. 고칠 때마다 $\hat{f}_{a_i} > \epsilon n$인지 살핀다.
3. 그렇다면 $a_i$을 큰손 후보 모임에 넣는다.

모자라게 세지 않는 성질 덕에 큰손을 놓치지 않는다(거짓 음성이 없다). 거짓 양성은 잡 $\delta$으로 가둬진다.

### 구간 묻기

두 갈래 쪼갬 위에 Count-Min 스케치를 세우면 구간 묻기 $\sum_{j=l}^{r} f_j$을 공간 $O(\frac{\log U}{\epsilon} \log \frac{1}{\delta})$으로 어긋남 $\epsilon n \log U$ 안에서 답할 수 있다.

### 안쪽 곱 어림

잦음 벡터가 $f$과 $g$인 흐름 둘이 주어지면 Count-Min 스케치 둘을 지니며 다음을 셈해 안쪽 곱 $\langle f, g \rangle = \sum_j f_j g_j$을 어림할 수 있다:

$$
\widehat{\langle f, g \rangle} = \min_{i=1}^{d} \sum_{k=1}^{w} C_f[i][k] \cdot C_g[i][k]
$$

## 합칠 수 있음

Count-Min 스케치는 **합칠 수 있다**. 같은 흩기 함수로 세운 스케치 $S_1$과 $S_2$이 있으면 원소마다의 합 $S_1 + S_2$이 합친 흐름의 올바른 스케치이다. 이 성질 덕에 다음이 가능하다:

- **나눠 셈하기**: 마디마다 그 자리 스케치를 지니고 가운데 조정자가 때때로 스케치를 합친다.
- **나란히 다루기**: 흐름을 일꾼들에게 갈라 주고 따로 스케치한 뒤 결과를 합친다.

## Count 스케치와 견주기

| 성질 | Count-Min 스케치 | Count 스케치 |
|---|---|---|
| 어긋남 갈래 | 한쪽($\hat{f}_j \geq f_j$) | 양쪽 |
| 어긋남 가둠 | $\epsilon n$ | $\epsilon \sqrt{F_2}$ |
| 공간 | $O(\frac{1}{\epsilon} \log \frac{1}{\delta})$ | $O(\frac{1}{\epsilon^2} \log \frac{1}{\delta})$ |
| 모으기 | 줄들의 최소 | 줄들의 가운뎃값 |
| 큰손 | 거짓 음성 없음 | 거짓 음성 가능 |

## 딥러닝과의 관계

- **특징 흩기**: 기계 배움의 흩기 재주는 흩기 함수로 특징을 붙박인 크기의 벡터로 옮기며 이는 CMS의 고침 걸음과 곧바로 같다. Count-Min 스케치 살피기가 특징 흩기의 어긋남 가둠을 준다.
- **기울기 스케치**: 나눠 하는 깊은 배움에서 기울기 벡터를 스케치로 눌러 담아 주고받기를 아끼며 모은다.
- **잦음 바탕 낱말 다듬기**: 말 모델은 앞손질에서 어느 토막을 낱말에 담을지 정하려 스케치의 잦음 어림을 쓴다.

## 요약

Count-Min 스케치는 공간 $O(\frac{1}{\epsilon} \log \frac{1}{\delta})$으로 한쪽 어긋남만 보장하는 공간 효율 좋은 잦음 어림을 준다. 단순함, 합칠 수 있음, 튼튼한 이론 보장 덕에 실제로 가장 중요한 흐름 자료 얼개 가운데 하나가 되었다. 핵심 통찰은 서로 독립인 여러 흩기 쏘기의 최소를 잡으면 셈틀 값을 부풀리는 부딪침 잡소리를 다스릴 수 있다는 것이다.

## 참고 문헌

- [An Improved Data Stream Summary: The Count-Min Sketch (Cormode and Muthukrishnan, 2005)](https://doi.org/10.1016/j.jalgor.2003.12.001)
- [Data Streams: Algorithms and Applications (Muthukrishnan)](https://www.cs.rutgers.edu/~muthu/stream-1-1.ps)


## 연습문제

**연습문제 1.**
Count-Min 스케치 자료 얼개와 그 공간-맞음 맞바꿈을 밝혀라.

??? success "연습문제 1 풀이"
    Count-Min 스케치는 $d \times w$ 셈틀 배열과 저마다 $\{1, \ldots, w\}$으로 옮기는 독립 흩기 함수 $h_1, \ldots, h_d$ $d$개를 쓴다. 것 $x$을 넣으려면 모든 $i$에 대해 $\text{table}[i][h_i(x)]$을 하나 올린다. $x$의 잦음을 물으면 $\hat{f}_x = \min_i \text{table}[i][h_i(x)]$을 돌려준다. 이는 늘 넘치게 센다. $\hat{f}_x \geq f_x$이다. $w = \lceil e/\epsilon \rceil$, $d = \lceil \ln(1/\delta) \rceil$이면 $N$이 온 셈일 때 확률 $\geq 1 - \delta$으로 $\hat{f}_x \leq f_x + \epsilon N$이다. 공간: 셈틀 $O((1/\epsilon) \log(1/\delta))$개.

---

**연습문제 2.**
마르코프 부등식으로 Count-Min 스케치의 어긋남 가둠을 밝혀라.

??? success "연습문제 2 풀이"
    줄 $i$에 대해: $E[\text{table}[i][h_i(x)] - f_x] = \sum_{y \neq x} f_y \cdot P[h_i(y) = h_i(x)] = (N - f_x)/w \leq N/w$. 마르코프에서 $w = e/\epsilon$일 때 $P[\text{table}[i][h_i(x)] - f_x > \epsilon N] \leq 1/(\epsilon w) = 1/e$이다. 독립인 줄 $d$개의 최소를 잡으면 $d = \ln(1/\delta)$일 때 $P[\hat{f}_x - f_x > \epsilon N] \leq (1/e)^d = \delta$이다.

---

**연습문제 3.**
Count-Min 스케치와 Count 스케치를 견주어라. 저마다의 이점은 무엇인가?

??? success "연습문제 3 풀이"
    Count-Min 스케치: 늘 넘치게 세며 줄들의 최소를 쓴다. Count 스케치: $\pm 1$의 아무 부호를 쓰고 줄들의 가운뎃값을 잡아 치우치지 않은 어림을 준다. Count-Min이 더 단순하고 점 묻기와 구간 묻기를 쉽게 받쳐 준다. Count 스케치는 치우치지 않은 어림을 받쳐 주고 음의 고침(지우기)도 다룰 수 있다. Count 스케치는 $|\hat{f}_x - f_x| \leq \epsilon \|f_{-x}\|_2$을 이루며(어긋남이 남은 잦음의 L2 크기에 매인다) 큰손이 판을 칠 때 Count-Min의 L1 가둠보다 빡빡하다.

---

**연습문제 4.**
Count-Min 스케치는 그물 오감 지켜보기에 어떻게 쓰이는가?

??? success "연습문제 4 풀이"
    그물 스위치는 초당 수백만 꾸러미를 다룬다. 흐름마다 정확히 세려면 기억이 너무 많이 든다. $w = 10^4, d = 5$인 Count-Min 스케치는 200KB쯤을 쓰며 어떤 흐름의 꾸러미 수든 99% 믿음으로 온 오감의 $\leq 0.01\%$ 어긋남 안에서 어림한다. 쓰임새: (1) 큰손 찾기(문턱을 넘는 흐름 찾기), (2) 큰물 공격 찾기(흐름 수의 갑작스러운 치솟음), (3) 오감 다루기(길 잡기를 가장 좋게 하려 코끼리 흐름 가려내기).