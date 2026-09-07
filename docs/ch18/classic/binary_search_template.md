# 이분 찾기 틀

보통의 이분 찾기는 정렬된 배열에서 특정한 값을 찾는다. 그러나 여러 문제는 조건이 거짓에서 참으로(또는 그 반대로) 바뀌는 **경계**를 찾아야 한다. 넓힌 이분 찾기 틀은 그런 문제를 모두 한결같이 다룬다. 곧 단조 술어가 주어질 때 그것을 채우는 가장 작은(또는 가장 큰) 번호를 찾는다.

이 틀은 이분 찾기를 올바로 짜기 어렵게 만드는, 어긋나기 쉬운 테두리 다루기의 세부, 곧 하나 차이 어긋남, 끝나지 않는 되풀이, 잘못된 가운뎃점 반올림을 없애 준다.

## 넓힌 문제

찾을 자리 $\{0, 1, \ldots, n-1\}$과, **한 방향으로만 바뀌는** 참거짓 잣대 $\text{condition}(m)$이 있다고 하자. 곧 다음을 채우는 문턱 $k$이 있다.

$$
\text{condition}(m) = \begin{cases} \text{false} & \text{if } m < k \\ \text{true} & \text{if } m \ge k \end{cases}
$$

목표는 $\text{condition}(m)$이 참이 되는 가장 작은 $m$을 찾는 것이다. 이를 **맨 왼쪽 참** 문제라 한다.

## 틀

```python
def binary_search_template(lo, hi, condition):
    """
    [lo, hi]에서 조건을 채우는 가장 작은 값 찾기.

    매개변수
    ----------
    lo : int
        찾을 자리의 아래 테두리(포함).
    hi : int
        찾을 자리의 위 테두리(포함).
    condition : callable
        단조 술어. 곧 문턱값 아래에서는 거짓,
        문턱값 위와 그 자리에서는 참이다.

    반환값
    -------
    int
        [lo, hi]에서 다음을 채우는 가장 작은 값 m,
        condition(m)이 참인 값. 그런 값이 없으면 hi + 1.
    """
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if condition(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

### 핵심 설계 선택

1. **되풀이 조건 `lo < hi`**(`lo <= hi`가 아님): `lo == hi`일 때 되풀이가 끝나고 그때 답은 `lo`이다.
2. **`hi = mid`**(`hi = mid - 1`이 아님): 조건이 참이면 `mid`이 답일 수 있으므로 찾을 자리에 남겨 둔다.
3. **`lo = mid + 1`**: 조건이 거짓이면 `mid`은 결코 답이 아니므로 뺀다.
4. **`mid = lo + (hi - lo) // 2`**: 내림하므로 `lo < hi`일 때 `mid < hi`가 되어 끝나지 않는 되풀이를 막는다.

!!! warning "끝나지 않는 되풀이 덫"
    `mid = lo + (hi - lo) // 2`에 `hi = mid - 1`을 함께 쓰면 되풀이가 답을 놓칠 수 있다. 같은 가운뎃점 식에 `lo = mid`을 쓰면 `hi - lo == 1`일 때 되풀이가 끝나지 않는다. 위의 틀은 두 함정을 모두 피한다.

## 옳음의 증명

**되돌이 안 바뀜.** 되돌이가 돌 때마다 그 첫머리에서 답($\text{condition}(m)$이 참인 가장 작은 $m$)은 $[\text{lo}, \text{hi}]$ 안에 있다.

**첫자리매김.** 처음 범위가 찾을 자리 전체를 덮으므로 불변량이 성립한다.

**지킴.** $\text{mid} = \lfloor (\text{lo} + \text{hi}) / 2 \rfloor$이라 하자.

- $\text{condition}(\text{mid})$이 참이면 답은 많아야 $\text{mid}$이므로 $\text{hi} = \text{mid}$으로 두면 안 바뀜이 지켜진다.
- $\text{condition}(\text{mid})$이 거짓이면 답은 적어도 $\text{mid} + 1$이므로 $\text{lo} = \text{mid} + 1$으로 두면 안 바뀜이 지켜진다.

**끝남.** $\text{hi} - \text{lo}$은 음이 아닌 정수이며 되돌이마다 반드시 줄어든다($\text{lo} < \text{hi}$이면 $\text{mid} < \text{hi}$이기 때문이다). $\text{lo} = \text{hi}$이 되면 되돌이가 끝나고, 안 바뀜에 따라 `lo`이 답임이 보장된다. $\square$

**때 복잡도.** 되돌이마다 찾을 자리가 반으로 주므로 이 틀은 $O(\log(\text{hi} - \text{lo}))$ 번 돈다. `condition`을 한 번 부르는 데 $O(C)$ 때가 들면 모두 $O(C \log(\text{hi} - \text{lo}))$이다.

## 가장 오른쪽 거짓 변종

$\text{condition}(m)$이 거짓인 **가장 큰** $m$을 찾으려면 거울처럼 뒤집은 틀을 쓴다.

```python
def binary_search_rightmost_false(lo, hi, condition):
    """
    [lo, hi]에서 조건이 거짓인 가장 큰 값 찾기.

    모든 값에 대해 조건이 참이면 lo - 1을 돌려준다.
    """
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2  # 올림
        if condition(mid):
            hi = mid - 1
        else:
            lo = mid
    return lo
```

`lo = mid`일 때 끝나지 않는 되풀이를 막으려 가운뎃점을 **올림**한다(`(hi - lo + 1) // 2`)는 점에 유의하라.

## 응용

### 넣을 자리 찾기

정렬된 배열에서 정렬을 지키려면 `target`을 어느 번호에 넣어야 하는지 찾아라.

```python
def search_insert(nums, target):
    """정렬된 배열에서 찾는 값을 넣을 자리 찾기."""
    return binary_search_template(
        0, len(nums),
        lambda mid: mid == len(nums) or nums[mid] >= target
    )
```

### 정수 제곱근

$k^2 \le x$인 가장 큰 정수 $k$을 찾아라.

```python
def integer_sqrt(x):
    """이분 찾기로 floor(sqrt(x)) 셈하기."""
    if x < 0:
        raise ValueError("Square root of negative number")
    if x == 0:
        return 0
    # (k+1)^2 > x인 가장 작은 k를 찾아 k 돌려주기
    return binary_search_template(
        1, x,
        lambda mid: mid * mid > x
    ) - 1
```

### 처음 나쁜 판

$1$부터 $n$까지 번호가 매겨진 판 $n$개와, 단조인(처음 나쁜 판 뒤의 모든 판도 나쁜) 함수 `is_bad(v)`가 주어질 때 처음 나쁜 판을 찾아라.

```python
def first_bad_version(n, is_bad):
    """1..n번 판 가운데 처음 나쁜 판 찾기."""
    return binary_search_template(1, n, is_bad)
```

### 짐을 실어 나를 담이

$d$일 안에 모든 짐을 나르는 데 필요한 최소 배 담이를 찾아라. "담이 $c$으로 $d$일 안에 모든 짐을 나를 수 있다"라는 술어는 $c$에 대해 단조이다.

```python
def ship_within_days(weights, days):
    """주어진 날 안에 모든 무게를 나를 최소 담이 찾기."""
    def can_ship(capacity):
        day_count, current_load = 1, 0
        for w in weights:
            if current_load + w > capacity:
                day_count += 1
                current_load = 0
            current_load += w
        return day_count <= days

    return binary_search_template(
        max(weights), sum(weights), can_ship
    )
```

## 이 틀을 쓸 때

문제가 다음 성질을 가질 때면 언제나 이 틀을 쓸 수 있다:

1. **단조 술어**: 찾을 자리 전체에서 조건이 거짓에서 참으로 정확히 한 번 바뀐다.
2. **마디 지어진 찾을 자리**: 구간 $[\text{lo}, \text{hi}]$을 미리 안다.
3. **효율적인 값매김**: `condition(mid)`을 살피는 데 다항 시간이 든다.

!!! tip "이분 찾기 문제 알아보기"
    문제가 "X를 채우는 최솟값"이나 "Y를 넘지 않는 최댓값"을 묻고 X나 Y가 답에 대해 단조라면, 답을 두고 이분 찾기를 하는 것이 알맞을 가능성이 높다.

## 요약

두루 쓰는 이분 찾기 틀은 이분 찾기의 온갖 갈래를 한 무늬로 줄인다. 한 방향으로만 바뀌는 잣대를 매기고, 찾을 테두리를 잡은 뒤, 틀이 넘어가는 자리를 찾게 하면 된다. 옳음 밝히기는 답이 늘 이제의 테두리 안에 있음을 보이는 되돌이 안 바뀜에 기대고, 끝남은 찾을 자리가 반드시 줄어드는 데서 따라 나온다. 이 틀은 $O(\log n)$ 번 돌며 그때마다 잣대를 한 번 부른다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 2장. MIT Press.

## 연습문제

**연습문제 1.**
이분 찾기 틀의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    이분 찾기 틀은 나누어 다스리기 틀을 쓴다. 문제를 더 작은 잔문제로 쪼개고, 되부르며 풀고, 그 결과를 아우른다. 때 복잡도는 잔문제의 크기와 아우르는 값을 다스리는 되돌이 식이 정한다. 흔히 으뜸 정리나 되부름 나무 살피기로 닫힌 꼴의 복잡도를 얻는다. $\square$

---

**연습문제 2.**
이분 찾기 틀의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    되돌이 식은 그 알고리즘이 어떻게 나누는지에 달려 있다(잔문제의 수 $a$, 크기를 줄이는 값 $b$, 아우르는 값 $f(n)$). 으뜸 정리를 쓴다. $f(n)$을 $n^{\log_b a}$과 견주어 어느 갈래인지 가린다. $f(n) = \Theta(n^{\log_b a})$이면(둘째 갈래) $T(n) = \Theta(n^{\log_b a} \log n)$이다. $\square$

---

**연습문제 3.**
이분 찾기 틀이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    막무가내로 하는 길은 흔히 $O(n^2)$ 이상이 든다. 나누어 다스리는 길은 되부르며 쪼개어 군더더기 셈을 줄이므로 복잡도가 더 낮다. 들임 크기가 $n = 10^6$이면 $O(n^2) = 10^{12}$과 $O(n \log n) = 2 \times 10^7$의 차이는 $50{,}000$ 곱절이다. $\square$

---

**연습문제 4.**
이분 찾기 틀의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    밑 자리는 더 나눌 수 없을 만큼 작은 들임을 다룬다(흔히 $n \leq 1$이나 $n \leq 2$). 이때는 옳은 결과를 곧바로 돌려주어야 한다. 밑 자리가 제대로 없으면 되부름이 끝나지 않는다. 밑 자리를 더 크게 잡고($n \leq 10$ 따위) 더 단순한 알고리즘으로 갈아타면 같은 점근 복잡도를 지키면서 되부름 덤을 줄여 참으로 더 빠르게 할 수 있다. $\square$
