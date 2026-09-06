# 범위 합 질의

접두사 합 질의는 색인 1부터 $i$까지의 누적 합을 돌려준다. 실제로는 아무 부분 범위 $[l, r]$의 합이 필요할 때가 많다. 이 쪽은 이진 색인 트리(BIT)가 접두사 질의 둘을 아울러 **범위 합 질의**에 답하는 법을 보이고, 이 기법의 모서리 경우와 넓힘을 살핀다.

## 접두사 합으로 귀착하기

색인 $l$부터 $r$까지(둘 다 포함)의 원소 합은 접두사 합 둘의 차로 나타낼 수 있다.

$$
\text{rangeSum}(l, r) = \text{prefix}(r) - \text{prefix}(l - 1)
$$

여기서 $\text{prefix}(i) = \sum_{j=1}^{i} a[j]$이고 관례로 $\text{prefix}(0) = 0$이다.

$\text{prefix}(r)$이 $a[1]$부터 $a[r]$까지를 모두 담고 $\text{prefix}(l-1)$을 빼면 $a[1]$부터 $a[l-1]$까지가 없어져 꼭 $a[l] + a[l+1] + \cdots + a[r]$만 남으므로 이 항등식이 성립한다.

!!! warning "경계의 경우: l = 1"
    $l = 1$이면 공식이 $\text{prefix}(0) = 0$을 셈한다. BIT 질의 함수는 $i = 0$일 때 0을 돌려주어 올바로 다루어야 한다. while 고리의 조건이 `i > 0`이므로 $i = 0$이 들어오면 고리에 들어가지 않고 저절로 0을 돌려준다.

## 한 걸음씩 따라가기

배열 $a = [1, 3, 5, 7, 9]$을 생각하자.

**질의: rangeSum(2, 4)**

$$
\text{rangeSum}(2, 4) = \text{prefix}(4) - \text{prefix}(1)
$$

- $\text{prefix}(4) = a[1]+a[2]+a[3]+a[4] = 1+3+5+7 = 16$
- $\text{prefix}(1) = a[1] = 1$
- $\text{rangeSum}(2, 4) = 16 - 1 = 15$

접두사 질의마다 $O(\log n)$이 들므로 범위 합 질의는 모두 $O(\log n)$이다 (상수 배가 2이지만 두 질의가 같은 BIT를 함께 쓴다).

## 올바름의 논증

범위 합 공식은 접두사 합의 **망원경 성질**에 기댄다.

$$
\sum_{j=l}^{r} a[j] = \sum_{j=1}^{r} a[j] - \sum_{j=1}^{l-1} a[j]
$$

이 항등식은 (양수든 음수든 0이든) 어떤 값에 대해서도, 올바른 어떤 색인 $1 \leq l \leq r \leq n$에 대해서도 성립한다. 요구되는 것은 덧셈이 결합적이고 역원이 있다는 것(곧 아벨 군에서 일한다는 것)뿐이며, 정수와 부동소수점 덧셈이 이를 만족한다.

## 구현

```python
"""
이진 색인 트리로 하는 범위 합 질의.

범위 질의를 접두사 질의 둘로 귀착하는 것을 경계 다루기와
두루 갖춘 시험과 함께 보인다.
"""


# === 범위 질의를 하는 펜윅 트리 ===

class FenwickTree:
    """점 갱신과 범위 합 질의를 받쳐 주는 BIT."""

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i: int, delta: int) -> None:
        """자리 i에 delta를 더한다."""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def prefix(self, i: int) -> int:
        """색인 1부터 i까지 원소의 합을 돌려준다."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def range_sum(self, l: int, r: int) -> int:
        """색인 l부터 r까지(둘 다 포함) 원소의 합을 돌려준다.

        항등식 sum(l, r) = prefix(r) - prefix(l - 1)을 쓴다.
        l = 1이면 prefix(0)이 저절로 0을 돌려준다.
        """
        return self.prefix(r) - self.prefix(l - 1)


# === 시연 ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9]
    n = len(data)
    ft = FenwickTree(n)
    for i, v in enumerate(data, 1):
        ft.update(i, v)

    print(f"Array: {data}")
    print()

    # 여러 가지 범위 질의
    queries = [(1, 3), (2, 5), (2, 4), (1, 1), (1, 5), (3, 3)]
    for l, r in queries:
        result = ft.range_sum(l, r)
        expected = sum(data[l - 1:r])
        print(f"  rangeSum({l}, {r}) = prefix({r}) - prefix({l - 1}) "
              f"= {ft.prefix(r)} - {ft.prefix(l - 1)} = {result}  "
              f"{'OK' if result == expected else 'MISMATCH'}")

    # 갱신한 뒤
    print()
    print("After adding 10 to position 3:")
    ft.update(3, 10)
    data[2] += 10  # 배열에 그대로 담아 둔다
    for l, r in [(1, 3), (2, 5), (3, 3)]:
        result = ft.range_sum(l, r)
        expected = sum(data[l - 1:r])
        print(f"  rangeSum({l}, {r}) = {result}  "
              f"{'OK' if result == expected else 'MISMATCH'}")
```

**출력:**
```
Array: [1, 3, 5, 7, 9]

  rangeSum(1, 3) = prefix(3) - prefix(0) = 9 - 0 = 9  OK
  rangeSum(2, 5) = prefix(5) - prefix(1) = 25 - 1 = 24  OK
  rangeSum(2, 4) = prefix(4) - prefix(1) = 16 - 1 = 15  OK
  rangeSum(1, 1) = prefix(1) - prefix(0) = 1 - 0 = 1  OK
  rangeSum(1, 5) = prefix(5) - prefix(0) = 25 - 0 = 25  OK
  rangeSum(3, 3) = prefix(3) - prefix(2) = 9 - 4 = 5  OK

After adding 10 to position 3:
  rangeSum(1, 3) = 19  OK
  rangeSum(2, 5) = 34  OK
  rangeSum(3, 3) = 15  OK
```

## 복잡도

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| 범위 합 질의 | $O(\log n)$ | $O(1)$ |

범위 합은 저마다 $O(\log n)$인 접두사 질의를 두 번 한다. $O(\log n) + O(\log n) = O(\log n)$이므로 전체 복잡도는 로그 그대로이다.

## 넓히기

!!! tip "한 점의 값 꺼내기"
    원소 하나 $a[i]$을 얻으려면 $\text{rangeSum}(i, i) = \text{prefix}(i) - \text{prefix}(i-1)$을 셈한다. 범위 합 질의와 같은 $O(\log n)$이 든다. $O(1)$에 얻으려면 BIT와 함께 본디 배열의 사본을 따로 지킨다.

## 참고 문헌

- Fenwick, P. M. (1994). A New Data Structure for Cumulative Frequency Tables. *Software: Practice and Experience*, 24(3), 327-336.


## 연습문제

**연습문제 1.**
범위 합 질의의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 범위 합 질의를 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
범위 합 질의가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.