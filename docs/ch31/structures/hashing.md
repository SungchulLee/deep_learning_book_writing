# 바깥 기억 흩기

기억 속 흩기 표는 연산마다 기대 때 $O(1)$을 이루지만 어느 기억 자리든 아무렇게 닿는 것이 똑같이 빠르다고 여긴다. 흩기 표가 원반에 있으면 다른 원반 쪽에 닿는 더듬기마다 들고남 한 번이 든다. **바깥 기억 흩기**는 열쇠를 저마다 원반 덩이 하나를 차지하는 **통**으로 옮겨 흩기를 바깥 기억 모형에 맞추며, 찾기, 넣기, 지우기마다 기대 들고남 $O(1)$번을 이룬다.

## 붙박인 바깥 기억 흩기

가장 단순한 바깥 기억 흩기는 통을 붙박인 $m$개 두고 저마다 열쇠 $B$개를 담는 원반 덩이 하나에 담는다.

흩기 함수가 열쇠마다 통으로 옮긴다:

$$
h(k) = k \bmod m
$$

통마다 원반 쪽 하나를 차지한다. 열쇠 $k$ 찾기는 통 $h(k)$을 담은 쪽 하나를 읽으며 들고남이 꼭 1번이다.

### 넘침 다루기

통이 담는 양 $B$을 넘으면 넘침 쪽을 사슬처럼 잇는다. 찾기가 으뜸 쪽과 넘침 쪽을 함께 읽어야 할 수 있다:

$$
\text{Expected I/O per lookup} = 1 + \frac{\alpha - 1}{B} \text{ (when } \alpha > B\text{)}
$$

여기서 $\alpha = N/m$은 통마다 평균 열쇠 개수다. 기대 들고남을 $O(1)$으로 두려면 $\alpha = O(B)$이어야 하며, 이는 통마다 평균으로 쪽을 상수 개만 담는다는 뜻이다.

## 늘릴 수 있는 흩기

붙박인 흩기는 $m$을 미리 골라야 하는데 $N$이 헤아릴 수 없이 늘면 말썽이다. **늘릴 수 있는 흩기**는 필요할 때 두 배가 되는 바뀌는 목록을 써서 전체를 다시 흩는 일을 피한다.

### 얼개

- 칸이 $2^d$개인 **목록**이며 $d$은 **온 자리 깊이**다.
- 목록 칸마다 **통**(원반 쪽)을 가리킨다.
- 목록 칸 여럿이 같은 통을 가리킬 수 있다.
- 통마다 흩기값의 몇 비트가 그 칸을 가려내는지 나타내는 **그 자리 깊이** $d_b \le d$을 가진다.

### 찾기

열쇠 $k$을 찾으려면:

1. $h(k)$을 셈하고 앞 $d$비트로 목록 번호를 얻는다.
2. 가리개를 따라 통으로 간다.
3. 통을 읽는다(통에 들고남 1번, 목록이 기억에 안 들어가면 1번 더).

### 가르기

통이 넘치면:

1. 통의 그 자리 깊이 $d_b$을 1 올린다.
2. 통을 둘로 가르고 $(d_b)$번째 비트로 열쇠를 다시 나눈다.
3. $d_b > d$이면 목록을 두 배로 늘린다($d$을 1 올린다).

목록을 두 배로 하면 가리개 수가 두 배가 되지만 새 통은 생기지 않는다. 기억에서만 하는 연산이다. 넣기 $N$번의 기대 가르기 횟수는 $O(N/B)$이다.

### 들고남 복잡도

| 연산 | 기대 들고남 |
|---|---|
| 찾기 | $O(1)$(목록이 원반에 있으면 2) |
| 넣기(가르기 없음) | $O(1)$ |
| 넣기(가르기 있음) | 고루 나누어 $O(1)$ |
| 목록 두 배로 늘리기 | $O(2^d / B)$(드물다) |

## 선형 흩기

**선형 흩기**는 목록을 아예 두지 않는다. 대신 미리 정한 선형 차례로 통을 가르며 흩기 함수 무리 $h_0, h_1, h_2, \ldots$을 쓴다:

$$
h_i(k) = k \bmod (2^i \cdot m_0)
$$

그리고 $m_0$은 처음 통의 개수다.

### 가르기 셈속

**가르기 가리개** $p$이 다음에 가를 통을 좇는다. 온 채움 비율이 문턱을 넘으면:

1. 통 $p$의 열쇠를 통 $p$과 새 통 $p + 2^i \cdot m_0$에 다시 나누어 통 $p$을 가른다.
2. $p$을 $p + 1$으로 옮긴다.
3. $p$이 $2^i \cdot m_0$에 이르면 $i$을 1 올리고 $p$을 0으로 되돌린다(새 바퀴).

이점은 가르기가 목록 덧짐 없이 통 하나씩 차례로 일어난다는 것이다.

### 들고남 복잡도

| 연산 | 기대 들고남 |
|---|---|
| 찾기 | $O(1)$ |
| 넣기 | 고루 나누어 $O(1)$ |
| 가르기 | $O(1)$(옛 통 읽기 + 통 둘 적기) |

## 바깥 기억 흩기 방식 견주기

| 성질 | 붙박임 | 늘릴 수 있음 | 선형 |
|---|---|---|---|
| 목록 | 없음 | 칸 $2^d$개 | 없음 |
| 늘어남 다루기 | 아니오(붙박인 $m$) | 예(목록이 두 배로) | 예(선형 가르기) |
| 찾기 들고남 | $O(1)$ | $O(1)$ | 기댓값 $O(1)$ |
| 가장 나쁜 경우 찾기 | 넘침이 있으면 $O(N/m)$ | $O(1)$(넘침 사슬 없음) | 채움을 다스리면 $O(1)$ |
| 공간 쓰임 | 채움 비율에 매임 | 평균 69%쯤 | 문턱으로 다스림 |

## 보기: 바깥 기억 흩기 표 흉내내기

```python
"""
바깥 기억 흩기 흉내내기.

목록 바탕 통 다루기와 연산마다 기대 들고남 O(1)번을 보이는
늘릴 수 있는 흩기를 보여 준다.
"""

import math

# ===================================================================
# 늘릴 수 있는 흩기 표
# ===================================================================

class ExtendibleHashTable:
    """통이 담는 양이 붙박인 늘릴 수 있는 흩기 표."""

    def __init__(self, bucket_capacity: int = 4):
        self.bucket_capacity = bucket_capacity
        self.global_depth = 1
        self.directory = [[] for _ in range(2)]
        self.bucket_depths = [1, 1]
        self.io_count = 0

    def _hash(self, key: int) -> int:
        """넉넉한 비트를 돌려주는 흩기 함수."""
        return hash(key) & ((1 << 32) - 1)

    def _dir_index(self, key: int) -> int:
        """global_depth 비트로 목록 번호를 얻는다."""
        return self._hash(key) & ((1 << self.global_depth) - 1)

    def lookup(self, key: int) -> bool:
        """열쇠를 찾는다. 찾으면 True를 돌려준다."""
        idx = self._dir_index(key)
        self.io_count += 1  # 통 읽기 = 들고남 1번
        return key in self.directory[idx]

    def insert(self, key: int):
        """흩기 표에 열쇠를 넣는다."""
        idx = self._dir_index(key)
        bucket = self.directory[idx]

        if key in bucket:
            return

        if len(bucket) < self.bucket_capacity:
            bucket.append(key)
            self.io_count += 1  # 통 적기
            return

        # 통이 가득 찼다. 갈라야 한다
        local_depth = self.bucket_depths[idx]

        if local_depth == self.global_depth:
            # 목록을 두 배로 늘린다
            self.global_depth += 1
            new_dir = [None] * (1 << self.global_depth)
            new_depths = [0] * (1 << self.global_depth)
            for i in range(len(self.directory)):
                new_dir[i] = self.directory[i]
                new_dir[i + len(self.directory)] = self.directory[i]
                new_depths[i] = self.bucket_depths[i]
                new_depths[i + len(self.directory)] = self.bucket_depths[i]
            self.directory = new_dir
            self.bucket_depths = new_depths

        # 통을 가른다
        new_depth = local_depth + 1
        old_bucket = bucket + [key]
        bucket0 = []
        bucket1 = []

        for k in old_bucket:
            if (self._hash(k) >> local_depth) & 1 == 0:
                bucket0.append(k)
            else:
                bucket1.append(k)

        # 목록 칸을 고친다
        idx = self._dir_index(key)
        step = 1 << new_depth
        base0 = idx & ((1 << new_depth) - 1) & ~(1 << local_depth)
        base1 = base0 | (1 << local_depth)

        for i in range(base0, len(self.directory), step):
            self.directory[i] = bucket0
            self.bucket_depths[i] = new_depth
        for i in range(base1, len(self.directory), step):
            self.directory[i] = bucket1
            self.bucket_depths[i] = new_depth

        self.io_count += 2  # 옛 통 읽기 + 새 통 둘 적기

    @property
    def stats(self) -> dict:
        """흩기 표의 통계를 돌려준다."""
        unique_buckets = len(set(id(b) for b in self.directory))
        total_keys = sum(
            len(b) for b in {id(b): b for b in self.directory}.values()
        )
        return {
            "global_depth": self.global_depth,
            "directory_size": len(self.directory),
            "num_buckets": unique_buckets,
            "total_keys": total_keys,
            "io_count": self.io_count,
        }


# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    B = 4  # 통이 담는 양(덩이 크기)
    ht = ExtendibleHashTable(bucket_capacity=B)

    # 열쇠를 넣는다
    keys = list(range(0, 50, 3))
    for k in keys:
        ht.insert(k)

    s = ht.stats
    print(f"Extendible Hash Table (bucket capacity B={B})")
    print(f"  Keys inserted:  {len(keys)}")
    print(f"  Global depth:   {s['global_depth']}")
    print(f"  Directory size: {s['directory_size']}")
    print(f"  Num buckets:    {s['num_buckets']}")
    print(f"  Total I/Os:     {s['io_count']}")
    print(f"  I/O per insert: {s['io_count'] / len(keys):.2f}")
    print()

    # 찾기 시험
    ht.io_count = 0
    for k in keys:
        assert ht.lookup(k)
    print(f"  Lookups:        {len(keys)}")
    print(f"  Lookup I/Os:    {ht.io_count}")
    print(f"  I/O per lookup: {ht.io_count / len(keys):.2f}")
```

??? example "보기 내놓기"

    ```
    Extendible Hash Table (bucket capacity B=4)
      넣은 열쇠:  17
      온 자리 깊이:   4
      목록 크기: 16
      통 개수:    10
      온 들고남:     27
      넣기마다 들고남: 1.59

      찾기:        17
      찾기 들고남:    17
      찾기마다 들고남: 1.00
    ```

    찾기마다 들고남이 꼭 1번 든다(통 하나 읽기). 넣기는 이따금 가르기 때문에 평균이 조금 더 들지만 고루 나눈 비용은 $O(1)$ 그대로다.

## 언제 바깥 기억 흩기를 쓸까

바깥 기억 흩기는 원반에 놓인 자료의 **점 묻기**(정확한 열쇠 찾기)에 안성맞춤이다. 연산마다 기대 들고남 $O(1)$번을 이루며 이는 가장 좋은 값이다. 그러나 **구간 묻기**는 효율 좋게 받쳐 주지 못한다. 그럴 때는 결과 $K$개를 내는 데 들고남 $O(\log_B N + K/B)$번이 드는 [B 나무](btree.md)를 쓴다.

## 참고 문헌

- Fagin, R. et al. "Extendible Hashing: A Fast Access Method for Dynamic Files," *ACM TODS*, 4(3), 1979.
- Litwin, W. "Linear Hashing: A New Tool for File and Table Addressing," *VLDB*, 1980.
- Vitter, J. S. *Algorithms and Data Structures for External Memory*, 2008.


## 연습문제

**연습문제 1.**
바깥 기억 흩기와 점 묻기의 들고남 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    바깥 기억 흩기는 흩기 함수 $h: \text{열쇠} \to \{1, \ldots, B_N\}$으로 열쇠를 원반 덩이로 옮기며 $B_N = \lceil N/(\alpha B) \rceil$은 통의 개수, $\alpha$은 채움 비율이다. 통마다 원반 덩이 하나다. 점 묻기: 열쇠를 흩고 덩이 하나를 읽는다. 들고남 $O(1)$번(넘침이 없다고 여긴 기댓값). 넣기: 흩고 덩이를 읽고 넣고 적는다. 들고남 $O(1)$번. 점 묻기에서는 B 나무($O(\log_B N)$)보다 낫지만 구간 묻기는 받쳐 주지 못한다.

---

**연습문제 2.**
늘릴 수 있는 흩기를 밝히고 통 넘침을 어떻게 다루는지 말하여라.

??? success "연습문제 2 풀이"
    늘릴 수 있는 흩기는 통이 넘칠 때 크기가 두 배가 되는 목록을 쓰되 넘친 통만 가른다. 목록은 흩기값의 앞 $d$비트를 통으로 옮긴다. (그 자리 깊이가 $d_b$인) 통 $b$이 넘치면, $d_b < d$이면 $b$을 통 둘로 가르고 목록 가리개를 고친다. $d_b = d$이면 목록을 두 배로 늘린 뒤($d \leftarrow d+1$) 가른다. 목록은 기억에 들어가고(크기 $O(2^d)$) 통마다 원반 덩이 하나다. 점 묻기는 늘 들고남 1번이 든다(기억에서 목록 찾기 + 덩이 1개 읽기).

---

**연습문제 3.**
바깥 기억 흩기와 B 나무 색인을 견주어라. 각각 언제 낫는가?

??? success "연습문제 3 풀이"
    바깥 기억 흩기: 점 묻기 $O(1)$, 넣기 $O(1)$. 구간 묻기나 차례 훑기는 못 한다. B 나무: 점 묻기와 넣기가 $O(\log_B N)$이지만 구간 묻기($O(\log_B N + K/B)$)와 차례 훑기를 받쳐 준다. 흩기를 쓸 곳: 정확한 열쇠 찾기(두르기, 겹침 없애기, 흩기 이음). B 나무를 쓸 곳: 구간 묻기, 차례 닿기, 앞자락 찾기. 많은 자료 바탕이 둘 다 쓴다. 으뜸 열쇠 찾기에는 흩기 색인, 구간 조건에는 B 나무 색인을 쓴다.

---

**연습문제 4.**
바깥 기억 흩기는 나눠 익히기의 박아 넣기 표 찾기에 어떻게 쓰이는가?

??? success "연습문제 4 풀이"
    큰 박아 넣기 표(칸이 수십억 개)는 원반에 담거나 여러 기계에 흩는다. 박아 넣기마다 낱것 번호를 흩어 조각이나 덩이를 찾는다. 바깥 기억 흩기 덕에 찾기마다 들고남 $O(1)$번이면 된다. 실제로는 (1) 박아 넣기 표를 흩기 가르기로 여러 SSD에 조각내고, (2) 묶음 찾기를 흩기값으로 줄 세워 조각마다 차례로 읽고, (3) 자주 닿는 박아 넣기를 GPU 기억에 둔다. 박아 넣기 찾기가 성능 병목인 추천 시스템(유튜브, 틱톡)의 등뼈다.