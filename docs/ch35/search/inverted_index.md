# 뒤집은 색인

글월 뭉치가 주어지면 찾기 엔진은 물음 낱말을 담은 글월을 모두 빠르게 찾아야 한다. 막무가내로 하면 물음마다 온 글월을 훑으므로 $O(N \cdot D)$이 든다. 여기서 $N$은 글월 개수이고 $D$은 글월의 평균 길이다. **뒤집은 색인**은 뭉치를 미리 다듬어, 물음이 온 뭉치 크기가 아니라 들어맞는 글월 개수에 견주는 때에 돌게 한다.

## 얼개

뒤집은 색인은 낱말마다 **딸림 목록**, 곧 그 낱말을 담은 글월 번호를 매긴 목록을 맞댄다.

- **사전**: 서로 다른 온 낱말의 해시 표나 매긴 배열.
- **딸림 목록**: 낱말 $t$마다 $t$이 나오는 글월 번호의 목록 $[d_1, d_2, \ldots, d_k]$.

때에 따라 딸림 항목마다 **낱말 잦기** $\text{tf}(t, d)$과 $d$ 안에서 $t$이 나오는 자리를 함께 갈무리한다.

## 세우기

낱말 조각이 모두 $T$개인 글월 $N$개가 주어지면:

1. 글월마다 낱말로 **조각낸다**.
2. (낱말, 글월번호) 짝마다 그 낱말의 딸림 목록에 글월 번호를 더한다.
3. 딸림 목록을 글월 번호로 매긴다.

$$
T_{\text{세우기}} = O(T \log T), \quad S = O(T)
$$

## 물음 다루기

### 낱말 하나 물음

사전에서 낱말을 찾아 그 딸림 목록을 돌려준다. 해시 표를 쓰면 다음과 같다.

$$
T_{\text{하나 물음}} = O(1 + |\text{딸림}|)
$$

### 불 AND 물음

낱말 $t_1, t_2$에 대해 두 딸림 목록의 사귐을 구한다. 목록이 매겨져 있으므로 아우르기 바탕 사귐을 쓴다.

$$
T_{\text{AND}} = O(|P_1| + |P_2|)
$$

여기서 $|P_i|$은 낱말 $t_i$의 딸림 목록 길이다.

### 불 OR 물음

딸림 목록을 아우른다(합집합).

$$
T_{\text{OR}} = O(|P_1| + |P_2|)
$$

## TF-IDF 점수 매기기

걸맞음으로 열매에 등수를 매기려 (낱말, 글월) 짝마다 **TF-IDF** 점수를 매긴다.

$$
\text{tf-idf}(t, d) = \text{tf}(t, d) \cdot \log \frac{N}{\text{df}(t)}
$$

여기서 $\text{tf}(t, d)$은 낱말 $t$이 글월 $d$에 나오는 횟수, $\text{df}(t)$은 $t$을 담은 글월 개수, $N$은 온 글월 개수다.

IDF 인자 $\log(N / \text{df}(t))$은 ("the" 같은) 흔한 낱말의 무게를 낮추고 드물면서 알림이 많은 낱말의 무게를 높인다.

!!! tip "코사인 닮음"
    물음 $q$과 글월 $d$의 닮음을 셈하려면 둘을 TF-IDF 벡터로 나타내고 코사인 닮음을 셈한다. $\cos(q, d) = \frac{q \cdot d}{\|q\| \cdot \|d\|}$.

## 구현

```python
"""
뒤집은 색인 -- 세우기, 불 물음, TF-IDF 점수 매기기.

글월 뭉치에서 뒤집은 색인을 세우고, 불 AND/OR 물음을 받쳐 주며,
TF-IDF 점수로 열매에 등수를 매긴다.
"""

from __future__ import annotations
import math
from collections import defaultdict


# === 뒤집은 색인 ==============================================================

class InvertedIndex:
    """TF-IDF 점수 매기기를 곁들인 뒤집은 색인."""

    def __init__(self):
        self.index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_count = 0
        self.doc_lengths: dict[int, int] = {}

    def add_document(self, doc_id: int, text: str) -> None:
        """글월을 조각내고 딸림 목록을 세워 색인에 넣는다."""
        tokens = text.lower().split()
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_count += 1

        # 낱말 잦기를 센다
        tf: dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        for term, freq in tf.items():
            self.index[term].append((doc_id, freq))

    def search_and(self, terms: list[str]) -> list[int]:
        """불 AND: 온 낱말을 모두 담은 글월 번호를 돌려준다."""
        if not terms:
            return []
        sets = []
        for term in terms:
            term = term.lower()
            doc_ids = {doc_id for doc_id, _ in self.index.get(term, [])}
            sets.append(doc_ids)
        result = sets[0]
        for s in sets[1:]:
            result &= s
        return sorted(result)

    def search_or(self, terms: list[str]) -> list[int]:
        """불 OR: 낱말 가운데 하나라도 담은 글월 번호를 돌려준다."""
        result: set[int] = set()
        for term in terms:
            term = term.lower()
            for doc_id, _ in self.index.get(term, []):
                result.add(doc_id)
        return sorted(result)

    def tfidf_rank(self, query: list[str]) -> list[tuple[int, float]]:
        """물음 낱말에 대한 TF-IDF 점수로 글월에 등수를 매긴다."""
        scores: dict[int, float] = defaultdict(float)
        for term in query:
            term = term.lower()
            postings = self.index.get(term, [])
            if not postings:
                continue
            df = len(postings)
            idf = math.log(self.doc_count / df)
            for doc_id, tf in postings:
                scores[doc_id] += tf * idf

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked


# === 메인 =====================================================================

if __name__ == "__main__":
    idx = InvertedIndex()

    documents = {
        0: "the quick brown fox jumps over the lazy dog",
        1: "the fox hunts the rabbit in the forest",
        2: "a lazy dog sleeps in the sun",
        3: "the quick rabbit runs from the fox",
    }

    for doc_id, text in documents.items():
        idx.add_document(doc_id, text)

    print(f"Indexed {idx.doc_count} documents\n")

    # 불 물음
    and_result = idx.search_and(["fox", "the"])
    print(f"AND('fox', 'the'): docs {and_result}")

    or_result = idx.search_or(["lazy", "rabbit"])
    print(f"OR('lazy', 'rabbit'): docs {or_result}")

    # TF-IDF 등수 매기기
    print("\nTF-IDF ranking for 'fox rabbit':")
    for doc_id, score in idx.tfidf_rank(["fox", "rabbit"]):
        print(f"  Doc {doc_id}: score={score:.3f}  \"{documents[doc_id][:40]}...\"")
```

**출력:**

```
Indexed 4 documents

AND('fox', 'the'): docs [0, 1, 3]
OR('lazy', 'rabbit'): docs [0, 1, 2, 3]

TF-IDF ranking for 'fox rabbit':
  Doc 1: score=0.981  "the fox hunts the rabbit in the forest..."
  Doc 3: score=0.981  "the quick rabbit runs from the fox..."
  Doc 0: score=0.288  "the quick brown fox jumps over the lazy ..."
```

글월 1과 3이 "fox"와 "rabbit"을 모두 담아 등수가 가장 높다. 글월 0은 "fox"만 담아 점수가 낮다. TF-IDF 점수는 낱말이 드문 정도를 옳게 비춘다. (글월 넷 가운데 둘에만 나오는) "rabbit"이 흔한 낱말보다 점수에 더 이바지한다.

## 참고 문헌

- Manning, C.D., Raghavan, P., and Schutze, H. *Introduction to Information Retrieval*. Cambridge University Press, 2008
- Zobel, J. and Moffat, A. "Inverted Files for Text Search Engines." *ACM Computing Surveys*, 2006

## 연습문제

**연습문제 1.**
다음 글월 셋에 대한 뒤집은 색인을 세워라: D1="the cat sat", D2="the dog sat", D3="cat and dog". 낱말마다 딸림 목록을 보여라.

??? success "연습문제 1 풀이"
    낱말과 딸림 목록(글월 번호): "the" -> [D1, D2], "cat" -> [D1, D3], "sat" -> [D1, D2], "dog" -> [D2, D3], "and" -> [D3]. 물음 "cat AND dog"에 답하려면 딸림 목록 [D1, D3]과 [D2, D3]의 사귐을 구해 [D3]을 얻는다. 글월 D3이 두 낱말을 모두 담는다. 낱말 잦기를 덧붙이면 "the" -> [(D1,1), (D2,1)], "cat" -> [(D1,1), (D3,1)]처럼 된다. 이러면 불 물음뿐 아니라 TF-IDF 등수 매기기도 받쳐 준다. $\square$

---

**연습문제 2.**
길이가 $m$과 $n$($m \le n$)인, 매긴 딸림 목록 둘의 사귐을 구하는 값싼 알고리즘을 밝혀라. 때 복잡도는 얼마인가?

??? success "연습문제 2 풀이"
    아우르기 바탕 사귐을 쓴다. 목록마다 손가락질을 하나씩 두고 지금 원소를 견준다. 같으면 열매에 더하고 손가락질 둘을 함께 옮긴다. 왼쪽이 작으면 왼쪽 손가락질을, 아니면 오른쪽 손가락질을 옮긴다. 때: $O(m + n)$. $m \ll n$이면 두 갈래 찾기가 더 낫다. 짧은 목록의 원소마다 긴 목록에서 두 갈래로 찾는다. 때: $O(m \log n)$. $m \ll n / \log n$일 때 더 좋다. 맞춰 가는 길로는 달려가며 찾기(지수 찾기 + 두 갈래 찾기)가 있다. 짧은 목록의 원소마다 긴 목록을 지수로 더듬어 언저리를 찾은 뒤 두 갈래로 찾는다. 어림 경우 $O(m \log(n/m))$으로, 두 끝점 사이를 매끄럽게 잇는다. $\square$

---

**연습문제 3.**
딸림 목록을 옥죄면 뒤집은 색인의 곳간 크기가 줄어드는 까닭을 풀어라. 차이 엮기와 너비가 들쭉날쭉한 바이트 엮기를 밝혀라.

??? success "연습문제 3 풀이"
    딸림 목록은 매긴 글월 번호를 갈무리하며 그 값이 한결같이 늘어난다. **차이 엮기**: 절대 번호 대신 잇단 번호의 차이를 갈무리한다. 목록 [3, 5, 20, 21, 23]이면 차이는 [3, 2, 15, 1, 2]다. 차이는 더 작은 수이므로 비트가 덜 든다. **너비가 들쭉날쭉한 바이트 엮기**: 차이마다 들쭉날쭉한 바이트 수로 엮는다. 바이트마다 가장 높은 비트를 이어짐 깃발로 쓴다(1 = 바이트가 더 있음, 0 = 마지막 바이트). 작은 차이(128 미만)는 1바이트를 쓰고 큰 차이는 2~4바이트를 쓴다. 차이 [3, 2, 15, 1, 2]는 저마다 1바이트(자료 비트 7개)에 들어가 온통 5바이트이며, 32비트 정수로 하면 20바이트다. 옥죄기 비는 누리 잣대 색인에서 흔히 4~8배다. 더 사나운 얼개(PForDelta, Simple-9, SIMD 바탕)는 더 빨리 풀면서도 더 좋은 비를 이룬다. $\square$

---

**연습문제 4.**
어떤 찾기 엔진이 글월 100억 개를 색인한다. 낱말 "the"이 글월 80억 개에 나온다. 이 낱말을 남달리 다루어야 하는 까닭과, 멈춤말 다루기가 색인 크기와 물음 됨됨이에 미치는 바를 따져라.

??? success "연습문제 4 풀이"
    "the"의 딸림 목록에는 항목이 80억 개 있어 옥죄어도 약 8 GB를 차지한다. 이 낱말 하나가 색인에서 어울리지 않게 큰 몫을 차지한다. 다루는 길: (1) **멈춤말 없애기**: "the"을 색인에서 아예 뺀다. 색인 크기가 크게 준다. "the"이 든 물음은 그 낱말을 눈여겨보지 않는다(보기로 "the matrix"이 그저 "matrix"가 된다). 멈춤말이 뜻을 지니는 물음("The Who", "to be or not to be")에서는 그른 열매를 낳을 수 있다. (2) **켜 있는 색인**: "the"을 색인하되 더 낮은 켜에 갈무리한다. 사귐 물음(AND)에서는 온 글월이 들어맞으므로 "the"의 딸림 목록을 건너뛴다. 자리가 대수로운 어구 물음이나 가까움 물음에서만 닿는다. (3) **잦기로 쳐 내기**: "the"의 딸림 목록에 등수가 높은 글월만 남긴다(보기로 페이지랭크 상위 100만). 등수가 낮은 글월은 등수 매기기에 보탬이 되지 않기 때문이다. $\square$

---

**연습문제 5.**
뒤집은 색인과 바로 선 색인을 견주어라. 각각 언제 알맞으며 찾기 엔진은 둘을 어떻게 함께 쓰는가?

??? success "연습문제 5 풀이"
    **뒤집은 색인**: 낱말을 글월에 맞댄다. 물음 낱말을 담은 글월을 찾는 데 값싸다(낱말마다 $O(1)$ 찾기 + 딸림 목록 훑기). 물음 다루기에 꼭 있어야 한다. **바로 선 색인**: 글월을 그 낱말(자리, 잦기와 함께)에 맞댄다. 글월 켜의 특징(보기로 글월 길이, 어느 글월의 낱말 잦기)을 셈하는 데 값싸다. 점수 매기기와 미리보기 글 짓기에 꼭 있어야 한다. 찾기 엔진은 둘을 함께 쓴다. (1) 뒤집은 색인이 물음에 들어맞는 후보 글월을 가려낸다(되찾기 마디). (2) 바로 선 색인이 후보마다 자세한 걸맞음 점수를 셈하고(등수 매기기 마디) 찾기 열매에 보이는 미리보기 글을 짓는다. 뒤집은 색인이 더 큰 얼개로(누리 잣대 엔진에서 약 100 TB) 원반이나 SSD에 갈무리된다. 바로 선 색인은 더 작고(돌려주는 글월만 점수를 매기면 된다) 흔히 기억에 맞대어 쓴다. $\square$
