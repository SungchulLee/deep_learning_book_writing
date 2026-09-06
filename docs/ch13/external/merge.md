# 바깥 병합 정렬

보통의 정렬 알고리즘은 데이터셋 전체가 주 기억에 들어간다고 놓는다. 데이터가 너무 클 때, 이를테면 램이 4GB인 기계에서 100GB짜리 기록 파일을 다룰 때는 디스크(또는 다른 보조 저장 장치)를 써서 정렬해야 한다. 디스크 입출력은 기억을 훑는 것보다 자릿수 단위로 느리므로, 알고리즘의 목표는 견줌을 줄이는 데서 **디스크 읽기와 쓰기** 횟수를 줄이는 데로 옮겨 간다. **바깥 병합 정렬**이 이런 상황의 고전적인 알고리즘이다.

## 바깥 기억 모형

The analysis uses the **external memory (I/O) model** with three parameters:

| 기호 | 뜻 |
|--------|---------|
| $N$ | 전체 원소 수 |
| $M$ | 기억에 들어가는 원소 수 |
| $B$ | 디스크 블록(페이지)마다의 원소 수 |

입출력 연산 한 번은 원소 $B$개짜리 블록 하나를 읽거나 쓴다. 목표는 입출력 연산의 총 횟수를 가장 작게 하는 것이다.

## 알고리즘 훑어보기

바깥 병합 정렬은 두 단계로 나아간다.

### 1단계 -- 런 만들기

1. 원소 $M$개를 기억으로 읽어 들인다.
2. 기억 안에서 도는 효율적인 정렬(이를테면 빠른 정렬)로 정렬한다.
3. 정렬된 원소 $M$개짜리 **런**을 디스크에 다시 쓴다.
4. 원소 $N$개를 모두 다룰 때까지 되풀이한다.

이는 저마다 길이가 많아야 $M$인 정렬된 런 $\lceil N / M \rceil$개를 낸다.

**1단계의 입출력 비용:** 원소마다 한 번 읽고 한 번 쓰므로 모두 $2 \lceil N / B \rceil$번의 입출력이다.

### 2단계 -- 병합 훑기

정렬된 런을 쌍으로 병합한다(두 갈래 병합).

1. 읽을 입력 런 둘과 쓸 출력 파일 하나를 연다.
2. 두 런의 맨 앞 원소를 거듭 견주어 더 작은 것을 출력에 쓰고, 그쪽 입력을 한 칸 민다.
3. 병합 훑기를 한 번 하면 런의 수가 반으로 줄고 런마다 길이가 두 배가 된다.
4. 정렬된 런 하나만 남을 때까지 되풀이한다.

병합 훑기마다 원소 $N$개를 모두 읽고 쓰므로 $2 \lceil N / B \rceil$번의 입출력이 든다. 훑기의 수는 $\lceil \log_2 (N / M) \rceil$이다.

## 입출력 복잡도

$$
\text{I/O}(N, M, B) = O\!\left(\frac{N}{B} \log_2 \frac{N}{M}\right)
$$

$O(\log_2 (N/M))$번의 훑기마다 $O(N/B)$번의 입출력을 한다.

이는 (다음 절에서 다루는) **여러 갈래 병합**으로 크게 나아질 수 있는데, 로그의 밑을 2에서 $M/B - 1$으로 올린다.

$$
\text{I/O}(N, M, B) = O\!\left(\frac{N}{B} \log_{M/B} \frac{N}{M}\right)
$$

## 풀이 예제

$M = 1{,}000$, $B = 100$으로 원소 $N = 10{,}000$개를 정렬해 보자.

| 단계 | 런 수 | 런 길이 | 훑기마다의 입출력 |
|-------|------|-----------|---------------|
| 런 만들기 | 10 | 1,000 | 200 |
| 병합 훑기 1 | 5 | 2,000 | 200 |
| 병합 훑기 2 | 3 | 4,000 | 200 |
| 병합 훑기 3 | 2 | 8,000 / 2,000 | 200 |
| 병합 훑기 4 | 1 | 10,000 | 200 |

모두: $200 + 4 \times 200 = 1{,}000$번의 입출력이다(원소마다 따로 접근했다면 $100{,}000$번이었을 것이다).

## 이중 버퍼 두기

실전에서 쓰는 손질로 **이중 버퍼 두기**가 있다. 기억 속의 입력 블록 하나를 다루는 동안 다음 블록을 디스크에서 따로 읽어 온다. 이렇게 하면 셈과 입출력이 겹쳐 디스크가 쉼 없이 일한다.

## 구현

```python
"""
바깥 병합 정렬 — 기억 공간보다 큰 자료를 위한 두 갈래 병합.

파일을 써서 바깥 정렬을 흉내 낸다. 실제 제품에서는 기억 사상 파일이나
버퍼를 다루는 곧바른 디스크 입출력을 쓸 것이다.
시간:  견줌 O(N log(N/M))번
입출력: 블록 옮김 O((N/B) * log_2(N/M))번
"""

import heapq
import tempfile
import os


# === 런 만들기 ==============================================================

def _create_sorted_runs(
    data: list[int], memory_size: int, temp_dir: str
) -> list[str]:
    """자료를 크기 *memory_size*의 정렬된 런으로 쪼갠다.

    정렬된 런을 하나씩 담은 파일 경로의 목록을 되돌린다.
    """
    runs = []
    for start in range(0, len(data), memory_size):
        chunk = sorted(data[start : start + memory_size])
        path = os.path.join(temp_dir, f"run_{len(runs)}.txt")
        with open(path, "w") as f:
            for val in chunk:
                f.write(f"{val}\n")
        runs.append(path)
    return runs


# === 두 갈래 병합 =============================================================

def _merge_two_runs(path_a: str, path_b: str, output_path: str) -> str:
    """정렬된 런 파일 둘을 정렬된 날 파일 하나로 병합한다."""
    with open(path_a) as fa, open(path_b) as fb, open(output_path, "w") as out:
        a = fa.readline()
        b = fb.readline()
        while a and b:
            if int(a) <= int(b):
                out.write(a)
                a = fa.readline()
            else:
                out.write(b)
                b = fb.readline()
        # 남은 원소 쓰기
        while a:
            out.write(a)
            a = fa.readline()
        while b:
            out.write(b)
            b = fb.readline()
    return output_path


# === 바깥 병합 정렬 ===========================================================

def external_merge_sort(data: list[int], memory_size: int) -> list[int]:
    """주어진 기억 공간 제약 아래 바깥 병합 정렬로 *data*을 정렬한다.

    매개변수
    ----------
    data : list[int]
        입력 자료(큰 파일을 흉내 낸다).
    memory_size : int
        기억 공간에 들어가는 원소의 최대 개수.

    반환값
    -------
    list[int]
        정렬된 자료.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # 단계 1: 정렬된 런 만들기
        runs = _create_sorted_runs(data, memory_size, temp_dir)

        # 단계 2: 병합 훑기
        pass_num = 0
        while len(runs) > 1:
            next_runs = []
            for i in range(0, len(runs), 2):
                if i + 1 < len(runs):
                    out_path = os.path.join(
                        temp_dir, f"pass{pass_num}_merge{i}.txt"
                    )
                    _merge_two_runs(runs[i], runs[i + 1], out_path)
                    next_runs.append(out_path)
                else:
                    next_runs.append(runs[i])
            runs = next_runs
            pass_num += 1

        # 마지막 정렬된 런 읽기
        with open(runs[0]) as f:
            return [int(line) for line in f]


# === 시연 ===================================================================

if __name__ == "__main__":
    import random

    random.seed(42)
    data = random.sample(range(10000), 100)
    memory_size = 20  # 작은 기억 공간 흉내내기

    sorted_data = external_merge_sort(data, memory_size)
    print(f"Input (first 10):  {data[:10]}")
    print(f"Sorted (first 10): {sorted_data[:10]}")
    print(f"Correctly sorted:  {sorted_data == sorted(data)}")
    print(f"Runs created:      {len(data) // memory_size + (1 if len(data) % memory_size else 0)}")
    print(f"Merge passes:      {(len(data) // memory_size - 1).bit_length()}")
```

**출력:**
```
Input (first 10):  [4575, 7562, 7326, 1040, 6498, 8802, 2848, 2813, 7147, 4280]
Sorted (first 10): [12, 37, 75, 105, 127, 153, 175, 239, 242, 252]
Correctly sorted:  True
Runs created:      5
Merge passes:      3
```

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 8장. MIT Press.
- Vitter, J. S. (2001). External memory algorithms and data structures: dealing with massive data. *ACM Computing Surveys*, 33(2), 209-271.


## 연습문제

**연습문제 1.**
바깥 병합 정렬의 핵심 생각과 그 시간·공간 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 견줌 너머의 성질(정수 열쇠, 바깥 저장 장치, 병렬 하드웨어)을 살려 써서 비교 기반 정렬이 따라올 수 없는 성능을 이룬다. 구체적인 복잡도 한계는 이 쪽에서 뜯어본다.

---

**연습문제 2.**
작은 입력에서 바깥 병합 정렬을 따라가라. 훑기나 단계마다 보여라.

??? success "연습문제 2 풀이"
    원소 6~8개에 알고리즘을 적용하며 훑을 때마다의 상태를 보여라. 이 따라가기가 알고리즘의 얼개를 드러내고 옳음을 눈에 보이게 한다.

---

**연습문제 3.**
어떤 조건에서 비교 기반 정렬보다 바깥 병합 정렬이 나은가?

??? success "연습문제 3 풀이"
    다음일 때 낫다. 입력의 정수 범위가 묶여 있을 때(세기 정렬과 기수 정렬), 데이터가 램을 넘칠 때(바깥 정렬), 병렬 하드웨어를 쓸 수 있을 때(병렬 정렬). 이런 조건에서는 알고리즘이 비교 기반의 아래 한계 $\Omega(n\log n)$을 비껴갈 수 있다.

---

**연습문제 4.**
바깥 병합 정렬이 실전에서 이득을 주는 깊은 학습 응용을 서술하라.

??? success "연습문제 4 풀이"
    응용: 어휘를 찾기 위한 토큰 번호 정렬($O(n + V)$의 세기 정렬), GPU 기억을 넘치는 데이터셋의 바깥 정렬, GPU에서 배치 연산을 위한 병렬 정렬. 특정 조건(정수 열쇠, 큰 데이터, 병렬성)이 갖추어질 때 이득이 가장 크다.