# N-여왕(쌓임)
```python
n = 10

queens_locations = [None for i in range(n+1)]

def check_feasibility(level):
    for i in range(1, level):
        if queens_locations[i] == queens_locations[level]:
            return False # 될 수 없다
        if abs( queens_locations[i] - queens_locations[level] ) == level - i:
            return False # 될 수 없다
    return True # 될 수 있다

def queens(level, verbose):
    # 바탕 경우
    if not check_feasibility(level):
        if verbose:
            print("base cases : cannot place n queens :", level, cols)
        return False # 여왕 n개를 놓을 수 없다
    elif level == n:
        if verbose:
            print("base cases : can place n queens    :", level, cols)
        return True # 여왕 n개를 놓을 수 있다
    
    # 되돌이 경우
    for i in range(1, n+1):
        queens_locations[level+1] = i
        if verbose:
            print("recursion cases                    :", level+1, cols)
        if queens(level+1, verbose):
            return True # 여왕 n개를 놓을 수 있다
    return False # 여왕 n개를 놓을 수 없다

def main():
    
    # return True # 여왕 n개를 놓을 수 있다
    # return False # 여왕 n개를 놓을 수 없다
    result = queens(0, verbose=0)
    print()
    print(result)
    
    # 참이면 cols에 여왕의 자리 앎이 들어 있다
    if result:
        print(queens_locations)
    
    
if __name__ == "__main__":
    main()
```

**출력:**
```

참
[None, 1, 3, 6, 8, 10, 5, 9, 2, 4, 7]
```

# 참고 문헌

[[알고리즘] 제2-3강 Recursion의 응용: n queens problem](https://www.youtube.com/watch?v=xKGbWC-DPT4&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=6)

## 연습문제

**연습문제 1.**
되짚기와 가지 뻗어 묶기의 차이를 설명하라. 가지 뻗어 묶기는 언제 나은가?

??? success "연습문제 1 풀이"
    **되짚기**는 상태 공간 나무를 짜임새 있게 살피며 제약을 어기는 갈래를 쳐 낸다(될 수 있는지 살피기). 옳은 풀이를 모두 찾거나 처음 하나를 찾는다. **가지 뻗어 묶기**는 가장 좋게 하기를 더한다. 곧 아래 나무마다 가장 좋을 수 있는 풀이의 묶음을 셈해 그 묶음이 여태 가장 좋은 풀이보다 나쁘면 갈래를 쳐 낸다. 아무 옳은 풀이가 아니라 가장 좋은 풀이가 필요한 가장 좋게 하기 문제(떠돌이 장수, 배낭)에서는 가지 뻗어 묶기가 낫다. 묶는 함수로 찾기 나무의 훨씬 큰 몫을 쳐 낼 수 있다.

---

**연습문제 2.**
N-여왕 문제의 상태 공간 나무를 적고 부딪침을 어떻게 효율 좋게 알아내는지 설명하라.

??? success "연습문제 2 풀이"
    상태 공간 나무의 깊이는 $N$이다(가로줄마다 한 층). 층마다 그 가로줄의 여왕이 놓일 세로줄을 고른다. 마디마다 자식이 최대 $N$개이다(세로줄마다 하나). 부딪침은 다음을 살펴 알아낸다. (1) **세로줄 부딪침**: 세로줄 $c$을 이미 썼다. (2) **대각선 부딪침**: $|\text{row}_1 - \text{row}_2| = |\text{col}_1 - \text{col}_2|$. 효율 좋게 좇으려면 참거짓 배열 셋 `cols[c]`, `diag1[row+col]`, `diag2[row-col]`을 써서 여왕을 놓을 때마다 $O(1)$에 부딪침을 살핀다.

---

**연습문제 3.**
되돌이 대신 또렷한 쌓임을 써서 N-여왕 풀개를 짜라.

??? success "연습문제 3 풀이"
    ```python
    def n_queens_stack(n):
        solutions = []
        stack = [(0, [])]  # (가로줄, 여왕 세로줄)
        while stack:
            row, queens = stack.pop()
            if row == n:
                solutions.append(queens[:])
                continue
            for col in range(n):
                if all(col != qc and abs(row - qr) != abs(col - qc)
                       for qr, qc in enumerate(queens)):
                    stack.append((row + 1, queens + [col]))
        return solutions
    ```
    또렷한 쌓임은 (가로줄, 어중간한 풀이) 짝의 목록으로 되돌이 부름 쌓임을 그대로 흉내낸다.

---

**연습문제 4.**
배정 문제의 가지 뻗어 묶기에서 묶는 함수는 어떻게 도는가?

??? success "연습문제 4 풀이"
    배정 문제에서는 값 행렬 $C$을 두고 일꾼 $n$명에게 일 $n$가지를 맡긴다. 어중간한 배정(앞선 $k$명을 맡긴 상태)의 아래 묶음은 (이미 맡긴 일꾼의 실제 값) + (남은 가로줄마다 아직 맡기지 않은 세로줄로 좁힌 최소 값의 합)이다. 이 묶음은 $O((n-k) \cdot (n-k))$ 시간에 셈한다. 이 아래 묶음이 여태 가장 좋은 온전한 배정 값을 넘으면 그 아래 나무 전체를 쳐 낸다. 더 빡빡한 묶음은 아직 맡기지 않은 일꾼과 일의 줄인 값 행렬에 헝가리 방법을 쓴다.
