# 문자열 출력하기

문자열을 한 글자씩 출력하는 것은 수열에 대한 재귀의 가장 단순한 예 중 하나이다. 재귀적 접근은 첫 글자를 처리한 뒤 남은 부분 문자열에 대해 재귀한다. 원소 하나를 처리하고 나머지에 재귀하는 이 패턴이 수열에 대한 선형 재귀의 토대이다.

## 재귀적 구조

- **기저 사례**: 문자열이 비어 있다 — 출력할 것이 없으므로 반환한다
- **재귀 사례**: 첫 글자를 출력한 뒤 나머지를 재귀적으로 출력한다

```python
"""Print a string character by character using recursion."""


# === 재귀적 출력 ===

def print_string_recursive(string):
    """Print each character of string using recursion."""
    if string == "":
        return
    print(string[0], end="")
    print_string_recursive(string[1:])


# === 내장 함수와 비교 ===

def print_string_builtin(string):
    """Print string using Python built-in."""
    print(string)


# === 메인 ===

if __name__ == "__main__":
    text = "Recursion breaks a problem into smaller subproblems"
    print("Print string using recursion:")
    print_string_recursive(text)
    print()
    print()
    print("Print string using built-in:")
    print_string_builtin(text)
```

**출력:**
```
Print string using recursion:
Recursion breaks a problem into smaller subproblems

Print string using built-in:
Recursion breaks a problem into smaller subproblems
```

## 복잡도

각 재귀 호출은 글자 하나를 처리하고 길이 $n - 1$의 새 부분 문자열을 만든다.

$$
T(n) = T(n - 1) + O(1), \quad T(0) = O(1)
$$

이는 $O(n)$의 시간 복잡도를 준다. 공간 복잡도는 재귀 스택을 위한 $O(n)$에, 부분 문자열 복사를 위한 총 $O(n^2)$이 더해진다(파이썬의 문자열 슬라이싱이 새 문자열을 만들기 때문이다).

## 참고 자료

[Recursion의 개념과 기본 예제들](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)


## 연습문제

**연습문제 1.**
문자열 출력하기에 대해 기저 사례와 재귀 사례를 찾아라. 모든 유효한 입력에 대해 재귀가 종료됨을 증명하라.

??? success "연습문제 1 풀이"
    기저 사례는 가장 작은 유효한 입력을 직접 처리한다. 재귀 사례는 호출마다 감소하는 잘 정의된 척도로 문제 크기를 줄인다. 이 척도가 (기저 사례의 문턱값으로) 아래로 유계이면서 엄격히 감소하므로 재귀는 반드시 유한한 단계 안에 종료된다.

---

**연습문제 2.**
문자열 출력하기의 시간 복잡도에 대한 점화식을 유도하고 풀어라.

??? success "연습문제 2 풀이"
    $T(n)$을 재귀 호출과 호출당 일의 함수로 표현한다. 이 점화식은 분기 계수, 부분문제 크기의 감소, 결합 비용을 담는다. 펼치기, 마스터 정리, 치환 중 하나로 풀어 닫힌 형태를 얻는다.

---

**연습문제 3.**
$n = 8$일 때 문자열 출력하기의 재귀 트리를 그려라. 각 층에서의 일과 전체 일을 계산하라.

??? success "연습문제 3 풀이"
    트리의 깊이는 입력이 기저 사례까지 얼마나 빨리 줄어드는지로 정해진다. 각 층에서 모든 노드의 일을 더한다. 모든 층에 걸친 총합이 실행 시간을 준다. $n = 8$이면 트리가 작아서 전부 열거할 수 있다.

---

**연습문제 4.**
재귀 구현을 반복 버전으로 변환하라. 공간 복잡도를 비교하라.

??? success "연습문제 4 풀이"
    호출 스택을 명시적 스택이나 반복 변수로 대체한다. 꼬리 재귀 형태는 while 반복문으로 곧바로 바뀐다. 꼬리가 아닌 형태는 호출 스택을 흉내 내기 위해 명시적 스택이 필요하다. 반복 버전은 보통 $O(\text{depth})$의 스택 공간을 아낀다.