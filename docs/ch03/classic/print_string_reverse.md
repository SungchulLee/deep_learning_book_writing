# 문자열을 거꾸로 출력하기

재귀로 문자열을 거꾸로 출력하는 것은 선형 재귀 패턴의 미묘하지만 중요한 변형을 보여준다. 첫 글자를 처리하고 나머지에 재귀하는 대신, 출력하기 *전에* 재귀 호출을 하면 출력이 역순으로 나온다. "재귀 후 출력"이라는 이 패턴은 후위 순서 처리가 필요한 재귀 알고리즘 곳곳에 나타난다.

## 재귀적 구조

- **기저 사례**: 문자열이 비어 있다 — 출력할 것이 없으므로 반환한다
- **재귀 사례**: 마지막 글자를 출력한 뒤 남은 앞부분을 재귀적으로 거꾸로 출력한다

다른 접근으로는 `string[-1]`을 출력하고 `string[:-1]`에 재귀하여 처리 순서를 뒤집는 방법이 있다.

```python
"""Print a string in reverse using recursion."""


# === 재귀적 역순 출력 ===

def print_reverse_recursive(string):
    """Print each character of string in reverse using recursion."""
    if string == "":
        return
    print(string[-1], end="")
    print_reverse_recursive(string[:-1])


# === 내장 함수와 비교 ===

def print_reverse_builtin(string):
    """Print reversed string using Python slicing."""
    print(string[::-1])


# === 메인 ===

if __name__ == "__main__":
    text = "Recursion breaks a problem into smaller subproblems"
    print("Print string reversely using recursion:")
    print_reverse_recursive(text)
    print()
    print()
    print("Print string reversely using built-in:")
    print_reverse_builtin(text)
```

**출력:**
```
Print string reversely using recursion:
smelborpbus rellams otni melborp a skaerb noisruceR

Print string reversely using built-in:
smelborpbus rellams otni melborp a skaerb noisruceR
```

## 복잡도

점화식은 정방향 출력과 같다.

$$
T(n) = T(n - 1) + O(1), \quad T(0) = O(1)
$$

이는 $O(n)$의 시간 복잡도와 $O(n)$의 스택 공간을 준다. 부분 문자열 슬라이싱은 중간 문자열을 만드는 데 총 $O(n^2)$의 일을 더한다.

## 참고 자료

[Recursion의 개념과 기본 예제들](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)


## 연습문제

**연습문제 1.**
문자열을 거꾸로 출력하기에 대해 기저 사례와 재귀 사례를 찾아라. 모든 유효한 입력에 대해 재귀가 종료됨을 증명하라.

??? success "연습문제 1 풀이"
    기저 사례는 가장 작은 유효한 입력을 직접 처리한다. 재귀 사례는 호출마다 감소하는 잘 정의된 척도로 문제 크기를 줄인다. 이 척도가 (기저 사례의 문턱값으로) 아래로 유계이면서 엄격히 감소하므로 재귀는 반드시 유한한 단계 안에 종료된다.

---

**연습문제 2.**
문자열을 거꾸로 출력하기의 시간 복잡도에 대한 점화식을 유도하고 풀어라.

??? success "연습문제 2 풀이"
    $T(n)$을 재귀 호출과 호출당 일의 함수로 표현한다. 이 점화식은 분기 계수, 부분문제 크기의 감소, 결합 비용을 담는다. 펼치기, 마스터 정리, 치환 중 하나로 풀어 닫힌 형태를 얻는다.

---

**연습문제 3.**
$n = 8$일 때 문자열을 거꾸로 출력하기의 재귀 트리를 그려라. 각 층에서의 일과 전체 일을 계산하라.

??? success "연습문제 3 풀이"
    트리의 깊이는 입력이 기저 사례까지 얼마나 빨리 줄어드는지로 정해진다. 각 층에서 모든 노드의 일을 더한다. 모든 층에 걸친 총합이 실행 시간을 준다. $n = 8$이면 트리가 작아서 전부 열거할 수 있다.

---

**연습문제 4.**
재귀 구현을 반복 버전으로 변환하라. 공간 복잡도를 비교하라.

??? success "연습문제 4 풀이"
    호출 스택을 명시적 스택이나 반복 변수로 대체한다. 꼬리 재귀 형태는 while 반복문으로 곧바로 바뀐다. 꼬리가 아닌 형태는 호출 스택을 흉내 내기 위해 명시적 스택이 필요하다. 반복 버전은 보통 $O(\text{depth})$의 스택 공간을 아낀다.