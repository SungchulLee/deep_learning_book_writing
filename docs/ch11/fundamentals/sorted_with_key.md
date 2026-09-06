# key를 쓰는 붙박이 sorted
# 클래스를 쓰는 정렬 보기

```python
class Employee():
    def __init__(self, name, age, salary):
        self.name = name
        self.age = age
        self.salary = salary

    def __repr__(self):
        return '({}, {}, ${})'.format(self.name, self.age, self.salary)
```

```python
e1 = Employee('Carl', 37, 70000)
e2 = Employee('Sarah', 29, 80000)
e3 = Employee('John', 43, 90000)
```

```python
employees = [e1, e2, e3]
```

```python
if 1: 
    # 이름으로 소팅하기
    def e_sort(emp): 
        return emp.name
elif 1: 
    # 나이로 소팅하기
    def e_sort(emp):
        return emp.age
elif 1:
    # 월급으로 소팅하기
    def e_sort(emp):
        return emp.salary
```

```python
sorted(employees, key=e_sort)
```

**출력:**
```
[(Carl, 37, $70000), (John, 43, $90000), (Sarah, 29, $80000)]
```

```python
sorted(employees, key=e_sort, reverse=True)
```

**출력:**
```
[(Sarah, 29, $80000), (John, 43, $90000), (Carl, 37, $70000)]
```

```python
sorted(employees, key=lambda e: e.salary)
```

**출력:**
```
[(Carl, 37, $70000), (Sarah, 29, $80000), (John, 43, $90000)]
```

```python
sorted(employees, key=lambda e: e.salary, reverse=True)
```

**출력:**
```
[(John, 43, $90000), (Sarah, 29, $80000), (Carl, 37, $70000)]
```

# 참고 문헌

[Corey Schafer](https://www.youtube.com/watch?v=D3JvDWO-BY4&index=20&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU)

## 연습문제

**연습문제 1.**
이 알고리즘의 시간 복잡도는 얼마인가?

??? success "연습문제 1 풀이"
    시간 복잡도는 최악과 평균의 경우 $O(n^2)$, (입력이 이미 정렬되었고 조기 종료를 쓰는) 최선의 경우 $O(n)$이다. 제자리에서 정렬하므로 공간 복잡도는 $O(1)$이다.

---

**연습문제 2.**
입력 `[5, 3, 1, 4, 2]`에서 알고리즘을 따라가며 한 번 훑을 때마다의 상태를 보여라.

??? success "연습문제 2 풀이"
    1회: `[3, 1, 4, 2, 5]`. 2회: `[1, 3, 2, 4, 5]`. 3회: `[1, 2, 3, 4, 5]`. 4회: 자리바꿈이 없어 조기 종료한다.

---

**연습문제 3.**
이 정렬 알고리즘은 안정적인가? 그 까닭을 밝혀라.

??? success "연습문제 3 풀이"
    그렇다. 이 알고리즘은 왼쪽 원소가 오른쪽보다 엄밀히 클 때만 이웃한 원소를 맞바꾸므로 안정적이다. 같은 원소는 결코 맞바꾸지 않아 본디 상대 순서가 지켜진다.

---

**연습문제 4.**
실제로 이 알고리즘을 언제 쓰겠는가?

??? success "연습문제 4 풀이"
    더 복잡한 알고리즘의 짐이 점근적 이점을 넘어서는 아주 작은 배열($n < 20$)에서, 거의 일차에 가까운 성능을 내는 거의 정렬된 데이터에서, 또는 간단해서 가르치는 보기로 쓴다.
