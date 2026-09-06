# 동적 배열
1 매달의 지출이 아래와 같다고 하자.
```
January - 2200
February - 2350
March - 2600
April - 2130
May - 2190
```
이 월별 지출을 담는 리스트를 만들고 그것을 써서 다음을 알아내라.

1 2월에는 1월보다 몇 달러를 더 썼는가?

2 그해 1분기(첫 세 달)의 총지출을 구하라.

3 어느 달엔가 정확히 2000달러를 썼는지 알아내라

4 6월이 막 끝났고 지출이 1980달러이다. 이 항목을 월별 지출 리스트에 추가하라

5 4월에 산 물건을 반품하여
200달러를 환불받았다. 이를 반영하여 월별 지출 리스트를
고쳐라

```python
expenses = [2200, 2350, 2600, 2130, 2190]
print(expenses)
```

**출력:**
```
[2200, 2350, 2600, 2130, 2190]
```

```python
# 1 2월에는 1월보다 몇 달러를 더 썼는가?
print(expenses[1] - expenses[0])
```

**출력:**
```
150
```

```python
# 2 총지출을 구하라
# 그 해의 1분기(처음 세 달) 동안의 지출이다.
print(sum(expenses[:3]))
```

**출력:**
```
7150
```

```python
# 3 정확히 2000달러를 쓴 달이 있는지 알아내라
print(2000 in expenses)
```

**출력:**
```
False
```

```python
# 4 6월이 막 끝났고 지출은 1980달러이다.
# 이 항목을 월별 지출 목록에 추가하라
expenses.append(1980)
print(expenses)
```

**출력:**
```
[2200, 2350, 2600, 2130, 2190, 1980]
```

```python
# 5 4월에 산 물건을 반품하여
# 200달러를 환불받았다. 이에 맞추어 월별 지출 목록을
# 고쳐라
expenses[3] -= 200 
print(expenses)
```

**출력:**
```
[2200, 2350, 2600, 1930, 2190, 1980]
```

2 좋아하는 마블 슈퍼히어로의 리스트가 있다.
```
heros = ['spider man','thor','hulk','iron man','captain america']
```
이것을 써서 다음을 알아내라.

1 리스트의 길이

2 이 리스트의 끝에 'black panther'를 추가하라

3 'black panther'를 'hulk' 뒤에 넣어야 한다는 것을 깨달았다.
   그러니 먼저 리스트에서 지운 뒤 'hulk' 뒤에 추가하라
   
4 이제 thor와 hulk는 쉽게 화를 내서 마음에 들지 않는다 :)
   그래서 리스트에서 thor와 hulk를 지우고 (멋있으니까) doctor strange로 바꾸려 한다.
   이를 코드 한 줄로 하라.
   
5 히어로 리스트를 알파벳순으로 정렬하라 (힌트: dir() 함수로 리스트에서 쓸 수 있는 함수를 모두 살펴보라)

```python
heros = ['spider man','thor','hulk','iron man','captain america']
print(heros)
```

**출력:**
```
['spider man', 'thor', 'hulk', 'iron man', 'captain america']
```

```python
# 1 목록의 길이
print(len(heros))
```

**출력:**
```
5
```

```python
# 2 이 목록의 끝에 'black panther'를 추가하라
heros.append('black panther')
print(heros)
```

**출력:**
```
['spider man', 'thor', 'hulk', 'iron man', 'captain america', 'black panther']
```

```python
# 3 'black panther'는 'hulk' 뒤에 넣어야 함을 깨달았다.
# 먼저 목록에서 지우고 'hulk' 뒤에 다시 넣어라
heros.pop()
idx = heros.index('hulk')
heros.insert(idx+1, 'black panther')
print(heros)
```

**출력:**
```
['spider man', 'thor', 'hulk', 'black panther', 'iron man', 'captain america']
```

```python
# 4 thor와 hulk는 쉽게 화를 내서 마음에 들지 않는다 :)
# 그래서 목록에서 thor와 hulk를 지우고
# doctor strange로 바꾸려 한다(멋지니까).
# 이를 코드 한 줄로 하라.
heros = ['doctor strange' if (hero == 'thor' or hero == 'hulk') 
         else hero
         for hero in heros]
print(heros)
```

**출력:**
```
['spider man', 'doctor strange', 'doctor strange', 'black panther', 'iron man', 'captain america']
```

```python
# 5 heros 목록을 알파벳순으로 정렬하라
# (힌트. dir() 함수로 list에서 쓸 수 있는 함수를 모두 나열해 보라)
heros.sort()
print(heros)
```

**출력:**
```
['black panther', 'captain america', 'doctor strange', 'doctor strange', 'iron man', 'spider man']
```

3 1과 어떤 최댓값 사이의 모든 홀수로 이루어진 리스트를 만들어라.
최댓값은 input() 함수를 써서 
사용자에게서 입력받아야 한다.

```python
max_num = int(input('Type some posive number : '))
lst = [i for i in range(1, max_num+1) if i%2==1]
print(lst)
```

**출력:**
```
Type some posive number : 20
[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
```

# 참고 문헌

[Arrays - Data Structures & Algorithms Tutorials in Python #3](https://www.youtube.com/watch?v=gDqQf4Ekr2A) [github](https://github.com/codebasics/data-structures-algorithms-python/blob/master/data_structures/2_Arrays/2_arrays_exercise.md)

## 연습문제

**연습문제 1.**
동적 배열의 배로 늘리기 전략을 설명하고, 그것이 왜 상각 $O(1)$ 삽입을 달성하는지 설명하라.

??? success "연습문제 1 풀이"
    배열이 가득 차면 크기가 두 배인 배열을 새로 할당하고 원소를 복사한다. $i$번째 크기 조정은 $2^i$개의 원소를 복사한다. 삽입 $n$번에 대한 총 복사량은 $\sum_{i=0}^{\log n} 2^i = 2n - 1 = O(n)$이다. 상각 비용은 삽입당 $O(n)/n = O(1)$이다.

---

**연습문제 2.**
`append` 연산 한 번의 최악의 경우 시간 복잡도는 얼마인가? 그런 일은 언제 일어나는가?

??? success "연습문제 2 풀이"
    배열이 가득 차서 크기를 조정해야 할 때 $O(n)$이다. (배로 늘린다면) 삽입 $n/2$번마다 한 번 일어나므로 드물다. 상각 비용은 여전히 $O(1)$이다.

---

**연습문제 3.**
성장 인수 2배와 1.5배를 비교하라. 무엇을 주고 무엇을 얻는가?

??? success "연습문제 3 풀이"
    2배는 분석이 더 단순하지만 메모리를 최대 50%까지 낭비한다. 1.5배는 메모리 활용이 더 낫지만(최대 33% 낭비) 상각 상수가 조금 더 크다. 파이썬의 `list`는 크기 조정이 더 잦아지는 대가로 메모리 효율을 더욱 높이려고 약 1.125배의 성장 인수를 쓴다.

---

**연습문제 4.**
`append`, `pop`, `__getitem__` 연산을 갖춘 간단한 동적 배열 클래스를 구현하라.

??? success "연습문제 4 풀이"
    ```python
    class DynamicArray:
        def __init__(self):
            self._data = [None] * 4
            self._size = 0
        def append(self, val):
            if self._size == len(self._data):
                new = [None] * (2 * len(self._data))
                new[:self._size] = self._data[:self._size]
                self._data = new
            self._data[self._size] = val
            self._size += 1
    ```
