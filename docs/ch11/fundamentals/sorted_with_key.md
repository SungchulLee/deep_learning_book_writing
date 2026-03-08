# Built-in sorted with key

# Sorting Example using Class

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

**Output:**
```
[(Carl, 37, $70000), (John, 43, $90000), (Sarah, 29, $80000)]
```

```python
sorted(employees, key=e_sort, reverse=True)
```

**Output:**
```
[(Sarah, 29, $80000), (John, 43, $90000), (Carl, 37, $70000)]
```

```python
sorted(employees, key=lambda e: e.salary)
```

**Output:**
```
[(Carl, 37, $70000), (Sarah, 29, $80000), (John, 43, $90000)]
```

```python
sorted(employees, key=lambda e: e.salary, reverse=True)
```

**Output:**
```
[(John, 43, $90000), (Sarah, 29, $80000), (Carl, 37, $70000)]
```

# Reference

[Corey Schafer](https://www.youtube.com/watch?v=D3JvDWO-BY4&index=20&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU)
