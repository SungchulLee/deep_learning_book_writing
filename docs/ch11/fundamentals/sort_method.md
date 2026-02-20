# List Method sort


```python
a = [-9, 1, 8, 2, -7, 3, 6, 4, 5]
a.sort()
a
```

**Output:**
```
[-9, -7, 1, 2, 3, 4, 5, 6, 8]
```


```python
a = [-9, 1, 8, 2, -7, 3, 6, 4, 5]
a.sort(reverse=True)
a
```

**Output:**
```
[8, 6, 5, 4, 3, 2, 1, -7, -9]
```


List Method sort는 리스트 자체를 inplace로 소트합니다.
inplace로 소트한다는 표시로 None을 return합니다.


```python
a = [9, 1, 8, 2, 7, 3, 6, 4, 5]
a = a.sort() 
print(a)
```

**Output:**
```
None
```


# Reference

[Corey Schafer](https://www.youtube.com/watch?v=D3JvDWO-BY4&index=20&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU)
