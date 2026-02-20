# List Method sort with key


```python
a = [-9, 1, 8, 2, -7, 3, 6, 4, 5]
a.sort(key=abs)
a
```

**Output:**
```
[1, 2, 3, 4, 5, 6, -7, 8, -9]
```


```python
a = [-9, 1, 8, 2, -7, 3, 6, 4, 5]
a.sort(key=lambda x: x**2+10*x+5)
print(a)
f = lambda x: x**2+10*x+5
fa = [f(i) for i in a]
print(fa)
```

**Output:**
```
[-7, -9, 1, 2, 3, 4, 5, 6, 8]
[-16, -4, 16, 29, 44, 61, 80, 101, 149]
```


```python
def g(x):
    return x**2+10*x+5 

a = [-9, 1, 8, 2, -7, 3, 6, 4, 5]
a.sort(key=g)
print(a)
ga = [g(i) for i in a]
print(ga)
```

**Output:**
```
[-7, -9, 1, 2, 3, 4, 5, 6, 8]
[-16, -4, 16, 29, 44, 61, 80, 101, 149]
```


```python
import numpy as np
a = [-9, 1, 8, 2, -7, 3, 6, 4, 5]
a.sort(key=np.cos, reverse=True)
print(a)
print(np.cos(np.array(a)))
```

**Output:**
```
[6, -7, 1, 5, 8, 2, 4, -9, 3]
[ 0.96017029  0.75390225  0.54030231  0.28366219 -0.14550003 -0.41614684
 -0.65364362 -0.91113026 -0.9899925 ]
```


# Reference

[Corey Schafer](https://www.youtube.com/watch?v=D3JvDWO-BY4&index=20&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU)
