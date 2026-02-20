# Dynamic Arrays


1 Let us say your expense for every month are listed below,
```
January - 2200
February - 2350
March - 2600
April - 2130
May - 2190
```
Create a list to store these monthly expenses and using that find out,

1 In Feb, how many dollars you spent extra compare to January?

2 Find out your total expense in first quarter (first three months) of the year.

3 Find out if you spent exactly 2000 dollars in any month

4 June month just finished and your expense is 1980 dollar. Add this item to our monthly expense list

5 You returned an item that you bought in a month of April and
got a refund of 200$. Make a correction to your monthly expense list
based on this


```python
expenses = [2200, 2350, 2600, 2130, 2190]
print(expenses)
```

**Output:**
```
[2200, 2350, 2600, 2130, 2190]
```


```python
# 1 In Feb, how many dollars you spent extra compare to January?
print(expenses[1] - expenses[0])
```

**Output:**
```
150
```


```python
# 2 Find out your total expense 
# in first quarter (first three months) of the year.
print(sum(expenses[:3]))
```

**Output:**
```
7150
```


```python
# 3 Find out if you spent exactly 2000 dollars in any month
print(2000 in expenses)
```

**Output:**
```
False
```


```python
# 4 June month just finished and your expense is 1980 dollar. 
# Add this item to our monthly expense list
expenses.append(1980)
print(expenses)
```

**Output:**
```
[2200, 2350, 2600, 2130, 2190, 1980]
```


```python
# 5 You returned an item that you bought in a month of April and
# got a refund of 200$. Make a correction to your monthly expense list
# based on this
expenses[3] -= 200 
print(expenses)
```

**Output:**
```
[2200, 2350, 2600, 1930, 2190, 1980]
```


2 You have a list of your favourite marvel super heros.
```
heros = ['spider man','thor','hulk','iron man','captain america']
```
Using this find out,

1 Length of the list

2 Add 'black panther' at the end of this list

3 You realize that you need to add 'black panther' after 'hulk',
   so remove it from the list first and then add it after 'hulk'
   
4 Now you don't like thor and hulk because they get angry easily :)
   So you want to remove thor and hulk from list and replace them with doctor strange (because he is cool).
   Do that with one line of code.
   
5 Sort the heros list in alphabetical order (Hint. Use dir() functions to list down all functions available in list)


```python
heros = ['spider man','thor','hulk','iron man','captain america']
print(heros)
```

**Output:**
```
['spider man', 'thor', 'hulk', 'iron man', 'captain america']
```


```python
# 1 Length of the list
print(len(heros))
```

**Output:**
```
5
```


```python
# 2 Add 'black panther' at the end of this list
heros.append('black panther')
print(heros)
```

**Output:**
```
['spider man', 'thor', 'hulk', 'iron man', 'captain america', 'black panther']
```


```python
# 3 You realize that you need to add 'black panther' after 'hulk', 
# so remove it from the list first and then add it after 'hulk'
heros.pop()
idx = heros.index('hulk')
heros.insert(idx+1, 'black panther')
print(heros)
```

**Output:**
```
['spider man', 'thor', 'hulk', 'black panther', 'iron man', 'captain america']
```


```python
# 4 Now you don't like thor and hulk because they get angry easily :) 
# So you want to remove thor and hulk from list and 
# replace them with doctor strange (because he is cool). 
# Do that with one line of code.
heros = ['doctor strange' if (hero == 'thor' or hero == 'hulk') 
         else hero
         for hero in heros]
print(heros)
```

**Output:**
```
['spider man', 'doctor strange', 'doctor strange', 'black panther', 'iron man', 'captain america']
```


```python
# 5 Sort the heros list in alphabetical order 
# (Hint. Use dir() functions to list down all functions available in list)
heros.sort()
print(heros)
```

**Output:**
```
['black panther', 'captain america', 'doctor strange', 'doctor strange', 'iron man', 'spider man']
```


3 Create a list of all odd numbers between 1 and a max number.
Max number is something you need to take from a user 
using input() function.


```python
max_num = int(input('Type some posive number : '))
lst = [i for i in range(1, max_num+1) if i%2==1]
print(lst)
```

**Output:**
```
Type some posive number : 20
[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
```


# Reference

[Arrays - Data Structures & Algorithms Tutorials in Python #3](https://www.youtube.com/watch?v=gDqQf4Ekr2A) [github](https://github.com/codebasics/data-structures-algorithms-python/blob/master/data_structures/2_Arrays/2_arrays_exercise.md)
