# Subset Generation


<img src="img/Screen Shot 2022-05-02 at 6.07.29 PM.png" width=50%>


```python
def print_power_set(initial_set):
    pre = set() 
    post = initial_set 
    print_pre_post(pre, post)
    
    
def print_pre_post(pre, post):
    if len(post) == 0:
        print(pre)
        return
    
    t = post.pop()
    
    pre_add_t = pre.copy()
    pre_add_t.add(t)
    post_copy = post.copy()
    
    print_pre_post(pre, post)
    print_pre_post(pre_add_t, post_copy)
        

def main():
    initial_set = {0,1,2}
    print_power_set(initial_set)
    
        
if __name__ == "__main__":
    main()
```

**Output:**
```
set()
{2}
{1}
{1, 2}
{0}
{0, 2}
{0, 1}
{0, 1, 2}
```


```python
def main():
    initial_set = {0,1,2}
    for i in range(2**len(initial_set)):
        print(f"{format(i, f'0{len(initial_set)}b')}")
    
        
if __name__ == "__main__":
    main()
```

**Output:**
```
000
001
010
011
100
101
110
111
```


```python
def main():
    initial_set = {0,1,2}
    initial_list = list(initial_set)
    for i in range(2**len(initial_set)):
        binary_representation = f"{format(i, '03b')}"
        tmp = set()
        for i, rep in enumerate(binary_representation):
            if rep == "1":
                tmp.add(initial_list[i])
        print(tmp)
    
        
if __name__ == "__main__":
    main()
```

**Output:**
```
set()
{2}
{1}
{1, 2}
{0}
{0, 2}
{0, 1}
{0, 1, 2}
```


<img src="img/Screen Shot 2022-05-02 at 6.50.43 PM.png" width=50%>


# Reference

[[알고리즘] 제2-4강 멱집합 (powerset)](https://www.youtube.com/watch?v=nkeMRRIVW9s&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=7)
