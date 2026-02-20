# Permutations and Combinations


# buggy code


```python
def print_permutation(initial_list, k):
    if len(initial_list) < k:
        return
    
    pre = [] 
    post = initial_list 
    print_permutation_pre_post(pre, post, k)
    
    
def print_permutation_pre_post(pre, post, k):
    if len(pre) == k:
        print(pre)
        return
    
    for i, item in enumerate(post):
        del post[i]      # there are bugs in these two lines
        pre.append(item) # there are bugs in these two lines
        
        print_permutation_pre_post(pre, post, k)
        

def main():
    initial_list = [0,1,2]
    print_permutation(initial_list, 2)
    
        
if __name__ == "__main__":
    main()
```

**Output:**
```
[0, 1]
```


# bug fixed, but too much memory usage


```python
def print_permutation(initial_list, k):
    if len(initial_list) < k:
        return
    
    pre = [] 
    post = initial_list 
    print_permutation_pre_post(pre, post, k)
    
    
def print_permutation_pre_post(pre, post, k):
    if len(pre) == k:
        print(pre)
        return
    
    for i, item in enumerate(post):
        post_copy = post.copy()
        del post_copy[i]
        pre_copy = pre.copy()
        pre_copy.append(item)

        print_permutation_pre_post(pre_copy, post_copy, k)
        

def main():
    initial_list = [0,1,2]
    print_permutation(initial_list, 2)
    
        
if __name__ == "__main__":
    main()
```

**Output:**
```
[0, 1]
[0, 2]
[1, 0]
[1, 2]
[2, 0]
[2, 1]
```


# use global variable to reduce memory usage (in progress)


```python
def print_permutation(i, k):
    """
    i : index you can change lst memebrs
    k : lenth of final printed list
    """
    if i > k:
        print(lst[:k])
        return
    
    for j in range(i,len(lst)):
        swap(i, j) # after all these recursive ops, we would like to have
        print_permutation_after_member_change(i, k) # lst[i:]
        swap(i, j)                                  # unchanged
    
    
def print_permutation_after_member_change(i, k):
    """
    i : index you can change lst memebrs, lst members are already changed
    k : lenth of final printed list
    """
    print_permutation(i+1, k)
    
    
def swap(i, j):
    lst[i], lst[j] = lst[j], lst[i]
        

def main():
    global lst
    lst = [0,1,2]
    print_permutation(0, 2)
    
        
if __name__ == "__main__":
    main()
```

**Output:**
```
[0, 1]
[0, 2]
[1, 0]
[1, 2]
[2, 1]
[2, 0]
```


# Reference

[[알고리즘] 제2-5강 순열(permutation)](https://www.youtube.com/watch?v=MjW10t9ppok&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=8)
