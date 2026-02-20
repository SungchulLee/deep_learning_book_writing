"""
Insertion Sort - In-Place
"""

def insertion_sort(lst):
    for right in range(1,len(lst)): 
        for left in range(right-1,-1,-1): 
            if lst[left] > lst[right]:
                #print(lst, "--->", lst[left], lst[right], "--->", end=' ')
                lst[left], lst[right] = lst[right], lst[left]
                #print(lst)
                right = left   
            else:
                break        
    return lst


lst = [-1, 3, 8, -5, 7, 4, 10]
print(insertion_sort(lst))
