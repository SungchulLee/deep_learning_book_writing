"""
Bubble Sort with Key
"""

def swap(a, b, key=lambda x: x):
    if key(a) < key(b):
        return a, b, True # last return item is flag that input is already sorted
    else:
        return b, a, False # last return item is flag that input is already sorted


def first_bubble_up(lst, key=lambda x: x):
    flag = True
    for i in range(len(lst)-1):
        lst[i], lst[i+1], flag_temp = swap(lst[i], lst[i+1], key=key)
        flag = flag and flag_temp
    return lst, flag # last return item is flag that input is already sorted


def bubble_sort(lst, key=lambda x: x):
    for i in range(len(lst)-1):
        if i == 0:
            lst, flag = first_bubble_up(lst, key=key)
        else:
            lst[:-i], flag = first_bubble_up(lst[:-i], key=key)
        if flag:
            break
    return lst


lst = [3,-5,2,-7,9]
bubble_sort(lst,key=abs)
