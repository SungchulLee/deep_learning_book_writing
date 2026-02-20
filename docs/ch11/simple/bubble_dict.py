"""
Bubble Sort for Dictionary
"""

def swap(a, b, key):
    if a[key] < b[key]:
        return a, b, True # last return item is flag that input is already sorted
    else:
        return b, a, False # last return item is flag that input is already sorted


def first_bubble_up(lst, key):
    flag = True
    for i in range(len(lst)-1):
        lst[i], lst[i+1], flag_temp = swap(lst[i], lst[i+1], key=key)
        flag = flag and flag_temp
    return lst, flag # last return item is flag that input is already sorted


def bubble_sort(lst, key):
    for i in range(len(lst)-1):
        if i == 0:
            lst, flag = first_bubble_up(lst, key=key)
        else:
            lst[:-i], flag = first_bubble_up(lst[:-i], key=key)
        if flag:
            break
    return lst


lst = [
        { 'name': 'mona',   'transaction_amount': 1000, 'device': 'iphone-10'},
        { 'name': 'dhaval', 'transaction_amount': 400,  'device': 'google pixel'},
        { 'name': 'kathy',  'transaction_amount': 200,  'device': 'vivo'},
        { 'name': 'aamir',  'transaction_amount': 800,  'device': 'iphone-8'},
    ]
#bubble_sort(lst,key='name')
#bubble_sort(lst,key='transaction_amount')
bubble_sort(lst,key='device')
