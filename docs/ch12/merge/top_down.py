"""
Top-Down
"""

def merge_sort(arr):
    if len(arr)<=1:
        return
    
    mid = len(arr) // 2
    left = arr[:mid] 
    right = arr[mid:] 

    merge_sort(left)
    merge_sort(right)

    merge_sort_two_sorted_arrays(left, right, arr)


def merge_sort_two_sorted_arrays(left, right, arr):
    len_left = len(left) 
    len_right = len(right) 
    i = j = k = 0
    while (i<len_left) and (j<len_right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1 
        else:
            arr[k] = right[j]
            j += 1 
        k += 1
        
    while (i<len_left): 
        arr[k] = left[i]
        i += 1
        k += 1
        
    while (j<len_right): 
        arr[k] = right[j]
        j += 1
        k += 1


if __name__ == '__main__':
    test_cases = (
        [],
        [3],
        [3,5],
        [5,3],
        [3,5,2],
        [5,3,4],
        [10,3,15,7,8,23,98,29],
        [9,8,7,2],
        [1,2,3,4,5]
    )
    for arr in test_cases:
        merge_sort(arr)
        print(arr)
