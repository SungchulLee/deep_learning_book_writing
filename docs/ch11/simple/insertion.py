"""
Insertion Sort
"""

# ======================================================================

def insert(lst, item):
    """
    input  : lst  : sorted list
             item : item to insert into the given sorted list
    output : out  : new sorted list of lst + [item]
    """
    pointer = len(lst) - 1 # 처음에는 포인터가 주어진 리스트의 맨 마지막 아이템을 포인팅한다.
    while True:
        if pointer >= 0:
            if lst[pointer] > item:
                pointer -= 1 # 포인터는 오른쪽에서 왼쪽으로 한칸씩 움직인다.
            else:
                out = lst[:pointer+1] + [item] + lst[pointer+1:]
                return out 
        else:
            out = [item] + lst
            return out


def insertion_sort(lst):
    """
    input  : lst : unsorted list
    output : out : sorted list of the given unsorted list
    """ 
    
    # Base Cases
    if (len(lst)<=1):
        out = lst
        return out
    if (len(lst)==2) and (lst[0]<=lst[1]):
        out = lst
        return out
    if (len(lst)==2) and (lst[0]>lst[1]):
        out = lst[::-1]
        return out
    
    # Recursion 
    lst[:-1] = insertion_sort(lst[:-1])
    out = insert(lst[:-1], lst[-1])
    return out


# === Main ===
if __name__ == "__main__":
    lst = [-1, 3, 8, -5, 7, 4, 10]
    print(insertion_sort(lst))
