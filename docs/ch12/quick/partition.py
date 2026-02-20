"""
Partition
"""

def start_pointer_move(start_pointer, lst):
    """
    input  : start_pointer : start를 포인팅하는 포인터(>=1)
             lst           : unsorted list
    output : start_pointer : 주어진 start_pointer의 리스트 멤버 값을 
                             주어진 리스트의 pivot=lst[0] 값과 비교
                             리스트 멤버 값이 작으면 큰 값이 나올 때까지 한칸씩 오른쪽으로 이동
                             처음으로 리스트 멤버 값이 큰 리스트 멤버 
                             혹은 리스트의 마지막 멤버를 포인팅하는 포인터                       
    """
    pivot_value = lst[0]
    start_value = lst[start_pointer]
    while True:
        if start_value <= pivot_value:
            start_pointer += 1
            try:
                start_value = lst[start_pointer]
            except IndexError:
                start_pointer -= 1
                break       
        else:
            break    
    return start_pointer


def end_pointer_move(end_pointer, lst):
    """
    input  : end_pointer : end를 포인팅하는 포인터(<=len(lst)-1)
             lst         : unsorted list
    output : end_pointer : 주어진 end_pointer의 리스트 멤버 값을 
                           주어진 리스트의 pivot=lst[0] 값과 비교
                           리스트 멤버 값이 크면 작은 값이 나올 때까지 한칸씩 왼쪽으로 이동
                           처음으로 리스트 멤버 값이 작은 리스트 멤버 
                           혹은 리스트의 1번 멤버 (0번 멤버는 pivot임) 를 포인팅하는 포인터                       
    """
    pivot_value = lst[0]
    end_value = lst[end_pointer]
    while True:
        if (end_value >= pivot_value) and (end_pointer>=2):
            end_pointer -= 1
            end_value = lst[end_pointer]      
        else:
            break    
    return end_pointer


def swap_move(start_pointer, end_pointer, lst):
    """
    주어진 리스트의 start_pointer 와 end_pointer 리스트 멤버를 스왑                      
    """
    lst[end_pointer], lst[start_pointer] = lst[start_pointer], lst[end_pointer]
    return lst


def pivot_move(end_pointer, lst):
    """
    주어진 리스트의 end_pointer 와 0 (피봇 포인터) 리스트 멤버를 스왑                      
    """
    lst[0], lst[end_pointer] = lst[end_pointer], lst[0]
    return lst


def partition(lst):
    """
    input  : lst    : unsorted list
    output : left   : 주어진 리스트에서 pivot=lst[0]보다 작은 멤버들을 모아 만든 (정렬이 되지 않은) 리스트   
             center : [pivot] where pivot=lst[0]
             right  : 주어진 리스트에서 pivot=lst[0]보다 큰 멤버들을 모아 만든 (정렬이 되지 않은) 리스트  
    """ 
    
    # Base Cases
    if (len(lst)==2) and (lst[0]<=lst[1]):
        left, center, right = [lst[0]], [lst[1]], []
        return left, center, right 
    if (len(lst)==2) and (lst[0]>lst[1]):
        left, center, right = [lst[1]], [lst[0]], []
        return left, center, right 
    
    # 스타트 포인터와 엔드 포인터 초기화
    start_pointer = 1 
    end_pointer = len(lst) - 1
    
    while start_pointer < end_pointer:
        # 스타트 포인터와 엔드 포인터 업데이트
        start_pointer = start_pointer_move(start_pointer, lst)
        end_pointer = end_pointer_move(end_pointer, lst)
        if start_pointer < end_pointer: 
            # 스타트 포인터와 엔드 포인터가 만나거나 크로스하지 않는 경우
            lst = swap_move(start_pointer, end_pointer, lst)
        elif (start_pointer==end_pointer) and (start_pointer==1): 
            # 스타트 포인터와 엔드 포인터가 만나는 경우 - 만나는 지점 1
            left, center, right = [], [lst[0]], lst[1:]
            break
        elif (start_pointer==end_pointer) and (start_pointer==len(lst)-1): 
            # 스타트 포인터와 엔드 포인터가 만나는 경우 - 만나는 지점 len(lst)-1
            left, center, right = lst[1:], [lst[0]], []
            break
        else:
            # 스타트 포인터와 엔드 포인터가 크로스하는 경우
            lst = pivot_move(end_pointer, lst)
            left, center, right = lst[:end_pointer], [lst[end_pointer]], lst[start_pointer:]
            break
    return left, center, right


def quick_sort(lst):
    """
    input  : lst : unsorted list
    output : out : sorted list of the given unsorted list
    """ 
    
    # Base Cases
    if (len(lst)<=1):
        return lst
    if (len(lst)==2) and (lst[0]<=lst[1]):
        return lst
    if (len(lst)==2) and (lst[0]>lst[1]):
        return lst[::-1]
    
    # Recursion 
    left, center, right = partition(lst)
    if len(left)>=2:
        left = quick_sort(left)
    if len(right)>=2:
        right = quick_sort(right)
    return left + center + right


lst = [-1, 3, 8, -5, 7, 4, 10]
print(quick_sort(lst))
