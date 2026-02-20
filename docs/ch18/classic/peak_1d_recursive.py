"""
1D Peak - Recursion
"""

def compare(left, center, right):
    if left > center: # 왼쪽이 중앙보다 크면 왼쪽 영역 (left 포함) 에서 찿는다.
        return 'left'
    elif right > center: # 오른쪽이 중앙보다 크면 오른쪽 영역 (right 포함) 에서 찿는다.
        return 'right'
    else: # 왼쪽과 오른쪽 모두 중앙보다 작거나 같으면 center가 찿고자하는 1D Peak 이다.
        return 'center'


def peak(lst):
    
    # Base Cases
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return lst[0]
    elif (len(lst) == 2) and (lst[0] >= lst[1]):
        return lst[0] 
    elif len(lst) == 2:
        return lst[1] 
    
    # Recursion
    i = int(len(lst)/2) # center index
    left = lst[i-1]
    center = lst[i]
    right = lst[i+1]
    result = compare(left,center, right)
    
    if result == 'left':
        return peak(lst[:i]) 
    elif result == 'right':
        return peak(lst[i+1:])
    else:
        return center


#lst = [1,2,3,4,5]
#lst = [5,4,3,2,1]
lst = [1,2,3,4,5,4,3,2,1]
peak(lst)
