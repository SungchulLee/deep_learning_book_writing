"""
Heap Property
"""

from heapq import heapify, heappop, heappush


# === Functions ===



# === Main ===
if __name__ == "__main__":
    lst = [4,5,1,2,3]
    lst


    # min heap is default
    # 내부적으로는 인덱스 0에서 시작해 k번째 원소가 항상 자식 원소들(2k+1, 2k+2) 보다 작거나 같은 최소 힙의 형태로 정렬된다.
    heapify(lst) 
    lst


    a = heappop(lst) 
    print(a, lst)


    b = heappop(lst) 
    print(b, lst)


    heappush(lst,1) 
    print(lst)
