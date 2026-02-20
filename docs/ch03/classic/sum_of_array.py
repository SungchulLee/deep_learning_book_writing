"""
Sum of Array
"""

def compute_sum_of_array_using_recursion(lst):
    if len(lst) == 0:
        return 0    
    return compute_sum_of_array_using_recursion(lst[:-1]) + lst[-1]


def compute_sum_of_array_using_python(lst):
    return sum(lst)


def main():
    lsts = (
    [],
    [0],
    [2,0,],
    [2,1,-1],
    [2,5,-2,4],
    [1,2,3,4,5,6,7,8,9,10]
    )
    for lst in lsts:
        result_0 = compute_sum_of_array_using_recursion(lst)
        result_1 = compute_sum_of_array_using_python(lst)
        print(f"Sum of list {lst} using recursion : {result_0}")
        print(f"Sum of list {lst} using python    : {result_1}")
        print()

        
if __name__ == "__main__":
    main()
