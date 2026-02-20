"""
Change to Binary
"""

def change_to_binary_and_print_using_recursion(num):
    if num <= 1:
        print(num, end='') 
        return
    
    change_to_binary_and_print_using_recursion(num//2)
    print(num%2, end='') 


def change_to_binary_and_print_using_python(num):
    print(bin(num)[2:])


def main():
    nums = (
    0,
    10,
    2,
    25,
    1256
    )
    for num in nums:
        print(f"Print number as binary using recursion : ", end='')
        change_to_binary_and_print_using_recursion(num)
        print()
        
        print(f"Print number as binary using python    : ", end='')
        change_to_binary_and_print_using_python(num)
        print()

        
if __name__ == "__main__":
    main()
