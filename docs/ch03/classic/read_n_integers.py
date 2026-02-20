"""
Read N Integers
"""

class Read_N_Integers_Using_Recursion:    
    def __init__(self, n, scanner_in_generator):
        self.n = n
        self.generator = scanner_in_generator
        self.data = [None] * (n + 1)
    def read_n_integers_using_recursion(self, k):
        if k == 0:
            return 
        self.read_n_integers_using_recursion(k-1)
        self.data[k] = next(self.generator)    
    def run(self):
        self.read_n_integers_using_recursion(self.n)
        
        
def range_generator(n):
    for i in range(n):
        yield i


def main():
    n = 10
    scanner_in_generator = range_generator(n)
    
    obj = Read_N_Integers_Using_Recursion(n, scanner_in_generator)
    obj.run()
    print(obj.data)
    
        
if __name__ == "__main__":
    main()
