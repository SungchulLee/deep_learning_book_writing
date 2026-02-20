"""
Selection Sort
"""

class Selection_Sort:
    
    def __init__(self, lst):
        self.lst = lst.copy()
        self.last = len(self.lst) - 1 # last of left, boundary between left and right
        self.max = None # index of max in self.lst[:self.last+1]
        
    def find_max(self):
        max_value = max(self.lst[:self.last+1])
        self.max = self.lst.index(max_value)

    def move_max_to_right(self):
        self.lst[self.last], self.lst[self.max] = self.lst[self.max], self.lst[self.last]
        self.last -= 1
        
    def run(self):
        while self.last > 0:
            self.find_max()
            self.move_max_to_right()
        return self.lst

    
def main():
    lsts = [
        [],
        [1,2,3,4],
        [4,3,2,1],
        [1,3,2,4],
        [2,4,1,3],
        [1,2,3,4,4],
        [4,3,4,2,1],
        [1,4,3,4,2],
        [2,4,4,1,3],
    ]
    for lst in lsts:
        obj = Selection_Sort(lst)
        sorted_lst = obj.run()
        print(lst, '--->', sorted_lst)

    
if __name__ == "__main__":
    main()
