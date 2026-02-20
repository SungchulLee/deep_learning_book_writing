"""
Bubble Sort - Efficient
"""

class Bubble_Sort:
    
    def __init__(self, lst):
        self.lst = lst.copy()
        self.last = len(self.lst) - 1 # last of left, boundary between left and right
        self.idx = 0 # index of swap between self.lst[self.idx] and self.lst[self.idx+1]
        
        self.flag_move_bubble = False # flag that bubble is moved
        self.flag_sorted = False # flag that self.lst is sorted
        
    def move_bubble_once(self):
        if self.lst[self.idx] > self.lst[self.idx+1]:
            self.lst[self.idx], self.lst[self.idx+1] = self.lst[self.idx+1], self.lst[self.idx]
            self.flag_move_bubble = True
        self.idx += 1

    def move_bubble(self):
        while self.idx < self.last:
            self.move_bubble_once()
        if not self.flag_move_bubble: # if there is no bubble move,
            self.flag_sorted = True   # self.lst is already sorted
            return                    # get out of the move_bubble method
        self.last -= 1
        self.idx = 0
        self.flag_move_bubble = False
        
    def run(self):
        while self.last > 0:
            self.move_bubble()
            if self.flag_sorted: # if self.lst is already sorted,
                return self.lst  # return self.lst
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
        obj = Bubble_Sort(lst)
        sorted_lst = obj.run()
        print(lst, '--->', sorted_lst)

    
if __name__ == "__main__":
    main()
