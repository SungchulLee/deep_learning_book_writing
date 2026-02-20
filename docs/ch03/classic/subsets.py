"""
Subset Generation
"""

def print_power_set(initial_set):
    pre = set() 
    post = initial_set 
    print_pre_post(pre, post)
    
    
def print_pre_post(pre, post):
    if len(post) == 0:
        print(pre)
        return
    
    t = post.pop()
    
    pre_add_t = pre.copy()
    pre_add_t.add(t)
    post_copy = post.copy()
    
    print_pre_post(pre, post)
    print_pre_post(pre_add_t, post_copy)
        

def main():
    initial_set = {0,1,2}
    print_power_set(initial_set)
    
        
if __name__ == "__main__":
    main()


def main():
    initial_set = {0,1,2}
    for i in range(2**len(initial_set)):
        print(f"{format(i, f'0{len(initial_set)}b')}")
    
        
if __name__ == "__main__":
    main()


def main():
    initial_set = {0,1,2}
    initial_list = list(initial_set)
    for i in range(2**len(initial_set)):
        binary_representation = f"{format(i, '03b')}"
        tmp = set()
        for i, rep in enumerate(binary_representation):
            if rep == "1":
                tmp.add(initial_list[i])
        print(tmp)
    
        
if __name__ == "__main__":
    main()
