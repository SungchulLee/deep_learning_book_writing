"""
N-Queens
"""

n = 10

queens_locations = [None for i in range(n+1)]


def check_feasibility(level):
    for i in range(1, level):
        if queens_locations[i] == queens_locations[level]:
            return False # not feasible
        if abs( queens_locations[i] - queens_locations[level] ) == level - i:
            return False # not feasible
    return True # feasibile


def queens(level, verbose):
    # base cases
    if not check_feasibility(level):
        if verbose:
            print("base cases : cannot place n queens :", level, cols)
        return False # cannot place n queens
    elif level == n:
        if verbose:
            print("base cases : can place n queens    :", level, cols)
        return True # can place n queens
    
    #recursion cases
    for i in range(1, n+1):
        queens_locations[level+1] = i
        if verbose:
            print("recursion cases                    :", level+1, cols)
        if queens(level+1, verbose):
            return True # can place n queens
    return False # cannot place n queens


def main():
    
    # return True # can place n queens
    # return False # cannot place n queens
    result = queens(0, verbose=0)
    print()
    print(result)
    
    # If True, cols has info on queens location
    if result:
        print(queens_locations)
    
    
if __name__ == "__main__":
    main()
