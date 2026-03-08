"""
List Method sort with key
"""



# === Main ===
if __name__ == "__main__":
    a = [-9, 1, 8, 2, -7, 3, 6, 4, 5]
    a.sort(key=abs)
    a


    a = [-9, 1, 8, 2, -7, 3, 6, 4, 5]
    a.sort(key=lambda x: x**2+10*x+5)
    print(a)
    f = lambda x: x**2+10*x+5
    fa = [f(i) for i in a]
    print(fa)


    # ======================================================================

    def g(x):
        return x**2+10*x+5 

    a = [-9, 1, 8, 2, -7, 3, 6, 4, 5]
    a.sort(key=g)
    print(a)
    ga = [g(i) for i in a]
    print(ga)


    import numpy as np
    a = [-9, 1, 8, 2, -7, 3, 6, 4, 5]
    a.sort(key=np.cos, reverse=True)
    print(a)
    print(np.cos(np.array(a)))
