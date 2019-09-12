from numba import jit
import time
@jit(nopython = True)
def foo(x,y):
    s = 0
    for i in range(x,y):
        s += i
    return s
tt = time.time()
print(foo(1,100000000))
print('Time used: {} sec'.format(time.time()-tt))
