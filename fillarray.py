import numpy as np
from timeit import default_timer as timer
from numba import cuda,jit

# CPU
def FillCPU(a):
    for k in range(10000000):
        a[k]+=1

#GPU
@jit(target_backend='cuda')
def FillGPU(a):
    for k in range(10000000):
        a[k]+=1


# Main
a = np.ones(10000000,dtype=np.float64)
start = timer()
FillCPU(a)
print("On a CPU: ",timer()-start)

start = timer()
FillGPU(a)
print("On a GPU: ",timer()-start)