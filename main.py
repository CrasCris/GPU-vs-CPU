import numpy as np
from timeit import default_timer as timer
from numba import vectorize

@vectorize(["float32(float32,float32)"], target ='cuda')

def MultiplyMyVector(a,b):
    return a*b

def main():
    N =64000000
    A = np.ones(N,dtype=np.float32)
    B= np.ones(N,dtype=np.float32)
    C= np.ones(N,dtype=np.float32)

    start= timer()
    C = MultiplyMyVector(A,B)
    vectormultiply_time = timer()-start

    print("C[:6] ="+ str(C[:6]))
    print("C[-6:] =" + str(C[-6:]))
    print("This multiplication took %f seconds" % vectormultiply_time)

main()