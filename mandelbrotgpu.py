#Mandelbrot set -> f(z) = z^2 + c 
from __future__ import print_function,division,absolute_import
import numpy as np
from matplotlib.pylab import imshow,show
from timeit import default_timer as timer
from numba import cuda

@cuda.jit(device=True)
def mandelbrot(x,y,max_iters):
    c = complex(x,y)
    z = 0.0j
    i= 0

    for i in range(max_iters):
        z=z * z +c
        if(z.real * z.real + z.imag * z.imag)>=4:
            return i
        return 255

@cuda.jit
def create_fractal(min_x,max_x,min_y,max_y,image,iters):
    width = image.shape[1]
    height = image.shape[0]

    pixel_size_x = (max_x - min_x)/width
    pixel_size_y = (max_y - min_y)/height

    x,y =cuda.grid(2)

    if x< width and y < height:
        real = min_x + x*pixel_size_x
        imag = min_y + y*pixel_size_y
        color = mandelbrot(real,imag,iters)
        image[y,x]=color

image = np.zeros((500*10*2,750*10*2),dtype=np.uint8)

pixels = 500*10*2*750*10*2
nthreads= 32
nblocksy=((500*10*2)//nthreads)+1
nblocksx=((750*10*2)//nthreads)+1

s=timer()
create_fractal[(nblocksx,nblocksy),(nthreads,nthreads)](
    -2.0,1.0,-1.0,1.0,image,20
)
e=timer()
print("Mandelbrot on CPU: %f seconds" % (e-s))
imshow(image)
show()