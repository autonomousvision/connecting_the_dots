import numpy as np
cimport cython

# use c square root function
cdef extern from "math.h":
    float sqrt(float x)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

# 3 parameters:
# - float image
# - kernel size (actually this is the radius, kernel is 2*k+1)
# - small constant epsilon that is used to avoid division by zero
def normalize(float[:, :] img, int kernel_size = 4, float epsilon = 0.01):
    
    # image dimensions
    cdef Py_ssize_t M = img.shape[0]
    cdef Py_ssize_t N = img.shape[1]

    # create outputs and output views
    img_lcn  = np.zeros((M, N), dtype=np.float32)
    img_std  = np.zeros((M, N), dtype=np.float32)
    cdef float[:, :] img_lcn_view  = img_lcn
    cdef float[:, :] img_std_view  = img_std
    
    # temporary c variables
    cdef float tmp, mean, stddev
    cdef Py_ssize_t m, n, i, j
    cdef Py_ssize_t ks = kernel_size
    cdef float eps = epsilon
    cdef float num = (ks*2+1)**2

    # for all pixels do
    for m in range(ks,M-ks):
        for n in range(ks,N-ks):
            
            # calculate mean
            mean = 0;
            for i in range(-ks,ks+1):
                for j in range(-ks,ks+1):
                    mean += img[m+i, n+j]
            mean = mean/num
        
            # calculate std dev
            stddev = 0;
            for i in range(-ks,ks+1):
                for j in range(-ks,ks+1):
                    stddev = stddev + (img[m+i, n+j]-mean)*(img[m+i, n+j]-mean)
            stddev = sqrt(stddev/num)

            # compute normalized image (add epsilon) and std dev image
            img_lcn_view[m, n]  = (img[m, n]-mean)/(stddev+eps)
            img_std_view[m, n]  = stddev

    # return both
    return img_lcn, img_std
