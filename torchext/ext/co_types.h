#ifndef TYPES_H
#define TYPES_H

#ifdef __CUDA_ARCH__
#define CPU_GPU_FUNCTION __host__ __device__
#else
#define CPU_GPU_FUNCTION
#endif

#endif
