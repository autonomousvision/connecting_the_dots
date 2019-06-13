#ifndef COMMON_H
#define COMMON_H

#include "co_types.h"
#include <cmath>
#include <algorithm>

#if defined(_OPENMP)
#include <omp.h>
#endif


#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&) = delete;\
  classname& operator=(const classname&) = delete;


template <typename T>
CPU_GPU_FUNCTION
void fill(T* arr, int N, T val) {
  for(int idx = 0; idx < N; ++idx) {
    arr[idx] = val;
  }
}

template <typename T>
CPU_GPU_FUNCTION
void fill_zero(T* arr, int N) {
  for(int idx = 0; idx < N; ++idx) {
    arr[idx] = 0;
  }
}

template <typename T>
CPU_GPU_FUNCTION
inline T distance_euclidean(const T* q, const T* t, int N) {
  T out = 0;
  for(int idx = 0; idx < N; idx++) {
    T diff = q[idx] - t[idx];
    out += diff * diff;
  }
  return out;
}

template <typename T>
CPU_GPU_FUNCTION
inline T distance_l2(const T* q, const T* t, int N) {
  T out = distance_euclidean(q, t, N);
  out = std::sqrt(out);
  return out;
}




template <typename T>
struct FillFunctor {
  T* arr;
  const T val;

  FillFunctor(T* arr, const T val) : arr(arr), val(val) {}
  CPU_GPU_FUNCTION void operator()(const int idx) {
    arr[idx] = val;
  }
};

template <typename T>
CPU_GPU_FUNCTION
T mmin(const T& a, const T& b) {
#ifdef __CUDA_ARCH__
  return min(a, b);
#else
  return std::min(a, b);
#endif
}

template <typename T>
CPU_GPU_FUNCTION
T mmax(const T& a, const T& b) {
#ifdef __CUDA_ARCH__
  return max(a, b);
#else
  return std::max(a, b);
#endif
}

template <typename T>
CPU_GPU_FUNCTION
T mround(const T& a) {
#ifdef __CUDA_ARCH__
  return round(a);
#else
  return round(a);
#endif
}


#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
#endif


template <typename T>
CPU_GPU_FUNCTION
void matomic_add(T* addr, T val) {
#ifdef __CUDA_ARCH__
  atomicAdd(addr, val);
#else
#if defined(_OPENMP)
#pragma omp atomic
#endif
  *addr += val;
#endif
}

#endif
