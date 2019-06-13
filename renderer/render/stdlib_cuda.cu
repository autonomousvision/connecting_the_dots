#include "common_cuda.h"
#include "stdlib_cuda.h"

void device_synchronize() {
  cudaDeviceSynchronize();
}

float* device_malloc_f32(long N) {
  return device_malloc<float>(N);
}
int* device_malloc_i32(long N) {
  return device_malloc<int>(N);
}

void device_free_f32(float* dptr) {
  device_free(dptr);
}
void device_free_i32(int* dptr) {
  device_free(dptr);
}

void device_to_host_f32(const float* dptr, float* hptr, long N) {
  device_to_host(dptr, hptr, N);
}
void device_to_host_i32(const int* dptr, int* hptr, long N) {
  device_to_host(dptr, hptr, N);
}

float* host_to_device_malloc_f32(const float* hptr, long N) {
  return host_to_device_malloc(hptr, N);
}

int* host_to_device_malloc_i32(const int* hptr, long N) {
  return host_to_device_malloc(hptr, N);
}
