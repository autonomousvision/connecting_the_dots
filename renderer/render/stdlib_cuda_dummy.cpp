#include "stdlib_cuda.h"

float* device_malloc_f32(long N) { return nullptr; }
int* device_malloc_i32(long N) { return nullptr; }
void device_free_f32(float* dptr) {}
void device_free_i32(int* dptr) {}
float* host_to_device_malloc_f32(const float* hptr, long N) { return nullptr; }
int* host_to_device_malloc_i32(const int* hptr, long N) { return nullptr; }
void device_to_host_f32(const float* dptr, float* hptr, long N) {}
void device_to_host_i32(const int* dptr, int* hptr, long N) {}
