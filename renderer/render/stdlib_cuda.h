#ifndef STDLIB_CUDA
#define STDLIB_CUDA

void device_synchronize();

float* device_malloc_f32(long N);
int* device_malloc_i32(long N);

void device_free_f32(float* dptr);
void device_free_i32(int* dptr);

float* host_to_device_malloc_f32(const float* hptr, long N);
int* host_to_device_malloc_i32(const int* hptr, long N);

void device_to_host_f32(const float* dptr, float* hptr, long N);
void device_to_host_i32(const int* dptr, int* hptr, long N);

#endif
