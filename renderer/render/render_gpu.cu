#include "common_cuda.h"
#include "render_gpu.h"

template <typename T>
RendererGpu<T>::RendererGpu(const Camera<T> cam, const Shader<T> shader, Buffer<T> buffer) : BaseRenderer<T>(cam, shader, buffer) {
  if(buffer.depth != nullptr) {
    buffer_gpu.depth = device_malloc<T>(cam.num_pixel());
  }

  if(buffer.color != nullptr) {
    buffer_gpu.color = device_malloc<T>(cam.num_pixel() * 3);
  }

  if(buffer.normal != nullptr) {
    buffer_gpu.normal = device_malloc<T>(cam.num_pixel() * 3);
  }
}

template <typename T>
RendererGpu<T>::~RendererGpu() {
  device_free(buffer_gpu.depth);
  device_free(buffer_gpu.color);
  device_free(buffer_gpu.normal);
}

template <typename T>
void RendererGpu<T>::gpu_to_cpu() {
  if(buffer_gpu.depth != nullptr && this->buffer.depth != nullptr) {
    device_to_host(buffer_gpu.depth, this->buffer.depth, this->cam.num_pixel());
  }
  if(buffer_gpu.color != nullptr && this->buffer.color != nullptr) {
    device_to_host(buffer_gpu.color, this->buffer.color, this->cam.num_pixel() * 3);
  }
  if(buffer_gpu.normal != nullptr && this->buffer.normal != nullptr) {
    device_to_host(buffer_gpu.normal, this->buffer.normal, this->cam.num_pixel() * 3);
  }
}

template <typename T>
RenderInput<T> RendererGpu<T>::input_to_device(const RenderInput<T> input) {
  RenderInput<T> input_gpu;
  input_gpu.n_verts = input.n_verts;
  input_gpu.n_faces = input.n_faces;

  if(input.verts != nullptr) {
    input_gpu.verts = host_to_device_malloc(input.verts, input.n_verts * 3);
  }
  if(input.colors != nullptr) {
    input_gpu.colors = host_to_device_malloc(input.colors, input.n_verts * 3);
  }
  if(input.normals != nullptr) {
    input_gpu.normals = host_to_device_malloc(input.normals, input.n_verts * 3);
  }
  if(input.faces != nullptr) {
    input_gpu.faces = host_to_device_malloc(input.faces, input.n_faces * 3);
  }

  return input_gpu;
}

template <typename T>
void RendererGpu<T>::input_free_device(const RenderInput<T> input) {
  if(input.verts != nullptr) {
    device_free(input.verts);
  }
  if(input.colors != nullptr) {
    device_free(input.colors);
  }
  if(input.normals != nullptr) {
    device_free(input.normals);
  }
  if(input.faces != nullptr) {
    device_free(input.faces);
  }
}


template <typename T>
void RendererGpu<T>::render_mesh(RenderInput<T> input) {
  RenderInput<T> input_gpu = this->input_to_device(input);
  RenderMeshFunctor<T> functor(input_gpu, this->shader, this->cam, this->buffer_gpu);
  iterate_cuda(functor, this->cam.num_pixel());
  gpu_to_cpu();
  this->input_free_device(input_gpu);
}

template <typename T>
void RendererGpu<T>::render_mesh_proj(const RenderInput<T> input, const Camera<T> proj, const float* pattern, float d_alpha, float d_beta) {
  RenderInput<T> input_gpu = this->input_to_device(input);
  float* pattern_gpu = host_to_device_malloc(pattern, proj.num_pixel()*3);

  RenderProjectorFunctor<T> functor(input_gpu, this->shader, this->cam, proj, pattern_gpu, d_alpha, d_beta, this->buffer_gpu);
  iterate_cuda(functor, this->cam.num_pixel());

  gpu_to_cpu();
  this->input_free_device(input_gpu);
  device_free(pattern_gpu);
}

template class RendererGpu<float>;
