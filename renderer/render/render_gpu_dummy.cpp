#include "render_gpu.h"

template <typename T>
RendererGpu<T>::RendererGpu(const Camera<T> cam, const Shader<T> shader, Buffer<T> buffer) : BaseRenderer<T>(cam, shader, buffer) {
}

template <typename T>
RendererGpu<T>::~RendererGpu() {
}

template <typename T>
void RendererGpu<T>::gpu_to_cpu() {}

template <typename T>
RenderInput<T> RendererGpu<T>::input_to_device(const RenderInput<T> input) { return RenderInput<T>(); }

template <typename T>
void RendererGpu<T>::input_free_device(const RenderInput<T> input) {
  throw std::logic_error("Not implemented");
}

template <typename T>
void RendererGpu<T>::render_mesh(const RenderInput<T> input) {
  throw std::logic_error("Not implemented");
}

template <typename T>
void RendererGpu<T>::render_mesh_proj(const RenderInput<T> input, const Camera<T> proj, const float* pattern, float d_alpha, float d_beta) {
  throw std::logic_error("Not implemented");
}


template class RendererGpu<float>;
