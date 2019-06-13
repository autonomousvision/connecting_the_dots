#include <limits>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "render_cpu.h"
#include "common_cpu.h"

template <typename T>
void RendererCpu<T>::render_mesh(RenderInput<T> input) {
  RenderMeshFunctor<T> functor(input, this->shader, this->cam, this->buffer);
  iterate_omp_cpu(functor, this->cam.num_pixel(), n_threads);
}

template <typename T>
void RendererCpu<T>::render_mesh_proj(const RenderInput<T> input, const Camera<T> proj, const float* pattern, float d_alpha, float d_beta) {
  RenderProjectorFunctor<T> functor(input, this->shader, this->cam, proj, pattern, d_alpha, d_beta, this->buffer);
  iterate_omp_cpu(functor, this->cam.num_pixel(), this->n_threads);
}

template class RendererCpu<float>;
