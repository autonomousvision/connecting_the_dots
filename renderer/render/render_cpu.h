#ifndef RENDER_CPU_H
#define RENDER_CPU_H

#include "render.h"



template <typename T>
class RendererCpu : public BaseRenderer<T> {
public:
  const int n_threads;

  RendererCpu(const Camera<T> cam, const Shader<T> shader, Buffer<T> buffer, int n_threads) : BaseRenderer<T>(cam, shader, buffer), n_threads(n_threads) {
  }

  virtual ~RendererCpu() {
  }

  virtual void render_mesh(const RenderInput<T> input);
  virtual void render_mesh_proj(const RenderInput<T> input, const Camera<T> proj, const float* pattern, float d_alpha, float d_beta);
};

#endif
