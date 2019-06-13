#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <iostream>
#include <limits>
#include <cmath>

#include "co_types.h"

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void vec_fill(T* v, const T fill) {
  for(int idx = 0; idx < N; ++idx) {
    v[idx] = fill;
  }
}

template <>
CPU_GPU_FUNCTION
inline void vec_fill<float, 3>(float* v, const float fill) {
  v[0] = fill;
  v[1] = fill;
  v[2] = fill;
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void vec_add(const T* in1, const T* in2, T* out) {
  for(int idx = 0; idx < N; ++idx) {
    out[idx] = in1[idx] + in2[idx];
  }
}

template <>
CPU_GPU_FUNCTION
inline void vec_add<float, 3>(const float* in1, const float* in2, float* out) {
  out[0] = in1[0] + in2[0];
  out[1] = in1[1] + in2[1];
  out[2] = in1[2] + in2[2];
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void vec_add(const T lam1, const T* in1, const T lam2, const T* in2, T* out) {
  for(int idx = 0; idx < N; ++idx) {
    out[idx] = lam1 * in1[idx] + lam2 * in2[idx];
  }
}

template <>
CPU_GPU_FUNCTION
inline void vec_add<float, 3>(const float lam1, const float* in1, const float lam2, const float* in2, float* out) {
  out[0] = lam1 * in1[0] + lam2 * in2[0];
  out[1] = lam1 * in1[1] + lam2 * in2[1];
  out[2] = lam1 * in1[2] + lam2 * in2[2];
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void vec_sub(const T* in1, const T* in2, T* out) {
  for(int idx = 0; idx < N; ++idx) {
    out[idx] = in1[idx] - in2[idx];
  }
}

template <>
CPU_GPU_FUNCTION
inline void vec_sub<float, 3>(const float* in1, const float* in2, float* out) {
  out[0] = in1[0] - in2[0];
  out[1] = in1[1] - in2[1];
  out[2] = in1[2] - in2[2];
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void vec_add_scalar(const T* in, const T lam, T* out) {
  for(int idx = 0; idx < N; ++idx) {
    out[idx] = in[idx] + lam;
  }
}

template <>
CPU_GPU_FUNCTION
inline void vec_add_scalar<float, 3>(const float* in, const float lam, float* out) {
  out[0] = in[0] + lam;
  out[1] = in[1] + lam;
  out[2] = in[2] + lam;
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void vec_mul_scalar(const T* in, const T lam, T* out) {
  for(int idx = 0; idx < N; ++idx) {
    out[idx] = in[idx] * lam;
  }
}

template <>
CPU_GPU_FUNCTION
inline void vec_mul_scalar<float, 3>(const float* in, const float lam, float* out) {
  out[0] = in[0] * lam;
  out[1] = in[1] * lam;
  out[2] = in[2] * lam;
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void vec_div_scalar(const T* in, const T lam, T* out) {
  for(int idx = 0; idx < N; ++idx) {
    out[idx] = in[idx] / lam;
  }
}

template <>
CPU_GPU_FUNCTION
inline void vec_div_scalar<float, 3>(const float* in, const float lam, float* out) {
  out[0] = in[0] / lam;
  out[1] = in[1] / lam;
  out[2] = in[2] / lam;
}

template <typename T>
CPU_GPU_FUNCTION
inline void mat_dot_vec3(const T* M, const T* v, T* w) {
  w[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
  w[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
  w[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];
}

template <typename T>
CPU_GPU_FUNCTION
inline void matT_dot_vec3(const T* M, const T* v, T* w) {
  w[0] = M[0] * v[0] + M[3] * v[1] + M[6] * v[2];
  w[1] = M[1] * v[0] + M[4] * v[1] + M[7] * v[2];
  w[2] = M[2] * v[0] + M[5] * v[1] + M[8] * v[2];
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline T vec_dot(const T* in1, const T* in2) {
  T out = T(0);
  for(int idx = 0; idx < N; ++idx) {
    out += in1[idx] * in2[idx];
  }
  return out;
}

template <>
CPU_GPU_FUNCTION
inline float vec_dot<float, 3>(const float* in1, const float* in2) {
  return in1[0] * in2[0] + in1[1] * in2[1] + in1[2] * in2[2];
}

template <typename T>
CPU_GPU_FUNCTION
inline void vec_cross3(const T* u, const T* v, T* out) {
  out[0] = u[1] * v[2] - u[2] * v[1];
  out[1] = u[2] * v[0] - u[0] * v[2];
  out[2] = u[0] * v[1] - u[1] * v[0];
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline T vec_norm(const T* u) {
  T norm = T(0);
  for(int idx = 0; idx < N; ++idx) {
    norm += u[idx] * u[idx];
  }
  return std::sqrt(norm);
}

template <>
CPU_GPU_FUNCTION
inline float vec_norm<float, 3>(const float* u) {
  return std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void vec_normalize(const T* u, T* v) {
  T denom = vec_norm(u);
  vec_div_scalar(u, denom, v);
}

template <>
CPU_GPU_FUNCTION
inline void vec_normalize<float, 3>(const float* u, float* v) {
  vec_div_scalar(u, vec_norm(u), v);
}

template <typename T>
CPU_GPU_FUNCTION
void vertex_normal_3d(const T* a, const T* b, const T* c, T* no) {
  T e1[3];
  T e2[3];
  vec_sub(a, b, e1);
  vec_sub(c, b, e2);
  vec_cross3(e1, e2, no);
}

template <typename T>
CPU_GPU_FUNCTION
bool ray_triangle_intersect_3d(const T* orig, const T* dir, const T* v0, const T* v1, const T* v2, T* t, T* u, T* v, T eps = 1e-6) {
  T v0v1[3];
  vec_sub(v1, v0, v0v1);
  T v0v2[3];
  vec_sub(v2, v0, v0v2);
  T pvec[3];
  vec_cross3(dir, v0v2, pvec);
  T det = vec_dot(v0v1, pvec);

  if(fabs(det) < eps) return false;

  T inv_det = 1 / det;

  T tvec[3];
  vec_sub(orig, v0, tvec);
  *u = vec_dot(tvec, pvec) * inv_det;
  if(*u < 0 || *u > 1) return false;

  T qvec[3];
  vec_cross3(tvec, v0v1, qvec);
  *v = vec_dot(dir, qvec) * inv_det;
  if(*v < 0 || (*u + *v) > 1) return false;

  *t = vec_dot(v0v2, qvec) * inv_det;
  T w = 1 - *u - *v;
  *v = *u;
  *u = w;

  return true;
}

template <typename T>
CPU_GPU_FUNCTION
bool ray_triangle_mesh_intersect_3d(const T* orig, const T* dir, const int* faces, int n_faces, const T* vertices, int* face_idx, T* t, T* u, T* v) {
#ifdef __CUDA_ARCH__
  *t = 1e9;
#else
  *t = std::numeric_limits<T>::max();
#endif
  bool valid = false;
  for(int fidx = 0; fidx < n_faces; ++fidx) {
    const T* v0 = vertices + faces[fidx * 3 + 0] * 3;
    const T* v1 = vertices + faces[fidx * 3 + 1] * 3;
    const T* v2 = vertices + faces[fidx * 3 + 2] * 3;

    T ft, fu, fv;
    bool inter = ray_triangle_intersect_3d(orig, dir, v0,v1,v2, &ft,&fu,&fv);
    if(inter && ft < *t) {
      *face_idx = fidx;
      *t = ft;
      *u = fu;
      *v = fv;
      valid = true;
    }
  }

  return valid;
}

template <typename T>
CPU_GPU_FUNCTION
void reflectance_light_dir(const T* sp, const T* lp, T* l) {
  vec_sub(lp, sp, l);
  vec_normalize(l, l);
}

template <typename T>
CPU_GPU_FUNCTION
T reflectance_lambartian(const T* sp, const T* lp, const T* n) {
  T l[3];
  reflectance_light_dir(sp, lp, l);
  return vec_dot(l, n);
}

template <typename T>
CPU_GPU_FUNCTION
T reflectance_phong(const T* orig, const T* sp, const T* lp, const T* n, const T ka, const T kd, const T ks, const T alpha) {
  T l[3];
  reflectance_light_dir(sp, lp, l);

  T r[3];
  vec_add(2 * vec_dot(l, n), n, -1.f, l, r);
  vec_normalize(r,r); //needed?

  T v[3];
  vec_sub(orig, sp, v);
  vec_normalize(v, v);

  return ka + kd * vec_dot(l, n) + ks * std::pow(vec_dot(r, v), alpha);
}

#endif
