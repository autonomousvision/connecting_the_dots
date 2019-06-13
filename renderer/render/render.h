#ifndef RENDER_H
#define RENDER_H

#include <cmath>
#include <algorithm>

#include "co_types.h"
#include "common.h"
#include "geometry.h"


template <typename T>
struct Camera {
  const T fx;
  const T fy;
  const T px;
  const T py;
  const T R0, R1, R2, R3, R4, R5, R6, R7, R8;
  const T t0, t1, t2;
  const T C0, C1, C2;
  const int height;
  const int width;

  Camera(const T fx, const T fy, const T px, const T py, const T* R, const T* t, int width, int height) :
    fx(fx), fy(fy), px(px), py(py),
    R0(R[0]), R1(R[1]), R2(R[2]), R3(R[3]), R4(R[4]), R5(R[5]), R6(R[6]), R7(R[7]), R8(R[8]),
    t0(t[0]), t1(t[1]), t2(t[2]),
    C0(-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2])),
    C1(-(R[1] * t[0] + R[4] * t[1] + R[7] * t[2])),
    C2(-(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])),
    height(height), width(width)
  {
  }

  CPU_GPU_FUNCTION
  inline void to_cam(const T* x, T* y) const {
    y[0] = R0 * x[0] + R1 * x[1] + R2 * x[2] + t0;
    y[1] = R3 * x[0] + R4 * x[1] + R5 * x[2] + t1;
    y[2] = R6 * x[0] + R7 * x[1] + R8 * x[2] + t2;
  }

  CPU_GPU_FUNCTION
  inline void to_world(const T* x, T* y) const {
    y[0] = R0 * (x[0] - t0) + R3 * (x[1] - t1) + R6 * (x[2] - t2);
    y[1] = R1 * (x[0] - t0) + R4 * (x[1] - t1) + R7 * (x[2] - t2);
    y[2] = R2 * (x[0] - t0) + R5 * (x[1] - t1) + R8 * (x[2] - t2);
  }

  CPU_GPU_FUNCTION
  inline void to_ray(const int h, const int w, T* dir) const {
    T uhat[2];
    uhat[0] = (w - px) / fx;
    uhat[1] = (h - py) / fy;
    dir[0] = R0 * (uhat[0]) + R3 * (uhat[1]) + R6;
    dir[1] = R1 * (uhat[0]) + R4 * (uhat[1]) + R7;
    dir[2] = R2 * (uhat[0]) + R5 * (uhat[1]) + R8;
  }

  CPU_GPU_FUNCTION
  inline void to_2d(const T* xyz, T* u, T* v, T* d) const {
    T xyz_t[3];
    to_cam(xyz, xyz_t);
    *u = fx * xyz_t[0] + px * xyz_t[2];
    *v = fy * xyz_t[1] + py * xyz_t[2];
    *d = xyz_t[2];
    *u /= *d;
    *v /= *d;
  }

  CPU_GPU_FUNCTION
  inline void get_C(T* C) const {
    C[0] = C0;
    C[1] = C1;
    C[2] = C2;
  }

  CPU_GPU_FUNCTION
  inline int num_pixel() const {
    return height * width;
  }
};


template <typename T>
struct RenderInput {
  T* verts;
  T* colors;
  T* normals;
  int n_verts;
  int* faces;
  int n_faces;

  RenderInput() : verts(nullptr), colors(nullptr), normals(nullptr), n_verts(0), faces(nullptr), n_faces(0) {}
};

template <typename T>
struct Buffer {
  T* depth;
  T* color;
  T* normal;

  Buffer() : depth(nullptr), color(nullptr), normal(nullptr) {}
};

template <typename T>
struct Shader {
  const T ka;
  const T kd;
  const T ks;
  const T alpha;

  Shader(T ka, T kd, T ks, T alpha) : ka(ka), kd(kd), ks(ks), alpha(alpha) {}

  CPU_GPU_FUNCTION
  T operator()(const T* orig, const T* sp, const T* lp, const T* norm) const {
    return reflectance_phong(orig, sp, lp, norm, ka, kd, ks, alpha);
  }
};



template <typename T>
class BaseRenderer {
public:
  const Camera<T> cam;
  const Shader<T> shader;
  Buffer<T> buffer;

  BaseRenderer(const Camera<T> cam, const Shader<T> shader, Buffer<T> buffer) : cam(cam), shader(shader), buffer(buffer) {
  }

  virtual ~BaseRenderer() {}

  virtual void render_mesh(const RenderInput<T> input) = 0;
  virtual void render_mesh_proj(const RenderInput<T> input, const Camera<T> proj, const float* pattern, float d_alpha, float d_beta) = 0;
};



template <typename T>
struct RenderFunctor {
  const Camera<T> cam;
  const Shader<T> shader;
  Buffer<T> buffer;

  RenderFunctor(const Camera<T> cam, const Shader<T> shader, Buffer<T> buffer) : cam(cam), shader(shader), buffer(buffer) {}
};


template <typename T>
struct RenderMeshFunctor : public RenderFunctor<T> {
  const RenderInput<T> input;

  RenderMeshFunctor(const RenderInput<T> input, const Shader<T> shader, const Camera<T> cam, Buffer<T> buffer) : RenderFunctor<T>(cam, shader,buffer), input(input) {
  }

  CPU_GPU_FUNCTION void operator()(const int idx) {
    int h = idx / this->cam.width;
    int w = idx % this->cam.width;

    T orig[3];
    this->cam.get_C(orig);
    T dir[3];
    this->cam.to_ray(h, w, dir);

    int face_idx;
    T t, tu, tv;
    bool valid = ray_triangle_mesh_intersect_3d(orig, dir, this->input.faces, this->input.n_faces, this->input.verts, &face_idx, &t, &tu, &tv);

    if(this->buffer.depth != nullptr) {
      this->buffer.depth[idx] = valid ? t : -1;
    }

    if(!valid) {
      if(this->buffer.color != nullptr) {
        this->buffer.color[idx * 3 + 0] = 0;
        this->buffer.color[idx * 3 + 1] = 0;
        this->buffer.color[idx * 3 + 2] = 0;
      }
      if(this->buffer.normal != nullptr) {
        this->buffer.normal[idx * 3 + 0] = 0;
        this->buffer.normal[idx * 3 + 1] = 0;
        this->buffer.normal[idx * 3 + 2] = 0;
      }
    }
    else if(this->buffer.normal != nullptr || this->buffer.color != nullptr) {
      const int* face = input.faces + face_idx * 3;
      T tw = 1 - tu - tv;

      T norm[3];
      vec_fill(norm, 0.f);
      vec_add(1.f, norm, tu, this->input.normals + face[0] * 3, norm);
      vec_add(1.f, norm, tv, this->input.normals + face[1] * 3, norm);
      vec_add(1.f, norm, tw, this->input.normals + face[2] * 3, norm);
      if(vec_dot(norm, dir) > 0) {
        vec_mul_scalar(norm, -1.f, norm);
      }

      if(this->buffer.normal != nullptr) {
        this->buffer.normal[idx * 3 + 0] = norm[0];
        this->buffer.normal[idx * 3 + 1] = norm[1];
        this->buffer.normal[idx * 3 + 2] = norm[2];
      }

      if(this->buffer.color != nullptr) {
        T color[3];
        vec_fill(color, 0.f);
        vec_add(1.f, color, tu, this->input.colors + face[0] * 3, color);
        vec_add(1.f, color, tv, this->input.colors + face[1] * 3, color);
        vec_add(1.f, color, tw, this->input.colors + face[2] * 3, color);

        T sp[3];
        vec_add(1.f, orig, t, dir, sp);
        T reflectance = this->shader(orig, sp, orig, norm);

        this->buffer.color[idx * 3 + 0] = mmin(1.f, mmax(0.f, reflectance * color[0]));
        this->buffer.color[idx * 3 + 1] = mmin(1.f, mmax(0.f, reflectance * color[1]));
        this->buffer.color[idx * 3 + 2] = mmin(1.f, mmax(0.f, reflectance * color[2]));
      }
    }
  }
};

template <typename T, int n=3>
CPU_GPU_FUNCTION
inline void interpolate_linear(const T* im, T x, T y, int height, int width, T* out_vec) {
  int x1 = int(x);
  int y1 = int(y);
  int x2 = x1 + 1;
  int y2 = y1 + 1;

  T denom = (x2 - x1) * (y2 - y1);
  T t11 = (x2 - x) * (y2 - y);
  T t21 = (x - x1) * (y2 - y);
  T t12 = (x2 - x) * (y - y1);
  T t22 = (x - x1) * (y - y1);

  x1 = mmin(mmax(x1, int(0)), width-1);
  x2 = mmin(mmax(x2, int(0)), width-1);
  y1 = mmin(mmax(y1, int(0)), height-1);
  y2 = mmin(mmax(y2, int(0)), height-1);

  for(int idx = 0; idx < n; ++idx) {
    out_vec[idx] = (im[(y1 * width + x1) * 3 + idx] * t11 +
                    im[(y2 * width + x1) * 3 + idx] * t12 +
                    im[(y1 * width + x2) * 3 + idx] * t21 +
                    im[(y2 * width + x2) * 3 + idx] * t22) / denom;
  }
}

template <typename T>
struct RenderProjectorFunctor : public RenderFunctor<T> {
  const RenderInput<T> input;
  const Camera<T> proj;
  const float* pattern;
  const float d_alpha;
  const float d_beta;

  RenderProjectorFunctor(const RenderInput<T> input, const Shader<T> shader, const Camera<T> cam, const Camera<T> proj, const float* pattern, float d_alpha, float d_beta, Buffer<T> buffer) : RenderFunctor<T>(cam, shader, buffer), input(input), proj(proj), pattern(pattern), d_alpha(d_alpha), d_beta(d_beta) {
  }

  CPU_GPU_FUNCTION void operator()(const int idx) {
    int h = idx / this->cam.width;
    int w = idx % this->cam.width;

    T orig[3];
    this->cam.get_C(orig);
    T dir[3];
    this->cam.to_ray(h, w, dir);

    int face_idx;
    T t, tu, tv;
    bool valid = ray_triangle_mesh_intersect_3d(orig, dir, this->input.faces, this->input.n_faces, this->input.verts, &face_idx, &t, &tu, &tv);
    if(this->buffer.depth != nullptr) {
      this->buffer.depth[idx] = valid ? t : -1;
    }

    this->buffer.color[idx * 3 + 0] = 0;
    this->buffer.color[idx * 3 + 1] = 0;
    this->buffer.color[idx * 3 + 2] = 0;

    if(valid) {
      if(this->buffer.normal != nullptr) {
        const int* face = input.faces + face_idx * 3;
        T tw = 1 - tu - tv;

        T norm[3];
        vertex_normal_3d(
            this->input.verts + face[0] * 3,
            this->input.verts + face[1] * 3,
            this->input.verts + face[2] * 3,
            norm);
        vec_normalize(norm, norm);

        if(vec_dot(norm, dir) > 0) {
          vec_mul_scalar(norm, -1.f, norm);
        }

        T color[3];
        vec_fill(color, 0.f);
        vec_add(1.f, color, tu, this->input.colors + face[0] * 3, color);
        vec_add(1.f, color, tv, this->input.colors + face[1] * 3, color);
        vec_add(1.f, color, tw, this->input.colors + face[2] * 3, color);

        T sp[3];
        vec_add(1.f, orig, t, dir, sp);
        T reflectance = this->shader(orig, sp, orig, norm);

        this->buffer.normal[idx * 3 + 0] = mmin(1.f, mmax(0.f, reflectance * color[0]));
        this->buffer.normal[idx * 3 + 1] = mmin(1.f, mmax(0.f, reflectance * color[1]));
        this->buffer.normal[idx * 3 + 2] = mmin(1.f, mmax(0.f, reflectance * color[2]));
      }

      // get 3D point
      T pt[3];
      vec_mul_scalar(dir, t, pt);
      vec_add(orig, pt, pt);

      // get dir from proj
      T proj_orig[3];
      proj.get_C(proj_orig);
      T proj_dir[3];
      vec_sub(pt, proj_orig, proj_dir);
      vec_div_scalar(proj_dir, proj_dir[2], proj_dir);

      // check if it hit same tria
      int p_face_idx;
      T p_t, p_tu, p_tv;
      valid = ray_triangle_mesh_intersect_3d(proj_orig, proj_dir, this->input.faces, this->input.n_faces, this->input.verts, &p_face_idx, &p_t, &p_tu, &p_tv);
      // if(!valid || p_face_idx != face_idx) {
      //   return;
      // }

      T p_pt[3];
      vec_mul_scalar(proj_dir, p_t, p_pt);
      vec_add(proj_orig, p_pt, p_pt);
      T diff[3];
      vec_sub(p_pt, pt, diff);
      if(!valid || vec_norm(diff) > 1e-5) {
        return;
      }

      // get uv in proj
      T u,v,d;
      proj.to_2d(p_pt, &u,&v,&d);

      // if valid u,v than use it to inpaint
      if(u >= 0 && v >= 0 && u < this->proj.width && v < this->proj.height) {
        // int pattern_idx = ((int(v) * this->proj.width) + int(u)) * 3;
        // this->buffer.color[idx * 3 + 0] = pattern[pattern_idx + 0];
        // this->buffer.color[idx * 3 + 1] = pattern[pattern_idx + 1];
        // this->buffer.color[idx * 3 + 2] = pattern[pattern_idx + 2];
        interpolate_linear(pattern, u, v, this->proj.height, this->proj.width, this->buffer.color + idx * 3);

        // decay based on distance
        T decay = d_alpha + d_beta * d;
        decay *= decay;
        decay = mmax(decay, T(1));
        vec_div_scalar(this->buffer.color + idx * 3, decay, this->buffer.color + idx * 3);
      }
    }

  }
};




#endif
