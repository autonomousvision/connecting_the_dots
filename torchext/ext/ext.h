#pragma once

#include "common.h"


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT_CPU(x) CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


template <typename T, int dim=3>
struct NNFunctor {
  const T* in0; // nelem0 x dim
  const T* in1; // nelem1 x dim
  const long nelem0;
  const long nelem1;
  long* out; // nelem0

  NNFunctor(const T* in0, const T* in1, long nelem0, long nelem1, long* out) : in0(in0), in1(in1), nelem0(nelem0), nelem1(nelem1), out(out) {}

  CPU_GPU_FUNCTION void operator()(long idx0) {
    // idx0 \in [nelem0]

    const T* vec0 = in0 + idx0 * dim;

    T min_dist = 1e9;
    long min_arg = -1;
    for(long idx1 = 0; idx1 < nelem1; ++idx1) {
      const T* vec1 = in1 + idx1 * dim;
      T dist = 0;
      for(long didx = 0; didx < dim; ++didx) {
        T diff = vec0[didx] - vec1[didx];
        dist += diff * diff;
      }

      if(dist < min_dist) {
        min_dist = dist;
        min_arg = idx1;
      }
    }

    out[idx0] = min_arg;
  }
};

struct CrossCheckFunctor {
  const long* in0; // nelem0
  const long* in1; // nelem1
  const long nelem0;
  const long nelem1;
  uint8_t* out; // nelem0

  CrossCheckFunctor(const long* in0, const long* in1, long nelem0, long nelem1, uint8_t* out) : in0(in0), in1(in1), nelem0(nelem0), nelem1(nelem1), out(out) {}

  CPU_GPU_FUNCTION void operator()(long idx0) {
    // idx0 \in [nelem0]
    int idx1 = in0[idx0];
    out[idx0] = idx1 >=0 && in1[idx1] >= 0 && idx0 == in1[idx1];
    // out[idx0] = idx0 == in1[in0[idx0]];
  }
};

template <typename T, int dim=3>
struct ProjNNFunctor {
  // xyz0, xyz1 in coord sys of 1
  const T* xyz0; // bs x height x width x 3
  const T* xyz1; // bs x height x width x 3
  const T* K; // 3 x 3
  const long batch_size;
  const long height;
  const long width;
  const long patch_size;
  long* out; // bs x height x width

  ProjNNFunctor(const T* xyz0, const T* xyz1, const T* K, long batch_size, long height, long width, long patch_size, long* out)
    : xyz0(xyz0), xyz1(xyz1), K(K), batch_size(batch_size), height(height), width(width), patch_size(patch_size), out(out) {}

  CPU_GPU_FUNCTION void operator()(long idx0) {
    // idx0 \in [0, bs x height x width]

    const long bs = idx0 / (height * width);

    const T x = xyz0[idx0 * 3 + 0];
    const T y = xyz0[idx0 * 3 + 1];
    const T z = xyz0[idx0 * 3 + 2];
    const T d = K[6] * x + K[7] * y + K[8] * z;
    const T u = (K[0] * x + K[1] * y + K[2] * z) / d;
    const T v = (K[3] * x + K[4] * y + K[5] * z) / d;

    int u0 = u + 0.5;
    int v0 = v + 0.5;

    long min_idx1 = -1;
    T min_dist = 1e9;
    for(int pidx = 0; pidx < patch_size*patch_size; ++pidx) {
      int pu = pidx % patch_size;
      int pv = pidx / patch_size;

      int u1 = u0 + pu - patch_size/2;
      int v1 = v0 + pv - patch_size/2;

      if(u1 >= 0 && v1 >= 0 && u1 < width && v1 < height) {
        const long idx1 = (bs * height + v1) * width + u1;
        const T* xyz1n = xyz1 + idx1 * 3;
        const T d = (x-xyz1n[0]) * (x-xyz1n[0]) + (y-xyz1n[1]) * (y-xyz1n[1]) + (z-xyz1n[2]) * (z-xyz1n[2]);
        if(d < min_dist) {
          min_dist = d;
          min_idx1 = idx1;
        }
      }
    }

    out[idx0] = min_idx1;
  }
};


template <typename T, int dim=3>
struct XCorrVolFunctor {
  const T* in0; // channels x height x width
  const T* in1; // channels x height x width
  const long channels;
  const long height;
  const long width;
  const long n_disps;
  const long block_size;
  T* out; // nelem0

  XCorrVolFunctor(const T* in0, const T* in1, long channels, long height, long width, long n_disps, long block_size, T* out) : in0(in0), in1(in1), channels(channels), height(height), width(width), n_disps(n_disps), block_size(block_size), out(out) {}

  CPU_GPU_FUNCTION void operator()(long oidx) {
    // idx0 \in [n_disps x height x width]

    auto d = oidx / (height * width);
    auto h = (oidx / width) % height;
    auto w = oidx % width;

    long block_size2 = block_size * block_size;

    T val = 0;
    for(int c = 0; c < channels; ++c) {
      // compute means
      T mu0 = 0;
      T mu1 = 0;
      for(int bh = 0; bh < block_size; ++bh) {
        long h0 = h + bh - block_size / 2;
        h0 = mmax(long(0), mmin(height-1, h0));
        for(int bw = 0; bw < block_size; ++bw) {
          long w0 = w + bw - block_size / 2;
          long w1 = w0 - d;
          w0 = mmax(long(0), mmin(width-1, w0));
          w1 = mmax(long(0), mmin(width-1, w1));
          long idx0 = (c * height + h0) * width + w0;
          long idx1 = (c * height + h0) * width + w1;
          mu0 += in0[idx0] / block_size2;
          mu1 += in1[idx1] / block_size2;
        }
      }

      // compute stds and dot product
      T sigma0 = 0;
      T sigma1 = 0;
      T dot = 0;
      for(int bh = 0; bh < block_size; ++bh) {
        long h0 = h + bh - block_size / 2;
        h0 = mmax(long(0), mmin(height-1, h0));
        for(int bw = 0; bw < block_size; ++bw) {
          long w0 = w + bw - block_size / 2;
          long w1 = w0 - d;
          w0 = mmax(long(0), mmin(width-1, w0));
          w1 = mmax(long(0), mmin(width-1, w1));
          long idx0 = (c * height + h0) * width + w0;
          long idx1 = (c * height + h0) * width + w1;
          T v0 = in0[idx0] - mu0;
          T v1 = in1[idx1] - mu1;

          dot += v0 * v1;
          sigma0 += v0 * v0;
          sigma1 += v1 * v1;
        }
      }

      T norm = sqrt(sigma0 * sigma1) + 1e-8;
      val += dot / norm;
    }

    out[oidx] = val;
  }
};




const int PHOTOMETRIC_LOSS_MSE = 0;
const int PHOTOMETRIC_LOSS_SAD = 1;
const int PHOTOMETRIC_LOSS_CENSUS_MSE = 2;
const int PHOTOMETRIC_LOSS_CENSUS_SAD = 3;

template <typename T, int type>
struct PhotometricLossForward {
  const T* es;  // batch_size x channels x height x width;
  const T* ta;
  const int block_size;
  const int block_size2;
  const T eps;
  const int batch_size;
  const int channels;
  const int height;
  const int width;
  T* out;  // batch_size x channels x height x width;

  PhotometricLossForward(const T* es, const T* ta, int block_size, T eps, int batch_size, int channels, int height, int width, T* out) :
    es(es), ta(ta), block_size(block_size), block_size2(block_size*block_size), eps(eps), batch_size(batch_size), channels(channels), height(height), width(width), out(out) {}

  CPU_GPU_FUNCTION void operator()(int outidx) {
    // outidx \in [0, batch_size x height x width]

    int w = outidx % width;
    int h = (outidx / width) % height;
    int n = outidx / (height * width);

    T loss = 0;
    for(int bidx = 0; bidx < block_size2; ++bidx) {
      int bh = bidx / block_size;
      int bw = bidx % block_size;
      int h0 = h + bh - block_size / 2;
      int w0 = w + bw - block_size / 2;

      h0 = mmin(height-1, mmax(0, h0));
      w0 = mmin(width-1, mmax(0, w0));

      for(int c = 0; c < channels; ++c) {
        int inidx = ((n * channels + c) * height + h0) * width + w0;
        if(type == PHOTOMETRIC_LOSS_SAD || type == PHOTOMETRIC_LOSS_MSE) {
          T diff = es[inidx] - ta[inidx];
          if(type == PHOTOMETRIC_LOSS_MSE) {
            loss += diff * diff / block_size2;
          }
          else if(type == PHOTOMETRIC_LOSS_SAD) {
            loss += fabs(diff) / block_size2;
          }
        }
        else if(type == PHOTOMETRIC_LOSS_CENSUS_SAD || type == PHOTOMETRIC_LOSS_CENSUS_MSE) {
          int inidxc = ((n * channels + c) * height + h) * width + w;
          T des = es[inidx] - es[inidxc];
          T dta = ta[inidx] - ta[inidxc];
          T h_des = 0.5 * (1 + des / sqrt(des * des + eps));
          T h_dta = 0.5 * (1 + dta / sqrt(dta * dta + eps));
          T diff = h_des - h_dta;
          // printf("%d,%d %d,%d: des=%f, dta=%f, h_des=%f, h_dta=%f, diff=%f\n", h,w, h0,w0, des,dta, h_des,h_dta, diff);
          // printf("%d,%d %d,%d: h_des=%f = 0.5 * (1 + %f / %f); %f, %f, %f\n", h,w, h0,w0, h_des, des, sqrt(des * des + eps), des*des, des*des+eps, eps);
          if(type == PHOTOMETRIC_LOSS_CENSUS_MSE) {
            loss += diff * diff / block_size2;
          }
          else if(type == PHOTOMETRIC_LOSS_CENSUS_SAD) {
            loss += fabs(diff) / block_size2;
          }
        }
      }
    }

    out[outidx] = loss;
  }
};

template <typename T, int type>
struct PhotometricLossBackward {
  const T* es;  // batch_size x channels x height x width;
  const T* ta;
  const T* grad_out;
  const int block_size;
  const int block_size2;
  const T eps;
  const int batch_size;
  const int channels;
  const int height;
  const int width;
  T* grad_in;  // batch_size x channels x height x width;

  PhotometricLossBackward(const T* es, const T* ta, const T* grad_out, int block_size, T eps, int batch_size, int channels, int height, int width, T* grad_in) :
    es(es), ta(ta), grad_out(grad_out), block_size(block_size), block_size2(block_size*block_size), eps(eps), batch_size(batch_size), channels(channels), height(height), width(width), grad_in(grad_in) {}

  CPU_GPU_FUNCTION void operator()(int outidx) {
    // outidx \in [0, batch_size x height x width]

    int w = outidx % width;
    int h = (outidx / width) % height;
    int n = outidx / (height * width);

    for(int bidx = 0; bidx < block_size2; ++bidx) {
      int bh = bidx / block_size;
      int bw = bidx % block_size;
      int h0 = h + bh - block_size / 2;
      int w0 = w + bw - block_size / 2;

      h0 = mmin(height-1, mmax(0, h0));
      w0 = mmin(width-1, mmax(0, w0));

      const T go = grad_out[outidx];

      for(int c = 0; c < channels; ++c) {
        int inidx = ((n * channels + c) * height + h0) * width + w0;
        if(type == PHOTOMETRIC_LOSS_SAD || type == PHOTOMETRIC_LOSS_MSE) {
          T diff = es[inidx] - ta[inidx];
          T grad = 0;
          if(type == PHOTOMETRIC_LOSS_MSE) {
            grad = 2 * diff;
          }
          else if(type == PHOTOMETRIC_LOSS_SAD) {
            grad = diff < 0 ? -1 : (diff > 0 ? 1 : 0);
          }
          grad = grad / block_size2 * go;
          matomic_add(grad_in + inidx, grad);
        }
        else if(type == PHOTOMETRIC_LOSS_CENSUS_SAD || type == PHOTOMETRIC_LOSS_CENSUS_MSE) {
          int inidxc = ((n * channels + c) * height + h) * width + w;
          T des = es[inidx] - es[inidxc];
          T dta = ta[inidx] - ta[inidxc];
          T h_des = 0.5 * (1 + des / sqrt(des * des + eps));
          T h_dta = 0.5 * (1 + dta / sqrt(dta * dta + eps));
          T diff = h_des - h_dta;

          T grad_loss = 0;
          if(type == PHOTOMETRIC_LOSS_CENSUS_MSE) {
            grad_loss = 2 * diff;
          }
          else if(type == PHOTOMETRIC_LOSS_CENSUS_SAD) {
            grad_loss = diff < 0 ? -1 : (diff > 0 ? 1 : 0);
          }
          grad_loss = grad_loss / block_size2;

          T tmp = des * des + eps;
          T grad_heaviside = 0.5 * eps / sqrt(tmp * tmp * tmp);

          T grad = go * grad_loss * grad_heaviside;
          matomic_add(grad_in + inidx, grad);
          matomic_add(grad_in + inidxc, -grad);
        }
      }
    }
  }
};



