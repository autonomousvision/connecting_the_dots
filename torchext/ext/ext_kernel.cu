#include <ATen/ATen.h>

#include "ext.h"
#include "common_cuda.h"

void nn_kernel(at::Tensor in0, at::Tensor in1, at::Tensor out) {
  auto nelem0 = in0.size(0);
  auto nelem1 = in1.size(0);
  auto dim = in0.size(1);

  AT_DISPATCH_FLOATING_TYPES(in0.scalar_type(), "nn", ([&] {
    iterate_cuda(
      NNFunctor<scalar_t>(in0.data<scalar_t>(), in1.data<scalar_t>(), nelem0, nelem1, out.data<long>()),
      nelem0);
  }));
}


void crosscheck_kernel(at::Tensor in0, at::Tensor in1, at::Tensor out) {
  auto nelem0 = in0.size(0);
  auto nelem1 = in1.size(0);

  iterate_cuda(
    CrossCheckFunctor(in0.data<long>(), in1.data<long>(), nelem0, nelem1, out.data<uint8_t>()),
    nelem0);
}

void proj_nn_kernel(at::Tensor xyz0, at::Tensor xyz1, at::Tensor K, int patch_size, at::Tensor out) {
  auto batch_size = xyz0.size(0);
  auto height = xyz0.size(1);
  auto width = xyz0.size(2);

  AT_DISPATCH_FLOATING_TYPES(xyz0.scalar_type(), "proj_nn", ([&] {
    iterate_cuda(
      ProjNNFunctor<scalar_t>(xyz0.data<scalar_t>(), xyz1.data<scalar_t>(), K.data<scalar_t>(), batch_size, height, width, patch_size, out.data<long>()),
      batch_size * height * width);
  }));
}

void xcorrvol_kernel(at::Tensor in0, at::Tensor in1, int n_disps, int block_size, at::Tensor out) {
  auto channels = in0.size(0);
  auto height = in0.size(1);
  auto width = in0.size(2);

  AT_DISPATCH_FLOATING_TYPES(in0.scalar_type(), "xcorrvol", ([&] {
    iterate_cuda(
      XCorrVolFunctor<scalar_t>(in0.data<scalar_t>(), in1.data<scalar_t>(), channels, height, width, n_disps, block_size, out.data<scalar_t>()),
      n_disps * height * width, 512);
  }));
}



void photometric_loss_forward_kernel(at::Tensor es, at::Tensor ta, int block_size, int type, float eps, at::Tensor out) {
  auto batch_size = es.size(0);
  auto channels = es.size(1);
  auto height = es.size(2);
  auto width = es.size(3);

  AT_DISPATCH_FLOATING_TYPES(es.scalar_type(), "photometric_loss_forward_cuda", ([&] {
    if(type == PHOTOMETRIC_LOSS_MSE) {
      iterate_cuda(
          PhotometricLossForward<scalar_t, PHOTOMETRIC_LOSS_MSE>(es.data<scalar_t>(), ta.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, out.data<scalar_t>()),
          out.numel());
    }
    else if(type == PHOTOMETRIC_LOSS_SAD) {
      iterate_cuda(
          PhotometricLossForward<scalar_t, PHOTOMETRIC_LOSS_SAD>(es.data<scalar_t>(), ta.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, out.data<scalar_t>()),
          out.numel());
    }
    else if(type == PHOTOMETRIC_LOSS_CENSUS_MSE) {
      iterate_cuda(
          PhotometricLossForward<scalar_t, PHOTOMETRIC_LOSS_CENSUS_MSE>(es.data<scalar_t>(), ta.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, out.data<scalar_t>()),
          out.numel());
    }
    else if(type == PHOTOMETRIC_LOSS_CENSUS_SAD) {
      iterate_cuda(
          PhotometricLossForward<scalar_t, PHOTOMETRIC_LOSS_CENSUS_SAD>(es.data<scalar_t>(), ta.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, out.data<scalar_t>()),
          out.numel());
    }
  }));
}

void photometric_loss_backward_kernel(at::Tensor es, at::Tensor ta, at::Tensor grad_out, int block_size, int type, float eps, at::Tensor grad_in) {
  auto batch_size = es.size(0);
  auto channels = es.size(1);
  auto height = es.size(2);
  auto width = es.size(3);

  AT_DISPATCH_FLOATING_TYPES(es.scalar_type(), "photometric_loss_backward_cuda", ([&] {
    if(type == PHOTOMETRIC_LOSS_MSE) {
      iterate_cuda(
          PhotometricLossBackward<scalar_t, PHOTOMETRIC_LOSS_MSE>(es.data<scalar_t>(), ta.data<scalar_t>(), grad_out.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, grad_in.data<scalar_t>()),
          grad_out.numel());
    }
    else if(type == PHOTOMETRIC_LOSS_SAD) {
      iterate_cuda(
          PhotometricLossBackward<scalar_t, PHOTOMETRIC_LOSS_SAD>(es.data<scalar_t>(), ta.data<scalar_t>(), grad_out.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, grad_in.data<scalar_t>()),
          grad_out.numel());
    }
    else if(type == PHOTOMETRIC_LOSS_CENSUS_MSE) {
      iterate_cuda(
          PhotometricLossBackward<scalar_t, PHOTOMETRIC_LOSS_CENSUS_MSE>(es.data<scalar_t>(), ta.data<scalar_t>(), grad_out.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, grad_in.data<scalar_t>()),
          grad_out.numel());
    }
    else if(type == PHOTOMETRIC_LOSS_CENSUS_SAD) {
      iterate_cuda(
          PhotometricLossBackward<scalar_t, PHOTOMETRIC_LOSS_CENSUS_SAD>(es.data<scalar_t>(), ta.data<scalar_t>(), grad_out.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, grad_in.data<scalar_t>()),
          grad_out.numel());
    }
  }));
}
