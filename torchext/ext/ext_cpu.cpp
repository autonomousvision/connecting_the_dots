#include <torch/extension.h>

#include <iostream>

#include "ext.h"

template <typename FunctorT>
void iterate_cpu(FunctorT functor, int N) {
  for(int idx = 0; idx < N; ++idx) {
    functor(idx);
  }
}

at::Tensor nn_cpu(at::Tensor in0, at::Tensor in1) {
  CHECK_INPUT_CPU(in0);
  CHECK_INPUT_CPU(in1);

  auto nelem0 = in0.size(0);
  auto nelem1 = in1.size(0);
  auto dim = in0.size(1);

  AT_ASSERTM(dim == in1.size(1), "in0 and in1 have to be the same shape");
  AT_ASSERTM(dim == 3, "dim hast to be 3");
  AT_ASSERTM(in0.dim() == 2, "in0 has to be N0 x 3");
  AT_ASSERTM(in1.dim() == 2, "in1 has to be N1 x 3");

  auto out = at::empty({nelem0}, torch::CPU(at::kLong));

  AT_DISPATCH_FLOATING_TYPES(in0.scalar_type(), "nn", ([&] {
    iterate_cpu(
      NNFunctor<scalar_t>(in0.data<scalar_t>(), in1.data<scalar_t>(), nelem0, nelem1, out.data<long>()),
      nelem0);
  }));

  return out;
}


at::Tensor crosscheck_cpu(at::Tensor in0, at::Tensor in1) {
  CHECK_INPUT_CPU(in0);
  CHECK_INPUT_CPU(in1);

  AT_ASSERTM(in0.dim() == 1, "");
  AT_ASSERTM(in1.dim() == 1, "");

  auto nelem0 = in0.size(0);
  auto nelem1 = in1.size(0);

  auto out = at::empty({nelem0}, torch::CPU(at::kByte));

  iterate_cpu(
    CrossCheckFunctor(in0.data<long>(), in1.data<long>(), nelem0, nelem1, out.data<uint8_t>()),
    nelem0);

  return out;
}


at::Tensor proj_nn_cpu(at::Tensor xyz0, at::Tensor xyz1, at::Tensor K, int patch_size) {
  CHECK_INPUT_CPU(xyz0);
  CHECK_INPUT_CPU(xyz1);
  CHECK_INPUT_CPU(K);

  auto batch_size = xyz0.size(0);
  auto height = xyz0.size(1);
  auto width = xyz0.size(2);

  AT_ASSERTM(xyz0.size(0) == xyz1.size(0), "");
  AT_ASSERTM(xyz0.size(1) == xyz1.size(1), "");
  AT_ASSERTM(xyz0.size(2) == xyz1.size(2), "");
  AT_ASSERTM(xyz0.size(3) == xyz1.size(3), "");
  AT_ASSERTM(xyz0.size(3) == 3, "");
  AT_ASSERTM(xyz0.dim() == 4, "");
  AT_ASSERTM(xyz1.dim() == 4, "");

  auto out = at::empty({batch_size, height, width}, torch::CPU(at::kLong));

  AT_DISPATCH_FLOATING_TYPES(xyz0.scalar_type(), "proj_nn", ([&] {
    iterate_cpu(
      ProjNNFunctor<scalar_t>(xyz0.data<scalar_t>(), xyz1.data<scalar_t>(), K.data<scalar_t>(), batch_size, height, width, patch_size, out.data<long>()),
      batch_size * height * width);
  }));

  return out;
}


at::Tensor xcorrvol_cpu(at::Tensor in0, at::Tensor in1, int n_disps, int block_size) {
  CHECK_INPUT_CPU(in0);
  CHECK_INPUT_CPU(in1);

  auto channels = in0.size(0);
  auto height = in0.size(1);
  auto width = in0.size(2);

  auto out = at::empty({n_disps, height, width}, in0.options());

  AT_DISPATCH_FLOATING_TYPES(in0.scalar_type(), "xcorrvol", ([&] {
    iterate_cpu(
      XCorrVolFunctor<scalar_t>(in0.data<scalar_t>(), in1.data<scalar_t>(), channels, height, width, n_disps, block_size, out.data<scalar_t>()),
      n_disps * height * width);
  }));

  return out;
}




at::Tensor photometric_loss_forward(at::Tensor es, at::Tensor ta, int block_size, int type, float eps) {
  CHECK_INPUT_CPU(es);
  CHECK_INPUT_CPU(ta);

  auto batch_size = es.size(0);
  auto channels = es.size(1);
  auto height = es.size(2);
  auto width = es.size(3);

  auto out = at::empty({batch_size, 1, height, width}, es.options());

  AT_DISPATCH_FLOATING_TYPES(es.scalar_type(), "photometric_loss_forward_cpu", ([&] {
    if(type == PHOTOMETRIC_LOSS_MSE) {
      iterate_cpu(
          PhotometricLossForward<scalar_t, PHOTOMETRIC_LOSS_MSE>(es.data<scalar_t>(), ta.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, out.data<scalar_t>()),
          out.numel());
    }
    else if(type == PHOTOMETRIC_LOSS_SAD) {
      iterate_cpu(
          PhotometricLossForward<scalar_t, PHOTOMETRIC_LOSS_SAD>(es.data<scalar_t>(), ta.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, out.data<scalar_t>()),
          out.numel());
    }
    else if(type == PHOTOMETRIC_LOSS_CENSUS_MSE) {
      iterate_cpu(
          PhotometricLossForward<scalar_t, PHOTOMETRIC_LOSS_CENSUS_MSE>(es.data<scalar_t>(), ta.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, out.data<scalar_t>()),
          out.numel());
    }
    else if(type == PHOTOMETRIC_LOSS_CENSUS_SAD) {
      iterate_cpu(
          PhotometricLossForward<scalar_t, PHOTOMETRIC_LOSS_CENSUS_SAD>(es.data<scalar_t>(), ta.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, out.data<scalar_t>()),
          out.numel());
    }
  }));

  return out;
}

at::Tensor photometric_loss_backward(at::Tensor es, at::Tensor ta, at::Tensor grad_out, int block_size, int type, float eps) {
  CHECK_INPUT_CPU(es);
  CHECK_INPUT_CPU(ta);
  CHECK_INPUT_CPU(grad_out);

  auto batch_size = es.size(0);
  auto channels = es.size(1);
  auto height = es.size(2);
  auto width = es.size(3);

  CHECK_INPUT_CPU(ta);
  auto grad_in = at::zeros({batch_size, channels, height, width}, grad_out.options());

  AT_DISPATCH_FLOATING_TYPES(es.scalar_type(), "photometric_loss_backward_cpu", ([&] {
    if(type == PHOTOMETRIC_LOSS_MSE) {
      iterate_cpu(
          PhotometricLossBackward<scalar_t, PHOTOMETRIC_LOSS_MSE>(es.data<scalar_t>(), ta.data<scalar_t>(), grad_out.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, grad_in.data<scalar_t>()),
          grad_out.numel());
    }
    else if(type == PHOTOMETRIC_LOSS_SAD) {
      iterate_cpu(
          PhotometricLossBackward<scalar_t, PHOTOMETRIC_LOSS_SAD>(es.data<scalar_t>(), ta.data<scalar_t>(), grad_out.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, grad_in.data<scalar_t>()),
          grad_out.numel());
    }
    else if(type == PHOTOMETRIC_LOSS_CENSUS_MSE) {
      iterate_cpu(
          PhotometricLossBackward<scalar_t, PHOTOMETRIC_LOSS_CENSUS_MSE>(es.data<scalar_t>(), ta.data<scalar_t>(), grad_out.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, grad_in.data<scalar_t>()),
          grad_out.numel());
    }
    else if(type == PHOTOMETRIC_LOSS_CENSUS_SAD) {
      iterate_cpu(
          PhotometricLossBackward<scalar_t, PHOTOMETRIC_LOSS_CENSUS_SAD>(es.data<scalar_t>(), ta.data<scalar_t>(), grad_out.data<scalar_t>(), block_size, eps, batch_size, channels, height, width, grad_in.data<scalar_t>()),
          grad_out.numel());
    }
  }));

  return grad_in;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nn_cpu", &nn_cpu, "nn_cpu");
  m.def("crosscheck_cpu", &crosscheck_cpu, "crosscheck_cpu");
  m.def("proj_nn_cpu", &proj_nn_cpu, "proj_nn_cpu");

  m.def("xcorrvol_cpu", &xcorrvol_cpu, "xcorrvol_cpu");

  m.def("photometric_loss_forward", &photometric_loss_forward);
  m.def("photometric_loss_backward", &photometric_loss_backward);
}
