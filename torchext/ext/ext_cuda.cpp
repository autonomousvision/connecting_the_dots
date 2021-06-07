#include <torch/extension.h>

#include <iostream>

#include "ext.h"

void nn_kernel(at::Tensor in0, at::Tensor in1, at::Tensor out);

at::Tensor nn_cuda(at::Tensor in0, at::Tensor in1) {
  CHECK_INPUT_CUDA(in0);
  CHECK_INPUT_CUDA(in1);

  auto nelem0 = in0.size(0);
  auto dim = in0.size(1);

  AT_ASSERTM(dim == in1.size(1), "in0 and in1 have to be the same shape");
  AT_ASSERTM(dim == 3, "dim hast to be 3");
  AT_ASSERTM(in0.dim() == 2, "in0 has to be N0 x 3");
  AT_ASSERTM(in1.dim() == 2, "in1 has to be N1 x 3");

  auto out = at::empty({nelem0}, torch::CUDA(at::kLong));

  nn_kernel(in0, in1, out);

  return out;
}


void crosscheck_kernel(at::Tensor in0, at::Tensor in1, at::Tensor out);

at::Tensor crosscheck_cuda(at::Tensor in0, at::Tensor in1) {
  CHECK_INPUT_CUDA(in0);
  CHECK_INPUT_CUDA(in1);

  AT_ASSERTM(in0.dim() == 1, "");
  AT_ASSERTM(in1.dim() == 1, "");

  auto nelem0 = in0.size(0);
  auto out = at::empty({nelem0}, torch::CUDA(at::kByte));
  crosscheck_kernel(in0, in1, out);

  return out;
}

void proj_nn_kernel(at::Tensor xyz0, at::Tensor xyz1, at::Tensor K, int patch_size, at::Tensor out);

at::Tensor proj_nn_cuda(at::Tensor xyz0, at::Tensor xyz1, at::Tensor K, int patch_size) {
  CHECK_INPUT_CUDA(xyz0);
  CHECK_INPUT_CUDA(xyz1);
  CHECK_INPUT_CUDA(K);

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

  auto out = at::empty({batch_size, height, width}, torch::CUDA(at::kLong));

  proj_nn_kernel(xyz0, xyz1, K, patch_size, out);

  return out;
}

void xcorrvol_kernel(at::Tensor in0, at::Tensor in1, int n_disps, int block_size, at::Tensor out);

at::Tensor xcorrvol_cuda(at::Tensor in0, at::Tensor in1, int n_disps, int block_size) {
  CHECK_INPUT_CUDA(in0);
  CHECK_INPUT_CUDA(in1);

  // auto channels = in0.size(0);
  auto height = in0.size(1);
  auto width = in0.size(2);

  auto out = at::empty({n_disps, height, width}, in0.options());

  xcorrvol_kernel(in0, in1, n_disps, block_size, out);

  return out;
}



void photometric_loss_forward_kernel(at::Tensor es, at::Tensor ta, int block_size, int type, float eps, at::Tensor out);

at::Tensor photometric_loss_forward(at::Tensor es, at::Tensor ta, int block_size, int type, float eps) {
  CHECK_INPUT_CUDA(es);
  CHECK_INPUT_CUDA(ta);

  auto batch_size = es.size(0);
  auto height = es.size(2);
  auto width = es.size(3);

  auto out = at::empty({batch_size, 1, height, width}, es.options());
  photometric_loss_forward_kernel(es, ta, block_size, type, eps, out);

  return out;
}


void photometric_loss_backward_kernel(at::Tensor es, at::Tensor ta, at::Tensor grad_out, int block_size, int type, float eps, at::Tensor grad_in);

at::Tensor photometric_loss_backward(at::Tensor es, at::Tensor ta, at::Tensor grad_out, int block_size, int type, float eps) {
  CHECK_INPUT_CUDA(es);
  CHECK_INPUT_CUDA(ta);
  CHECK_INPUT_CUDA(grad_out);

  auto batch_size = es.size(0);
  auto channels = es.size(1);
  auto height = es.size(2);
  auto width = es.size(3);

  auto grad_in = at::zeros({batch_size, channels, height, width}, grad_out.options());
  photometric_loss_backward_kernel(es, ta, grad_out, block_size, type, eps, grad_in);

  return grad_in;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nn_cuda", &nn_cuda, "nn_cuda");
  m.def("crosscheck_cuda", &crosscheck_cuda, "crosscheck_cuda");
  m.def("proj_nn_cuda", &proj_nn_cuda, "proj_nn_cuda");

  m.def("xcorrvol_cuda", &xcorrvol_cuda, "xcorrvol_cuda");

  m.def("photometric_loss_forward", &photometric_loss_forward);
  m.def("photometric_loss_backward", &photometric_loss_backward);
}
