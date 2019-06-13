import torch
from . import ext_cpu
from . import ext_cuda

class NNFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, in0, in1):
    args = (in0, in1)
    if in0.is_cuda:
      out = ext_cuda.nn_cuda(*args)
    else:
      out = ext_cpu.nn_cpu(*args)
    return out

  @staticmethod
  def backward(ctx, grad_out):
    return None, None

def nn(in0, in1):
  return NNFunction.apply(in0, in1)


class CrossCheckFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, in0, in1):
    args = (in0, in1)
    if in0.is_cuda:
      out = ext_cuda.crosscheck_cuda(*args)
    else:
      out = ext_cpu.crosscheck_cpu(*args)
    return out

  @staticmethod
  def backward(ctx, grad_out):
    return None, None

def crosscheck(in0, in1):
  return CrossCheckFunction.apply(in0, in1)

class ProjNNFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, xyz0, xyz1, K, patch_size):
    args = (xyz0, xyz1, K, patch_size)
    if xyz0.is_cuda:
      out = ext_cuda.proj_nn_cuda(*args)
    else:
      out = ext_cpu.proj_nn_cpu(*args)
    return out

  @staticmethod
  def backward(ctx, grad_out):
    return None, None, None, None

def proj_nn(xyz0, xyz1, K, patch_size):
  return ProjNNFunction.apply(xyz0, xyz1, K, patch_size)



class XCorrVolFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, in0, in1, n_disps, block_size):
    args = (in0, in1, n_disps, block_size)
    if in0.is_cuda:
      out = ext_cuda.xcorrvol_cuda(*args)
    else:
      out = ext_cpu.xcorrvol_cpu(*args)
    return out

  @staticmethod
  def backward(ctx, grad_out):
    return None, None, None, None

def xcorrvol(in0, in1, n_disps, block_size):
  return XCorrVolFunction.apply(in0, in1, n_disps, block_size)




class PhotometricLossFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, es, ta, block_size, type, eps):
    args = (es, ta, block_size, type, eps)
    ctx.save_for_backward(es, ta)
    ctx.block_size = block_size
    ctx.type = type
    ctx.eps = eps
    if es.is_cuda:
      out = ext_cuda.photometric_loss_forward(*args)
    else:
      out = ext_cpu.photometric_loss_forward(*args)
    return out

  @staticmethod
  def backward(ctx, grad_out):
    es, ta = ctx.saved_tensors
    block_size = ctx.block_size
    type = ctx.type
    eps = ctx.eps
    args = (es, ta, grad_out.contiguous(), block_size, type, eps)
    if grad_out.is_cuda:
      grad_es = ext_cuda.photometric_loss_backward(*args)
    else:
      grad_es = ext_cpu.photometric_loss_backward(*args)
    return grad_es, None, None, None, None

def photometric_loss(es, ta, block_size, type='mse', eps=0.1):
  type = type.lower()
  if type == 'mse':
    type = 0
  elif type == 'sad':
    type = 1
  elif type == 'census_mse':
    type = 2
  elif type == 'census_sad':
    type = 3
  else:
    raise Exception('invalid loss type')
  return PhotometricLossFunction.apply(es, ta, block_size, type, eps)

def photometric_loss_pytorch(es, ta, block_size, type='mse', eps=0.1):
  type = type.lower()
  p = block_size // 2
  es_pad = torch.nn.functional.pad(es, (p,p,p,p), mode='replicate')
  ta_pad = torch.nn.functional.pad(ta, (p,p,p,p), mode='replicate')
  es_uf = torch.nn.functional.unfold(es_pad, kernel_size=block_size)
  ta_uf = torch.nn.functional.unfold(ta_pad, kernel_size=block_size)
  es_uf = es_uf.view(es.shape[0], es.shape[1], -1, es.shape[2], es.shape[3])
  ta_uf = ta_uf.view(ta.shape[0], ta.shape[1], -1, ta.shape[2], ta.shape[3])
  if type == 'mse':
    ref = (es_uf - ta_uf)**2
  elif type == 'sad':
    ref = torch.abs(es_uf - ta_uf)
  elif type == 'census_mse' or type == 'census_sad':
    des = es_uf - es.unsqueeze(2)
    dta = ta_uf - ta.unsqueeze(2)
    h_des = 0.5 * (1 + des / torch.sqrt(des * des + eps))
    h_dta = 0.5 * (1 + dta / torch.sqrt(dta * dta + eps))
    diff = h_des - h_dta
    if type == 'census_mse':
      ref = diff * diff
    elif type == 'census_sad':
      ref = torch.abs(diff)
  else:
    raise Exception('invalid loss type')
  ref = ref.view(es.shape[0], -1, es.shape[2], es.shape[3])
  ref = torch.sum(ref, dim=1, keepdim=True) / block_size**2
  return ref
