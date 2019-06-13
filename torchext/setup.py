from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
import os

this_dir = os.path.dirname(os.path.realpath(__file__))

include_dirs = [
]

nvcc_args = [
  '-arch=sm_30',
  '-gencode=arch=compute_30,code=sm_30',
  '-gencode=arch=compute_35,code=sm_35',
]

setup(
  name='ext',
  ext_modules=[
    CppExtension('ext_cpu', ['ext/ext_cpu.cpp']),
    CUDAExtension('ext_cuda', ['ext/ext_cuda.cpp', 'ext/ext_kernel.cu'], extra_compile_args={'cxx': [], 'nvcc': nvcc_args}),
  ],
  cmdclass={'build_ext': BuildExtension},
  include_dirs=include_dirs
)
