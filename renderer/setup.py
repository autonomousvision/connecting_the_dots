from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import platform
import os
import json

this_dir = os.path.dirname(__file__)

with open('../config.json') as fp:
  config = json.load(fp)

extra_compile_args = ['-O3', '-std=c++11']

print('using cuda')
cuda_lib_dir = config['CUDA_LIBRARY_DIR']
cuda_lib = 'cudart'

sources = ['cyrender.pyx']
extra_objects = [
  os.path.join(this_dir, 'render/render_cpu.cpp.o'),
]
library_dirs = []
libraries = ['m']
extra_objects.append(os.path.join(this_dir, 'render/render_gpu.cu.o'))
extra_objects.append(os.path.join(this_dir, 'render/stdlib_cuda.cu.o'))
library_dirs.append(cuda_lib_dir)
libraries.append(cuda_lib)

setup(
  name="cyrender",
  cmdclass= {'build_ext': build_ext},
  ext_modules=[
    Extension('cyrender',
      sources,
      extra_objects=extra_objects,
      language='c++',
      library_dirs=library_dirs,
      libraries=libraries,
      include_dirs=[
        np.get_include(),
      ],
      extra_compile_args=extra_compile_args,
      # extra_link_args=extra_link_args
    )
  ]
)
