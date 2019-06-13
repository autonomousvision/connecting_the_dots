from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import platform
import os

this_dir = os.path.dirname(__file__)


extra_compile_args = ['-O3', '-std=c++11']
extra_link_args = []

print('using openmp')
extra_compile_args.append('-fopenmp')
extra_link_args.append('-fopenmp')

sources = ['hyperdepth.pyx']
extra_objects = []
library_dirs = []
libraries = ['m']

setup(
  name="hyperdepth",
  cmdclass= {'build_ext': build_ext},
  ext_modules=[
    Extension('hyperdepth',
      sources,
      extra_objects=extra_objects,
      language='c++',
      library_dirs=library_dirs,
      libraries=libraries,
      include_dirs=[
        np.get_include(),
      ],
      extra_compile_args=extra_compile_args,
      extra_link_args=extra_link_args
    )
  ]
)




