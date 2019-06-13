cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, malloc
from libcpp cimport bool
from libcpp.string cimport string
from cpython cimport PyObject, Py_INCREF

CREATE_INIT = True # workaround, so cython builds a init function

np.import_array()


ctypedef unsigned char  uint8_t

cdef extern from "rf/train.h":
  cdef cppclass TrainParameters:
    int n_trees;
    int max_tree_depth;
    int n_test_split_functions;
    int n_test_thresholds;
    int n_test_samples;
    int min_samples_to_split;
    int min_samples_for_leaf;
    int print_node_info;
    TrainParameters();


cdef extern from "hyperdepth.h":
  void train(int row_from, int row_to, TrainParameters params, const uint8_t* ims, const float* disps, int n, int h, int w, int n_disp_bins, int depth_switch, int n_threads, string forest_prefix);
  void eval(int row_from, int row_to, const uint8_t* ims, const float* disps, int n, int h, int w, int n_disp_bins, int depth_switch, int n_threads, string forest_prefix, float* out);
  



cdef class TrainParams:
  cdef TrainParameters params;
  
  def __cinit__(self, int n_trees=6, int max_tree_depth=8, int n_test_split_functions=50, int n_test_thresholds=10, int n_test_samples=4096, int min_samples_to_split=16, int min_samples_for_leaf=8, int print_node_info=100):
    self.params.n_trees = n_trees
    self.params.max_tree_depth = max_tree_depth
    self.params.n_test_split_functions = n_test_split_functions
    self.params.n_test_thresholds = n_test_thresholds
    self.params.n_test_samples = n_test_samples
    self.params.min_samples_to_split = min_samples_to_split
    self.params.min_samples_for_leaf = min_samples_for_leaf
    self.params.print_node_info = print_node_info

  def __str__(self):
    return f'n_trees={self.params.n_trees}, max_tree_depth={self.params.max_tree_depth}, n_test_split_functions={self.params.n_test_split_functions}, n_test_thresholds={self.params.n_test_thresholds}, n_test_samples={self.params.n_test_samples}, min_samples_to_split={self.params.min_samples_to_split}, min_samples_for_leaf={self.params.min_samples_for_leaf}'


def train_forest(TrainParams params, uint8_t[:,:,::1] ims, float[:,:,::1] disps, int n_disp_bins=10, int depth_switch=0, int n_threads=18, str forest_prefix='forest', int row_from=-1, int row_to=-1):
  cdef int n = ims.shape[0]
  cdef int h = ims.shape[1]
  cdef int w = ims.shape[2]

  if row_from < 0:
    row_from = 0
  if row_to > h or row_to < 0:
    row_to = h

  if n != disps.shape[0] or h != disps.shape[1] or w != disps.shape[2]:
    raise Exception('ims.shape != disps.shape')

  train(row_from, row_to, params.params, &ims[0,0,0], &disps[0,0,0], n, h, w, n_disp_bins, depth_switch, n_threads, forest_prefix.encode())


def eval_forest(uint8_t[:,:,::1] ims, float[:,:,::1] disps, int n_disp_bins=10, int depth_switch=0, int n_threads=18, str forest_prefix='forest', int row_from=-1, int row_to=-1):
  cdef int n = ims.shape[0]
  cdef int h = ims.shape[1]
  cdef int w = ims.shape[2]

  if n != disps.shape[0] or h != disps.shape[1] or w != disps.shape[2]:
    raise Exception('ims.shape != disps.shape')

  if row_from < 0:
    row_from = 0
  if row_to > h or row_to < 0:
    row_to = h

  out = np.empty((n, h, w, 3), dtype=np.float32)
  cdef float[:,:,:,::1] out_view = out
  eval(row_from, row_to, &ims[0,0,0], &disps[0,0,0], n, h, w, n_disp_bins, depth_switch, n_threads, forest_prefix.encode(), &out_view[0,0,0,0])
  return out
