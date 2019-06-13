cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, malloc
from libcpp cimport bool
from cpython cimport PyObject, Py_INCREF

CREATE_INIT = True # workaround, so cython builds a init function

np.import_array()


ctypedef unsigned char  uint8_t

cdef extern from "render/render.h":
  cdef cppclass Camera[T]:
    const T fx;
    const T fy;
    const T px;
    const T py;
    const T R0, R1, R2, R3, R4, R5, R6, R7, R8;
    const T t0, t1, t2;
    const T C0, C1, C2;
    const int height;
    const int width;
    Camera(const T fx, const T fy, const T px, const T py, const T* R, const T* t, int width, int height) 

  cdef cppclass RenderInput[T]:
    T* verts;
    T* radii;
    T* colors;
    T* normals;
    int n_verts;
    int* faces;
    int n_faces;
  
    T* tex_coords;
    T* tex;
    int tex_height;
    int tex_width;
    int tex_channels;

    RenderInput();

  cdef cppclass Buffer[T]:
    T* depth;
    T* color;
    T* normal;
    Buffer();

  cdef cppclass Shader[T]:
    const T ka;
    const T kd;
    const T ks;
    const T alpha; 
    Shader(T ka, T kd, T ks, T alpha)

  cdef cppclass BaseRenderer[T]:
    const Camera[T] cam;
    const Shader[T] shader;
    Buffer[T] buffer;
    BaseRenderer(const Camera[T] cam, const Shader[T] shader, Buffer[T] buffer)
    void render_mesh(const RenderInput[T] input); 
    void render_mesh_proj(const RenderInput[T] input, const Camera[T] proj, const float* pattern, float d_alpha, float d_beta);


cdef extern from "render/render_cpu.h":
  cdef cppclass RendererCpu[T](BaseRenderer[T]):
    RendererCpu(const Camera[T] cam, const Shader[T] shader, Buffer[T] buffer, int n_threads)
    void render_mesh(const RenderInput[T] input); 
    void render_mesh_proj(const RenderInput[T] input, const Camera[T] proj, const float* pattern, float d_alpha, float d_beta);

cdef extern from "render/render_gpu.h":
  cdef cppclass RendererGpu[T](BaseRenderer[T]):
    RendererGpu(const Camera[T] cam, const Shader[T] shader, Buffer[T] buffer)
    void render_mesh(const RenderInput[T] input); 
    void render_mesh_proj(const RenderInput[T] input, const Camera[T] proj, const float* pattern, float d_alpha, float d_beta);


cdef class PyCamera:
  cdef Camera[float]* cam;
  
  def __cinit__(self, float fx, float fy, float px, float py, float[:,::1] R, float[::1] t, int width, int height):
   if R.shape[0] != 3 or R.shape[1] != 3:
     raise Exception('invalid R matrix')
   if t.shape[0] != 3:
     raise Exception('invalid t vector')

   self.cam = new Camera[float](fx,fy, px,py, &R[0,0], &t[0], width, height)

  def __dealloc__(self):
    del self.cam


cdef class PyRenderInput:
  cdef RenderInput[float] input;
  cdef verts
  cdef colors
  cdef normals
  cdef faces

  def __cinit__(self, float[:,::1] verts=None, float[:,::1] colors=None, float[:,::1] normals=None, int[:,::1] faces=None):
   self.input = RenderInput[float]()
   if verts is not None:
     self.set_verts(verts)
   if normals is not None:
     self.set_normals(normals)
   if colors is not None:
     self.set_colors(colors)
   if faces is not None:
     self.set_faces(faces)

  def set_verts(self, float[:,::1] verts):
    if verts.shape[1] != 3:
      raise Exception('verts has to be a Nx3 matrix')
    self.verts = verts
    cdef float[:,::1] verts_view = self.verts
    self.input.verts = &verts_view[0,0]
    self.input.n_verts = self.verts.shape[0]

  def set_colors(self, float[:,::1] colors):
    if colors.shape[1] != 3:
      raise Exception('colors has to be a Nx3 matrix')
    self.colors = colors
    cdef float[:,::1] colors_view = self.colors
    self.input.colors = &colors_view[0,0]

  def set_normals(self, float[:,::1] normals):
    if normals.shape[1] != 3:
      raise Exception('normals has to be a Nx3 matrix')
    self.normals = normals
    cdef float[:,::1] normals_view = self.normals
    self.input.normals = &normals_view[0,0]

  def set_faces(self, int[:,::1] faces):
    if faces.shape[1] != 3:
      raise Exception('faces has to be a Nx3 matrix')
    self.faces = faces
    cdef int[:,::1] faces_view = self.faces
    self.input.faces = &faces_view[0,0]
    self.input.n_faces = self.faces.shape[0]

cdef class PyShader:
  cdef Shader[float]* shader

  def __cinit__(self, float ka, float kd, float ks, float alpha):
    self.shader = new Shader[float](ka, kd, ks, alpha)

  def __dealloc__(self):
    del self.shader


cdef class PyRenderer:
  cdef BaseRenderer[float]* renderer

  cdef Buffer[float] buffer
  cdef depth_buffer
  cdef color_buffer
  cdef normal_buffer

  def depth(self):
   return self.depth_buffer

  def color(self):
   return self.color_buffer

  def normal(self):
   return self.normal_buffer

  def __cinit__(self, PyCamera cam, PyShader shader, engine='cpu', int n_threads=1):
    self.depth_buffer = np.empty((cam.cam[0].height, cam.cam[0].width), dtype=np.float32)
    self.color_buffer = np.empty((cam.cam[0].height, cam.cam[0].width, 3), dtype=np.float32)
    self.normal_buffer = np.empty((cam.cam[0].height, cam.cam[0].width, 3), dtype=np.float32)

    cdef float[:,::1] dbv = self.depth_buffer
    cdef float[:,:,::1] cbv = self.color_buffer
    cdef float[:,:,::1] nbv = self.normal_buffer
    self.buffer.depth = &dbv[0,0]
    self.buffer.color = &cbv[0,0,0]
    self.buffer.normal = &nbv[0,0,0]

    if engine == 'cpu':
      self.renderer = new RendererCpu[float](cam.cam[0], shader.shader[0], self.buffer, n_threads)
    elif engine == 'gpu':
      self.renderer = new RendererGpu[float](cam.cam[0], shader.shader[0], self.buffer)
    else:
      raise Exception('invalid engine')

  def __dealloc__(self):
    del self.renderer

  def mesh(self, PyRenderInput input):
    self.renderer.render_mesh(input.input)

  def mesh_proj(self, PyRenderInput input, PyCamera proj, float[:,:,::1] pattern, float d_alpha=1, float d_beta=0):
    if pattern.shape[0] != proj.cam[0].height or pattern.shape[1] != proj.cam[0].width or pattern.shape[2] != 3:
      raise Exception(f'pattern has to be a {proj.cam[0].height}x{proj.cam[0].width}x3 tensor')
    self.renderer.render_mesh_proj(input.input, proj.cam[0], &pattern[0,0,0], d_alpha, d_beta)

