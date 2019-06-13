INCLUDE_DIR = 
C = gcc -c
C_FLAGS = -O3 -msse -msse2 -msse3 -msse4.2 -fPIC -Wall
CXX = g++ -c
CXX_FLAGS = -O3 -std=c++11 -msse -msse2 -msse3 -msse4.2 -fPIC -Wall
CUDA = nvcc -c
CUDA_FLAGS = -x cu -Xcompiler -fPIC -arch=sm_30 -std=c++11 --expt-extended-lambda


PYRENDER_DEPENDENCIES = setup.py \
  render/render_cpu.cpp.o \
  render/stdlib_cuda_dummy.cpp.o \
  render/render_gpu_dummy.cpp.o

PYRENDER_DEPENDENCIES += render/render_gpu.cu.o \
    render/stdlib_cuda.cu.o

all: pyrender

clean:
	rm render/*.o

pyrender: $(PYRENDER_DEPENDENCIES)
	cd pyrender; \
	python setup.py build_ext --inplace

%.c.o: %.c
	$(C) $(C_FLAGS) -o $@ $< $(INCLUDE_DIR)
%.cpp.o: %.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $< $(INCLUDE_DIR)
%.cu.o: %.cu
	$(CUDA) -o $@ $< $(CUDA_FLAGS) $(INCLUDE_DIR)
