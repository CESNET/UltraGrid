NVCC=@CUDA_INSTALL_PATH@/bin/nvcc
CUDA_MEMCHECK=@CUDA_INSTALL_PATH@/bin/cuda-memcheck

TESTS=opengl_interop
check_PROGRAMS = opengl_interop

NVCC_CFLAGS = @COMMON_CFLAGS@ $(DEFAULT_INCLUDES) $(INCLUDES) \
	-gencode arch=compute_20,code=sm_20 \
	-gencode arch=compute_11,code=sm_11

opengl_interop_SOURCES = view.c main.c
opengl_interop_DEPENDENCIES = image.cu.o
opengl_interop_LDADD = image.cu.o ../../libgpujpeg.la
opengl_interop_CFLAGS = @COMMON_CFLAGS@ $(DEFAULT_INCLUDES) $(INCLUDES)
opengl_interop_LDFLAGS = -L@CUDA_INSTALL_LIB@ -lcuda -lcudart -lglut -lGLU -lGLEW -lGL -lpthread 

all-local: tests
tests: check-TESTS

# Build gpujpeg library
../../libgpujpeg.la:
	@cd ../../; ./config.status --recheck --enable-opengl && make
    
# Set suffix for CUDA files
SUFFIXES=.cu

# Pattern rule for compiling CUDA files
%.cu.o: %.cu
	$(NVCC) $(NVCC_CFLAGS) -c $< -o $@;
    

