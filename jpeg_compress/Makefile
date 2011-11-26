# Copyright (c) 2011, Martin Srom
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#    # Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#    # Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Target executable
TARGET := jpeg_compress
# C files
CFILES := \
    main.c \
    jpeg_common.c \
    jpeg_encoder.c \
    jpeg_decoder.c \
    jpeg_table.c \
    jpeg_huffman_cpu_encoder.c \
    jpeg_huffman_cpu_decoder.c \
    jpeg_writer.c \
    jpeg_reader.c
# CUDA files
CUFILES := \
    jpeg_preprocessor.cu \
    jpeg_huffman_gpu_encoder.cu \
    jpeg_huffman_gpu_decoder.cu

# CUDA install path
CUDA_INSTALL_PATH ?= /usr/local/cuda

# Compilers
CC := gcc
LINK := g++ -fPIC
NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc

# Common flags
COMMONFLAGS += -I. -I$(CUDA_INSTALL_PATH)/include -O2
# C flags
CFLAGS += $(COMMONFLAGS) -std=c99
# CUDA flags
NVCCFLAGS += $(COMMONFLAGS) \
	-gencode arch=compute_20,code=sm_20 \
	-gencode arch=compute_11,code=sm_11
# Linker flags
LDFLAGS +=

# Do 32bit vs. 64bit setup
LBITS := $(shell getconf LONG_BIT)
ifeq ($(LBITS),64)
    # 64bit
    LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lnpp
else
    # 32bit
    LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib -lcudart
endif

# Build
build: $(TARGET)

# Clean
clean:
	rm -f *.o $(TARGET)

# Lists of object files
COBJS=$(CFILES:.c=.c.o)
CUOBJS=$(CUFILES:.cu=.cu.o)

# Build target
$(TARGET): $(COBJS) $(CUOBJS)
	$(LINK) $(COBJS) $(CUOBJS) $(LDFLAGS) -o $(TARGET);    

# Set suffix for CUDA files
.SUFFIXES: .cu

# Pattern rule for compiling C files
%.c.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

# Pattern rule for compiling CUDA files
%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@;

# Set file dependencies
main.c.o: main.c
jpeg_common.c.o: jpeg_common.c jpeg_common.h
jpeg_encoder.c.o: jpeg_encoder.c jpeg_encoder.h
jpeg_decoder.c.o: jpeg_decoder.c jpeg_decoder.h
jpeg_table.c.o: jpeg_table.c jpeg_table.h
jpeg_preprocessor.cu.o: jpeg_preprocessor.cu jpeg_preprocessor.h
jpeg_huffman_cpu_encoder.c.o: jpeg_huffman_cpu_encoder.c jpeg_huffman_cpu_encoder.h
jpeg_huffman_gpu_encoder.cu.o: jpeg_huffman_gpu_encoder.cu jpeg_huffman_gpu_encoder.h
jpeg_huffman_cpu_decoder.c.o: jpeg_huffman_cpu_decoder.c jpeg_huffman_cpu_decoder.h
jpeg_huffman_gpu_decoder.cu.o: jpeg_huffman_gpu_decoder.cu jpeg_huffman_gpu_decoder.h
jpeg_writer.c.o: jpeg_writer.c jpeg_writer.h
jpeg_reader.c.o: jpeg_reader.c jpeg_reader.h
